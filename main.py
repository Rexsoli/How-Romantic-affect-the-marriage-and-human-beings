import random
import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict, Optional, Set


# ============================================================
# Configuration
# ============================================================

@dataclass(slots=True)
class SimulationConfig:
    """模型参数配置；这些值通常在一次仿真过程中保持不变。"""

    # -------- Simulation scale --------
    initial_couples: int = 100                    # 初始家庭数量
    initial_property: float = 100.0               # 每个初始家庭的初始财富
    years: int = 150                              # 仿真总年份
    seed: int = 42                                # 随机种子

    # -------- Reproduction --------
    birth_prob_per_year: float = 0.9              # 有生育意愿后的年度实际出生概率
    initial_B_choices: List[float] = field(       # 创始个体 B 的候选值
        default_factory=lambda: [2.0, 2.5, 3.0, 3.5, 4.0]
    )

    # -------- Marriage module --------
    marriage_mode: str = "fermi"                  # 婚配概率模式："fermi" 或 "normal"
    fermi_kT: float = 0.3                         # Fermi 婚配公式中的对数财富比敏感度
    normal_sigma: float = 0.5                     # 正态婚配公式中的对数财富比尺度
    wealth_floor: float = 1.0                     # 最低财富感知尺度，用于计算对数财富比

    # -------- Genetic traits --------
    initial_D_choices: List[float] = field(       # 创始个体 D 的候选值，默认从 0 到 1 均匀覆盖
        default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0]
    )
    D_mutation_scale: float = 0.05                # D 遗传突变幅度
    B_mutation_scale: float = 1                # B 遗传突变幅度
    B_min: float = 0.0                            # B 下界
    B_max: float = 8.0                            # B 上界
    B_father_base_weight: float = 0.3             # B 遗传中父亲的基础权重
    D_father_base_weight: float = 0.5             # D 遗传/家庭 D 中父亲的基础权重
    parent_wealth_log_strength: float = 1.0       # 父母结婚财富对数比对遗传权重的影响强度
    parent_wealth_floor: float = 1.0              # 父母财富对数比的最低财富尺度

    # -------- Kinship / inheritance --------
    marriage_kinship_max_depth: int = 2           # 禁止婚配的血缘搜索深度
    inheritance_kinship_max_depth: int = 3        # 继承血亲搜索深度

    # -------- Economic module --------
    annual_wage_per_adult: float = 2.0            # 每个存活成年人的年工资
    asset_interest_rate: float = 0.03             # 家庭财富的年利息率
    child_base_living_cost: float = 4.0           # 每个未成年子女的基础年开支
    child_wealth_cost_rate: float = 0.015         # 子女随家庭财富增加的开支率
    adult_base_living_cost: float = 1.0           # 每个存活成年人的基础年开支
    adult_wealth_cost_rate: float = 0.0           # 成年人随家庭财富增加的开支率

# ============================================================
# Utility
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """把数值 x 限制在 [lo, hi] 区间内。"""
    return max(lo, min(hi, x))


# ============================================================
# Person / Couple
# ============================================================

@dataclass(slots=True)
class Person:
    """个体代理，保存人的年龄、财富、婚姻、生育和亲属信息。"""

    pid: int                         # 个体唯一编号
    sex: str                         # 性别，"M" 为男性，"F" 为女性
    age: int                         # 当前年龄，年度循环中每年增加 1
    wealth: float                    # 个人财富；已婚者主要财富进入家庭账户
    B: float                         # 生育倾向，决定家庭目标子女数
    D: float                         # 预留的可遗传沉默性状；当前只遗传和突变，不参与行为决策

    father_id: Optional[int] = None  # 父亲编号；初始虚拟祖先可为负数
    mother_id: Optional[int] = None  # 母亲编号；初始虚拟祖先可为负数

    married: bool = False            # 是否已经结婚；本模型不改嫁/再婚

    children_ids: List[int] = field(default_factory=list)  # 子女编号列表

    wealth_at_marriage: float = 0.0  # 结婚时个人财富，用于遗传 D 的父母权重


@dataclass(slots=True)
class DeadPersonRecord:
    """死亡个体的轻量亲属记录，只保留血缘搜索需要的信息。"""

    pid: int
    sex: str
    father_id: Optional[int]
    mother_id: Optional[int]
    children_ids: List[int] = field(default_factory=list)


@dataclass(slots=True)
class Couple:
    """家庭账户，婚后财产、生育和继承都围绕该对象运行。"""

    cid: int                         # 家庭唯一编号
    father_id: int                   # 丈夫/父亲编号
    mother_id: int                   # 妻子/母亲编号
    family_wealth: float             # 家庭共同账户财富
    family_B: float                  # 结婚时确定的家庭目标生育倾向
    family_D: float                  # 家庭财产分配时的男/女子女单人继承比

    children_ids: List[int] = field(default_factory=list)  # 家庭子女编号列表
    dissolved: bool = False          # 父母双亡后家庭账户关闭
    first_child_birth_year: Optional[int] = None  # 第一个孩子出生年份
    children_property_distributed: bool = False   # 孩子成年时的首次分配是否完成


# ============================================================
# Simulation
# ============================================================

class SocietySimulation:
    """社会演化仿真主类；config 保存模型参数，实例字段保存运行状态。"""

    def __init__(self, config: Optional[SimulationConfig] = None):
        """初始化配置、人口容器、编号计数器和创始家庭。"""
        self.config = config if config is not None else SimulationConfig()

        # 所有随机事件都使用实例级 RNG，保证整体可复现且避免 pair seed 锁死。
        self.rng = random.Random(self.config.seed)

        # people/couples 只保存当前活人和活跃家庭；死亡个体和关闭家庭移入档案表
        self.people: Dict[int, Person] = {}
        self.dead_people: Dict[int, DeadPersonRecord] = {}
        self.couples: Dict[int, Couple] = {}
        self.closed_couples: Dict[int, Couple] = {}
        # 只记录未关闭家庭中的成员，便于财富路由到家庭账户
        self.active_couple_by_pid: Dict[int, int] = {}
        # 婚配禁亲祖先集合缓存；人的祖先出生后不变，因此可以长期复用
        self.marriage_ancestor_cache: Dict[int, Set[int]] = {}

        # 下一个可用个体编号、家庭编号和当前年份
        self.next_pid = 1
        self.next_cid = 1
        self.year = 0
        # 无继承者财富先进入公共支票账户，下一年开始时再平均分配
        self.public_checking_account = 0.0
        # history 保存每年末的基础统计，便于检查人口、婚配、财富和公共账户是否异常
        self.history: List[Dict[str, float]] = []

        self._initialize_population()

    def _get_genealogy_record(self, pid: int):
        """按编号查找活人或死亡轻量亲属记录。"""
        person = self.people.get(pid)
        if person is not None:
            return person
        return self.dead_people.get(pid)

    # --------------------------------------------------------
    # Creation helpers
    # --------------------------------------------------------
    def _new_person(
        self,
        sex: str,
        age: int,
        wealth: float,
        B: float,
        D: float,
        father_id: int,
        mother_id: int,
    ) -> int:
        """创建一个新个体，写入 people，并返回该个体编号。"""
        # pid 是全局递增编号，避免与已有真实个体重复
        pid = self.next_pid
        self.next_pid += 1

        # B 和 D 在创建时立即截断到合法范围
        self.people[pid] = Person(
            pid=pid,
            sex=sex,
            age=age,
            wealth=wealth,
            B=clamp(B, self.config.B_min, self.config.B_max),
            D=clamp(D, 0.05, 0.95),
            father_id=father_id,
            mother_id=mother_id,
        )
        return pid

    def _new_couple(
        self,
        father_id: int,
        mother_id: int,
        family_wealth: float,
        family_B: float,
        family_D: float,
    ) -> int:
        """创建新家庭账户，并把夫妻登记到 active_couple_by_pid。"""
        # cid 是家庭账户编号，独立于个人编号
        cid = self.next_cid
        self.next_cid += 1

        # family_B 在婚姻形成时固定
        self.couples[cid] = Couple(
            cid=cid,
            father_id=father_id,
            mother_id=mother_id,
            family_wealth=family_wealth,
            family_B=family_B,
            family_D=family_D,
        )

        # married 一旦变为 True，本模型不会再改回未婚
        self.people[father_id].married = True
        self.people[mother_id].married = True
        self.active_couple_by_pid[father_id] = cid
        self.active_couple_by_pid[mother_id] = cid

        return cid

    def _initialize_population(self):
        """初始化创始家庭和唯一的虚拟祖先，避免创始人被误判为近亲。"""
        initial_D_choices = self.config.initial_D_choices
        for couple_idx in range(self.config.initial_couples):
            # 创始父母的 B 从给定候选集合中抽取，D 按候选集合循环分配，保证初始分布均匀
            father_B = self.rng.choice(self.config.initial_B_choices)
            mother_B = self.rng.choice(self.config.initial_B_choices)

            father_D = initial_D_choices[(2 * couple_idx) % len(initial_D_choices)]
            mother_D = initial_D_choices[(2 * couple_idx + 1) % len(initial_D_choices)]
            # 每对创始夫妻使用 4 个互不重复的负数虚拟父母编号，只用于亲属判断
            base_virtual_id = -4 * couple_idx - 1
            virtual_ids = tuple(base_virtual_id - i for i in range(4))
            father_parent_ids = virtual_ids[0:2]
            mother_parent_ids = virtual_ids[2:4]

            # 创始家庭的财富直接放入家庭账户，个人财富为 0
            father_id = self._new_person(
                sex="M",
                age=18,
                wealth=0.0,
                B=father_B,
                D=father_D,
                father_id=father_parent_ids[0],
                mother_id=father_parent_ids[1],
            )
            mother_id = self._new_person(
                sex="F",
                age=18,
                wealth=0.0,
                B=mother_B,
                D=mother_D,
                father_id=mother_parent_ids[0],
                mother_id=mother_parent_ids[1],
            )

            self.people[father_id].wealth_at_marriage = self.config.initial_property / 2.0
            self.people[mother_id].wealth_at_marriage = self.config.initial_property / 2.0

            # 创始家庭的 family_B / family_D 在家庭形成时固定
            family_B, family_D = self._family_traits(
                self.people[father_id],
                self.people[mother_id],
            )

            self._new_couple(
                father_id=father_id,
                mother_id=mother_id,
                family_wealth=self.config.initial_property,
                family_B=family_B,
                family_D=family_D,
            )

    # --------------------------------------------------------
    # Dynamic parental influence
    # --------------------------------------------------------
    def _parental_weight(self, father: Person, mother: Person, base_father_weight: float) -> float:
        """根据外部基础权重和父母结婚财富对数比，计算父亲遗传影响权重。"""
        base = clamp(base_father_weight, 1e-9, 1.0 - 1e-9)
        cfg = self.config
        wealth_x = math.log(
            (father.wealth_at_marriage + cfg.parent_wealth_floor)
            / (mother.wealth_at_marriage + cfg.parent_wealth_floor)
        )
        logit = math.log(base / (1.0 - base)) + cfg.parent_wealth_log_strength * wealth_x

        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1.0 + exp_logit)

    def _parental_trait(
        self,
        father: Person,
        mother: Person,
        father_value: float,
        mother_value: float,
        base_father_weight: float,
        lo: float,
        hi: float,
        mutation_scale: float = 0.0,
    ) -> float:
        """按父母权重混合一个遗传性状，可选加入突变。"""
        w_f = self._parental_weight(father, mother, base_father_weight)
        value = w_f * father_value + (1.0 - w_f) * mother_value
        if mutation_scale > 0:
            value += self.rng.uniform(-mutation_scale, mutation_scale)
        return clamp(value, lo, hi)

    def _family_traits(self, father: Person, mother: Person) -> tuple[float, float]:
        """生成新家庭固定的 family_B 和 family_D；不加入突变。"""
        cfg = self.config
        return (
            self._parental_trait(
                father, mother, father.B, mother.B,
                cfg.B_father_base_weight, cfg.B_min, cfg.B_max,
            ),
            self._parental_trait(
                father, mother, father.D, mother.D,
                cfg.D_father_base_weight, 0.0, 1.0,
            ),
        )

    def _child_traits(self, father: Person, mother: Person) -> tuple[float, float]:
        """生成子代 B 和 D；二者都按父母权重继承并加入突变。"""
        cfg = self.config
        return (
            self._parental_trait(
                father, mother, father.B, mother.B,
                cfg.B_father_base_weight, cfg.B_min, cfg.B_max,
                cfg.B_mutation_scale,
            ),
            self._parental_trait(
                father, mother, father.D, mother.D,
                cfg.D_father_base_weight, 0.0, 1.0,
                cfg.D_mutation_scale,
            ),
        )

    # --------------------------------------------------------
    # Marriage restriction
    # --------------------------------------------------------
    def blood_related_within_marriage_limit(self, pid1: int, pid2: int) -> bool:
        """检查两人是否在婚配禁止血缘深度内存在共同祖先。"""
        a1 = self._marriage_ancestor_set(pid1)
        a2 = self._marriage_ancestor_set(pid2)
        return not a1.isdisjoint(a2)

    def _marriage_ancestor_set(self, pid: int) -> Set[int]:
        """返回婚配禁亲深度内的祖先集合；结果会缓存。"""
        if pid in self.marriage_ancestor_cache:
            return self.marriage_ancestor_cache[pid]

        dist = self._ancestor_distances(
            pid,
            max_depth=self.config.marriage_kinship_max_depth,
        )
        ancestors = set(dist.keys())
        ancestors.discard(pid)
        self.marriage_ancestor_cache[pid] = ancestors
        return ancestors

    # --------------------------------------------------------
    # Kinship-distance based heir search
    # --------------------------------------------------------
    def _parent_ids(self, pid: int) -> List[int]:
        """返回某个个体的父母编号；虚拟祖先或不存在编号返回空列表。"""
        p = self._get_genealogy_record(pid)
        if p is None:
            return []

        # parents 只包含非 None 的父母编号
        parents = []
        if p.father_id is not None:
            parents.append(p.father_id)
        if p.mother_id is not None:
            parents.append(p.mother_id)
        return parents

    def _ancestor_distances(self, pid: int, max_depth: Optional[int] = None) -> Dict[int, int]:
        """向上搜索祖先，返回祖先编号到源个体的代际距离。"""
        if max_depth is None:
            max_depth = self.config.inheritance_kinship_max_depth

        # dist 记录每个已发现祖先的最短距离，源个体距离为 0
        dist = {pid: 0}
        # queue 用于广度优先搜索祖先图
        queue = [(pid, 0)]
        head = 0

        while head < len(queue):
            current, d = queue[head]
            head += 1

            if d >= max_depth:
                continue

            for par in self._parent_ids(current):
                nd = d + 1
                # 如果同一祖先可由多条路径到达，保留更短距离
                if par not in dist or nd < dist[par]:
                    dist[par] = nd
                    queue.append((par, nd))

        return dist

    def _nearest_living_blood_relatives(
        self,
        source_pids: List[int],
        exclude_pids: Optional[Set[int]] = None,
        max_depth: Optional[int] = None,
    ) -> List[int]:
        """从来源个体出发做局部亲属 BFS，寻找最近一层存活血亲。"""
        if max_depth is None:
            max_depth = self.config.inheritance_kinship_max_depth

        if exclude_pids is None:
            exclude_pids = set()

        excluded = set(source_pids) | set(exclude_pids)
        visited = set(source_pids)
        frontier = list(source_pids)

        for _ in range(max_depth):
            next_frontier = []
            heirs = []

            for pid in frontier:
                for nb in self._kin_neighbors(pid):
                    if nb in visited:
                        continue

                    visited.add(nb)
                    next_frontier.append(nb)

                    person = self.people.get(nb)
                    if person is not None and nb not in excluded:
                        heirs.append(nb)

            if heirs:
                return heirs

            frontier = next_frontier

        return []

    def _kin_neighbors(self, pid: int) -> List[int]:
        """返回一个人的直接亲属邻居：父母和子女。"""
        p = self._get_genealogy_record(pid)
        if p is None:
            return []

        neighbors = []

        if self._get_genealogy_record(p.father_id) is not None:
            neighbors.append(p.father_id)
        if self._get_genealogy_record(p.mother_id) is not None:
            neighbors.append(p.mother_id)

        for cid in p.children_ids:
            if self._get_genealogy_record(cid) is not None:
                neighbors.append(cid)

        return neighbors

    # --------------------------------------------------------
    # Wealth routing
    # --------------------------------------------------------
    def _active_couple_of_person(self, pid: int) -> Optional[Couple]:
        """返回个体所属的未关闭家庭；若不存在则返回 None。"""
        cid = self.active_couple_by_pid.get(pid)
        if cid is None:
            return None

        couple = self.couples.get(cid)
        if couple is None or couple.dissolved:
            # 若映射已经过期，顺手清理，避免后续财富路由错误
            self.active_couple_by_pid.pop(pid, None)
            return None

        return couple

    def _dissolve_couple_membership(self, couple: Couple):
        """关闭家庭账户时移除夫妻到 active couple 的映射。"""
        self.active_couple_by_pid.pop(couple.father_id, None)
        self.active_couple_by_pid.pop(couple.mother_id, None)

    def _receive_wealth(self, pid: int, amount: float):
        """给某人发放财富；若其属于未关闭家庭，则进入家庭账户。"""
        if amount <= 0:
            return

        # 已婚且家庭未关闭时，财富进入家庭账户；否则进入个人财富
        couple = self._active_couple_of_person(pid)
        if couple is not None:
            couple.family_wealth += amount
        elif pid in self.people:
            self.people[pid].wealth += amount

    def _transfer_to_public_checking_account(self, amount: float):
        """没有合适血亲继承者时，把财富转入社会公共支票账户。"""
        if amount <= 0:
            return
        self.public_checking_account += amount

    def _redistribute_public_checking_account(self):
        """每年开始时，把上一年累积的公共支票账户余额平分给所有存活者。"""
        if self.public_checking_account <= 0:
            return

        # living 是当前年份分配时仍存活的人
        living = list(self.people.keys())
        if not living:
            return

        share = self.public_checking_account / len(living)
        self.public_checking_account = 0.0
        for pid in living:
            self._receive_wealth(pid, share)

    # --------------------------------------------------------
    # Economic module
    # --------------------------------------------------------
    def _process_economics(self):
        """
        年度经济结算。

        账户逻辑：
        1. 家庭旧财富先产生资产收益；
        2. 所有存活成年人先获得个人工资和个人资产收益；
        3. 已婚者若属于未关闭家庭，则把个人账户余额搬入家庭账户，并清空个人账户；
        4. 家庭账户支付已婚成年人和未成年子女生活成本；
        5. 单身成年人支付个人生活成本。
        """
        cfg = self.config
        people = self.people
        couples = self.couples
        active_map = self.active_couple_by_pid
        r = cfg.asset_interest_rate
        wage = cfg.annual_wage_per_adult

        # --------------------------------------------------------
        # 1. 家庭旧财富产生资产收益
        # --------------------------------------------------------
        for couple in couples.values():
            couple.family_wealth *= (1.0 + r)

        # --------------------------------------------------------
        # 2. 所有成年个体先进入个人账户：工资 + 个人资产收益
        # --------------------------------------------------------
        for p in people.values():
            if p.age < 18:
                continue
            p.wealth += wage + r * p.wealth

        # --------------------------------------------------------
        # 3. 已婚者个人账户搬入家庭账户
        # --------------------------------------------------------
        for p in people.values():
            if not p.married or p.wealth <= 0:
                continue

            cid = active_map.get(p.pid)
            if cid is None:
                continue

            couple = couples.get(cid)
            if couple is not None:
                couple.family_wealth += p.wealth
                p.wealth = 0.0

        # --------------------------------------------------------
        # 4. 家庭支付已婚成年人和未成年子女生活成本
        # --------------------------------------------------------
        for couple in couples.values():
            W = couple.family_wealth

            num_adults = int(couple.father_id in people) + int(couple.mother_id in people)

            num_minors = 0
            for cid in couple.children_ids:
                child = people.get(cid)
                if child is not None and child.age < 18:
                    num_minors += 1

            adult_cost = cfg.adult_base_living_cost + cfg.adult_wealth_cost_rate * W
            child_cost = cfg.child_base_living_cost + cfg.child_wealth_cost_rate * W

            total_cost = num_minors * child_cost + num_adults * adult_cost
            couple.family_wealth = max(0.0, W - total_cost)

        # --------------------------------------------------------
        # 5. 单身成年人支付个人生活成本
        # --------------------------------------------------------
        for p in people.values():
            if p.married or p.age < 18:
                continue

            W = p.wealth
            adult_cost = cfg.adult_base_living_cost + cfg.adult_wealth_cost_rate * W
            p.wealth = max(0.0, W - adult_cost)

        # --------------------------------------------------------
        # 6. 父母双亡且持有个人遗产的未成年人，也会消耗个人财富
        # --------------------------------------------------------
        for p in people.values():
            if p.married or p.age >= 18 or p.wealth <= 0:
                continue

            # 仍有父母存活时，未成年子女的生活成本由家庭账户承担
            if p.father_id in people or p.mother_id in people:
                continue

            W = p.wealth
            child_cost = cfg.child_base_living_cost + cfg.child_wealth_cost_rate * W
            p.wealth = max(0.0, W - child_cost)

    def _can_afford_additional_child(self, couple: Couple) -> bool:
        """
        检查家庭在当前财富水平下是否能承担多一个孩子。

        如果新增孩子后的年度开支超过收入，则当前财富必须能覆盖缺口。
        """
        cfg = self.config
        people = self.people

        # 按 _process_economics 的同一顺序预测：家庭资产收益 + 存活父母个人收入并入家庭。
        # 如果以后修改年度经济流程，这里的生育承受能力预测也要同步修改。
        W = couple.family_wealth * (1.0 + cfg.asset_interest_rate)
        num_adults = 0
        for pid in (couple.father_id, couple.mother_id):
            p = people.get(pid)
            if p is None:
                continue
            num_adults += 1
            W += p.wealth + cfg.annual_wage_per_adult + cfg.asset_interest_rate * p.wealth

        num_minors = 0
        for cid in couple.children_ids:
            child = people.get(cid)
            if child is not None and child.age < 18:
                num_minors += 1

        adult_cost = cfg.adult_base_living_cost + cfg.adult_wealth_cost_rate * W
        child_cost = cfg.child_base_living_cost + cfg.child_wealth_cost_rate * W
        total_cost = num_adults * adult_cost + (num_minors + 1) * child_cost

        return W - total_cost >= 0.0

    # --------------------------------------------------------
    # Birth logic
    # --------------------------------------------------------
    def _has_additional_child_desire(self, couple: Couple) -> bool:
        """判断家庭是否仍有追加子女意愿，不检查财富承受能力。"""
        family_B = couple.family_B
        living_children = 0
        for cid in couple.children_ids:
            if cid in self.people:
                living_children += 1

        base_n = math.floor(family_B)
        frac = family_B - base_n

        if living_children < base_n:
            return True
        if living_children == base_n:
            return frac > 0.0

        return False

    def _should_try_birth(self, couple: Couple) -> bool:
        """根据经济承受能力和 family_B 判断家庭本年是否想尝试生育。"""
        # 先检查经济承受能力，不能承受时不进入生育意愿判断
        if not self._can_afford_additional_child(couple):
            return False

        # family_B 的整数部分是目标存活子女数，fractional part 是额外存活子女目标概率
        family_B = couple.family_B
        n = 0
        for cid in couple.children_ids:
            if cid in self.people:
                n += 1

        base_n = math.floor(family_B)
        frac = family_B - base_n

        if n < base_n:
            return True
        if n == base_n:
            return self.rng.random() < frac

        return False

    def _give_birth(self, couple: Couple):
        """给家庭新增一个孩子，并登记父母、祖先和家庭子女关系。"""
        father = self.people[couple.father_id]
        mother = self.people[couple.mother_id]

        # 新生儿性别随机，B/D 按父母遗传规则生成
        child_sex = "M" if self.rng.random() < 0.5 else "F"
        child_B, child_D = self._child_traits(father, mother)
        # 新生儿出生时年龄为 0，个人财富为 0
        child_id = self._new_person(
            sex=child_sex,
            age=0,
            wealth=0.0,
            B=child_B,
            D=child_D,
            father_id=father.pid,
            mother_id=mother.pid,
        )

        father.children_ids.append(child_id)
        mother.children_ids.append(child_id)
        couple.children_ids.append(child_id)

        # 第一个孩子出生年份用于 18 年后的首次财产分配
        if couple.first_child_birth_year is None:
            couple.first_child_birth_year = self.year

    # --------------------------------------------------------
    # Inheritance
    # --------------------------------------------------------
    def _transfer_single_wealth_to_nearest_relatives(self, pid: int):
        """未婚者死亡后，把个人财富转给最近血亲。"""
        person = self.people[pid]

        if person.wealth <= 0:
            return

        heirs = self._nearest_living_blood_relatives(
            source_pids=[pid],
            exclude_pids={pid},
        )

        if heirs:
            # 最近血亲可能有多个，财富平均分配
            share = person.wealth / len(heirs)
            for hid in heirs:
                self._receive_wealth(hid, share)
        else:
            # 没有血亲时进入社会公共支票账户
            self._transfer_to_public_checking_account(person.wealth)

        person.wealth = 0.0

    def _redistribute_childless_family_wealth(self, couple: Couple):
        """无子女家庭关闭时，把家庭财富转给父母双方最近血亲。"""
        heirs = self._nearest_living_blood_relatives(
            source_pids=[couple.father_id, couple.mother_id],
            exclude_pids={couple.father_id, couple.mother_id},
        )

        if heirs:
            # 多个同距离血亲平分家庭财富
            share = couple.family_wealth / len(heirs)
            for hid in heirs:
                self._receive_wealth(hid, share)
        else:
            # 没有血亲时进入社会公共支票账户
            self._transfer_to_public_checking_account(couple.family_wealth)

    # --------------------------------------------------------
    # Property distribution
    # --------------------------------------------------------
    def _distribute_current_family_wealth(
        self,
        couple: Couple,
        allow_childless_fallback: bool,
    ) -> bool:
        """分配家庭当前账户财富；成功分配返回 True。"""
        if couple.family_wealth <= 0:
            return False

        if len(couple.children_ids) == 0:
            # 没有子女时，只有最终清算场景允许走血亲/公共支票账户规则
            if not allow_childless_fallback:
                return False

            self._redistribute_childless_family_wealth(couple)
            couple.family_wealth = 0.0
            return True

        living_children = [
            self.people[cid]
            for cid in couple.children_ids
            if cid in self.people
        ]
        if not living_children:
            if not allow_childless_fallback:
                return False

            self._redistribute_childless_family_wealth(couple)
            couple.family_wealth = 0.0
            return True

        male_children = [child for child in living_children if child.sex == "M"]
        female_children = [child for child in living_children if child.sex == "F"]

        W = couple.family_wealth
        D = clamp(couple.family_D, 1e-9, 1.0 - 1e-9)
        num_males = len(male_children)
        num_females = len(female_children)

        # family_D 是 [0, 1] 内的继承倾向：0.5 表示男女单人平分，
        # 大于 0.5 表示每个男性子女更多，小于 0.5 表示每个女性子女更多。
        if num_males > 0 and num_females > 0:
            denominator = D * num_males + (1.0 - D) * num_females
            male_share = D * W / denominator
            female_share = (1.0 - D) * W / denominator
            for child in male_children:
                self._receive_wealth(child.pid, male_share)
            for child in female_children:
                self._receive_wealth(child.pid, female_share)
        elif num_males > 0:
            share = W / num_males
            for child in male_children:
                self._receive_wealth(child.pid, share)
        else:
            share = W / num_females
            for child in female_children:
                self._receive_wealth(child.pid, share)

        couple.family_wealth = 0.0
        return True

    def _distribute_family_property(self, couple: Couple):
        """父母双方死亡后关闭家庭账户，并分配剩余家庭财产。"""
        self._distribute_current_family_wealth(
            couple,
            allow_childless_fallback=True,
        )
        couple.dissolved = True
        self._dissolve_couple_membership(couple)
        #self.closed_couples[couple.cid] = couple
        self.couples.pop(couple.cid, None)

    # --------------------------------------------------------
    # Marriage
    # --------------------------------------------------------
    def _marriage_letter_weight(self, male_wealth: float, female_wealth: float) -> float:
        """计算女性向某个男性投递情书的抽样权重。"""
        cfg = self.config
        # wealth_floor 既避免 log(0)，也表示低财富人群的最低财富感知尺度
        x = math.log((male_wealth + cfg.wealth_floor) / (female_wealth + cfg.wealth_floor))
        if cfg.marriage_mode == "normal":
            z = x / (cfg.normal_sigma * math.sqrt(2.0))
            return clamp(0.5 * (1.0 + math.erf(z)), 0.0, 1.0)

        exponent = clamp(-x / cfg.fermi_kT, -700.0, 700.0)
        return 1.0 / (1.0 + math.exp(exponent))

    def _female_choose_males(self, female: Person, males: List[Person]) -> List[Person]:
        """女性对所有男性独立投信；先按财富概率筛选，再检查血缘。"""

        accepted: List[Person] = []
        cfg = self.config

        for male in males:
            # 根据男女财富计算投信概率
            p = self._marriage_letter_weight(male.wealth, female.wealth)

            # 按年份和双方编号生成稳定随机数，避免循环顺序变化牵动婚配结果
            key = (
                cfg.seed
                + 0x9E3779B97F4A7C15 * (self.year + 1)
                + 0xBF58476D1CE4E5B9 * (male.pid + 1)
                + 0x94D049BB133111EB * (female.pid + 1)
            ) & 0xFFFFFFFFFFFFFFFF
            key = (key ^ (key >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
            key = (key ^ (key >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
            key = key ^ (key >> 31)
            u = (key >> 11) / float(1 << 53)

            # 先用稳定随机数过滤掉大部分无效 pair，再做较贵的血缘检查
            if u >= p:
                continue

            if self.blood_related_within_marriage_limit(male.pid, female.pid):
                continue

            accepted.append(male)

        if not accepted:
            return []

        return accepted

    def _marry(self, male_id: int, female_id: int):
        """执行结婚：合并个人财富，创建家庭账户，固定 family_B。"""
        male = self.people[male_id]
        female = self.people[female_id]

        # 结婚时财富用于 D 的父母权重计算
        male.wealth_at_marriage = male.wealth
        female.wealth_at_marriage = female.wealth

        # 婚姻形成时，夫妻已有个人财富一次性转入家庭账户；后续年度工资是新增收入。
        family_wealth = male.wealth + female.wealth
        family_B, family_D = self._family_traits(male, female)

        male.wealth = 0.0
        female.wealth = 0.0

        self._new_couple(
            father_id=male_id,
            mother_id=female_id,
            family_wealth=family_wealth,
            family_B=family_B,
            family_D=family_D,
        )

    def _perform_marriages(self) -> int:
        """运行婚配市场：女性逐对概率投递情书，男性选择最富候选者。"""
        males: List[Person] = []
        females: List[Person] = []

        for p in self.people.values():
            if p.married:
                continue

            if p.sex == "M":
                if p.age >= 18:
                    males.append(p)
            elif 18 <= p.age <= 35:
                females.append(p)

        # 男性按财富从高到低处理，符合财富优先的婚配顺序
        males.sort(key=lambda x: x.wealth, reverse=True)
        females.sort(key=lambda x: x.wealth, reverse=True)
        # incoming_proposals[male_id] 保存给该男性投递情书的女性
        incoming_proposals: Dict[int, List[Person]] = defaultdict(list)
        marriages = 0

        for female in females:
            for male in self._female_choose_males(female, males):
                incoming_proposals[male.pid].append(female)

        # matched_females 防止同一女性在同一年嫁给多个男性
        matched_females: Set[int] = set()
        for male in males:
            candidates = [
                female for female in incoming_proposals.get(male.pid, [])
                if female.pid not in matched_females
            ]
            if not candidates:
                continue

            # 男性从收到情书的女性中选择财富最高者
            chosen = max(candidates, key=lambda f: f.wealth)
            self._marry(male.pid, chosen.pid)
            matched_females.add(chosen.pid)
            marriages += 1

        return marriages

    # --------------------------------------------------------
    # Mortality
    # --------------------------------------------------------
    def _kill_person(self, pid: int):
        """将个体标记为死亡；若其未婚，则立即转移个人财富。"""
        p = self.people[pid]

        if not p.married:
            self._transfer_single_wealth_to_nearest_relatives(pid)
        else:
            couple = self._active_couple_of_person(pid)
            if couple is not None:
                couple.family_wealth += p.wealth
                p.wealth = 0.0
            self.active_couple_by_pid.pop(pid, None)

        self._compress_dead_person(pid)

    def _compress_dead_person(self, pid: int):
        """把死亡个体从完整 Person 压缩为轻量死亡亲属记录。"""
        p = self.people.get(pid)
        if p is None:
            return

        self.dead_people[pid] = DeadPersonRecord(
            pid=p.pid,
            sex=p.sex,
            father_id=p.father_id,
            mother_id=p.mother_id,
            children_ids=list(p.children_ids),
        )
        self.people.pop(pid, None)

    def _mortality_probability(self, age: int) -> float:
        """根据年龄返回年度死亡概率。"""
        if age >= 50:
            return 1.0
        if age <= 1:
            return 0.1
        if age <= 3:
            return 0.05
        if age <= 6:
            return 0.02
        if age <= 29:
            return 0.01

        # 30 到 49 岁死亡概率线性上升，50 岁达到确定死亡
        progress = (age - 30) / 20.0
        gamma = 3.0
        return 0.01 + 0.99 * (progress ** gamma)

    def _apply_mortality(self) -> int:
        """对所有存活个体抽取死亡事件，返回本年死亡人数。"""
        deaths = 0
        for pid, p in list(self.people.items()):
            if self.rng.random() < self._mortality_probability(p.age):
                self._kill_person(pid)
                deaths += 1
        return deaths

    # --------------------------------------------------------
    # Couple cleanup
    # --------------------------------------------------------
    def _cleanup_dead_couples(self):
        """检查所有未关闭家庭；只有父母双亡才关闭并清算家庭账户。"""
        people = self.people
        for c in list(self.couples.values()):
            if c.father_id not in people and c.mother_id not in people:
                self._distribute_family_property(c)

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------
    def _record_history(
        self,
        births: int,
        deaths: int,
        marriages: int,
        wealth_limited_child_desire_couples: int,
    ):
        """记录本年末的基础统计指标，用于检查仿真结果是否合理。"""
        alive = list(self.people.values())
        active_couples = list(self.couples.values())
        single_wealth = sum(p.wealth for p in alive if not p.married)
        family_wealth = sum(c.family_wealth for c in active_couples)
        total_wealth = single_wealth + family_wealth + self.public_checking_account
        couple_wealths = [
            c.family_wealth
            for c in active_couples
            if not c.children_property_distributed
        ]
        print(f"year:{self.year}, "
              f"Alive: {len(alive)}, "
              f"birth: {births}, "
              f"birth/death: {births/deaths*100:.1f}%, "
              f"wealth: {total_wealth:.2f},"
              f"Average_wealth: {total_wealth/len(alive):.2f}, ")
        print(
            "wealth_limited_child_desire_couples: "
            f"{wealth_limited_child_desire_couples}"
        )
        self.history.append({
            "year": self.year,
            "alive_population": len(alive),
            "num_couples_active": len(active_couples),
            "births": births,
            "deaths": deaths,
            "marriages": marriages,
            "wealth_limited_child_desire_couples": wealth_limited_child_desire_couples,
            "couple_wealths": couple_wealths,
            "total_wealth": total_wealth,
            "mean_wealth": total_wealth / len(alive) if alive else 0.0,
            "public_checking_account": self.public_checking_account,
        })

    # --------------------------------------------------------
    # Main evolution loop
    # --------------------------------------------------------
    def step(self):
        """推进仿真一年，按固定顺序执行公共账户、年龄、死亡、经济、生育和婚配。"""
        self.year += 1

        # 0. 上一年进入公共支票账户的财富，先平分给当前存活人口
        self._redistribute_public_checking_account()

        # 1. 所有存活个体年龄增加 1；女性生育状态由年龄动态判断
        for p in self.people.values():
            p.age += 1

        # 2. 根据年龄死亡概率抽取死亡
        deaths = self._apply_mortality()

        # 3. 检查父母双亡家庭并清算；死亡者个人财富已在 _kill_person 中并入家庭
        self._cleanup_dead_couples()

        # 4. 生育前先结算家庭和单身成年人的年度经济
        self._process_economics()

        # 5. 处理孩子成年首次分配和本年生育
        births = 0
        wealth_limited_child_desire_couples = 0
        for couple in list(self.couples.values()):
            if (
                couple.first_child_birth_year is not None
                and self.year - couple.first_child_birth_year >= 18
                and not couple.children_property_distributed
            ):
                # 第一个孩子出生满 18 年时，只处理一次成年分配；无钱或无存活子女也标记完成
                self._distribute_current_family_wealth(
                    couple,
                    allow_childless_fallback=False,
                )
                couple.children_property_distributed = True

            mother = self.people.get(couple.mother_id)
            can_reproduce = (
                couple.father_id in self.people
                and mother is not None
                and 18 <= mother.age <= 35
                and (
                    couple.first_child_birth_year is None
                    or self.year - couple.first_child_birth_year < 18
                )
            )

            if (
                can_reproduce
                and self._has_additional_child_desire(couple)
                and not self._can_afford_additional_child(couple)
            ):
                wealth_limited_child_desire_couples += 1

            # 家庭有生育资格、有生育意愿且通过年度随机抽取时才出生
            if can_reproduce and self._should_try_birth(couple):
                if self.rng.random() < self.config.birth_prob_per_year:
                    self._give_birth(couple)
                    births += 1

        # 6. 本年婚配市场
        marriages = self._perform_marriages()

        # 7. 年末记录基础统计
        self._record_history(
            births=births,
            deaths=deaths,
            marriages=marriages,
            wealth_limited_child_desire_couples=wealth_limited_child_desire_couples,
        )

    def run(self):
        """连续运行指定年份数。"""
        for _ in range(self.config.years):
            self.step()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # 程序入口处集中设置本次仿真的模型参数。
    config = SimulationConfig(
        # -------- Simulation scale --------
        initial_couples=100,
        initial_property=100,
        years=250,
        seed=123,

        # -------- Reproduction --------
        birth_prob_per_year=0.5,
        initial_B_choices=[1, 2, 3, 4, 5, 6],

        # -------- Marriage module --------
        # "fermi" or "normal"
        marriage_mode="fermi",
        fermi_kT=0.00005,
        normal_sigma=0.15,
        wealth_floor=3,

        # -------- Genetic traits --------
        initial_D_choices=[0.05, 0.25, 0.5, 0.75, 0.95],
        D_mutation_scale=0.1,
        B_mutation_scale=0.5,
        B_min=1.0,
        B_max=6.0,
        B_father_base_weight=0.3,
        D_father_base_weight=0.3,
        parent_wealth_log_strength=1.0,
        parent_wealth_floor=3.0,

        # -------- Kinship / inheritance --------
        marriage_kinship_max_depth=2,
        inheritance_kinship_max_depth=3,
        # -------- Economic module --------
        annual_wage_per_adult=3.0,
        asset_interest_rate=0.05,
        child_base_living_cost=1.0,
        child_wealth_cost_rate=0.01,
        adult_base_living_cost=2.0,
        adult_wealth_cost_rate=0.02,
    )

    # Main.py 不做记录和展示，只推进核心仿真状态
    sim = SocietySimulation(config)
    sim.run()

    import matplotlib.pyplot as plt
