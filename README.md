# How-Romantic and Family Property Distribution and Population Evolution Simulation

## 1. Model Objects

The program runs a yearly agent-based simulation built around individuals and family households.

- `Person` represents an individual. It stores sex, age, personal wealth, fertility tendency `B`, inheritance tendency `D`, parents, marriage status, and children.
- `DeadPersonRecord` is a lightweight kinship record for a deceased individual. It keeps only the person id, sex, parent ids, and child ids, and is used for kinship searches.
- `Couple` represents a family account. It stores the father, mother, family wealth, household fertility target `family_B`, household inheritance ratio `family_D`, and children.
- `SimulationConfig` stores the parameter settings for one simulation run.
- `SocietySimulation` handles initialization, yearly evolution, marriage, fertility, death, and property distribution.

Once a marriage forms, individuals do not divorce or remarry. A family is not dissolved when only one parent dies. The family account is closed only after both parents have died, and then the final property distribution is performed.

In the runtime state, `people` stores only currently living people. Deceased individuals are compressed from full `Person` objects into `DeadPersonRecord` objects and moved into `dead_people`. Similarly, `couples` stores only active family accounts that have not been closed. Yearly economics, marriage, death, and public-account redistribution scan only living people or active families. The deceased archive is read only when kinship checks or inheritance searches need historical family links.

Founder `D` values are no longer sampled randomly. They are assigned by cycling through `initial_D_choices`. The default candidate values are `[0.0, 0.25, 0.5, 0.75, 1.0]`, so the initial generation covers different inheritance-system tendencies from 0 to 1.

## 2. Age and Fertility

- Women are fertile only when `18 <= age <= 35`.
- Men can enter the marriage market after `age >= 18` and do not have a 35-year upper bound.
- Unmarried adults continue to go through single-person economic settlement. Men older than 35 can still enter the marriage market, while women older than 35 no longer enter the marriage or fertility window.
- Household fertility is jointly constrained by the mother's age, whether the father is alive, whether the family has been closed, the 18-year fertility window, household target `family_B`, the yearly birth probability, and economic affordability. In the current model, `family_B` means the target number of living children, not cumulative births.

## 3. Marriage Rules

Marriage happens only between unmarried individuals.

- A female candidate must be alive, unmarried, and between ages 18 and 35.
- A male candidate must be alive, unmarried, and at least 18 years old.
- Close-kin marriage restrictions are controlled by `marriage_kinship_max_depth`; the current main program sets this to 2.
- For each eligible man, each eligible woman independently decides whether to send a letter according to a wealth-ratio probability. Each man then marries the wealthiest woman among those who sent him a letter.
- The random value for a letter is generated from `SimulationConfig.seed`, the current year, and the two person ids. This keeps the full simulation reproducible, prevents the same male-female pair from being locked by one fixed random threshold across years, and avoids changing unrelated pair outcomes when the marriage loop order is adjusted.

Marriage probability is controlled by `marriage_mode`. The current program keeps two modes: `fermi` and `normal`.

Marriage probability does not use the absolute wealth difference directly. It uses the logarithmic wealth ratio:

```text
x = log((male_wealth + wealth_floor) / (female_wealth + wealth_floor))
```

Here `wealth_floor` is the minimum perceived wealth scale. It avoids division by zero and `log(0)`, and it also represents a minimum effective wealth baseline for low-wealth people in the marriage market. A larger `wealth_floor` makes small wealth differences among low-wealth people less sensitive. This form makes marriage probability respond to relative wealth differences rather than absolute wealth differences.

When letters are actually generated, the program first calculates the wealth probability using either the `fermi` or `normal` formula and uses a random draw to filter out most male-female pairs. Only after the random draw passes does it check whether the two people fall within the prohibited kinship range. Ancestor sets for marriage kinship checks are cached to avoid repeatedly searching the same person's ancestors inside the nested marriage loops.

### 3.1 `fermi`

Let:

```text
P = 1 / (1 + exp(-x / fermi_kT))
```

`fermi_kT` controls how strongly the logarithmic wealth ratio affects marriage probability. A larger `fermi_kT` makes the probability change more smoothly with wealth-ratio differences. A smaller `fermi_kT` makes the probability change more sharply.

### 3.2 `normal`

Let:

```text
P = normal_cdf(x / normal_sigma)
```

`normal_sigma` controls the scale of the logarithmic wealth ratio. The higher the man's wealth relative to the woman's wealth, the higher the probability that she sends a letter. The lower the man's relative wealth, the lower the probability.

## 4. Fertility and Economic Affordability

Families, unmarried adults, and orphaned minors who hold personal inherited wealth first go through yearly economic settlement. Fertility and marriage are processed afterward.

The yearly economic process currently follows these account-flow steps.

Family assets first earn interest:

```text
family_wealth = family_wealth * (1 + asset_interest_rate)
```

All adult individuals first receive annual wage income and interest on their personal account:

```text
person_wealth = person_wealth + annual_wage_per_adult + asset_interest_rate * person_wealth
```

Then married individuals move any personal-account balance into their active family account, and their personal account is reset to zero. When a marriage forms, the existing personal wealth of both spouses is also transferred once into the shared family account. This is asset merging, not annual wage income.

Yearly cost for each minor child:

```text
child_cost = child_base_living_cost + child_wealth_cost_rate * family_wealth
```

Yearly cost for each living adult:

```text
adult_cost = adult_base_living_cost + adult_wealth_cost_rate * family_wealth
```

Total yearly family living cost:

```text
total_living_cost = minor_children * child_cost + living_adults * adult_cost
```

The family account pays the living costs of adult parents and minor children:

```text
family_wealth = max(0, family_wealth - total_living_cost)
```

Unmarried adults pay adult living costs from their personal account:

```text
single_wealth = max(0, single_wealth - adult_cost)
```

If a minor has lost both parents and holds personal wealth, that inheritance is also used to pay the minor child's living cost:

```text
orphan_child_wealth = max(0, orphan_child_wealth - child_cost)
```

Therefore unmarried people can not only keep wealth but also continue accumulating wealth through wages and asset interest. This includes both men and women. Women older than 35 no longer participate in marriage or fertility, but their personal wealth is not forcibly transferred because of age. Inheritance or public-account rules are triggered only at death.

The household fertility target is computed using living children:

```text
living_children = number of children who are alive
```

If a child dies early, the family may continue reproducing until it reaches the target number of living children determined by `family_B`.

In the current model, wealth-related costs are added per household member. High-wealth families with many children therefore face stronger class-maintenance costs. This means every household member requires a maintenance cost corresponding to the family's class level, rather than a single fixed family-level class cost.

Adult-child property distribution happens after the year's economic settlement. Thus, in the year when the first child reaches 18 years since birth, the family first completes wage, asset-interest, and living-cost settlement, then distributes the family-account balance at that time to living children.

If adding one more child would create too large a yearly deficit and the family does not have enough wealth to cover that deficit, the family cannot attempt birth that year. The simulation also records, each year, how many couples wanted another child but were blocked by insufficient wealth.

## 5. Heredity Rules

The parental influence weights for offspring `B` and `D` are jointly determined by an external base weight and the logarithmic wealth ratio at parental marriage. Let the father's base weight be `p0`, the father's wealth at marriage be `Wf`, the mother's wealth at marriage be `Wm`, the minimum wealth scale be `parent_wealth_floor`, and the wealth influence strength be `parent_wealth_log_strength`:

```text
x = log((Wf + parent_wealth_floor) / (Wm + parent_wealth_floor))
father_weight = sigmoid(logit(p0) + parent_wealth_log_strength * x)
```

`p0` is controlled by external parameters. `B` uses `B_father_base_weight`, while `D` and `family_D` use `D_father_base_weight`. When parental marriage wealth is equal, `father_weight = p0`. When the father had higher relative wealth at marriage, the father's weight increases. When the mother had higher relative wealth at marriage, the father's weight decreases.

When a family forms, `family_B` is the weighted average of parental `B` values without mutation:

```text
family_B = father_weight_B * B_father + (1 - father_weight_B) * B_mother
```

Offspring `B` is generated from the same parental weights with genetic mutation:

```text
B_child = father_weight_B * B_father + (1 - father_weight_B) * B_mother + mutation
```

Offspring `D` is inherited using the parental `D` weights with genetic mutation:

```text
D_child = father_weight_D * D_father + (1 - father_weight_D) * D_mother + mutation
```

When a family forms, it also fixes a `family_D` used for later family property distribution. `family_D` is generated as a wealth-weighted average of parental `D` values at marriage, without mutation:

```text
family_D = father_weight_D * D_father + (1 - father_weight_D) * D_mother
```

The current version has temporarily removed `D feedback`, so `D` is no longer adjusted by marriage-market feedback. But `D` already participates in family property distribution: `family_D` is a household inheritance-tendency parameter in `[0, 1]`. `0.5` means equal per-person inheritance for sons and daughters. Values above `0.5` mean each son receives more than each daughter. Values below `0.5` mean each daughter receives more than each son.

## 6. Family Property Distribution

Family property has two main distribution moments.

### 6.1 First Distribution: Children Reaching Adulthood

When 18 years have passed since the family's first child was born, if the family has not already performed adult-child distribution, the full current family-account balance is distributed to living children. The distribution uses the family's `family_D`, not each child's own `D`.

Let distributable wealth be `W`, the number of living male children be `Nm`, and the number of living female children be `Nf`. Male children have weight `family_D`, and female children have weight `1 - family_D`:

```text
denominator = family_D * Nm + (1 - family_D) * Nf
male_child_share = family_D * W / denominator
female_child_share = (1 - family_D) * W / denominator
```

If there are only male children or only female children, children of the same sex split the wealth equally. `family_D = 0.5` means sons and daughters receive the same per-person share. `family_D = 2/3` means each son receives twice as much as each daughter. `family_D = 0.75` means each son receives three times as much as each daughter.

After the distribution, the family account is reset to zero, but the family itself continues to exist. The family can still accumulate new wealth later through the yearly economic module.

If a married child receives adult-child distribution, inheritance, or public-account redistribution, the wealth enters that child's active family account. If the recipient is unmarried, it enters personal wealth. The current model uses marital wealth pooling and does not distinguish a separate post-marriage personal inheritance account.

Adult-child distribution is processed only once, when 18 years have passed since the first child was born. Even if the family account is zero that year, or there are no living children to receive the distribution, the program marks the distribution as completed. It will not delay and perform a new adult-child distribution later just because the family accumulates new wealth again.

### 6.2 Second Distribution: Remaining Wealth After Both Parents Die

When both parents have died, the family account is closed. If the family has living children, the remaining family wealth is distributed to them using the same `family_D` rule.

If there are no living children, the program starts from both parents and performs a local BFS over the kinship graph using parent-child links. It searches for the nearest living relatives within three kinship generations and splits the wealth equally among them.

If there is no eligible heir within the three-generation kinship range, the wealth enters the social public checking account.

## 7. Wealth Transfer for Single People

When an unmarried individual dies, their personal wealth is transferred from that individual through a local BFS over the kinship graph using parent-child links. It goes to the nearest living relatives within three kinship generations.

If there are no eligible relatives, the wealth enters the social public checking account.

Unmarried people do not transfer wealth merely because they age out of the marriage or fertility window. Single men and single women both keep personal wealth and continue through single-person economic settlement. Only death triggers transfer to nearest blood relatives; if no eligible heir exists, the wealth enters the social public checking account.

## 8. Social Public Checking Account

When property cannot find an eligible heir, it enters `public_checking_account`.

The social public checking account is not distributed immediately in the same year. It is distributed at the start of the next year, equally among all people alive at that time.

If a living person already belongs to an active family, their public-account share enters the family account. Otherwise, it enters personal wealth.

## 9. Yearly Execution Order

Each year is processed in this order:

1. Wealth that entered the social public checking account in the previous year is distributed equally to the currently living population.
2. Everyone ages by 1 year; female fertility status is determined dynamically by age.
3. Deaths are sampled according to age-based mortality.
4. Families are checked to see whether both parents are dead. If both are dead, the family account is closed and remaining property is distributed.
5. Active families, unmarried adults, and orphaned minors with personal inherited wealth go through yearly economic settlement.
6. The simulation first checks adult-child distribution after 18 years since first birth, then lets families attempt birth for the current year.
7. The marriage market is processed.
8. End-of-year `history` is recorded, including living population, active family count, births, deaths, marriages, total wealth, mean wealth, public checking account balance, couples blocked from having another child by insufficient wealth, and family-wealth lists used for Lorenz-curve analysis.

## 10. Current Main Program Parameters

The current `__main__` block creates a `SimulationConfig` and passes it into `SocietySimulation`. The main parameters are:

- `initial_couples = 100`
- `initial_property = 100`
- `years = 300`
- `seed = 123`
- `birth_prob_per_year = 0.5`
- `initial_B_choices = [1, 2, 3, 4, 5, 6]`
- `marriage_mode = "fermi"`
- `fermi_kT = 0.00005`
- `normal_sigma = 0.15`
- `wealth_floor = 3`
- `initial_D_choices = [0.05, 0.25, 0.5, 0.75, 0.95]`
- `D_mutation_scale = 0.1`
- `B_mutation_scale = 0.5`
- `B_min = 1.0`
- `B_max = 6.0`
- `B_father_base_weight = 0.3`
- `D_father_base_weight = 0.3`
- `parent_wealth_log_strength = 1.0`
- `parent_wealth_floor = 3.0`
- `marriage_kinship_max_depth = 2`
- `inheritance_kinship_max_depth = 3`
- `annual_wage_per_adult = 3.0`
- `asset_interest_rate = 0.05`
- `child_base_living_cost = 1.0`
- `child_wealth_cost_rate = 0.01`
- `adult_base_living_cost = 2.0`
- `adult_wealth_cost_rate = 0.02`

## 11. Output and Lorenz Curve

The main program records yearly console statistics and, after the simulation finishes, plots a Lorenz curve for married-couple family wealth.

The Lorenz curve uses active couples' `family_wealth`, excluding families that have already completed adult-child property distribution because those family accounts may have been reset to zero after distribution. The x-axis is cumulative married couples, and the y-axis is cumulative married-couple wealth:

```text
x = cumulative married couples (%)
y = cumulative married-couple wealth (%)
```

Thus a point on the curve answers: "what share of married-couple wealth is owned by the bottom x% of couples?"

The program plots several years for comparison and prints the final year's bottom 10% and bottom 50% wealth shares.
