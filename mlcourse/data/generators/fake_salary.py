# %%
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nbex.interactive import session
from numpy.random import default_rng

from mlcourse.config import Config

# %%
config = Config()

# %%
sns.set_theme(style="darkgrid")

# %%
rng = default_rng(seed=42)


# %%
def generate_linear_relation(x, slope=1.5, offset=1, random_scale=1.0, rng=rng):
    random_offsets = rng.normal(size=x.shape, scale=random_scale)
    y_linear = slope * x + offset
    return y_linear + random_offsets


# %%
def linear_salaries_for_ages(
    ages, slope=550, offset=20_000, random_scale=500, multiplier=1.0
):
    return (
        generate_linear_relation(
            ages, slope=slope, offset=offset, random_scale=random_scale
        )
        * multiplier
    )


# %%
ages = rng.uniform(low=20, high=65, size=5_000)
ages[:20]

# %%
linear_salaries = generate_linear_relation(
    ages, slope=550, offset=20_000, random_scale=500
)

# %%
if session.is_interactive:
    sns.scatterplot(x=ages[:500], y=linear_salaries[:500])

# %%
if session.is_interactive:
    sns.scatterplot(x=ages, y=linear_salaries, alpha=0.15)

# %%
if session.is_interactive:
    sns.regplot(x=ages[:500], y=linear_salaries[:500], line_kws={"color": "red"})


# %%
def simple_stepwise_conditions_for_ages(ages):
    return [
        ages < 20,
        (20 <= ages) & (ages < 50),
        50 <= ages,
    ]


# %%
def simple_stepwise_values_for_ages(ages):
    # np.ones_like(ages)
    return [20_000, 40_000, 60_000]


# %%
def simple_stepwise_salaries_for_ages(ages):
    return np.select(
        simple_stepwise_conditions_for_ages(ages), [20_000, 40_000, 60_000]
    )


# %%
def print_simple_stepwise_choices(ages):
    print("Ages:")
    pprint(ages)
    print("\nConditions:")
    pprint(simple_stepwise_conditions_for_ages(ages))
    print("\nValues:")
    pprint(simple_stepwise_values_for_ages(ages))
    print("\nSalaries for ages:")
    pprint(simple_stepwise_salaries_for_ages(ages))
    print("\n")


# %%
if session.is_interactive:
    print_simple_stepwise_choices(np.array([18]))
    print_simple_stepwise_choices(np.array([40]))
    print_simple_stepwise_choices(np.array([18, 40]))

# %%
if session.is_interactive:
    print_simple_stepwise_choices(np.array([18, 40, 70, 35, 90, 15]))


# %%
# Salaries approximately taken from
# https://www.indeed.com/career-advice/pay-salary/average-salary-by-age


# %%
def stepwise_conditions_for_ages(ages):
    return [
        ages < 20,
        (20 <= ages) & (ages < 25),
        (25 <= ages) & (ages < 35),
        (35 <= ages) & (ages < 45),
        (45 <= ages) & (ages < 55),
        (55 <= ages) & (ages < 65),
        65 <= ages,
    ]


# %%
def stepwise_values_for_ages(ages):
    ones = np.ones_like(ages)
    return [
        22_000 * ones,
        30_000 * ones,
        41_000 * ones,
        51_000 * ones,
        52_500 * ones,
        50_500 * ones,
        47_500 * ones,
    ]


# %%
stepwise_values = [22_000, 30_000, 41_000, 51_000, 52_500, 50_500, 47_500]


# %%
def deterministic_stepwise_salaries_for_ages(ages):
    return np.select(stepwise_conditions_for_ages(ages), stepwise_values_for_ages(ages))


# %%
# def deterministic_stepwise_salaries_for_ages(ages):
#     return np.select(stepwise_conditions_for_ages(ages), stepwise_values)


# %%
def print_stepwise_choices(ages):
    print("Ages:")
    pprint(ages)
    print("\nConditions:")
    pprint(stepwise_conditions_for_ages(ages))
    print("\nValues:")
    pprint(stepwise_values_for_ages(ages))
    print("\nSalaries for ages:")
    pprint(deterministic_stepwise_salaries_for_ages(ages))


# %%
if session.is_interactive:
    print_stepwise_choices(np.array([19]))

# %%
if session.is_interactive:
    sns.scatterplot(
        x=ages[:500], y=deterministic_stepwise_salaries_for_ages(ages[:500])
    )


# %%
def deterministic_interpolated_salaries_for_ages(ages):
    return np.interp(
        ages,
        [18, 22.5, 30, 40, 50, 60, 70],
        [22_000, 30_000, 41_000, 51_000, 52_500, 50_500, 47_500],
    )


# %%
if session.is_interactive:
    sns.scatterplot(
        x=ages[:500], y=deterministic_interpolated_salaries_for_ages(ages[:500])
    )


# %%
def stepwise_salaries_for_ages(ages, rng=rng, random_scale=500, multiplier=1.0):
    random_offsets = rng.normal(size=ages.shape, scale=random_scale)
    # r_min, r_max = np.amin(random_offsets), np.amax(random_offsets)
    # print(r_min, r_max)
    base_salary = deterministic_stepwise_salaries_for_ages(ages) + random_offsets
    return base_salary * multiplier


# %%
stepwise_salaries = stepwise_salaries_for_ages(ages, random_scale=500)

# %%
if session.is_interactive:
    sns.scatterplot(x=ages[:500], y=stepwise_salaries[:500])


# %%
if session.is_interactive:
    sns.scatterplot(x=ages, y=stepwise_salaries, alpha=0.15)


# %%
def interpolated_salaries_for_ages(ages, rng=rng, random_scale=500, multiplier=1.0):
    random_offsets = rng.normal(size=ages.shape, scale=random_scale)
    # r_min, r_max = np.amin(random_offsets), np.amax(random_offsets)
    # print(r_min, r_max)
    base_salary = random_offsets + deterministic_interpolated_salaries_for_ages(ages)
    return base_salary * multiplier


# %%
interpolated_salaries = interpolated_salaries_for_ages(ages)


# %%
if session.is_interactive:
    sns.scatterplot(x=ages[:500], y=interpolated_salaries[:500])


# %%
if session.is_interactive:
    sns.scatterplot(x=ages, y=interpolated_salaries, alpha=0.15)


# %%
if session.is_interactive:
    sns.scatterplot(x=ages, y=linear_salaries, alpha=0.15)
    sns.scatterplot(x=ages, y=stepwise_salaries, alpha=0.15)
    sns.scatterplot(x=ages, y=interpolated_salaries, alpha=0.15)

# %%
linear_salaries_df = pd.DataFrame(
    {"age": np.round(ages), "salary": np.round(linear_salaries)}
)
stepwise_salaries_df = pd.DataFrame(
    {"age": np.round(ages), "salary": np.round(stepwise_salaries)}
)
interpolated_salaries_df = pd.DataFrame(
    {"age": np.round(ages), "salary": np.round(interpolated_salaries)}
)

# %%
linear_salaries_df

# %%
if session.is_interactive:
    sns.scatterplot(data=linear_salaries_df, x="age", y="salary", alpha=0.25)
    sns.scatterplot(data=stepwise_salaries_df, x="age", y="salary", alpha=0.25)
    sns.scatterplot(data=interpolated_salaries_df, x="age", y="salary", alpha=0.25)

# %%
linear_salaries_df.to_csv(config.data_dir_path / "generated/linear_salaries.csv")
stepwise_salaries_df.to_csv(config.data_dir_path / "generated/stepwise_salaries.csv")
interpolated_salaries_df.to_csv(
    config.data_dir_path / "generated/interpolated_salaries.csv"
)


# %%
def multivar_salaries_for_ages(ages, education_levels):
    base_salary = interpolated_salaries_for_ages(ages)
    edu_inc = 100 * ages * education_levels
    return base_salary + edu_inc


# %%
education_levels = rng.integers(low=0, high=3, size=ages.shape)
multivar_salaries = multivar_salaries_for_ages(ages, education_levels)

# %%
if session.is_interactive:
    sns.scatterplot(x=ages, y=multivar_salaries, hue=education_levels)

# %%
multivar_salaries_df = pd.DataFrame(
    {"age": np.round(ages), "edu_lvl": education_levels, "salary": multivar_salaries}
)

# %%
if session.is_interactive:
    sns.scatterplot(
        data=multivar_salaries_df, x="age", y="salary", hue="edu_lvl", alpha=0.25
    )

# %%
multivar_salaries_df.to_csv(config.data_dir_path / "generated/multivar_salaries.csv")

# %%
if session.is_interactive:
    sns.pairplot(data=multivar_salaries_df)

# %%
if session.is_interactive:
    grid = sns.pairplot(
        data=multivar_salaries_df,
        vars=["age", "salary", "edu_lvl"],
        hue="edu_lvl",
        diag_kind="hist",
        height=3,
        aspect=1,
    )


# %%
def salaries_for_professions(ages, professions):
    return np.choose(
        professions,
        [
            interpolated_salaries_for_ages(ages, random_scale=4_000, multiplier=0.75),
            interpolated_salaries_for_ages(ages, random_scale=2_500, multiplier=1.1),
            linear_salaries_for_ages(
                ages, slope=200, offset=30_000, random_scale=2_500, multiplier=0.66
            ),
            stepwise_salaries_for_ages(ages, random_scale=2_500, multiplier=1.25),
            stepwise_salaries_for_ages(ages, random_scale=2_500, multiplier=0.9),
            linear_salaries_for_ages(
                ages, slope=400, offset=45_000, random_scale=4_000, multiplier=0.9
            ),
        ],
    )


# %%
multidist_ages = rng.uniform(low=20, high=65, size=10_000)
professions = rng.integers(size=multidist_ages.shape, low=0, high=5, endpoint=True)
multidist_salaries = salaries_for_professions(
    ages=multidist_ages, professions=professions
)

# %%
if session.is_interactive:
    sns.scatterplot(x=multidist_ages, y=multidist_salaries, alpha=0.15)

# %%
if session.is_interactive:
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.scatterplot(
        x=multidist_ages,
        y=multidist_salaries,
        hue=professions,
        style=professions,
        # palette="flare",
        ax=ax,
        alpha=0.5,
    )

# %%
if session.is_interactive:
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.scatterplot(
        x=multidist_ages,
        y=multidist_salaries,
        hue=professions,
        # style=professions,
        # palette="coolwarm",
        # palette="seismic",
        palette="gist_rainbow",
        ax=ax,
        alpha=0.5,
    )


# %%
multidist_salaries_df = pd.DataFrame(
    {
        "age": np.round(multidist_ages),
        "profession": professions,
        "salary": np.round(multidist_salaries),
    }
)

# %%
multidist_salaries_df.to_csv(config.data_dir_path / "generated/multidist_salaries.csv")

# %%
if session.is_interactive:
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.scatterplot(
        data=multidist_salaries_df,
        x="age",
        y="salary",
        hue="profession",
        palette="gist_rainbow",
        ax=ax,
        alpha=0.5,
    )


# %%
if session.is_interactive:
    fig, axes = plt.subplots(
        ncols=2,
        nrows=3,
        figsize=(9, 9),
        # sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.5, "wspace": 0.2},
    )
    for row in range(3):
        for col in range(2):
            profession = col + 2 * row
            ax = axes[row, col]
            ax.set_title(f"profession={profession}")
            sns.scatterplot(
                data=multidist_salaries_df.loc[
                    multidist_salaries_df["profession"] == profession
                ],
                x="age",
                y="salary",
                ax=ax,
                alpha=0.25,
            )

# %%
if session.is_interactive:
    sns.pairplot(data=multidist_salaries_df)

# %%

# %%
if session.is_interactive:
    grid = sns.pairplot(
        data=multidist_salaries_df, hue="profession", height=2, aspect=2
    )

# %%
