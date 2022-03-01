# %%
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nbex.interactive import session
from numpy.random import default_rng

from mlcourse.data.generators.fake_salary import (
    ages,
    education_levels,
    
    linear_salaries, 
    stepwise_salaries, 
    interpolated_salaries, 
    multivar_salaries,
    
    multidist_ages,
    professions,
    
    multidist_salaries,
)

# %%
sns.set_theme(style="darkgrid")

# %%
sns.scatterplot(x=ages[:500], y=linear_salaries[:500]);

# %%
sns.scatterplot(x=ages, y=linear_salaries, alpha=0.15);

# %%
sns.regplot(x=ages[:500], y=linear_salaries[:500], line_kws={"color": "red"})

# %%
# Salaries approximately taken from
# https://www.indeed.com/career-advice/pay-salary/average-salary-by-age


# %%
sns.scatterplot(x=ages[:500], y=interpolated_salaries[:500]);


# %%
sns.scatterplot(x=ages, y=interpolated_salaries, alpha=0.15);


# %%
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
sns.scatterplot(data=linear_salaries_df, x="age", y="salary", alpha=0.25)
sns.scatterplot(data=stepwise_salaries_df, x="age", y="salary", alpha=0.25)
sns.scatterplot(data=interpolated_salaries_df, x="age", y="salary", alpha=0.25);

# %%
sns.scatterplot(x=ages, y=multivar_salaries, hue=education_levels)

# %%
multivar_salaries_df = pd.DataFrame(
    {"age": np.round(ages), "edu_lvl": education_levels, "salary": multivar_salaries}
)

# %%
sns.scatterplot(
    data=multivar_salaries_df, x="age", y="salary", hue="edu_lvl", alpha=0.25
)

# %%
sns.pairplot(data=multivar_salaries_df);

# %%
grid = sns.pairplot(
    data=multivar_salaries_df,
    vars=["age", "salary", "edu_lvl"],
    hue="edu_lvl",
    diag_kind="hist",
    height=3,
    aspect=1,
)

# %%
sns.scatterplot(x=multidist_ages, y=multidist_salaries, alpha=0.15);

# %%
fig, ax = plt.subplots(figsize=(9, 8))
sns.scatterplot(
    x=multidist_ages,
    y=multidist_salaries,
    hue=professions,
    style=professions,
    # palette="flare",
    ax=ax,
    alpha=0.5,
);

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
