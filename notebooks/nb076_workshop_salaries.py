# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 078a - Workshop: Salaries
#
# The module `mlcourse.data.generators.fake_salary` contains a number of synthetic datasets
# that represent salary as a function of ages and education level, or ages and profession.
#
# Analyze how `linear_salaries`, `stepwise_salaries`, `interpolated_salaries` and `multivar_salaries` depend on `ages` and `education_levels` and train regression models (at least linear and decision tree models) that model these dependencies.
#
# Do the same for `multidist_ages`, `professions`, and `multidist_salaries`.
#
# *Hint:* The `fake_salary` module contains a number of plots that show the relatinships; to display them run the file as main module or interactively in VS Code. Please try to solve the exercises yourself before looking at the plots.

# %%
import matplotlib.pyplot as plt

# %%
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
ages.shape, education_levels.shape

# %%
(linear_salaries.shape,
 stepwise_salaries.shape,
 interpolated_salaries.shape,
 multivar_salaries.shape)

# %%
