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
# # California Housing Prices
#
# In this workshop, we'll use one of Scikit-Learn's "real world" datasets to predict house prices in California.

# %%
import sklearn
import sklearn.datasets
import numpy as np
import seaborn as sns

# %% [markdown]
# ## Load the Dataset
#
# You can get the dataset with the `fetch_california_housing()` function.

# %%
housing = sklearn.datasets.fetch_california_housing()

# %% [markdown]
# Using `dir` you can see the features; as usual `DESCR` contains a description of the dataset:

# %%
dir(housing)

# %%
housing.feature_names

# %%
print(sklearn.datasets.fetch_california_housing().DESCR)

# %% [markdown]
# By supplying the `return_X_y` argument you can get back just the features and target values as numpy arrays.

# %%
x, y = sklearn.datasets.fetch_california_housing(return_X_y=True)

# %%
x.shape, y.shape

# %% [markdown]
# ## Analyze using Plots
#
# Analyze the dataset using PyPlot and Seaborn plots. Plot, e.g., house age vs. price:

# %%
sns.scatterplot(x=x[:, 1], y=y);

# %% [markdown]
# By adjusting the plot parameters, you can gain more insight:

# %%
sns.scatterplot(x=x[:, 1], y=y, alpha=0.1);

# %% [markdown]
# Also plot scatterplots of features against each other; use hue with a third feature or the target value to increase the amount of information displayed in the plot.

# %%
sns.scatterplot(x=x[:, 1], y=x[:, 7], hue=y, alpha=0.4);

# %% [markdown]
# Use different plots to understand the dataset. The dataset has some obvious problems. Which ones can you identify? What measures could you take to fix these problems or at least lessen their impact on the machine learning tasks?

# %% [markdown]
# ## Building Models
#
# Train at least a linear regression and a decision tree model to predict housing prices in California. Experiment with different hyperparameter settings to see what maximum performance you can get from each model type. Is overfitting a problem?

# %% [markdown]
# I hope you have fun with this exercise.

# %%
