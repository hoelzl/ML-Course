# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
#
#  <h1 style="text-align:center;">Machine Learning for Programmers</h1>
#  <h2 style="text-align:center;">Evaluating Algorithms (Part 1)</h2>
#  <h3 style="text-align:center;">Dr. Matthias Hölzl</h3>

# %% [markdown] slideshow={"slide_type": "subslide"}
#  # What's the goal of ML
#
#  Turning raw data into usable information...
#
#  - ... by analyzing the data or looking at examples
#  - ... so that our results improve if we provide additional data

# %% [markdown] slideshow={"slide_type": "subslide"}
#
#  # How does this look like?
#
#  - Data:
#
#    - Many items (100s to billions)
#
#    - Fixed set of features per item (10s to thousands / millions)
#
#  - Information:
#
#    - Assign a number to each item
#
#      - From a small fixed set (clustering, classification)
#
#      - From an (in theory) infinite set (regression)
#
#    - Generate a complex output for each item
#
#      - translation, text generation (GPT-2/3, T5)
#
#      - images based on text, other images (Dall-E)

# %% [markdown] slideshow={"slide_type": "slide"}
#  # Classifying Approaches to ML
#
#  ## By Data/Environment Provided
#
#  - Unsupervised
#
#  - Supervised
#
#  - Reinforcement
#
#
#  But also...
#
#  - Semi-supervised / weakly supervised
#
#  - Self-supervised

# %% [markdown] slideshow={"slide_type": "subslide"}
#  ## By solution algorithm
#
#  - Parametric: Try to compress training data into settings for a fixed number
#    of parameters
#
#    - Linear and logistic regression
#
#    - Neural networks
#
#  - Non-parametric: Do not try to represent the training data as a fixed number
#    of parameters
#
#     - K-nearest-neighbors
#
#     - Decision trees
#
#     - Support Vector Machines
#
#  - Both have hyperparameters

# %% [markdown] slideshow={"slide_type": "subslide"}
#  # Parameters vs. Hyperparameters
#
#  - Hyperparameters are values that the designer sets before instantiating the
#    algorithm:
#
#    - Architecture and number of nodes in a neural network
#
#  - Parameters are values that the algorithm sets during training (fitting)
#
#    - Slope of the solution line in linear regression

# %% [markdown] slideshow={"slide_type": "subslide"}
#  ## Supervised Learning: Training a DL Classifier
#
#  <br/>
#  <img src="img/ag/Figure-01-008.png" style="width: 100%;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
#  ## Supervised Learning: Evaluation/Test
#
#  <img src="img/ag/Figure-01-009.png" style="width: 70%; padding: 20px;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
#  ## Clustering
#
#  <img src="img/ag/Figure-01-013.jpg" style="float: right;width: 40%;"/>
#
#  - Many documents
#  - Figure out which ones are "similar"
#
#  Clustering is often done via unsupervised methods.

# %% [markdown] slideshow={"slide_type": "subslide"}
#  ## Classification
#
#  <img src="img/ag/Figure-01-022.png" style="float: right;width: 40%;"/>
#
#  - Many documents, fixed set of labels
#  - Assign one or more labels to each document

# %% [markdown] slideshow={"slide_type": "subslide"}
#  ## Regression
#
#  <img src="img/ag/Figure-01-011.png" style="float: right;width: 40%;"/>
#
#  - Learn numerical relationships
#  - "How does salary depend on age?"

# %% [markdown] slideshow={"slide_type": "subslide"}
#  ## Geocoding / toponym resolution
#
#  <img src="img/france.jpg" style="float: right;width: 40%;"/>
#
#  <div style="float: left; width: 60%;">
#
#  <br/>
#
#  - Figure out coordinates from occurrences of names in text
#  - Reverse geocoding: Figure out place name from coordinates
#  </div>

# %% [markdown] slideshow={"slide_type": "subslide"}
#  ## (Extractive) Question Answering
#
#  <img src="img/question-mark.jpg" style="float: right;width: 30%;"/>
#
#
#  - One document, one question about the document
#  - Extract the answer from the document

# %% [markdown] slideshow={"slide_type": "slide"}
# # Evaluating Regression Performance
#
# How can we determine the quality of our solution for a regression problem?

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-09-008.png" style="width: 40%; padding: 20px;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-09-009.png" style="width: 80%; padding: 20px;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-09-010.png" style="width: 40%; padding: 20px;"/>

# %% slideshow={"slide_type": "subslide"}
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# %% slideshow={"slide_type": "subslide"}
rng = np.random.default_rng(42)
x = rng.uniform(size=(150, 1), low=0.0, high=10.0)

# %% slideshow={"slide_type": "subslide"}
plt.figure(figsize=(20, 1), frameon=False)
plt.yticks([], [])
plt.scatter(x, np.zeros_like(x), alpha=0.4);


# %% slideshow={"slide_type": "slide"}
def lin(x):
    return 0.85 * x - 1.5


# %% slideshow={"slide_type": "subslide"}
def fun(x):
    return 2 * np.sin(x) + 0.1 * x ** 2 - 2


# %% slideshow={"slide_type": "subslide"}
x_plot = np.linspace(0, 10, 500)
plt.figure(figsize=(20, 8))
sns.lineplot(x=x_plot, y=lin(x_plot))
sns.lineplot(x=x_plot, y=fun(x_plot));


# %% slideshow={"slide_type": "subslide"}
def randomize(fun, x):
    return fun(x) + rng.normal(size=x.shape, scale=0.5)


# %% slideshow={"slide_type": "subslide"}
plt.figure(figsize=(20, 8))
sns.lineplot(x=x_plot, y=randomize(lin, x_plot))
sns.lineplot(x=x_plot, y=randomize(fun, x_plot));

# %% slideshow={"slide_type": "subslide"}
plt.figure(figsize=(20, 8))
sns.scatterplot(x=x_plot, y=randomize(lin, x_plot))
sns.scatterplot(x=x_plot, y=randomize(fun, x_plot));

# %% slideshow={"slide_type": "subslide"}
plt.figure(figsize=(20, 8))
x_vec = x.reshape(-1)
sns.scatterplot(x=x_vec, y=randomize(lin, x_vec))
sns.scatterplot(x=x_vec, y=randomize(fun, x_vec));

# %% slideshow={"slide_type": "subslide"}
plt.figure(figsize=(20, 8))
x_vec = x.reshape(-1)
sns.regplot(x=x_vec, y=randomize(lin, x_vec))
sns.regplot(x=x_vec, y=randomize(fun, x_vec));

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Fitting a Linear Regressor

# %%
x_train, x_test = x[:100], x[100:]
y1_train = randomize(lin, x_train).reshape(-1)
y1_test = randomize(lin, x_test).reshape(-1)
y2_train = randomize(fun, x_train.reshape(-1))
y2_test = randomize(fun, x_test).reshape(-1)

# %%
x_train.shape, x_test.shape, y1_train.shape, y1_test.shape, y2_train.shape, y2_test.shape

# %% slideshow={"slide_type": "subslide"}
from sklearn.linear_model import LinearRegression
lr1 = LinearRegression()
lr1.fit(x_train, y1_train)
lr1_pred = lr1.predict(x_test)

# %% slideshow={"slide_type": "subslide"}
plt.figure(figsize=(20, 8))
plt.scatter(x_train, y1_train, alpha=0.6);
plt.scatter(x_test, lr1_pred, c="green");
plt.plot(x_test, lr1_pred, c="red")
plt.scatter(x_test, y1_test, c="orange");

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Computing the Error:
#
# First try: Compute distance between prediction and true values:

# %%
error_per_sample = y1_test - lr1_pred
error = error_per_sample.mean()
error

# %% [markdown] slideshow={"slide_type": "fragment"}
# The positive and negative error cancel out!

# %% [markdown] slideshow={"slide_type": "subslide"}
# Better: try with absolute distance or squared distance:

# %%
abs_error_per_sample = np.abs(y1_test - lr1_pred)
square_error_per_sample = (y1_test - lr1_pred) ** 2

abs_error_per_sample.mean(), square_error_per_sample.mean()

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error

mean_absolute_error(y1_test, lr1_pred), mean_squared_error(y1_test, lr1_pred)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Linear Regression and Non-Linear Relations

# %%
from sklearn.linear_model import LinearRegression
lr2 = LinearRegression()
lr2.fit(x_train, y2_train)
lr2_pred = lr2.predict(x_test)

# %% slideshow={"slide_type": "subslide"}
plt.figure(figsize=(20, 8))
plt.scatter(x_train, y2_train)
plt.plot(x_test, lr2_pred, c="red")
plt.scatter(x_test, lr2_pred, c="green");
plt.scatter(x_test, y2_test, c="orange");

# %% slideshow={"slide_type": "subslide"}
mean_absolute_error(y1_test, lr2_pred), mean_squared_error(y1_test, lr2_pred)

# %%
