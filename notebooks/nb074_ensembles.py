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

# %% [markdown] slideshow={"slide_type": "slide"}
# # Ensembles

# %% slideshow={"slide_type": "subslide"}
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

sns.set_theme()

# %% slideshow={"slide_type": "subslide"}
rng = np.random.default_rng(42)

x = rng.uniform(size=(150, 1), low=0.0, high=10.0)
x_train, x_test = x[:100], x[100:]

x_plot = np.linspace(0, 10, 500).reshape(-1, 1)


# %% slideshow={"slide_type": "subslide"}
def lin(x):
    return 0.85 * x - 1.5


# %% slideshow={"slide_type": "-"}
def fun(x):
    return 2 * np.sin(x) + 0.1 * x ** 2 - 2


# %% slideshow={"slide_type": "-"}
def randomize(fun, x, scale=0.5):
    return fun(x) + rng.normal(size=x.shape, scale=scale)


# %% slideshow={"slide_type": "subslide"}
def evaluate_non_random_regressor(reg_type, f_y, *args, **kwargs):
    reg = reg_type(*args, **kwargs)

    y_train = f_y(x_train).reshape(-1)
    y_test = f_y(x_test).reshape(-1)

    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)

    x_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.lineplot(x=x_plot[:, 0], y=reg.predict(x_plot), ax=ax)
    sns.lineplot(x=x_plot[:, 0], y=f_y(x_plot[:, 0]), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_train, ax=ax)
    plt.show()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(
        "\nNo randomness:      " f"MAE = {mae:.2f}, MSE = {mse:.2f}, RMSE = {rmse:.2f}"
    )

    return reg


# %% slideshow={"slide_type": "subslide"}
def plot_graphs(f_y, reg, reg_rand, reg_chaos, y_train, y_rand_train, y_chaos_train):
    x_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.lineplot(x=x_plot[:, 0], y=reg.predict(x_plot), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_train, ax=ax)

    sns.lineplot(x=x_plot[:, 0], y=reg_rand.predict(x_plot), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_rand_train, ax=ax)

    sns.lineplot(x=x_plot[:, 0], y=reg_chaos.predict(x_plot), ax=ax)
    sns.scatterplot(x=x_train[:, 0], y=y_chaos_train, ax=ax)

    sns.lineplot(x=x_plot[:, 0], y=f_y(x_plot[:, 0]), ax=ax)
    plt.show()   


# %% slideshow={"slide_type": "subslide"}
def print_evaluation(y_test, y_pred, y_rand_test, y_rand_pred, y_chaos_test, y_chaos_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mae_rand = mean_absolute_error(y_rand_test, y_rand_pred)
    mae_chaos = mean_absolute_error(y_chaos_test, y_chaos_pred)

    mse = mean_squared_error(y_test, y_pred)
    mse_rand = mean_squared_error(y_rand_test, y_rand_pred)
    mse_chaos = mean_squared_error(y_chaos_test, y_chaos_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_rand = np.sqrt(mean_squared_error(y_rand_test, y_rand_pred))
    rmse_chaos = np.sqrt(mean_squared_error(y_chaos_test, y_chaos_pred))

    print(
        "\nNo randomness:      " f"MAE = {mae:.2f}, MSE = {mse:.2f}, RMSE = {rmse:.2f}"
    )
    print(
        "Some randomness:    "
        f"MAE = {mae_rand:.2f}, MSE = {mse_rand:.2f}, RMSE = {rmse_rand:.2f}"
    )
    print(
        "Lots of randomness: "
        f"MAE = {mae_chaos:.2f}, MSE = {mse_chaos:.2f}, RMSE = {rmse_chaos:.2f}"
    )


# %% slideshow={"slide_type": "subslide"}
def evaluate_regressor(reg_type, f_y, *args, **kwargs):
    reg = reg_type(*args, **kwargs)
    reg_rand = reg_type(*args, **kwargs)
    reg_chaos = reg_type(*args, **kwargs)
    
    y_train = f_y(x_train).reshape(-1)
    y_test = f_y(x_test).reshape(-1)
    y_pred = reg.fit(x_train, y_train).predict(x_test)
    
    y_rand_train = randomize(f_y, x_train).reshape(-1)
    y_rand_test = randomize(f_y, x_test).reshape(-1)
    y_rand_pred = reg_rand.fit(x_train, y_rand_train).predict(x_test)

    y_chaos_train = randomize(f_y, x_train, 1.5).reshape(-1)
    y_chaos_test = randomize(f_y, x_test, 1.5).reshape(-1)
    y_chaos_pred = reg_chaos.fit(x_train, y_chaos_train).predict(x_test)

    plot_graphs(f_y, reg, reg_rand, reg_chaos, y_train, y_rand_train, y_chaos_train)
    print_evaluation(y_test, y_pred, y_rand_test, y_rand_pred, y_chaos_test, y_chaos_pred)


# %% [markdown] slideshow={"slide_type": "slide"}
# # Ensembles, Random Forests, Gradient Boosted Trees

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Ensemble Methods
#
# Idea: combine several estimators to improve their overal performance.
#
# - Averaging methods: 
#   - Independent estimators, average predictions
#   - Reduces variance (overfitting)
#   - Bagging, random forests
# - Boosting methods:
#   - Train estimators sequentially
#   - Each estimator is trained to reduce the bias of its (combined) predecessors

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Bagging
#
# - Averaging method: build several estimators of the same type, average their results
# - Needs some way to introduce differences between estimators
#   - Otherwise variance is not reduced
#   - Train on random subsets of the training data
# - Reduce overfitting
# - Work best with strong estimators (e.g., decision trees with (moderately) large depth)

# %% [markdown]
# ### Random Forests
#
# - Bagging classifier/regressor using decision trees
# - For each tree in the forest:
#   - Subset of training data
#   - Subset of features
# - Often significant reduction in variance (overfitting)
# - Sometimes increase in bias

# %% slideshow={"slide_type": "subslide"}
from sklearn.ensemble import RandomForestRegressor

# %% slideshow={"slide_type": "subslide"}
evaluate_non_random_regressor(RandomForestRegressor, lin, random_state=42);

# %% slideshow={"slide_type": "subslide"}
evaluate_non_random_regressor(RandomForestRegressor, fun, random_state=42);

# %% slideshow={"slide_type": "subslide"}
evaluate_non_random_regressor(
    RandomForestRegressor, fun, n_estimators=25, criterion="absolute_error", random_state=42
);

# %% slideshow={"slide_type": "subslide"}
evaluate_regressor(RandomForestRegressor, lin, random_state=42);

# %% slideshow={"slide_type": "subslide"}
evaluate_regressor(
    RandomForestRegressor, lin, n_estimators=500, max_depth=3, random_state=42
)

# %% slideshow={"slide_type": "subslide"}
evaluate_regressor(
    RandomForestRegressor, lin, n_estimators=500, min_samples_leaf=6, random_state=42
)

# %% slideshow={"slide_type": "subslide"}
evaluate_regressor(RandomForestRegressor, fun, random_state=42)

# %% slideshow={"slide_type": "subslide"}
evaluate_regressor(
    RandomForestRegressor,
    fun,
    n_estimators=1000,
    min_samples_leaf=6,
    random_state=43,
    n_jobs=-1,
)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Gradient Boosted Trees
#
# - Boosting method for both regression and classification
# - Requires differentiable loss function

# %% slideshow={"slide_type": "subslide"}
from sklearn.ensemble import GradientBoostingRegressor

# %% slideshow={"slide_type": "subslide"}
evaluate_non_random_regressor(GradientBoostingRegressor, lin);

# %% slideshow={"slide_type": "subslide"}
evaluate_non_random_regressor(GradientBoostingRegressor, fun);

# %% slideshow={"slide_type": "subslide"}
evaluate_regressor(GradientBoostingRegressor, lin);

# %% slideshow={"slide_type": "subslide"}
evaluate_regressor(GradientBoostingRegressor, lin, n_estimators=200, learning_rate=0.05, loss="absolute_error");

# %% slideshow={"slide_type": "subslide"}
evaluate_regressor(GradientBoostingRegressor, lin, n_estimators=500, learning_rate=0.01,
                   loss="absolute_error", subsample=0.1, random_state=46);

# %% slideshow={"slide_type": "subslide"}
evaluate_regressor(GradientBoostingRegressor, fun, n_estimators=500, learning_rate=0.01,
                   loss="absolute_error", subsample=0.1, random_state=44);

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Multiple Features

# %% slideshow={"slide_type": "subslide"}
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=1)

# %%
x, y, coef = make_regression(n_samples=250, n_features=4, n_informative=1, coef=True, random_state=42)
x.shape, y.shape, coef

# %% slideshow={"slide_type": "subslide"}
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 12))
for i, ax in enumerate(axs.reshape(-1)):
    sns.scatterplot(x=x[:, i], y=y, ax=ax)

# %% slideshow={"slide_type": "subslide"}
x, y, coef = make_regression(n_samples=250, n_features=20, n_informative=10, coef=True, random_state=42)
x.shape, y.shape, coef

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %% slideshow={"slide_type": "subslide"}
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 12))
for i in range(2):
    sns.scatterplot(x=x[:, i], y=y, ax=axs[0, i]);
for i in range(2):
    sns.scatterplot(x=x[:, i + 6], y=y, ax=axs[1, i]);

# %% slideshow={"slide_type": "subslide"}
lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)
y_lr_pred = lr_clf.predict(x_test)

mean_absolute_error(y_test, y_lr_pred), mean_squared_error(y_test, y_lr_pred)

# %%
lr_clf.coef_.astype(np.int32), coef.astype(np.int32)

# %% slideshow={"slide_type": "subslide"}
dt_clf = DecisionTreeRegressor()
dt_clf.fit(x_train, y_train)
y_dt_pred = dt_clf.predict(x_test)

mean_absolute_error(y_test, y_dt_pred), mean_squared_error(y_test, y_dt_pred)

# %% slideshow={"slide_type": "subslide"}
rf_clf = RandomForestRegressor()
rf_clf.fit(x_train, y_train)
y_rf_pred = rf_clf.predict(x_test)

mean_absolute_error(y_test, y_rf_pred), mean_squared_error(y_test, y_rf_pred)

# %% slideshow={"slide_type": "subslide"}
gb_clf = GradientBoostingRegressor()
gb_clf.fit(x_train, y_train)
y_gb_pred = gb_clf.predict(x_test)

mean_absolute_error(y_test, y_gb_pred), mean_squared_error(y_test, y_gb_pred)

# %% slideshow={"slide_type": "subslide"}
x, y, coef = make_regression(n_samples=250, n_features=20, n_informative=10, noise=100.0, coef=True, random_state=42)
x.shape, y.shape, coef

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %% slideshow={"slide_type": "subslide"}
lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)
y_lr_pred = lr_clf.predict(x_test)

mean_absolute_error(y_test, y_lr_pred), mean_squared_error(y_test, y_lr_pred)

# %% slideshow={"slide_type": "subslide"}
dt_clf = DecisionTreeRegressor()
dt_clf.fit(x_train, y_train)
y_dt_pred = dt_clf.predict(x_test)

mean_absolute_error(y_test, y_dt_pred), mean_squared_error(y_test, y_dt_pred)

# %% slideshow={"slide_type": "subslide"}
rf_clf = RandomForestRegressor()
rf_clf.fit(x_train, y_train)
y_rf_pred = rf_clf.predict(x_test)

mean_absolute_error(y_test, y_rf_pred), mean_squared_error(y_test, y_rf_pred)

# %% slideshow={"slide_type": "subslide"}
gb_clf = GradientBoostingRegressor()
gb_clf.fit(x_train, y_train)
y_gb_pred = gb_clf.predict(x_test)

mean_absolute_error(y_test, y_gb_pred), mean_squared_error(y_test, y_gb_pred)

# %% slideshow={"slide_type": "subslide"}
x, y, coef = make_regression(n_samples=250, n_features=20, n_informative=10, noise=100.0,
                             coef=True, random_state=42)
y += (20 * x[:, 1]) ** 2
x.shape, y.shape, coef

# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

# %% slideshow={"slide_type": "subslide"}
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 12))
for i in range(2):
    sns.scatterplot(x=x[:, i], y=y, ax=axs[0, i]);
for i in range(2):
    sns.scatterplot(x=x[:, i + 6], y=y, ax=axs[1, i]);

# %% slideshow={"slide_type": "subslide"}
lr_clf = LinearRegression()
lr_clf.fit(x_train, y_train)
y_lr_pred = lr_clf.predict(x_test)

mean_absolute_error(y_test, y_lr_pred), mean_squared_error(y_test, y_lr_pred)

# %% slideshow={"slide_type": "subslide"}
dt_clf = DecisionTreeRegressor()
dt_clf.fit(x_train, y_train)
y_dt_pred = dt_clf.predict(x_test)

mean_absolute_error(y_test, y_dt_pred), mean_squared_error(y_test, y_dt_pred)

# %% slideshow={"slide_type": "subslide"}
rf_clf = RandomForestRegressor()
rf_clf.fit(x_train, y_train)
y_rf_pred = rf_clf.predict(x_test)

mean_absolute_error(y_test, y_rf_pred), mean_squared_error(y_test, y_rf_pred)

# %% slideshow={"slide_type": "subslide"}
gb_clf = GradientBoostingRegressor()
gb_clf.fit(x_train, y_train)
y_gb_pred = gb_clf.predict(x_test)

mean_absolute_error(y_test, y_gb_pred), mean_squared_error(y_test, y_gb_pred)

# %% [markdown] slideshow={"slide_type": "slide"}
#
#  ## Feature Engineering

# %% slideshow={"slide_type": "subslide"}
x = rng.uniform(size=(150, 1), low=0.0, high=10.0)
x_train, x_test = x[:100], x[100:]
x_plot = np.linspace(0, 10, 500)
x_train[:3]

# %% slideshow={"slide_type": "subslide"}
y_lin_train = lin(x_train).reshape(-1)
y_lin_test = lin(x_test).reshape(-1)
y_fun_train = fun(x_train.reshape(-1))
y_fun_test = fun(x_test).reshape(-1)

# %% slideshow={"slide_type": "subslide"}
x_squares = x * x
x_squares[:3]

# %% slideshow={"slide_type": "subslide"}
x_sins = np.sin(x)
x_sins[:3]

# %% slideshow={"slide_type": "subslide"}
x_train_aug = np.concatenate([x_train, x_train * x_train, np.sin(x_train)], axis=1)
x_train_aug[:3]

# %% slideshow={"slide_type": "subslide"}
x_test_aug = np.concatenate([x_test, x_test * x_test, np.sin(x_test)], axis=1)

# %% slideshow={"slide_type": "subslide"}
# from sklearn.linear_model import Ridge
# lr_aug_lin = Ridge()
lr_aug_lin = LinearRegression()
lr_aug_lin.fit(x_train_aug, y_lin_train);

# %% slideshow={"slide_type": "subslide"}
lr_aug_lin.coef_, lr_aug_lin.intercept_


# %% slideshow={"slide_type": "subslide"}
y_aug_lin_pred = lr_aug_lin.predict(x_test_aug)

# %% slideshow={"slide_type": "subslide"}
mean_absolute_error(y_lin_test, y_aug_lin_pred), mean_squared_error(
    y_lin_test, y_aug_lin_pred
)

# %% slideshow={"slide_type": "subslide"}
x_test.shape, x_plot.shape


# %% slideshow={"slide_type": "subslide"}
def train_and_plot_aug(f_y, scale=0.5):
    y_plot = f_y(x_plot)
    
    f_r = lambda x: randomize(f_y, x, scale=scale)
    y_train = f_r(x_train_aug[:, 0])
    y_test = f_r(x_test)
    
    lr_aug = LinearRegression() # Try with Ridge() as well...
    lr_aug.fit(x_train_aug, y_train)
    y_pred_test = lr_aug.predict(
                      np.concatenate([x_test, x_test * x_test, np.sin(x_test)], axis=1)
                   )
    x_plot2 = x_plot.reshape(-1, 1)
    y_pred_plot = lr_aug.predict(
                     np.concatenate([x_plot2, x_plot2 * x_plot2, np.sin(x_plot2)], axis=1)
                  )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=x_plot2[:, 0], y=y_plot, color="orange")
    sns.scatterplot(x=x_plot2[:, 0], y=y_pred_plot, color="red")
    sns.scatterplot(x=x_train_aug[:, 0], y=y_train, color="green")
    plt.show()

    mae_in = mean_absolute_error(y_test, y_pred_test)
    mse_in = mean_absolute_error(y_test, y_pred_test)
    rmse_in = np.sqrt(mse_in)

    y_nr = f_y(x_test)
    mae_true = mean_absolute_error(y_nr, y_pred_test)
    mse_true = mean_absolute_error(y_nr, y_pred_test)
    rmse_true = np.sqrt(mse_true)

    print(f"Vs. input: MAE: {mae_in:.2f}, MSE: {mse_in:.2f}, RMSE: {rmse_in:.2f}")
    print(f"True:      MAE: {mae_true:.2f}, MSE: {mse_true:.2f}, RMSE: {rmse_true:.2f}")
    print(f"Parameters: {lr_aug.coef_}, {lr_aug.intercept_}")


# %% slideshow={"slide_type": "subslide"}
train_and_plot_aug(lin)

# %% slideshow={"slide_type": "subslide"}
train_and_plot_aug(fun, scale=0.0)

# %% slideshow={"slide_type": "subslide"}
train_and_plot_aug(fun, scale=0.5)

# %% slideshow={"slide_type": "subslide"}
train_and_plot_aug(fun, scale=1.5)

# %% slideshow={"slide_type": "subslide"}
train_and_plot_aug(fun, scale=3)


# %%
def fun2(x): return 2.8 * np.sin(x) + 0.3 * x + 0.08 * x ** 2 - 2.5

train_and_plot_aug(fun2, scale=1.5)

# %%
train_and_plot_aug(lambda x: np.select([x<=6, x>6], [-0.5, 3.5]))

# %%
