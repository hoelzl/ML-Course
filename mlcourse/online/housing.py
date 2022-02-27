# %%
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# %%
california_housing = fetch_california_housing()

# %%
pprint(california_housing)

# %%
dir(california_housing)

# %%
pprint(california_housing.data)

# %%
pprint(california_housing.data[0])

# %%
print(california_housing.feature_names)

# %%
pprint(california_housing.target)

# %% [markdown]
# # Data Frames

# %%
simple_df = pd.DataFrame(
    data={"attr_1": [1, 2, 3], "attr_2": ["a", "b", "c"], "attr_3": [0.1, 0.5, 0.2]}
)

# %%
simple_df.info()

# %%
display(simple_df)

# %%
simple_df.describe()

# %%
all_data = np.concatenate(
    [california_housing.data, california_housing.target.reshape(-1, 1)], axis=1
)

# %%
all_data.shape

# %%
all_columns = [*california_housing.feature_names, "Target"]

# %%
pprint(all_columns, compact=True)
print("Length =", len(all_columns))

# %%
housing_df = pd.DataFrame.from_records(data=all_data, columns=all_columns)

# %%
display(housing_df)

# %%
housing_df.info()

# %%
housing_df.describe()

# %%
california_housing_v2 = fetch_california_housing(as_frame=True)

# %%
california_housing_v2.frame

# %%
california_housing.frame


# %%
x, y = california_housing.data, california_housing.target

# %%
pprint(x)
pprint(y)

# %%
plt.hist(x=x[0], bins=50)

# %%
plt.hist(x=x[1], bins=50)

# %%
sns.histplot(data=x[0])

# %%
sns.set_theme()

# %%
sns.histplot(data=x[0])

# %%
plt.hist(x=x[1], bins=50)

# %%
housing_df.hist(figsize=(15, 9))


# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)

# %%
x_train.shape, x_test.shape

# %%
y_train.shape, y_test.shape

# %%
pprint(y[:5], compact=True)
pprint(y_train[:5], compact=True)
pprint(y_test[:5], compact=True)


# %%
train_df, test_df = train_test_split(housing_df, test_size=0.25, random_state=42)

# %%
train_df

# %%
lat_idx = all_columns.index("Latitude")
print(lat_idx)
lng_idx = all_columns.index("Longitude")
print(lng_idx)

# %%
plt.scatter(x=x[:, lng_idx], y=x[:, lat_idx])

# %%
plt.scatter(x=x[:, lng_idx], y=x[:, lat_idx], alpha=0.15)

# %%
sns.scatterplot(x=x[:, lng_idx], y=x[:, lat_idx])

# %%
sns.scatterplot(x=x[:, lng_idx], y=x[:, lat_idx], alpha=0.55)

# %%
housing_df.plot(kind="scatter", x="Longitude", y="Latitude", figsize=(6, 6), alpha=0.5)

# %%
housing_df.plot(
    kind="scatter",
    x="Longitude",
    y="Latitude",
    alpha=0.4,
    s=housing_df["Population"] / 50,
    figsize=(8, 6),
    c="Target",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)
plt.legend()

# %%
sns.pairplot(housing_df[:1000])

# %%
sgd_regressor = SGDRegressor()

# %%
sgd_regressor.fit(x_train, y_train)

# %%
sgd_pred = sgd_regressor.predict(x_test)

# %%
mean_squared_error(y_test, sgd_pred)

# %%
tree_regressor = DecisionTreeRegressor()

# %%
tree_regressor.fit(x_train, y_train)

# %%
tree_predict = tree_regressor.predict(x_test)

# %%
mean_squared_error(y_test, tree_predict)

# %%
mean_squared_error(y_train, tree_regressor.predict(x_train))

# %%
rf_regressor = RandomForestRegressor()

# %%
rf_regressor.fit(x_train, y_train)

# %%
rf_predict = rf_regressor.predict(x_test)

# %%
mean_squared_error(y_test, rf_predict)

# %%
mean_squared_error(y_train, rf_regressor.predict(x_train))

# %%
scaler = StandardScaler()

# %%
scaler.fit(x_train)

# %%
x_train_scaled = scaler.transform(x_train)

# %%
x_train[0]

# %%
x_train_scaled[0]

# %%
# x_train_scaled = scaler.fit_transform(x_train)

# %%
sgd_scaled_regressor = SGDRegressor()

# %%
sgd_scaled_regressor.fit(x_train_scaled, y_train)

# %%
sgd_scaled_pred = sgd_scaled_regressor.predict(scaler.transform(x_test))

# %%
mean_squared_error(y_test, sgd_scaled_pred)

# %%
mean_squared_error(y_train, sgd_scaled_regressor.predict(scaler.transform(x_train)))

# %%
x_sample = np.arange(10)
y_sample = 2 * x_sample + 1

# %%
