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
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/hoelzl/ML-Course/blob/master/notebooks/nb078_solution_california_housing_prices.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% id="6Kg0HRecmOu1"
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_california_housing

# %% colab={"base_uri": "https://localhost:8080/"} id="ZM7Z5aR_vYCT" outputId="19a63be6-64c0-4d38-a1f0-f931ada07aa2"
california_housing = fetch_california_housing()

# %% colab={"base_uri": "https://localhost:8080/"} id="c3FymbbuveVB" outputId="1c68b826-e0c0-4b21-a6bb-54c3b8ae8179"
pprint(california_housing)

# %% colab={"base_uri": "https://localhost:8080/"} id="z_7SZl6Yvj8o" outputId="6fce7364-1592-4ec6-b6ee-c856f8169962"
print(california_housing.DESCR)

# %% colab={"base_uri": "https://localhost:8080/"} id="aD8gm5ljvtr_" outputId="c8301510-00a0-43d2-8d98-feb457dbc830"
california_housing.target

# %% colab={"base_uri": "https://localhost:8080/"} id="BQYsILwyv6h-" outputId="5a5036e8-6930-470a-c1a7-d35e9e4bd53b"
california_housing.data

# %% colab={"base_uri": "https://localhost:8080/"} id="Dclf183wwjIM" outputId="b114b769-9b9a-4e5f-a2cb-fb5b169d6b34"
california_housing.data.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="rieUe4yFwwr7" outputId="c77e225b-ae12-4125-fb0b-bd54eedfba01"
california_housing.target.shape

# %% id="kN3iDY-gxm5g"
simple_df = pd.DataFrame({"attr_1": [1, 2, 3], "attr_2": ["a", "b", "c"], "attr_3": [0.1, 0.5, 0.5]})

# %% colab={"base_uri": "https://localhost:8080/", "height": 143} id="YxKkUi76yncO" outputId="9cd51b02-c844-42e7-f75d-6f79131d9975"
simple_df

# %% colab={"base_uri": "https://localhost:8080/"} id="RaSb15R_yo50" outputId="e0471c44-07d9-4b44-8f09-c490c0802132"
simple_df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="CbKRRJ3Wy9hs" outputId="216d58a1-0c10-405a-8b8a-73c6abb3fd4f"
simple_df.describe()

# %% id="lRndgE2pzioK"
x, y = california_housing.data, california_housing.target

# %% colab={"base_uri": "https://localhost:8080/"} id="eNISIdfL0J6v" outputId="240fa0a5-85d4-4c92-ce6c-776ef2b6b778"
x

# %% colab={"base_uri": "https://localhost:8080/"} id="FL9mzBDg0PC9" outputId="7ced7c0d-9880-49be-b25d-5c0c111f58fb"
x[0]

# %% colab={"base_uri": "https://localhost:8080/"} id="ZTJUEgzm0VOl" outputId="e0983e80-7b83-4191-fa4f-ff7db3602c28"
x[0][1]

# %% colab={"base_uri": "https://localhost:8080/"} id="qzEkdwFv0Zcm" outputId="6956f079-5929-4e90-f20b-d56c3b8a62fe"
x[0, 1]

# %% colab={"base_uri": "https://localhost:8080/"} id="FO_HLxAQ0fLV" outputId="703bf492-d829-4958-9591-09ea2225ab77"
x[:, 1]

# %% colab={"base_uri": "https://localhost:8080/", "height": 112} id="0l0_cBPW0qgN" outputId="01317d91-0576-4d0c-945a-5603b4245c01"
pd.DataFrame.from_records([[1, 2, 3], ["a", "b", "c"]], columns=["attr_1", "attr_2", "attr_3"])

# %% id="cl2tE2441J4D"
housing_df = pd.DataFrame.from_records(x, columns=california_housing.feature_names)

# %% colab={"base_uri": "https://localhost:8080/"} id="zQQ5nDQm1vrJ" outputId="cdf27f63-f256-49b3-ff93-9970dd19b32c"

# %% colab={"base_uri": "https://localhost:8080/"} id="bJYiYqTv2CqY" outputId="29e1b027-7cc3-49a8-8228-dd767132bf7c"
housing_df.info()

# %% colab={"base_uri": "https://localhost:8080/", "height": 300} id="lKqmjdtY2HLv" outputId="0747b756-f7e2-4e00-a163-7b60da123e3a"
housing_df.describe()

# %% colab={"base_uri": "https://localhost:8080/"} id="iOJ46tx92OBv" outputId="a7f79b97-ab25-4749-eb45-626fcd67abae"
housing_df["MedInc"]

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="pI_YXJGS2hym" outputId="0b8ac7d4-8eae-4023-9085-bcadc5c2880b"
housing_df[["MedInc", "HouseAge"]]

# %% colab={"base_uri": "https://localhost:8080/", "height": 424} id="B0OPtQtA2n8V" outputId="f092e156-2a1a-4d38-81d8-497d832738e2"
housing_df[:100]

# %% colab={"base_uri": "https://localhost:8080/"} id="E8h1c_nE27fc" outputId="fcacb07e-8625-4d5a-acd9-6765841ae041"
housing_df.iloc[100]

# %% colab={"base_uri": "https://localhost:8080/", "height": 265} id="g6f8rvtM3Xh6" outputId="af085b4a-f65d-45ca-c30e-6fe45526faee"
plt.hist(x=x[:, 1], bins=50);

# %% colab={"base_uri": "https://localhost:8080/", "height": 553} id="8J56kl4_3wvZ" outputId="83c916e8-4e7d-4329-9df0-41eafecf1c61"
housing_df.hist(figsize=(15,9));

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="oDI3Ow_d4Mhf" outputId="c799545b-4a0a-4480-8295-977fe12a0100"
sns.pairplot(housing_df[:1000])

# %% colab={"base_uri": "https://localhost:8080/", "height": 284} id="KIb3MRRE5riR" outputId="bffcd374-a237-41cb-8dcd-b593f4eb3858"
plt.scatter(x=x[:, 7], y=x[:, 6], alpha=0.15)

# %% id="NhlozsdU6wG2"
from sklearn.linear_model import SGDRegressor

# %% id="WFu7QYQs7TJb"
sgd_regressor = SGDRegressor()

# %% id="o0b1VPrN7XWz"
from sklearn.model_selection import train_test_split

# %% id="X3zJYn-O7qRi"
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# %% colab={"base_uri": "https://localhost:8080/"} id="pnh_cdR_8P1g" outputId="107474df-b103-435d-d75c-8028fc69e8fc"
x_train.shape, x_test.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="bFFjwLL-8U1H" outputId="112f0625-3a6b-4edd-91f1-cd1eb3f62001"
y_train.shape, y_test.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="ykPzAaGa8csQ" outputId="54f9777b-fc07-4399-a2a9-ba87ec70da20"
print(y[:5])
print(y_train[:5])
print(y_test[:5])

# %% colab={"base_uri": "https://localhost:8080/"} id="bWCZhUGO8lJX" outputId="ef71fdae-da04-4d9f-b0d0-ab33413a8b47"
sgd_regressor.fit(x_train, y_train)

# %% id="J0aKOlzB80nG"
sgd_pred = sgd_regressor.predict(x_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="kEMv-hH986Zd" outputId="233cef00-59aa-41eb-ec4f-30b1501b5a81"
sgd_pred[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="0giMNfHy89K0" outputId="a7bf1fb9-da90-474f-e8a8-cf16c9b5a95c"
y_test[:10]

# %% id="uiWssI3p8_6k"
from sklearn.tree import DecisionTreeRegressor

# %% id="jfdRLJq29Qaz"
tree_regressor = DecisionTreeRegressor()

# %% colab={"base_uri": "https://localhost:8080/"} id="KWYKhZN39T50" outputId="77f86d5c-0764-45f6-b1cd-94bb0e26f443"
tree_regressor.fit(x_train, y_train)

# %% id="zCNcQMm49XkD"
tree_predict = tree_regressor.predict(x_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="46UixoPU9fij" outputId="3dc3857e-6298-48d1-ffe2-45cbe84c42ea"
tree_predict[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="UCYaiqnJ9hbU" outputId="93cdd585-6ec7-4c17-bb62-c8a8c53605a5"
y_test[:10]

# %% id="Dz5igseD9kOz"
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %% colab={"base_uri": "https://localhost:8080/"} id="3aPArYeH-UEZ" outputId="faaa77f6-068c-41b3-e602-be9498191457"
mean_absolute_error([1, 2, 3], [1, 1, 1])

# %% colab={"base_uri": "https://localhost:8080/"} id="BFlC7w-x-Zlw" outputId="30f098b2-6cda-4d27-88c2-d6af0c821a83"
mean_squared_error([1, 2, 3], [1, 1, 1])

# %% colab={"base_uri": "https://localhost:8080/"} id="skrG7k8c-jMH" outputId="f7b94fb4-912f-4876-fee4-3be0c8b0815b"
mean_absolute_error(y_test, tree_predict)

# %% colab={"base_uri": "https://localhost:8080/"} id="Ib5k8VSe-2o2" outputId="8632cf3e-3a14-4cc4-e124-bb12f9a39a4d"
mean_absolute_error(y_test, sgd_pred)

# %% id="FP_QaYqo-_1O"
from sklearn.ensemble import RandomForestRegressor

# %% id="TuVB-hXe_Gb9"
rf_regressor = RandomForestRegressor()

# %% colab={"base_uri": "https://localhost:8080/"} id="CZNigxeU_Oos" outputId="dde22edf-d85e-4474-abeb-1453ded7821f"
rf_regressor.fit(x_train, y_train)

# %% id="_PIKFCyE_S_L"
rf_pred = rf_regressor.predict(x_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="EBfqFCFa_YaE" outputId="45b31f0c-ee11-4029-a639-cce3bcd9aab7"
mean_absolute_error(y_test, rf_pred)

# %% id="JKPg5dGT_cFr"
from sklearn.preprocessing import StandardScaler

# %% id="uYBqRnJ4AKJI"
scaler = StandardScaler()

# %% colab={"base_uri": "https://localhost:8080/"} id="JYpT7AI9AeIw" outputId="bb131123-9a85-42b4-bc69-f6bdf39ff841"
scaler.fit(x_train)

# %% id="1mEcg2B5Ahsf"
x_train_scaled = scaler.transform(x_train)

# %% colab={"base_uri": "https://localhost:8080/"} id="TL4cY2aXAp7n" outputId="bd349dd8-3a0f-4635-8520-0592bae20b89"
x_train[0]

# %% colab={"base_uri": "https://localhost:8080/"} id="kwEq2fLnArmG" outputId="9de48c07-8c76-4627-ea2d-521892ec261e"
x_train_scaled[0]

# %% id="yI3iyZDAAwam"
sgd_scaled_regressor = SGDRegressor()

# %% colab={"base_uri": "https://localhost:8080/"} id="lQ1c8qt3BQM8" outputId="06c4e001-b054-43f3-8794-269ef024196e"
sgd_scaled_regressor.fit(x_train_scaled, y_train)

# %% id="j4oL-lt7BUNs"
sgd_scaled_pred = sgd_scaled_regressor.predict(scaler.transform(x_test))

# %% colab={"base_uri": "https://localhost:8080/"} id="6KR7A1UHBryb" outputId="791b3526-0dc9-4332-b216-174a943b6294"
sgd_scaled_pred[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="IsDQ3UPoBxzS" outputId="efa317cd-e0da-4614-bcfa-9aa58543d198"
mean_absolute_error(y_test, sgd_scaled_pred)

# %% colab={"base_uri": "https://localhost:8080/"} id="sVCgk-yhB4aK" outputId="9c8bfc56-f3ad-4a18-c0b2-2f048955990d"
mean_absolute_error(y_train, rf_regressor.predict(x_train))

# %% colab={"base_uri": "https://localhost:8080/"} id="PwcEy8ERC5_F" outputId="4d42679d-0cf5-4cd4-884a-082858056fbe"
mean_absolute_error(y_test, rf_pred)

# %% id="qXyGYLVvDAzO"
