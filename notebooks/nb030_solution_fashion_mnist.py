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
# <a href="https://colab.research.google.com/github/hoelzl/ML-Course/blob/master/notebooks/nb030_solution_fashion_mnist.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% id="z1-qVdLjhEQ4"
from sklearn.datasets import fetch_openml

# %% id="USJxPp7olMC4"
f_mnist = fetch_openml(data_id=40996)

# %% colab={"base_uri": "https://localhost:8080/"} id="Lb6daEN5lSR2" outputId="212397a8-6288-46ae-e799-814294218edb"
type(f_mnist)

# %% colab={"base_uri": "https://localhost:8080/"} id="AHSFWGyYliR-" outputId="33ea8cdd-412c-43e9-f828-5859c3e34cd9"
f_mnist.keys()

# %% colab={"base_uri": "https://localhost:8080/"} id="etSSb6rKllVG" outputId="2bdbd122-1312-483f-9c82-2b5d6912d2b5"
f_mnist.data

# %% colab={"base_uri": "https://localhost:8080/"} id="fcA5FjhAlqPN" outputId="412b1edc-4f4a-4500-d5c8-0638f871d099"
f_mnist.data.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="QKYyt9E4lt4F" outputId="767fd21f-3dce-4dab-c8a9-e865259b8eb8"
f_mnist.target

# %% colab={"base_uri": "https://localhost:8080/"} id="pZUaL54Tl1It" outputId="246da150-0a2f-46f0-baf7-73f44e47d1c6"
f_mnist.target_names

# %% colab={"base_uri": "https://localhost:8080/", "height": 146} id="W8KV98KYl7A1" outputId="1e287316-e81c-4da9-a7ee-5c470b0ca959"
f_mnist.DESCR

# %% colab={"base_uri": "https://localhost:8080/"} id="2ca0rhTcl9XF" outputId="327bb624-1a5a-40a6-d448-a894d82732ba"
print(f_mnist.DESCR)

# %% id="ggiNXV3kmCWs"
x, y = f_mnist.data, f_mnist.target

# %% id="PO1IK8zVmOWl"
import matplotlib.pyplot as plt

# %% colab={"base_uri": "https://localhost:8080/", "height": 283} id="Lifi7FosmaF8" outputId="c94d77c2-61f3-4c8c-878e-22d76b57bd1f"
plt.imshow(x[0].reshape(28, 28), cmap="binary")

# %% id="ljfvQ5hlmg9I"
import numpy as np

# %% id="C5W5OEr2mpJL"
y = y.astype(np.int32)

# %% id="aG9ADP_pmr_7"
x_train, x_test = x[:60_000], x[60_000:]
y_train, y_test = y[:60_000], y[60_000:]


# %% id="EWIO4-N0nAGq"
def show_item(index):
  image = x[index].reshape(28, 28)
  label = y[index]
  plt.imshow(image, cmap="binary")
  plt.show()
  print("Item =", index, "label =", label)


# %% colab={"base_uri": "https://localhost:8080/", "height": 283} id="1NrdBz0infQw" outputId="49368c0b-368d-4ccb-8c53-36949c7f7f22"
show_item(1)


# %% id="taQ33edfnhdR"
def show_items_with_label(label, num_items=3):
  items_shown = 0
  for i in range(len(y)):
    if y[i] == label:
      items_shown += 1
      show_item(i)
    if items_shown >= num_items:
      break


# %% colab={"base_uri": "https://localhost:8080/", "height": 816} id="nkBTvnjEoH9f" outputId="74a33021-6055-428b-8ae7-b91ce302cb56"
show_items_with_label(3)


# %% id="DNMohW88oLCf"
def show_all_item_kinds():
  for label in range(10):
    show_items_with_label(label)



# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="El3DFnffoVZ-" outputId="2f92b8b9-6422-47b5-a272-e3769bde9fba"
show_all_item_kinds()

# %% id="zAiXBx_HoW92"
trousers_train = y_train == 1
trousers_test = y_test == 1

# %% colab={"base_uri": "https://localhost:8080/"} id="3jTKWmcao4M0" outputId="82e1c47b-ce9e-4dd9-b2cd-d777f49ee00b"
trousers_test[:6]

# %% colab={"base_uri": "https://localhost:8080/"} id="Bfcdod6po8lc" outputId="89987d01-63d5-458a-e7b2-a0af12971307"
y_test[:6]

# %% id="WwJnx3kapAx9"
from sklearn.linear_model import SGDClassifier

# %% id="e1uINa0IpJaU"
sgd_clf = SGDClassifier(random_state=42)

# %% colab={"base_uri": "https://localhost:8080/"} id="5x_vnMwBpN8k" outputId="fc8368b0-48fa-4684-ebb2-9763b1d384fe"
sgd_clf.fit(x_train, trousers_train)

# %% id="qYC0gb9xpSV7"
trousers_predict = sgd_clf.predict(x_test)

# %% id="Fq-okPzPpbWj"
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# %% id="uFHx-JuGpeMK"
def print_scores(predictions):
    print(f"Accuracy:          {accuracy_score(trousers_test, predictions) * 100:.1f}%")
    print(f"Balanced Accuracy: {balanced_accuracy_score(trousers_test, predictions) * 100:.1f}%")
    print(f"Precision:         {precision_score(trousers_test, predictions, zero_division=0) * 100:.1f}%")
    print(f"Recall:            {recall_score(trousers_test, predictions, zero_division=0) * 100:.1f}%")
    print(f"F1:                {f1_score(trousers_test, predictions, zero_division=0) * 100:.1f}%")


# %% colab={"base_uri": "https://localhost:8080/"} id="o8QnH9yXpgfT" outputId="bdc79c8e-bc93-42c1-ed63-a11392918981"
print_scores(trousers_predict)

# %% id="YM8SVWFkplHi"
from sklearn.tree import DecisionTreeClassifier

# %% id="VHnijIv7ptdq"
dt_clf = DecisionTreeClassifier()

# %% colab={"base_uri": "https://localhost:8080/"} id="XXg2lX0MpvtM" outputId="95aea8b3-8310-401a-f661-d054e316a1dc"
dt_clf.fit(x_train, trousers_train)

# %% id="Wr2RVGYJp1KK"
trousers_predict_dt = dt_clf.predict(x_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="0FqfU5kNqKQS" outputId="9d5ef19e-1f71-4149-e474-8c3947bd2906"
print_scores(trousers_predict_dt)

# %% id="kaD8B_IcqNnf"
from sklearn.ensemble import RandomForestClassifier

# %% id="HtE__jHtqWP4"
rf_clf = RandomForestClassifier(random_state=42)

# %% colab={"base_uri": "https://localhost:8080/"} id="JQibuBIYqZUh" outputId="cda3cde7-52a1-4c4b-d30f-0754268b96f9"
rf_clf.fit(x_train, trousers_train)

# %% id="ikuFInXhqdP3"
trousers_predict_rf = rf_clf.predict(x_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="EZCgMXmIqyVv" outputId="9b641dfe-ad72-47b1-b4e4-c4152fe2c409"
print_scores(trousers_predict_rf)

# %% id="0fyGrAPuq1WH"
