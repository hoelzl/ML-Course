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
# <a href="https://colab.research.google.com/github/hoelzl/ML-Course/blob/master/notebooks/nb020_welcome_mnist_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% id="vdotYgt_WVO6"
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# %% id="CoUMrPkOW6Mn"
mnist = fetch_openml("mnist_784", version=1)

# %% id="8NXLWQ7KW_id"
X, y = mnist.data, mnist.target

# %% colab={"base_uri": "https://localhost:8080/"} id="aq9I337pXHDc" outputId="c377f988-48aa-4aa4-8042-2926f6c52e06"
X.shape, y.shape

# %% id="jQUsMkhbXJrU"
X_train, X_test = X[:60_000], X[60_000:]
y_train, y_test = y[:60_000].astype(np.int32), y[60_000:].astype(np.int32)

# %% colab={"base_uri": "https://localhost:8080/"} id="iFdnk9KTXP1D" outputId="3f8b7ade-d953-4284-abc3-82a66d36bda4"
X_train.shape, X_test.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="IA2HKxWcXZpK" outputId="38df9055-ef7a-4a1c-e077-12205dff712c"
y_train.shape, y_test.shape

# %% id="fRFi0pBNeDWW"
sgd_clf = SGDClassifier()

# %% colab={"base_uri": "https://localhost:8080/"} id="8_lp7VY7e1L1" outputId="e5f4a32f-4cb1-41d2-b07e-3c87ae0b824c"
y_train[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="9SjpRh96emVx" outputId="bb00833f-e094-4df7-cd98-341229c4cef3"
sgd_clf.fit(X_train, y_train)

# %% id="p981HdKge-3_"
pred_sgd = sgd_clf.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="pjabHSyeg9cP" outputId="4b1ce87a-a85e-420c-ed92-b008f17917a8"
y_test[:10]

# %% colab={"base_uri": "https://localhost:8080/"} id="E-ZwXBtahDiN" outputId="f2aa9c43-6a97-44b0-a016-40cd30a835fa"
pred_sgd[:10]


# %% id="a9QHwfsyhFk9"
def print_scores(predictions):
    print(f"Accuracy:          {accuracy_score(y_test, predictions) * 100:.1f}%")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, predictions) * 100:.1f}%")


# %% colab={"base_uri": "https://localhost:8080/"} id="UZ-Y-KdUhRhU" outputId="34c6830e-1680-44d9-c716-6b6da2ec0ed7"
print_scores(pred_sgd)

# %% colab={"base_uri": "https://localhost:8080/"} id="01vo8KV-hdt8" outputId="2456bb5c-64d7-43d6-8a48-0324730d849c"
print_scores(np.zeros((len(X_test),)))

# %% id="9wvUoqAah-0B"
dt_clf = DecisionTreeClassifier()

# %% colab={"base_uri": "https://localhost:8080/"} id="RjsP4WbBihTP" outputId="99184c51-3ede-4e2f-e9dd-27674416f668"
dt_clf.fit(X_train, y_train)

# %% id="i4N7cgdtimCV"
pred_dt = dt_clf.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="tJZzv_qIixuO" outputId="af5c4fec-f5e9-4200-c9bd-ebe504e09fed"
print_scores(pred_dt)

# %% id="n_RWQSOKi0Jk"
rf_clf = RandomForestClassifier()

# %% colab={"base_uri": "https://localhost:8080/"} id="4axMpHFqi-Vc" outputId="46abdc3a-cc60-4421-c10e-3172b06afc80"
rf_clf.fit(X_train, y_train)

# %% id="6D9TQJz-jCsD"
pred_rf = rf_clf.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="AT6C903ej2H4" outputId="ca8512d4-0a19-4511-937a-d4de727712aa"
print_scores(pred_rf)

# %% id="CobXWETpj4t4"
gbt_clf = GradientBoostingClassifier()

# %% colab={"base_uri": "https://localhost:8080/"} id="rcH7TTiMkLnO" outputId="90ee51d7-734e-4a3d-bbbf-26e807488d19"
gbt_clf.fit(X_train, y_train)

# %% id="75lUmoH2zZfY"
pred_gbt = gbt_clf.predict(X_test)

# %% colab={"base_uri": "https://localhost:8080/"} id="2pGjr1Ag06Om" outputId="f73166c0-4d88-4068-a554-d0c4b179c96a"
print_scores(pred_gbt)

# %% id="ZoR7VlSM0-uW"
