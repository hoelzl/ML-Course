# %%
from sklearn.datasets import fetch_openml

# %%
mnist = (
    globals()["mnist"]
    if "mnist" in globals()
    else fetch_openml("mnist_784", version=1)
)

# %%
type(mnist)

# %%
mnist.keys()

# %%
mnist["data"]

# %%
mnist.data

# %%
mnist.data.shape

# %%
28 * 28

# %%
mnist.target

# %%
type(mnist.data)

# %%
type(mnist.target)

# %%
x, y = mnist.data.to_numpy(), mnist.target.to_numpy()

# %%
type(x)

# %%
x.shape

# %%
y.shape

# %%
first_digit = x[0]
second_digit = x[1]

# %%
type(first_digit)

# %%
first_digit.shape

# %%
first_digit_image = first_digit.reshape(28, 28)
second_digit_image = second_digit.reshape(28, 28)

# %%
first_digit_image.shape

# %%
import matplotlib.pyplot as plt  # noqa: E402

# %%
plt.imshow(first_digit_image, cmap="binary")
plt.show()

# %%
plt.imshow(second_digit_image, cmap="binary")
plt.show()

# %%
y[:2]

# %%
import numpy as np  # noqa: E402

# %%
y = y.astype(np.int32)

# %%
x_train, x_test = x[:60_000], x[60_000:]

# %%
y_train, y_test = y[:60_000], y[60_000:]

# %%
np.array([1, 2, 3])

# %%
[1, 2, 3] == [1, 3, 3]  # noqa: B015

# %%
np.array([1, 2, 3]) == np.array([1, 3, 3])  # noqa: B015

# %%
lucky7_train = y_train == 7
lucky7_test = y_test == 7

# %%
lucky7_test[:3], y_test[:3]

# %%
plt.imshow(x_test[0].reshape(28, 28), cmap="binary")

# %%
from sklearn.linear_model import SGDClassifier  # noqa: E402

# %%
sgd_clf = SGDClassifier(random_state=42)

# %%
sgd_clf.fit(x_train, lucky7_train)

# %%
sgd_clf.predict([first_digit])

# %%
sgd_clf.predict(x_test[:3])

# %%
lucky7_predict = sgd_clf.predict(x_test)

# %%
correct_predictions = lucky7_predict == lucky7_test

# %%
correct_predictions[:10]

# %%
np.sum(correct_predictions)

# %%
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# %%
accuracy_score(lucky7_test, lucky7_predict)

# %%
balanced_accuracy_score(lucky7_test, lucky7_predict)

# %%
precision_score(lucky7_test, lucky7_predict)

# %%
f1_score(lucky7_test, lucky7_predict)


# %%
def print_scores(predictions):
    accuracy = accuracy_score(lucky7_test, predictions) * 100
    balanced_accuracy = balanced_accuracy_score(lucky7_test, predictions) * 100
    precision = (
        precision_score(lucky7_test, predictions, zero_division=0) * 100
    )
    recall = recall_score(lucky7_test, predictions, zero_division=0) * 100
    f1 = f1_score(lucky7_test, predictions, zero_division=0) * 100
    print(f"Accuracy:          {accuracy:.1f}%")
    print(f"Balanced Accuracy: {balanced_accuracy:.1f}%")
    print(f"Precision:         {precision:.1f}%")
    print(f"Recall:            {recall:.1f}%")
    print(f"F1:                {f1:.1f}%")


# %%
print_scores(lucky7_predict)

# %%
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

# %%
dt_clf = DecisionTreeClassifier()

# %%
dt_clf.fit(x_train, lucky7_train)

# %%
lucky7_predict_dt = dt_clf.predict(x_test)

# %%
print_scores(lucky7_predict_dt)

# %%
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

# %%
rf_clf = RandomForestClassifier(random_state=42)

# %%
rf_clf.fit(x_train, lucky7_train)

# %%
lucky7_predict_rf = rf_clf.predict(x_test)

# %%
print_scores(lucky7_predict_rf)

# %%
from sklearn.base import BaseEstimator  # noqa: E402


# %%
class UnluckyClassifier(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros((len(X),), dtype=bool)


# %%
unlucky_clf = UnluckyClassifier()

# %%
unlucky_clf.fit(x_train, lucky7_train)

# %%
unlucky_predict = unlucky_clf.predict(x_test)

# %%
print_scores(unlucky_predict)

# %% [markdown]
#
# # Multiclass prediction

# %%
rf_clf = RandomForestClassifier()

# %%
rf_clf.fit(x_train, y_train)

# %%
pred_rf = rf_clf.predict(x_test)


# %%
def print_scores(predictions):
    print(
        f"Accuracy:          {accuracy_score(y_test, predictions) * 100:.1f}%"
    )
    print(
        f"Balanced Accuracy: {balanced_accuracy_score(y_test, predictions) * 100:.1f}%"
    )


# %%
print_scores(pred_rf)

# %%
print_scores(np.zeros((len(x_test), 1)))

# %%
