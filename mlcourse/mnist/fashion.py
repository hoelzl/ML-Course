# %%
from sklearn.datasets import fetch_openml
import pickle

from mlcourse.config import Config

# %%
config = Config()
f_mnist_pkl_path = config.data_dir_path / "external/f_mnist_2.pkl"

# %%
if f_mnist_pkl_path.exists():
    with open(f_mnist_pkl_path, "rb") as file:
        f_mnist = pickle.load(file)

# %%
f_mnist = (
    globals()["f_mnist"] if "f_mnist" in globals() else fetch_openml(data_id=40996)
)

# %%
# f_mnist = fetch_openml(data_id=40996)
# with open(f_mnist_pkl_path, "wb") as file:
#     pickle.dump(f_mnist, file)

# %%
type(f_mnist)

# %%
f_mnist.keys()

# %%
f_mnist["data"]

# %%
f_mnist.data

# %%
f_mnist.data.shape

# %%
28 * 28

# %%
f_mnist.target

# %%
f_mnist.categories

# %%
type(f_mnist.data)

# %%
type(f_mnist.target)

# %%
x, y = f_mnist.data.to_numpy(), f_mnist.target.to_numpy()

# %%
type(x)

# %%
x.shape

# %%
y.shape

# %%
first_item = x[0]
second_item = x[1]

# %%
type(first_item)

# %%
first_item.shape

# %%
first_item_image = first_item.reshape(28, 28)
second_digit_image = second_item.reshape(28, 28)

# %%
first_item_image.shape

# %%
import matplotlib.pyplot as plt  # noqa: E402

# %%
plt.imshow(first_item_image, cmap="binary")
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
def show_item(index):
    image = x[index].reshape(28, 28)
    label = y[index]
    plt.imshow(image, cmap="binary")
    plt.show()
    print("Item", index, "Label =", label)


# %%
show_item(0)


# %%
def show_items_with_label(label, num_items=3):
    items_shown = 0
    for i in range(len(y)):
        if y[i] == label:
            items_shown += 1
            show_item(i)
        if items_shown >= num_items:
            break


# %%
show_items_with_label(1)


# %%
def show_all_item_kinds():
    for label in range(10):
        show_items_with_label(label)


# %%
show_all_item_kinds()

# %%
trousers_train = y_train == 1
trousers_test = y_test == 1

# %%
trousers_test[:3], y_test[:3]

# %%
plt.imshow(x_test[0].reshape(28, 28), cmap="binary")
plt.show()

# %%
from sklearn.linear_model import SGDClassifier  # noqa: E402

# %%
sgd_clf = SGDClassifier(random_state=42)

# %%
sgd_clf.fit(x_train, trousers_train)

# %%
sgd_clf.predict([first_item])

# %%
sgd_clf.predict(x_test[:3])

# %%
trousers_predict = sgd_clf.predict(x_test)

# %%
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


# %%
def print_scores(predictions):
    accuracy = accuracy_score(trousers_test, predictions) * 100
    balanced_accuracy = balanced_accuracy_score(trousers_test, predictions) * 100
    precision = precision_score(trousers_test, predictions, zero_division=0) * 100
    recall = recall_score(trousers_test, predictions, zero_division=0) * 100
    f1 = f1_score(trousers_test, predictions, zero_division=0) * 100
    print(f"Accuracy:          {accuracy:.1f}%")
    print(f"Balanced Accuracy: {balanced_accuracy:.1f}%")
    print(f"Precision:         {precision:.1f}%")
    print(f"Recall:            {recall:.1f}%")
    print(f"F1:                {f1:.1f}%")


# %%
print_scores(trousers_predict)

# %%
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

# %%
dt_clf = DecisionTreeClassifier()

# %%
dt_clf.fit(x_train, trousers_train)

# %%
trousers_predict_dt = dt_clf.predict(x_test)

# %%
print_scores(trousers_predict_dt)

# %%
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

# %%
rf_clf = RandomForestClassifier(random_state=42)

# %%
rf_clf.fit(x_train, trousers_train)

# %%
trousers_predict_rf = rf_clf.predict(x_test)

# %%
print_scores(trousers_predict_rf)

# %%
