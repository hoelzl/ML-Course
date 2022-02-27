# %%
import matplotlib.pyplot as plt
from nbex.interactive import session
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import train_test_split

# %%
mnist = globals().get("mnist") or fetch_openml("mnist_784", version=1)

# %%
mnist.data.to_numpy().shape

# %%
x = mnist.data.to_numpy().reshape(-1, 28, 28).astype(np.int32)
y = mnist.target.to_numpy().astype(np.int32)

# %%
def show_labeled_image(index):
    plt.imshow(x[index], cmap="binary")
    plt.show()
    y[index]


# %%
if session.is_interactive:
    show_labeled_image(5)

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# %%
x_train.dtype

# %%
x_train.shape, x_test.shape

# %%
y_train.shape, y_test.shape

# %%
x_mean = x_train.mean(axis=0)

# %%
x_mean.shape

# %%
if session.is_interactive:
    plt.imshow(x_mean, cmap="binary")

# %%
ideal_digits = np.zeros((10, 28, 28), dtype=np.int32)

# %%
for i in range(10):
    x_i = x_train[y_train == i]
    ideal_digits[i] = x_i.mean(axis=0)

# %%
plt.imshow(ideal_digits[0], cmap="binary")

# %%
if session.is_interactive:
    fig, ax = plt.subplots(2, 5, figsize=(15, 5))
    for i in range(10):
        ax.reshape(10)[i].imshow(ideal_digits[i], cmap="binary")


# %%
if session.is_interactive:
    plt.imshow(x_test[0], cmap="binary")

# %%
np.set_printoptions(precision=2)

# %%
x_test[0, 3:12, 3:12]

# %%
diff = ideal_digits[8] - x_test[0]
diff[3:12, 3:12]

# %%
plt.imshow(diff)

# %%
(diff * diff)[3:12, 3:12]

# %%
(diff * diff).sum()

# %%
diffs = ideal_digits - x_test[0]
diffs.shape

# %%
errors = (diffs * diffs).sum(axis=(1, 2))
print(errors.shape)
errors.argmin()

# %%
def compute_single_numpy_prediction(img):
    diffs = ideal_digits - img
    return (diffs * diffs).sum(axis=(1, 2)).argmin()


# %%
compute_single_numpy_prediction(x_test[0])

# %%
batched_ideal_digits = np.expand_dims(ideal_digits, axis=0)
batched_ideal_digits.shape

# %%
np.expand_dims(x_test, axis=1).shape

# %%
def compute_numpy_predictions(imgs):
    diffs = ideal_digits - np.expand_dims(imgs, axis=1)
    return (diffs * diffs).sum(axis=(2, 3)).argmin(axis=1)


# %%
pred_numpy = compute_numpy_predictions(x_test)

# %%
pred_numpy.shape


# %%
def print_scores(predictions):
    accuracy = accuracy_score(y_test, predictions) * 100
    balanced_accuracy = balanced_accuracy_score(y_test, predictions) * 100
    print(f"Accuracy:          {accuracy:.1f}%")
    print(f"Balanced Accuracy: {balanced_accuracy:.1f}%")
    print()
    for i in range(10):
        idx = y_test == i
        accuracy_i = accuracy_score(y_test[idx], predictions[idx]) * 100
        print(f"Accuracy for {i}:    {accuracy_i:.1f}%")
    print()


# %%
print_scores(pred_numpy)

# %%
if session.is_interactive:
    plt.imshow(x_mean, cmap="binary")

# %%
rf_clf = RandomForestClassifier()

# %%
x_train.shape

# %%
rf_clf.fit(x_train.reshape(-1, 28 * 28), y_train)

# %%
pred_rf = rf_clf.predict(x_test.reshape(-1, 28 * 28))

# %%
print_scores(pred_rf)

# %%
