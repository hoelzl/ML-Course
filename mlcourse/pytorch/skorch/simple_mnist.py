# %% [markdown]
#
# # Training Neural Networks using Skorch
#
# Skorch is a library for PyTorch that simplifies training in a Scikit Learn

# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetClassifier

from mlcourse.config import Config

# %%
config = Config()
mnist_pkl_path = config.data_dir_path / "external/mnist.pkl"

# %%
np.set_printoptions(precision=1)

# %%
if mnist_pkl_path.exists():
    with open(mnist_pkl_path, "rb") as file:
        mnist = pickle.load(file)

# %%
mnist = globals().get("mnist") or fetch_openml("mnist_784", version=1)

# %% [markdown]
#
# ## Transforming the data:
#
# Neural nets generally expect their inputs as `float32` (or possibly `float16`)
# values. Furthermore `skorch` expects classes to be stored as `int64` values,
# so we change the type of the arrays accordingly.


# %%
x = mnist.data.to_numpy().reshape(-1, 1, 28, 28).astype(np.float32)
y = mnist.target.to_numpy().astype(np.int64)
print("Shape of x:", x.shape, "- type of x:", x.dtype)
print("Shape of y:", y.shape, "- type of y:", y.dtype)

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=10_000, random_state=42
)
print("Shape of x train:", x_train.shape, "- shape of y_train:", y_train.shape)
print("Shape of x test: ", x_test.shape, "- shape of y_test: ", y_test.shape)

# %% [markdown]
#
# ## Normalize / Standardize
#
# Neural networks generally prefer their input to be in the range $(0, 1)$ or
# $(-1, 1)$ so we need to convert the integer array to floating point:

# %%
print(x_train[0, 0, 20:24, 10:14])

# %% [markdown]
#
# Don't use `Standard Scaler` for this data, since it will scale each feature
# independently:

# %%
scaler = StandardScaler()
x_train_scaled_with_scaler = scaler.fit_transform(x_train.reshape(-1, 28 * 28)).reshape(
    -1, 1, 28, 28
)
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
axs[0].imshow(x_train[0, 0], cmap="binary")
axs[1].imshow(x_train_scaled_with_scaler[0, 0], cmap="binary")

# %% [markdown]
#
# Since we know the range of the value, we can easily perform the processing
# manually:

# %%
x_train = x_train / 255.0
x_test = x_test / 255.0

# %%
print(x_train[0, 0, 20:24, 10:14])
plt.imshow(x_train[0, 0], cmap="binary")


# %%
mlp_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.Softmax(dim=1),
)

# %%
mlp_clf = NeuralNetClassifier(
    mlp_model, max_epochs=10, lr=0.1, iterator_train__shuffle=True
)

# %%
mlp_clf.fit(x_train, y_train)

# %%
y_pred_mlp = mlp_clf.predict(x_test)


# %%
def print_scores(y, y_pred):
    print(f"Accuracy:          {accuracy_score(y, y_pred) * 100:.1f}%")
    print(f"Balanced accuracy: {balanced_accuracy_score(y, y_pred) * 100:.1f}%")
    print(
        f"Precision (macro): {precision_score(y, y_pred, average='macro') * 100:.1f}%"
    )
    print(f"Recall (macro):    {recall_score(y, y_pred, average='macro') * 100:.1f}%")
    print(f"F1 (macro):        {f1_score(y, y_pred, average='macro') * 100:.1f}%")


# %%
print_scores(y_test, y_pred_mlp)

# %%
conv_model = nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5, stride=(2, 2)),
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=5, stride=(2, 2)),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(320, 60),
    nn.ReLU(),
    nn.Linear(60, 10),
    nn.Softmax(dim=1),
)

# %%
conv_clf = NeuralNetClassifier(
    conv_model, max_epochs=20, lr=0.1, iterator_train__shuffle=True
)

# %%
conv_clf.fit(x_train, y_train)

# %%
y_pred_conv = conv_clf.predict(x_test)

# %%
print_scores(y_test, y_pred_conv)

# %%
