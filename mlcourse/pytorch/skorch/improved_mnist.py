# %% [markdown]
#
# # Trainin Neural Networks using Skorch
#
# Skorch is a library for PyTorch that simplifies training in a Scikit Learn

# %%
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_model_summary import summary
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler
from torch.optim.lr_scheduler import CyclicLR

from mlcourse.config import Config

# %%
config = Config()
model_path = config.data_dir_path / "saved_models"
model_path.mkdir(parents=True, exist_ok=True)

# %%
np.set_printoptions(precision=1)

# %%
mnist = globals().get("mnist") or fetch_openml("mnist_784", version=1)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
x = mnist.data.to_numpy().reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
y = mnist.target.to_numpy().astype(np.int64)

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=10_000, random_state=42
)


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
class ConvNet(nn.Module):
    def __init__(self, kernels_1=10, kernels_2=20, hidden=60):
        self.kernels_2 = kernels_2
        super().__init__()
        self.conv1 = nn.Conv2d(1, kernels_1, kernel_size=5, stride=(2, 2))
        self.conv2 = nn.Conv2d(kernels_1, kernels_2, kernel_size=5, stride=(2, 2))
        self.fc1 = nn.Linear(16 * kernels_2, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 16 * self.kernels_2)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


# %%
net = ConvNet()
print(summary(net, torch.zeros((1, 1, 28, 28)), show_input=True))
print(summary(net, torch.zeros((1, 1, 28, 28)), show_input=False))


# %%
net2 = ConvNet(kernels_1=25, kernels_2=40, hidden=128)
print(summary(net2, torch.zeros((1, 1, 28, 28)), show_input=True))
print(summary(net2, torch.zeros((1, 1, 28, 28)), show_input=False))


# %%
conv_1_clf = NeuralNetClassifier(
    ConvNet, max_epochs=20, lr=0.1, iterator_train__shuffle=True, device=device
)

# %%
conv_1_clf.fit(x_train, y_train)

# %%
y_pred_conv_1 = conv_1_clf.predict(x_test)

# %%
print_scores(y_test, y_pred_conv_1)

# %%
conv_2_clf = NeuralNetClassifier(
    ConvNet,
    max_epochs=20,
    lr=0.1,
    iterator_train__shuffle=True,
    module__kernels_1=16,
    module__kernels_2=32,
    module__hidden=128,
    callbacks=[
        ("lr_scheduler", LRScheduler(policy=CyclicLR, base_lr=0.05, max_lr=0.15))
    ],
    device=device,
    verbose=True,
)

# %%
conv_2_clf.fit(
    x_train,
    y_train,
)

# %%
y_pred_conv_2 = conv_2_clf.predict(x_test)

# %%
print_scores(y_test, y_pred_conv_2)

# %%
search_clf = NeuralNetClassifier(
    ConvNet,
    max_epochs=10,
    lr=0.1,
    iterator_train__shuffle=True,
    callbacks=[
        ("lr_scheduler", LRScheduler(policy=CyclicLR, base_lr=0.05, max_lr=0.15))
    ],
    device=device,
    verbose=False,
)

# %%
search = RandomizedSearchCV(
    search_clf,
    n_iter=25,
    param_distributions=[
        {
            "module__kernels_1": [5, 10, 20],
            "module__kernels_2": [15, 30, 60],
            "module__hidden": [60, 120, 360],
            "callbacks__lr_scheduler__base_lr": [0.25, 0.1, 0.05],
            "callbacks__lr_scheduler__max_lr": [0.5],
        },
        {
            "module__kernels_1": [30, 60],
            "module__kernels_2": [60, 120, 240],
            "module__hidden": [360, 720],
            "callbacks__lr_scheduler__base_lr": [0.25, 0.1, 0.05],
            "callbacks__lr_scheduler__max_lr": [0.5],
        },
        {
            "module__kernels_1": [5, 10, 20],
            "module__kernels_2": [15, 30, 60],
            "module__hidden": [60, 120, 360],
            "callbacks__lr_scheduler__base_lr": [0.1, 0.05],
            "callbacks__lr_scheduler__max_lr": [0.25, 0.15],
        },
        {
            "module__kernels_1": [30, 60],
            "module__kernels_2": [60, 120, 240],
            "module__hidden": [360, 720],
            "callbacks__lr_scheduler__base_lr": [0.1, 0.05],
            "callbacks__lr_scheduler__max_lr": [0.25, 0.15],
        },
    ],
)

# %%
search.fit(x_train, y_train)

# %%
search.best_estimator_, search.best_params_


# %%
y_pred_search = search.predict(x_test)

# %%
print_scores(y_test, y_pred_search)

# %%
model_file = model_path / "mnist_conv_randomized_search.pkl"

# %%
joblib.dump(search, model_file)

# %%
search_loaded = joblib.load(model_file)

# %%
y_pred_loaded = search_loaded.predict(x_test)

# %%
print_scores(y_test, y_pred_loaded)

# %%
