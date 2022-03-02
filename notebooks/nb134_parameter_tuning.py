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
# # MNIST with CNN

# %%
import joblib
import numpy as np
from pytorch_model_summary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from mlcourse.config import Config
from sklearn.metrics import classification_report
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from torchvision.transforms.functional import InterpolationMode
from sklearn.model_selection import RandomizedSearchCV

# %%
config = Config()

# %%
input_size = 28 * 28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.005
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# %% slideshow={"slide_type": "subslide"}
mnist_transforms = transforms.Compose(
    [transforms.Resize((28, 28)), transforms.ToTensor()]
)

# %%
augmented_transforms = transforms.Compose(
    [
        transforms.RandomApply(
            [
                transforms.Resize((56, 56)),
                transforms.RandomResizedCrop(
                    28, (0.8, 1.0), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomAffine(
                            degrees=15.0,
                            translate=(0.08, 0.8),
                            interpolation=InterpolationMode.BICUBIC,
                        )
                    ],
                    0.5,
                ),
            ]
        ),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)

# %%
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=augmented_transforms, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=mnist_transforms, download=True
)

# %%
if "x_train" not in globals() or "y_train" not in globals():
    x_train = np.stack([x.numpy() for x, _ in train_dataset])
    y_train = np.array([y for _, y in train_dataset], dtype=np.int64)

# %%
type(x_train), type(y_train)

# %%
x_train.shape, y_train.shape

# %%
x_train.dtype, y_train.dtype

# %%
y_test = np.array([y for _, y in test_dataset])


# %%
class ConvNet(nn.Module):
    def __init__(self, kernels_1=10, kernels_2=20, hidden=60):
        super().__init__()
        self.kernels_2 = kernels_2
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
cnn_classifier = NeuralNetClassifier(
    ConvNet,
    # We added a softmax, so use NLLLoss instead of Cross Entropy
    # criterion=nn.CrossEntropyLoss,
    batch_size=100,
    max_epochs=10,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    iterator_train__shuffle=True,
    train_split=predefined_split(test_dataset),
    device=device,
    verbose=False,
)

# %%
cnn_classifier = NeuralNetClassifier(
    ConvNet,
    batch_size=100,
    max_epochs=10,
    lr=0.1,
    iterator_train__shuffle=True,
    device=device,
    verbose=False,
)

# %%
cnn_classifier.fit(x_train, y_train)

# %% [markdown]
#
# ## Parameter Search with Cross-validation
#
# `RandomizedSearchCV` and `GridSearchCV` are scikit-learn *estimators* that
# perform a search for hyperparameters that lead to the best evaluation metrics.
#
# They use n-fold *cross-validation* to estimate the performance of each
# setting.

# %%
search = RandomizedSearchCV(
    cnn_classifier,
    n_iter=3,  # In reality this should be much higher...
    cv=2,  # Use only two cross validation sets to save training time
    verbose=3,
    n_jobs=8,
    param_distributions=[
        {
            "module__kernels_1": [10, 20],
            "module__kernels_2": [30, 60],
            "module__hidden": [120, 180],
        },
        {
            "module__kernels_1": [30, 60],
            "module__kernels_2": [80, 120],
            "module__hidden": [360],
        },
    ],
)

# %%
search.fit(x_train, y_train)

# %%
search.best_estimator_, search.best_params_

# %%
y_pred_search = search.predict(test_dataset)

# %%
print(classification_report(y_test, y_pred_search))

# %%
model_file = config.model_dir_path / "mnist_conv_randomized_search.pkl"

# %%
joblib.dump(search, model_file)

# %%
search_loaded = joblib.load(model_file)

# %%
y_pred_loaded = search_loaded.predict(test_dataset)

# %%
print(classification_report(y_test, y_pred_loaded))

# %%
