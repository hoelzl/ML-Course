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
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from mlcourse.config import Config
from mlcourse.utils.data import show_dataset
from pytorch_model_summary import summary
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from skorch import NeuralNetClassifier
from skorch.callbacks import Checkpoint, EarlyStopping, LRScheduler
from skorch.helper import predefined_split
from torch.utils.data import Subset
from torchvision.transforms.functional import InterpolationMode

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
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=mnist_transforms, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=mnist_transforms, download=True
)

# %% 
partial_model = nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(10, 20, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    # nn.Linear(320, 60),
    # nn.ReLU(),
    # nn.Linear(60, 10),
)
print(summary(partial_model, torch.zeros((1, 1, 28, 28)), show_input=True))
print(summary(partial_model, torch.zeros((1, 1, 28, 28))))

# %%
conv_model = nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(10, 20, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(320, 60),
    nn.ReLU(),
    nn.Linear(60, 10),
    # nn.Softmax(dim=1),
)

# %%
print(summary(conv_model, torch.zeros((1, 1, 28, 28))))
print(summary(conv_model, torch.zeros((1, 1, 28, 28)), show_input=True))

# %%
cnn_classifier = NeuralNetClassifier(
    conv_model,
    criterion=nn.CrossEntropyLoss,
    batch_size=100,
    max_epochs=2,
    lr=0.1,
    iterator_train__shuffle=True,
    train_split=predefined_split(test_dataset),
    device=device,
)

# %%
cnn_classifier.fit(train_dataset, None)

# %%
cnn_classifier.partial_fit(train_dataset, None)

# %%
y_pred_cnn = cnn_classifier.predict(test_dataset)

# %%
y_test = np.array([y for _, y in test_dataset])

# %%
print(classification_report(y_test, y_pred_cnn))

# %%
print(confusion_matrix(y_test, y_pred_cnn))

# %%
plt.figure(figsize=(10, 8))
ax = plt.axes()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_cnn, ax=ax)

# %% [markdown]
#
# ## Finding Misclassified Images

# %%
def find_misclassified_images(y_pred=y_pred_cnn):
    return np.where(y_test != y_pred)[0]


# %%
find_misclassified_images(y_pred_cnn)

# %%
misclassified_ds = Subset(test_dataset, find_misclassified_images())

# %%
show_dataset(misclassified_ds)

# %% [markdown]
#
# ## Data Augmentation (V2)


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
augmented_train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=augmented_transforms, download=True
)

# %%
cnn_classifier = NeuralNetClassifier(
    conv_model,
    criterion=nn.CrossEntropyLoss,
    batch_size=100,
    max_epochs=2,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    iterator_train__shuffle=True,
    train_split=predefined_split(test_dataset),
    device=device,
)

# %%
cnn_classifier.fit(augmented_train_dataset, None)

# %% [markdown]
#
# ## Callbacks

# %%
step_lr_scheduler = LRScheduler(policy="StepLR", step_size=5, gamma=0.1)

# %%
checkpoint = Checkpoint(
    f_pickle="mnist_cnn.pkl",
    dirname=config.model_dir_path.as_posix(),
    monitor="valid_acc_best",
)

# %%
early_stopping = EarlyStopping(monitor="valid_acc", patience=5, lower_is_better=False)

# %%
cnn_classifier = NeuralNetClassifier(
    conv_model,
    criterion=nn.CrossEntropyLoss,
    batch_size=100,
    max_epochs=200,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    iterator_train__shuffle=True,
    train_split=predefined_split(test_dataset),
    callbacks=[step_lr_scheduler, checkpoint, early_stopping],
    device=device,
)

# %%
cnn_classifier.fit(augmented_train_dataset, None)

# %%
with open(config.model_dir_path / "mnist_cnn.pkl", "rb") as file:
    loaded_classifier = pickle.load(file)

# %%
y_pred_loaded = loaded_classifier.predict(test_dataset)

# %%
print(classification_report(y_test, y_pred_loaded))

# %%
print(confusion_matrix(y_test, y_pred_loaded))

# %% [markdown]
# ## Workshop Fashion MNIST mit CNN
#
# Trainieren Sie ein Konvolutionsnetz, das Bilder aus dem Fashion MNIST Datenset
# klassifizieren kann.
#
# (Zur Erinnerung: Das Torch `Dataset` für Fashion MNIST kann mit der Klasse
# `torchvision.datasets.FashionMNIST` erzeugt werden.)

# %%
