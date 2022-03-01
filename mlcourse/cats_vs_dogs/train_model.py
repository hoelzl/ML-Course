# %%
import pickle
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.utils
from mlcourse import data
from mlcourse.cats_vs_dogs.download_data import (
    ensure_dogs_vs_cats_data_exists,
    train_path,
)
from mlcourse.config import Config
from nbex.interactive import session
from PIL import Image
from torchvision import datasets, models, transforms
from pytorch_model_summary import summary
from skorch.callbacks import Checkpoint, EpochScoring, Freezer, LRScheduler
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms


# %% [md]

# This file fine-tunes a pretrained model to distinguish dogs and cats based on the
# kaggle dataset available from [this page](https://www.kaggle.com/c/dogs-vs-cats/data).

# %%
config = Config()
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
ensure_dogs_vs_cats_data_exists()

# %%
net_mean = [0.485, 0.456, 0.406]
net_std = [0.229, 0.224, 0.225]

# %% [markdown]
#
# See the [torchvision
# documentation](https://pytorch.org/vision/stable/transforms.html) for a
# description of the available transformations.

# %%
train_transform = transforms.Compose(
    [
        # Probably more common for Interception based networks, but should work
        # for resnet as well.
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(net_mean, net_std),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(net_mean, net_std),
    ]
)

# %%
sample_image_path = train_path / "cat/cat.1.jpg"
if session.is_interactive:
    sample_image = Image.open(sample_image_path)
    plt.imshow(sample_image)

# %%
if session.is_interactive:
    print(val_transform(sample_image))
    plt.imshow(train_transform(sample_image).permute(1, 2, 0))
    plt.show()
    plt.imshow(val_transform(sample_image).permute(1, 2, 0))


# %%
class ImageFilter:
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id

    def __call__(self, pathname: str) -> bool:
        path = Path(pathname)
        is_dog_or_cat = path.stem[:3] in ["cat", "dog"]
        try:
            image_number = int(path.stem[4:])
        except ValueError:
            image_number = -1
        is_in_range = self.min_id <= image_number <= self.max_id
        is_valid_type = path.suffix in [".jpg", ".png"]
        return is_dog_or_cat and is_in_range and is_valid_type


# %%
_filter_0_1000 = ImageFilter(0, 1000)

assert _filter_0_1000("dog.123.jpg")
assert _filter_0_1000("cat.0.png")
assert not _filter_0_1000("foo.123.png")
assert not _filter_0_1000("dog.100000.png")
assert not _filter_0_1000("cat.abc.png")
assert not _filter_0_1000("dog.123.gif")

# %%
# We're using only the training folder, since otherwise we would need to implement a
# second method of determining the labels.
dogs_vs_cats_path = train_path
assert dogs_vs_cats_path.exists()

# %%
train_ds = datasets.ImageFolder(
    root=dogs_vs_cats_path,
    transform=train_transform,
    is_valid_file=ImageFilter(0, 1999),
)
val_ds = datasets.ImageFolder(
    root=dogs_vs_cats_path,
    transform=val_transform,
    is_valid_file=ImageFilter(2000, 2499),
)

# %%
if session.is_interactive:
    for img, label in islice(train_ds, 2):
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Label: {label}")
        plt.show()
    # Show some dogs as well
    # for img, label in islice(train_ds, len(train_ds)-3, len(train_ds)-1):
    #     plt.imshow(img.permute(1, 2, 0))
    #     plt.title(f"Label: {label}")
    #     plt.show()

# %%
class_names = train_ds.classes
if session.is_interactive:
    print("Class names:", class_names)


# %%
class CatsVsDogsModel(nn.Module):
    def __init__(self, num_output_features, model_type=models.resnet18) -> None:
        super().__init__()
        model = model_type(pretrained=True)
        num_fc_inputs = model.fc.in_features
        model.fc = nn.Linear(num_fc_inputs, num_output_features)
        self.model = model

    def forward(self, x):
        return self.model(x)


# %%
if session.is_interactive:
    print(
        summary(
            CatsVsDogsModel(2),
            torch.unsqueeze(next(iter(train_ds))[0], 0),
            show_hierarchical=True,
        )
    )

# %%
lr_scheduler_ = LRScheduler(policy="StepLR", step_size=7, gamma=0.1)

# %%
checkpoint = Checkpoint(
    f_pickle="dogs_vs_cats.pkl",
    dirname=config.model_dir_path.as_posix(),
    monitor="valid_acc_best",
)

# %%
def is_fc_layer(name: str):
    lambda x: not x.startswith("model.fc")


# %%
freezer = Freezer(is_fc_layer)

# %%
net = NeuralNetClassifier(
    CatsVsDogsModel,
    criterion=nn.CrossEntropyLoss,
    lr=0.001,
    batch_size=16,
    max_epochs=10,
    module__num_output_features=2,
    optimizer=optim.SGD,
    optimizer__momentum=0.9,
    iterator_train__shuffle=True,
    train_split=predefined_split(val_ds),
    callbacks=[lr_scheduler_, checkpoint, freezer],
    device=device,
)

# %%
net.fit(train_ds, y=None)

# %%
