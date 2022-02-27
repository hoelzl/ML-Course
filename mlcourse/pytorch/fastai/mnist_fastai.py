# %%[markdown]
#
# # MNIST using fast.ai
#
# In contrast to the file `mnist.py` which uses standard PyTorch functionality
# for parts of the data loading, in this file we use a pure fast.ai workflow.
# The default `ImageDataLoaders` factory function assume that the image data is
# stored as files in a directory, so we'll first download the data.

from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
from fastai.data.block import CategoryBlock, DataBlock
from fastai.data.external import URLs, untar_data
from fastai.data.transforms import GrandparentSplitter, get_image_files, parent_label
from fastai.learner import Learner
from fastai.metrics import Precision, Recall, accuracy
from fastai.vision.augment import aug_transforms
from fastai.vision.core import PILImageBW
from fastai.vision.data import ImageBlock, ImageDataLoaders
from pytorch_model_summary import summary

from mlcourse.config import Config

# %%
config = Config()
mnist_root = config.data_dir_path / "external/mnist_fastai"
mnist_root.mkdir(parents=True, exist_ok=True)

# %%
mnist_dir = untar_data(URLs.MNIST, dest=mnist_root)
mnist_dir

# %%
Path.BASE_PATH = mnist_dir
print(mnist_dir.ls(), "\n")
pprint(sorted((mnist_dir / "training").ls()))

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# %%

# %%
mnist_dls_rgb = ImageDataLoaders.from_folder(
    mnist_dir,
    train="training",
    valid="testing",
    device=device,
)

# %%
mnist_block = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
    get_items=get_image_files,
    splitter=GrandparentSplitter(train_name="training", valid_name="testing"),
    get_y=parent_label,
    # batch_tfms=aug_transforms(mult=1.2, do_flip=False)
)

# %%
mnist_dls = mnist_block.dataloaders(mnist_dir)

# %%
mnist_dls.train.one_batch()[0].shape

# %%
mnist_dls.show_batch()

# %% [markdown]
#
# ## Multi-Layer Perceptron (Feed-Forward Network)

# %%
mlp_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.LogSoftmax(dim=1),
).to(device)

# %%
mlp_model(torch.zeros((1, 1, 28, 28)).to(device))

# %%
print(summary(mlp_model, torch.zeros((1, 1, 28, 28)).to(device), show_input=True))
print(summary(mlp_model, torch.zeros((1, 1, 28, 28)).to(device), show_input=False))

# %%
mlp_learner = Learner(
    mnist_dls,
    mlp_model,
    metrics=[accuracy, Precision(average="macro"), Recall(average="macro")],
)

# %%
mlp_learner.lr_find()

# %%
with mlp_learner.no_bar():
    mlp_learner.fit(n_epoch=4)

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
    nn.LogSoftmax(dim=1),
).to(device)

# %%
print(summary(conv_model, torch.zeros((1, 1, 28, 28)).to(device), show_input=True))
print(summary(conv_model, torch.zeros((1, 1, 28, 28)).to(device), show_input=False))

# %%
conv_learner = Learner(
    mnist_dls,
    conv_model,
    metrics=[accuracy, Precision(average="macro"), Recall(average="macro")],
)

# %%
conv_learner.lr_find()

# %%
with conv_learner.no_bar():
    conv_learner.fit(n_epoch=4)

# %%
mnist_block_2 = DataBlock(
    blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
    get_items=get_image_files,
    splitter=GrandparentSplitter(train_name="training", valid_name="testing"),
    get_y=parent_label,
    batch_tfms=aug_transforms(mult=2, do_flip=False),
)

# %%
mnist_dls_2 = mnist_block_2.dataloaders(mnist_dir)

# %%
conv_learner_2 = Learner(
    mnist_dls_2,
    conv_model,
    metrics=[accuracy, Precision(average="macro"), Recall(average="macro")],
)

# %%
with conv_learner_2.no_bar():
    conv_learner_2.fit(n_epoch=16)


# %%
conv_model_3 = nn.Sequential(
    nn.Conv2d(1, 10, kernel_size=3, stride=(2, 2), padding=1),
    nn.BatchNorm2d(10),
    nn.ReLU(),
    nn.Conv2d(10, 20, kernel_size=3, stride=(2, 2), padding=1),
    nn.BatchNorm2d(20),
    nn.ReLU(),
    nn.Conv2d(20, 40, kernel_size=3, stride=(2, 2), padding=1),
    nn.BatchNorm2d(40),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(640, 180),
    nn.BatchNorm1d(180),
    nn.ReLU(),
    nn.Linear(180, 60),
    nn.BatchNorm1d(60),
    nn.ReLU(),
    nn.Linear(60, 10),
    nn.LogSoftmax(dim=1),
).to(device)

# %%
conv_learner_3 = Learner(
    mnist_dls_2,
    conv_model_3,
    metrics=[accuracy, Precision(average="macro"), Recall(average="macro")],
)

# %%
with conv_learner_3.no_bar():
    conv_learner_3.fit(n_epoch=16)


# %%
