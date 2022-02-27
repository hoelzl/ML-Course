# %%
from pprint import pprint

import fastai.data.load as dl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.metrics import Precision, Recall, accuracy
from pytorch_model_summary import summary
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data.dataloader import DataLoader

from mlcourse.config import Config

# %%
config = Config()
mnist_dir = config.data_dir_path / "external"
mnist_dir.mkdir(parents=True, exist_ok=True)

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# %%[markdown]
#
# # Create the Dataset

# %%
mnist_train_dataset = torchvision.datasets.MNIST(mnist_dir, train=True, download=True)

# %%
type(mnist_train_dataset)

# %% [markdown]
#
# `torchvision.datasets.mnist.MNIST` is a PyTorch `Dataset`, which means that it
# defines `__getitem__()` and `__len__()` methods, that can be used by
# `DataLoader`s.

# %%
type(mnist_train_dataset).mro()

# %%
len(mnist_train_dataset)

# %%
mnist_train_dataset[0]

# %%
plt.imshow(mnist_train_dataset[0][0], cmap="binary")

# A `DataLoader` must eventually return PyTorch tensors, therefore we need to
# transfor the PIL `Image` objects into tensors. This can either be done in the
# `DataLoader` itself, or we already perform the conversion when building the
# `DataSet`.  The module `torchvision.transforms` provides a conversion from
# `Image` to tensor, as well as other useful operations. the `MNIST` constructor
# allows us to pass a transform that will be used to transform the data returned
# by the data set.


# %%
mnist_train_dataset = torchvision.datasets.MNIST(
    mnist_dir, train=True, download=True, transform=torchvision.transforms.ToTensor()
)
mnist_test_dataset = torchvision.datasets.MNIST(
    mnist_dir, train=False, download=True, transform=torchvision.transforms.ToTensor()
)

# %%
img_tensor = mnist_train_dataset[0][0]
type(img_tensor), img_tensor.dtype, img_tensor.shape, img_tensor.squeeze().shape

# %%
img_tensor[0, 10:14, 10:14]

# %%
plt.imshow(img_tensor.squeeze(), cmap="binary")

# %% [markdown]
#
# For PyTorch to use `Dataset` it has to be wrapped in a `DataLoader` that
# performs useful operations such as batching or shuffling of the data. (For
# `Dataset`s like the one used in this file, the `DataLoader` instantiates a
# `Sampler` based on the parameters passed to the `DataLoader`'s constructor to
# perfrom the actual work.) See [the PyTorch
# documentation](https://pytorch.org/docs/stable/data.html#module-torch.utils.data)
# for more information.
#
# For deep learning saved_models you typically want the `DataLoader` to shuffle the
# data, since you will train for multiple epochs, and you want to batch the
# data, in particular when you train on a GPU. To build a batch from a list of
# samples, the `collate_fn` (if supplied to the `DataLoader`'s constructor) is
# called on the list of items in the batch. This can be used to, e.g., convert
# pad sequential items to the same length in each batch.
#
# `DataLoader`s can load data in single- or multi-process modes. There are
# several intricacies to keep in mind when using multi-process data loading, in
# paricular when using CUDA. See [the PyTorch
# documentation](https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading)
# for a discussion.

# %%
mnist_train_data_loader = DataLoader(mnist_train_dataset, batch_size=16, shuffle=True)
mnist_test_data_loader = DataLoader(mnist_test_dataset, batch_size=32, drop_last=False)

# %%
type(mnist_train_data_loader)

# %%
len(mnist_train_data_loader)

# %%
len(mnist_train_data_loader) * mnist_train_data_loader.batch_size

# %%[markdown]
#
# If the batch size does not evenly divide the batch size, the `drop_last`
# argument determines whether the last batch will be dropped, or whether a batch
# with a smaller size will be passed to the model.

# %%
len(mnist_test_data_loader) * mnist_test_data_loader.batch_size

# %%
x, y = next(iter(mnist_train_data_loader))
x.shape, y.shape

# %%
image_batches = enumerate(mnist_train_data_loader)
batch_index, (batch_images, batch_labels) = next(image_batches)

(
    batch_index,
    batch_images.type(),
    batch_images.shape,
    batch_labels.type(),
    batch_labels.shape,
)

# %%
fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(10, 12))
images = iter(batch_images)
values = iter(batch_labels)
for i in range(4):
    for j in range(4):
        axs[i, j].imshow(X=next(images).squeeze(), cmap="binary")
        axs[i, j].set_title(f"Value: {next(values)}")

# %%[markdown]
#
# To make use of the fast.ai infrastructure for training and visualizing saved_models,
# we wrap the two `DataLoader` objects into a single `DataLoaders` object. To
# this end, we need to use fast.ai `DataLoader` objects. The fast.ai
# `DataLoader` class is compatible with PyTorch `DataLoader`, but provides
# several extensions.
#
# Adjust the batch sizes to the memory of your GPU.

# %%
mnist_train_fastai_data_loader = dl.DataLoader(
    mnist_train_dataset, batch_size=50, shuffle=True, device=device
)
mnist_test_fastai_data_loader = dl.DataLoader(
    mnist_test_dataset, batch_size=1000, drop_last=False, device=device
)

# %%
mnist_dls = DataLoaders(
    mnist_train_fastai_data_loader, mnist_test_fastai_data_loader, device=device
)

# %%[markdown]
#
# The `DataLoaders` class provides a few convenience methods, e.g.,
# `one_batch()`.
#
# To access the training and validation `DataLoader`s, use the `train` and
# `valid` properties:

# %%
type(mnist_dls.train), type(mnist_dls.valid)

# %%
x, y = mnist_dls.train.one_batch()
x.shape, y.shape


# %%
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, stride=(2, 2))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=(2, 2))
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


# %%
net = ConvNet()
print(summary(net, torch.zeros((1, 1, 28, 28)), show_input=True))
print(summary(net, torch.zeros((1, 1, 28, 28)), show_input=False))

# %%
learner = Learner(
    mnist_dls,
    ConvNet(),
    loss_func=F.nll_loss,
    metrics=[accuracy, Precision(average="macro"), Recall(average="macro")],
)

# %%[markdown]
#
# These are too many epochs, but we want to see the behavior of the net when it
# is trained for some time.

# %%
learner.fit(n_epoch=20)

# %%
pprint(list(learner.metrics))

# %%
model = learner.model
x, y = mnist_dls.valid.one_batch()
model, type(x), x.shape, type(y), y.shape

# %%
with torch.no_grad():
    pred_proba = model(x)

# %%
pred_proba.shape

# %%
pred = pred_proba.argmax(axis=1)

# %%
pred, y

# %%
(pred == y).sum()

# %%

x = torch.stack([img for img, label in mnist_dls.valid_ds], dim=0).to(device)
y = torch.tensor([label for img, label in mnist_dls.valid_ds])
x.shape, y.shape

# %%
with torch.no_grad():
    pred_proba = model(x).to(torch.device("cpu"))

# %%
pred_proba.shape

# %%
pred = pred_proba.argmax(axis=1)

# %%
pred, y

# %%
(pred == y).sum()

# %%
accuracy_score(y, pred)

# %%
precision_score(y, pred, average="macro")

# %%
precision_score(y, pred, average="micro")

# %%
recall_score(y, pred, average="macro")

# %%
recall_score(y, pred, average="micro")


# %%
class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, stride=(2, 2), padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=(2, 2), padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3, stride=(2, 2), padding=1)
        self.bn3 = nn.BatchNorm2d(40)
        self.fc1 = nn.Linear(640, 120)
        self.bn4 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 60)
        self.bn5 = nn.BatchNorm1d(60)
        self.fc3 = nn.Linear(60, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 640)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


# %%
net2 = ConvNet2()
print(summary(net2, torch.zeros((1, 1, 28, 28)), show_input=True))
print(summary(net2, torch.zeros((1, 1, 28, 28)), show_input=False))

# %%
learner2 = Learner(
    mnist_dls,
    ConvNet2(),
    loss_func=F.nll_loss,
    metrics=[accuracy, Precision(average="macro"), Recall(average="macro")],
)

# %%[markdown]
#
# These are too many epochs, but we want to see the behavior of the net when it
# is trained for some time.

# %%
learner2.fit(n_epoch=5)

# %%
model2 = learner2.model
with torch.no_grad():
    pred2 = model2(x).to(torch.device("cpu")).argmax(axis=1)

# %%
accuracy_score(y, pred), accuracy_score(y, pred2)

# %%
