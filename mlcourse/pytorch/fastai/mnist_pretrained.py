# %%[markdown]
#
# # Mnist Using A Pretrained Fast.Ai Model
#
# In contrast to the file `mnist.py` which uses standard PyTorch functionality
# for parts of the data loading, in this file we use a pure fast.ai workflow.
# In contrast to `minst_fastai.py` we use a pretrained model in this file.
#
# The default `ImageDataLoaders` factory function assume that the image data is
# stored as files in a directory, so we'll first download the data.

from pathlib import Path
from pprint import pprint

import torch
from fastai.data.external import URLs, untar_data
from fastai.metrics import Precision, Recall, accuracy
from fastai.vision.augment import Resize, aug_transforms
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import cnn_learner
from fastai.vision.models import resnet18

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
mnist_dls = ImageDataLoaders.from_folder(
    mnist_dir,
    train="training",
    valid="testing",
    device=device,
    batch_tfms=aug_transforms(mult=2, do_flip=False),
    item_tfms=Resize(224),
)

# %%
resnet_learner = cnn_learner(
    mnist_dls,
    resnet18,
    metrics=[accuracy, Precision(average="macro"), Recall(average="macro")],
)

# %%
with resnet_learner.no_bar():
    resnet_learner.fine_tune(1)

# %%
