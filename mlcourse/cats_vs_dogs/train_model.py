# %%
import logging
from pathlib import Path
from enum import Enum
from typing import Tuple, List, Dict

import kaggle
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils
from nbex.interactive import session

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.resnet import resnet34

from mlcourse.config import Config
from mlcourse.cats_vs_dogs.download_data import ensure_dogs_vs_cats_data_exists

# %% [md]

# This file fine-tunes a pretrained model to distinguish dogs and cats based on the
# kaggle dataset available from [this page](https://www.kaggle.com/c/dogs-vs-cats/data).

# %%
config = Config()
batch_size = 16

# %%
ensure_dogs_vs_cats_data_exists()

# %%
net_mean = [0.485, 0.456, 0.406]
net_std = [0.229, 0.224, 0.225]

# %%
data_transforms = {
    "train": transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(net_mean, net_std)]),
    "val":   transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
         transforms.Normalize(net_mean, net_std)])
}

# %%
max_image_number = 999


# %%
class ImageValidator:
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
is_valid_image = ImageValidator(0, 1000)

assert is_valid_image("dog.123.jpg")
assert is_valid_image("cat.0.png")
assert not is_valid_image("foo.123.png")
assert not is_valid_image("dog.100000.png")
assert not is_valid_image("cat.abc.png")
assert not is_valid_image("dog.123.gif")

# %%
# We're using only the training folder, since otherwise we would need to implement a
# second method of determining the labels.
dogs_vs_cats_path = config.data_dir_path / "external/dogs-vs-cats/train"
assert dogs_vs_cats_path.exists()

# %%
dogs_vs_cats_datasets = {
    tv: datasets.ImageFolder(root=dogs_vs_cats_path, transform=data_transforms[tv],
                             is_valid_file=ImageValidator(n_min, n_max)) for
    (tv, n_min, n_max) in [("train", 0, 999), ("val", 1000, 1499)]}

# %%
dogs_vs_cats_dataloaders = {
    tv: torch.utils.data.DataLoader(dogs_vs_cats_datasets[tv], batch_size=batch_size,
                                    shuffle=True, num_workers=config.max_processes)
    for tv in ["train", "val"]}

# %%
dogs_vs_cats_sizes = {
    tv: len(dogs_vs_cats_datasets[tv]) for tv in ["train", "val"]
}

# %%
if session.is_interactive:
    print(
        f"Dataset sizes: train = {dogs_vs_cats_sizes['train']}, val = "
        f"{dogs_vs_cats_sizes['val']}")

# %%
class_names = dogs_vs_cats_datasets["train"].classes

# %%
if session.is_interactive:
    print(f"Class names: {class_names}")


# %%
def convert_tensor_to_image(tensor):
    ts = tensor.numpy().transpose((1, 2, 0))
    ts = np.clip(net_std * ts + net_mean, 0, 1)
    return ts


# %%
def show_image(tensor, title="Images"):
    img = convert_tensor_to_image(tensor)
    plt.imshow(img)
    plt.title(title)
    plt.show()


# %%
raw_tensors, classes = next(iter(dogs_vs_cats_dataloaders["train"]))
image_tensors = torchvision.utils.make_grid(raw_tensors)
show_image(image_tensors)

# %%



# %%
class AnimalType(str, Enum):
    cat = "cat"
    dog = "dog"


# %%
def classify(image: PILImage, learner: Learner) -> Tuple[AnimalType, float]:
    with learner.no_bar():
        results = learner.predict(image)
    _, category, probabilities = results
    is_a_cat = category == 1
    animal_type = AnimalType.cat if is_a_cat else AnimalType.dog
    percent = np.round(100 * probabilities)
    return animal_type, percent[category]


# %%
if session.is_interactive:

    def evaluate_classification(images, animal_type):
        num_total, num_correct = 0, 0
        for image_file in images:
            image = PILImage.create(image_file)
            predicted_animal_type, _ = classify(image, loaded_learner)
            num_total += 1
            if predicted_animal_type == animal_type:
                num_correct += 1
        return num_total, num_correct


    image_files = config.data_dir_path / "external/cats-vs-dogs-small/test"
    cat_images = (image_files / "cat").glob("cat*.jpg")
    dog_images = (image_files / "dog").glob("dog*.jpg")

    num_cats, num_cats_correct = evaluate_classification(cat_images, AnimalType.cat)
    print(f"Classified {num_cats_correct} out of {num_cats} cat images correctly.")
    num_dogs, num_dogs_correct = evaluate_classification(dog_images, AnimalType.dog)
    print(f"Classified {num_dogs_correct} out of {num_dogs} dog images correctly.")

# %%
