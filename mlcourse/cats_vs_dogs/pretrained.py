# %%
from enum import Enum
from typing import Tuple

import numpy as np
from fastai.data.core import DataLoaders
from fastai.learner import Learner, load_learner
from fastai.metrics import accuracy, error_rate
from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    URLs,
    get_image_files,
    untar_data,
)
from fastai.vision.core import PILImage
from fastai.vision.learner import cnn_learner
from nbex.interactive import session
from torchvision.models.resnet import resnet34

from mlcourse.config import Config

# %%
config = Config()


# %%
def is_cat(filename: str) -> bool:
    """
    Returns true if filename is an image of a cat.

    In the dataset we are using this is indicated by the first letter of the
    filename; cats are labeled with uppercase letters, dogs with lowercase ones.
    """
    result = filename[0].isupper()
    # print(f"File: {filename}, initial: '{filename[0]}', result: {result}")
    return result


# %%
def create_dataloaders() -> DataLoaders:
    """
    Create the dataloaders for the cats vs. dogs dataset.
    """
    path = untar_data(URLs.PETS) / "images"
    dls = ImageDataLoaders.from_name_func(
        path,
        get_image_files(path),
        valid_pct=0.2,
        batch_size=8,
        seed=42,
        label_func=is_cat,
        item_tfms=Resize(224),
        device="cuda"
    )
    return dls


# %%
if session.is_interactive:
    _dls = create_dataloaders()
else:
    _dls = None


# %%
def train_model() -> Learner:
    dls = create_dataloaders()
    learner = cnn_learner(dls, resnet34, metrics=[error_rate, accuracy])
    new_dls = create_dataloaders()
    learner.dls = new_dls
    with learner.no_bar():
        learner.fine_tune(1)
    return learner


# %%
if session.is_interactive:
    learner = train_model()


# %%
model_file_path = config.model_dir_path / "cats-vs-dogs.pth"

# %%
if session.is_interactive:
    learner.dls.show_batch()
    learner.show_results()

# %%
if session.is_interactive:
    learner.export(model_file_path)

# %%
if session.is_interactive:
    loaded_learner = load_learner(model_file_path)


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
