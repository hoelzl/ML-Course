# %%
import logging
from pathlib import Path
from zipfile import ZipFile

import kaggle
from nbex.interactive import session

from mlcourse.config import Config

# %%
config = Config()

# %%
zip_path = config.data_dir_path / "external/zips/"
zip_file_path = zip_path / "dogs-vs-cats.zip"
extract_path = config.data_dir_path / "external/dogs-vs-cats"

logging.debug(f"Zip path: {zip_path.absolute()}")
logging.debug(f"Zip file path: {zip_file_path.absolute()!r}")
logging.debug(f"Extract path: {extract_path.absolute()!r}")
if session.is_interactive:
    print(f"Path for files is: '{extract_path.absolute()!r}'")


# %%
def download_dogs_vs_cats_zip_file_if_necessary(force=False):
    if force or not zip_file_path.exists():
        zip_path.mkdir(parents=True, exist_ok=True)
        print("Downloading dogs-vs-cats dataset.")
        kaggle.api.competition_download_files(competition="dogs-vs-cats", path=zip_path,
                                              force=force, quiet=False)
        print(f"Downloaded dataset to {zip_file_path}.")


# %%
intermediate_zip_path = zip_path / "dogs-vs-cats"
train_path = extract_path / "train"
test_path = extract_path / "test"


# %%
def unzip_dogs_vs_cats_zip_file():
    intermediate_zip_path.mkdir(exist_ok=True, parents=True)
    ZipFile(zip_file_path).extractall(intermediate_zip_path)


# %%
def unzip_training_data_if_necessary():
    if not train_path.exists():
        extract_path.mkdir(exist_ok=True, parents=True)
        train_zip_path = intermediate_zip_path / "train.zip"
        if not train_zip_path.exists():
            unzip_dogs_vs_cats_zip_file()
        ZipFile(train_zip_path).extractall(extract_path)


# %%
max_image_number = 999


# %%
def is_valid_image_file(path: Path) -> bool:
    is_dog_or_cat = path.stem[:3] in ["cat", "dog"]
    try:
        image_number = int(path.stem[4:])
    except ValueError:
        image_number = -1
    has_number = 0 <= image_number
    is_valid_type = path.suffix in [".jpg", ".png"]
    return is_dog_or_cat and has_number and is_valid_type


# %%
assert is_valid_image_file(Path("dog.123.jpg"))
assert is_valid_image_file(Path("cat.0.png"))
assert not is_valid_image_file(Path("foo.123.png"))
assert is_valid_image_file(Path("dog.100000.png"))
assert not is_valid_image_file(Path("catabcd.png"))
assert not is_valid_image_file(Path("dog.123.gif"))


# %%
def move_training_data_into_subfolders() -> None:
    """
    Move training files into subfolders according to their classes
    """
    for cls in ["dog", "cat"]:
        (train_path / cls).mkdir(exist_ok=True)
        for file in train_path.glob(f"{cls}*"):
            if file.is_file() and is_valid_image_file(file):
                new_path = file.parent / cls / file.name
                file.rename(new_path)


# %%
def unzip_test_data_if_necessary():
    if not test_path.exists():
        extract_path.mkdir(exist_ok=True, parents=True)
        test_zip_path = intermediate_zip_path / "test1.zip"
        if not test_zip_path.exists():
            unzip_dogs_vs_cats_zip_file()
        ZipFile(test_zip_path).extractall(extract_path)
        (extract_path / "test1").rename(extract_path / "test")


# %%
def ensure_dogs_vs_cats_data_exists(force_download=False):
    download_dogs_vs_cats_zip_file_if_necessary(force=force_download)
    unzip_training_data_if_necessary()
    move_training_data_into_subfolders()
    unzip_test_data_if_necessary()
