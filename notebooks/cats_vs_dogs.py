# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from fastai.vision.all import *

# %%
path = untar_data(URLs.PETS) / "images"


# %%
def is_cat(x):
    return x[0].isupper()


# %%
def create_dls():
     return ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=is_cat, item_tfms=Resize(224))


# %%
dls = create_dls()
dls.show_batch()

# %%
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.dls = create_dls()

# %%
learn.fine_tune(1)

# %%
learn.show_results()

# %%
