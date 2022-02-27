# %%
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mlcourse.config import Config

# %%
config = Config()

# %%
colors = Image.open(config.data_dir_path / "raw/colors.png")

# %%
colors

# %%
plt.imshow(colors)

# %%
color_array = np.array(colors)

# %%
type(color_array)

# %%
color_array.shape

# %%
color_array

# %%
color_array[:1]

# %%
plt.imshow(color_array[:1])

# %%
plt.imshow(color_array[:, :1])

# %%
plt.imshow(color_array[0])

# %%
color_array[0].shape

# %%
plt.imshow(color_array[0].reshape(1, 4, 3))

# %%
plt.imshow(np.expand_dims(color_array[0], axis=0))

# %%
layers = np.split(color_array, 3, axis=2)
len(layers)

# %%
layers[0].shape

# %%
plt.imshow(layers[0], cmap="binary")

# %%
list(permutations([1, 2, 3]))

# %%
fig, axes = plt.subplots(2, 3)
lin_axes = axes.reshape(-1)
for i, p in enumerate(permutations(layers)):
    lin_axes[i].imshow(np.concatenate(p, axis=2))

# %%
