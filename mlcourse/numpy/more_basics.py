# %%
import numpy as np

# %%
vector = np.array([1, 2, 3, 4])

# %%
vector

# %%
vector.shape

# %%
mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# %%
mat

# %%
mat.shape

# %%
mat.sum()

# %%
mat.sum(axis=0)

# %%
mat[0]

# %%
mat[1]

# %%
mat.sum(axis=1)

# %%
mat

# %%
mat.mean()

# %%
mat.mean(axis=1)

# %%
mat.mean(axis=0)

# %%
tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# %%
tensor

# %%
tensor.shape

# %%
list(range(10))

# %%
np.array(range(10))

# %%
range(1.0, 10.0)

# %%
np.arange(10)

# %%
np.arange(10.0)

# %%
np.arange(10).dtype

# %%
np.arange(10.0).dtype

# %%
np.arange(2, 10)

# %%
np.arange(2, 10, 3)

# %%
np.linspace(0.1, 1.0, 10)

# %%
np.arange(0.1, 1.1, 0.1)

# %%
np.linspace(0.0, 1.0, 10)

# %%
np.zeros(3)

# %%
np.zeros(3).shape

# %%
np.zeros((3,))

# %%
np.zeros((3, 3))

# %%
np.zeros((2, 3, 4))

# %%
np.ones(3)

# %%
np.ones((1, 2, 3))

# %%
np.ones(1, 2, 3)

# %%
id = np.eye(4)

# %%
id

# %%
np.eye(3, 4)

# %%
np.identity(3)

# %%
