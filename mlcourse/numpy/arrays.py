# %%
import numpy as np

# %% [markdown]
#
# # Attributes of arrays

# %%
int_vector = np.arange(36)
float_vector = np.arange(36.0)
small_float_vector = np.arange(36.0, dtype=np.float16)

rng = np.random.default_rng(42)
int_matrix = rng.integers(low=0, high=10, size=(3, 5))
tensor = rng.random(size=(3, 4, 5))

# %%
int_vector.dtype

# %%
float_vector.dtype

# %%
small_float_vector.dtype

# %%
int_matrix.dtype

# %%
tensor.dtype

# %%
int_vector.shape

# %%
small_float_vector.shape

# %%
int_matrix.shape

# %%
tensor.shape

# %%
int_vector.size

# %%
float_vector.size

# %%
small_float_vector.size

# %%
int_matrix.size

# %%
tensor.size

# %%
int_vector.itemsize

# %%
float_vector.itemsize

# %%
small_float_vector.itemsize

# %%
int_matrix.itemsize

# %%
tensor.itemsize

# %%
np.info(int_vector)

# %%
np.info(float_vector)

# %%
np.info(small_float_vector)

# %%
np.info(int_matrix)

# %%
np.info(tensor)

# %% [markdown]
#
# # Changing Shape and Size
#
# ## Changing the Shape

# %%
float_vector.shape

# %%
float_matrix = float_vector.reshape((6, 6))

# %%
float_matrix

# %%
float_matrix.shape

# %%
float_vector

# %%
np.info(float_matrix)

# %%
float_matrix.reshape(3, 12)

# %%
# float_vector.reshape(20, 20)

# %%
m1 = float_vector.reshape(3, 12)
m1

# %%
np.info(m1)

# %%
m2 = float_vector.reshape(3, 12, order="F")
m2

# %%
np.info(m2)

# %%
v = np.arange(3)
v

# %%
v.shape

# %%
v.reshape(3, 1)

# %%
v.reshape(-1, 1)

# %%
float_vector.shape

# %%
float_vector.reshape(2, 6, 3)

# %%
float_vector.reshape(2, -1, 3)

# %%
float_vector.reshape(2, 6, -1)

# %%
# float_vector.reshape(2, -1, -1)

# %%
v.reshape(3, 1)

# %%
v.reshape(1, 3)

# %%
v.reshape(1, 3, 1)

# %%
vt = v.reshape(1, 3, 1, 1)
vt

# %%
vt.squeeze()

# %%
vt.squeeze().shape

# %% [markdown]
# ## Changing the Size

# %%
# float_vector.resize(10, 10)

# %%
int_vector.resize(10, 10)

# %%
np.info(int_vector)

# %%
np.resize(float_vector, (10, 10))

# %%
float_vector

# %%
v = np.array([1, 2, 3])

# %%
np.pad(v, (2, 3), "constant", constant_values=(0, 4))

# %%
v

# %%
a = np.array([[10, 20], [30, 40], [50, 60]])
a

# %%
np.pad(a, ((1, 2), (3, 4)), "constant", constant_values=((11, 22), (33, 44)))

# %% [markdown]
# ## Transposing

# %%
float_matrix

# %%
float_matrix.T

# %%
int_matrix

# %%
int_matrix.T

# %%
tensor

# %%
np.set_printoptions(precision=2)

# %%
tensor

# %%
tensor.T

# %%
tensor.shape, tensor.T.shape

# %%
tensor[1, 2, 3], tensor.T[3, 2, 1]

# %%
tensor

# %%
tensor2 = tensor.transpose((1, 2, 0))
tensor2

# %%
tensor2.shape

# %%
tensor[1, 2, 3], tensor2[2, 3, 1]

# %%
