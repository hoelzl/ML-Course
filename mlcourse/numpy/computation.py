# %%
import math
import operator

import numpy as np
from IPython.display import display
from toolz import reduce

# %% [markdown]
# # Computation in NumPy
#
# Computations can be performed of tensors with compatible dimensions:

# %%
v1 = np.array([2, 4, 5])
v2 = np.arange(3.0)

# %%
v1 + v2

# %%
v1 * v2

# %%
v1 @ v2

# %%
v1.dot(v2)

# %%
m1 = np.arange(12).reshape(3, 4)
m2 = np.ones((3, 4))
m3 = np.array([[1, 3], [5, 7], [2, 4], [6, 8]])
print("m1.shape:", m1.shape, "m2.shape:", m2.shape, "m3.shape:", m3.shape)

# %%
m1 + m2

# %%
# m1 + m3

# %%
m1 @ m3

# %% [markdown]
# ## Boolean Operations

# %%
v1 = np.arange(4)
v2 = np.ones(4)

# %%
v1 == v2  # noqa B015

# %%
v1 <= v2  # noqa B015

# %%
# if v1 == v2: print("Done")

# %%
equals = v1 == v2
equals

# %%
equals.all()

# %%
equals.any()

# %%
if equals.any():
    print("Done")


# %% [markdown]
# ## Broadcasting (Part 1)
#
# Most operations in NumPy can be used with scalars as well:

# %%
v1 = np.arange(8)
v1

# %%
v1 + 5

# %%
3 + v1

# %%
v1 * 2

# %%
v1 ** 2

# %%
2 ** v1

# %%
v1 > 5  # noqa B015

# %% [markdown]
# ## Minimum, Maximum, Sum, ...

# %%
np.set_printoptions(precision=2)

# %%
rng = np.random.default_rng(42)
vec = rng.random(10)
vec

# %%
vec.max()

# %%
vec.argmax()

# %%
vec[vec.argmax()]

# %%
vec.min()

# %%
vec.argmin()

# %%
rng = np.random.default_rng(42)
arr = rng.random((3, 5))

# %%
arr.max()

# %%
arr.argmax()

# %%
arr.min()

# %%
arr.argmin()

# %%
arr.reshape(-1)[arr.argmin()]

# %%
# arr[np.unravel_index(arr.argmin(), arr.shape)]

# %%
arr

# %%
arr.sum()

# %%
arr.sum(axis=0)

# %%
arr.sum(axis=1)

# %%
arr.mean()

# %%
arr.mean(axis=0)

# %%
arr.mean(axis=1)

# %%
v1 = np.arange(2)
v2 = np.linspace(1, 3, 3)

# %%
np.concatenate([v1, v2])

# %%
m1 = np.arange(12).reshape(3, 4)
m2 = np.arange(12, 24).reshape(3, 4)

# %%
m1

# %%
m2

# %%
np.concatenate([m1, m2])

# %%
np.concatenate([m1, m2], axis=1)

# %%
m3 = np.arange(12, 15).reshape(3, -1)
m3

# %%
# np.concatenate([m1, m3])

# %%
np.concatenate([m1, m3], axis=1)

# %% [markdown]
# ## Indices for NumPy Arrays

# %%
vec = np.arange(10)

# %%
vec

# %%
vec[3]

# %%
vec[3:8]

# %%
vec[-1]

# %%
arr = np.arange(24).reshape(4, 6)

# %%
arr

# %%
arr[1]

# %%
arr[1][2]

# %%
arr[1, 2]

# %%
arr

# %%
arr[1:3]

# %%
arr[1:3][2:4]

# %%
arr[1:3, 2:4]

# %%
arr[:, 2:4]

# %%
# Danger!
arr[:2:4]

# %%
arr[:, 1:6:2]

# %% [markdown]
# ## Slices and Modifications
#
# It's possible to apply operations to slices. Modification of slices *changes
# the underlying array.*

# %%
arr = np.ones((3, 3))
arr

# %%
arr[1:, 1:] = 2.0

# %%
arr

# %%
lst = [1, 2, 3]
vec = np.array([1, 2, 3])

# %%
lst[:] = [99]

# %%
lst

# %%
vec[:] = [99]

# %%
vec

# %%
vec[:] = 11
vec

# %% [markdown]
# ## Danger!
# Don't use the `lst[:]` Idiom!

# %%
lst1 = list(range(10))
lst2 = lst1[:]
lst1[:] = [22] * 10
print(lst1)
print(lst2)

# %%
vec1 = np.arange(10)
vec2 = vec1[:]
vec1[:] = 22
print(vec1)
print(vec2)

# %%
vec1 = np.arange(10)
vec2 = vec1.copy()
vec1[:] = 22
print(vec1)
print(vec2)

# %% [markdown]
#
# Similar considerations hold for reshaped arrays:

# %%
vec = np.arange(4)
arr = vec.reshape(2, 2)
arr

# %%
arr[1, 1] = 10
vec[0] = 20
arr

# %%
vec

# %% [markdown]
# ### Boolean Operations on NumPy Arrays

# %%
bool_vec = np.array([True, False, True, False, True])

# %%
neg_vec = np.logical_not(bool_vec)
neg_vec

# %%
np.logical_and(bool_vec, neg_vec)

# %%
~bool_vec

# %%
bool_vec & neg_vec

# %%
bool_vec | neg_vec

# %% [markdown]
# ## Conditional Selection
#
# You can use a NumPy array with Boolean values as index value, if it has the
# same shape as the "value array". This will select all elements of the value
# array for which the index evaluates to true.

# %%
vec = np.arange(9)
bool_vec = vec % 3 == 0
print(vec)
print(bool_vec)

# %%
vec[bool_vec]

# %%
arr = np.arange(8).reshape(2, 4)
bool_arr = arr % 2 == 0
bool_arr

# %%
arr[bool_arr]

# %%
# Error!
# arr[bool_arr.reshape(-1)]

# %%
vec[vec % 2 > 0]

# %%
arr[arr < 5]

# %%
arr = np.arange(30).reshape(6, 5)
arr

# %%
arr[:, 1]

# %%
arr[:, 1] % 2 == 0  # noqa B015

# %%
arr[arr[:, 1] % 2 == 0]

# %%
arr[arr[:, 1] % 2 == 1]

# %% [markdown]
# ## Universal NumPy Functions
#
# NumPy offers a wealth of universal functions that work on NumPy arrays, lists,
# and often numbers

# %%
vec1 = rng.random(5)
vec2 = rng.random(5)
display(vec1)

list1 = list(vec1)
list2 = list(vec2)
display(list1)

matrix = np.arange(6).reshape(2, 3)
list_matrix = [[0, 1, 2], [3, 4, 5]]
display(matrix)
display(list_matrix)

# %%
vec1.sum()

# %%
# list1.sum()

# %%
reduce(operator.add, list1, 0)

# %%
reduce(operator.add, vec1, 0)

# %%
np.sum(vec1)

# %%
np.sum(list1)

# %%
np.sum(matrix)

# %%
np.sum(list_matrix)

# %%
np.sum(123)

# %%
np.sum(list_matrix, axis=0)

# %%
np.sin(vec1)

# %%
np.sin(list1)

# %%
np.sin(matrix)

# %%
np.sin(list_matrix)

# %%
np.sin(math.pi)

# %%
np.mean(vec1)

# %%
np.median(vec1)

# %%
np.std(vec1)

# %%
np.greater(vec1, vec2)

# %%
np.greater(list1, list2)

# %%
np.greater(vec1, list2)

# %%
display(vec1)
display(vec2)

# %%
np.maximum(vec1, vec2)

# %%
np.maximum(list1, list2)

# %%
np.maximum(list1, vec2)

# %% [markdown]
#
# A complete list of universal functions is
# [here](https://numpy.org/doc/stable/reference/ufuncs.html).

# %% [markdown]
#
# ## Broadcasting (Part 2)

# %%
arr = np.arange(16).reshape(2, 2, 4)
print(f"arr.shape: {arr.shape}")
arr

# %%
arr * arr

# %%
3 * arr

# %%
vec1 = np.arange(3)
display(vec1)
print(f"vec1.shape: {vec1.shape}")
print(f"arr.shape:  {arr.shape}")
# arr * vec1

# %%
vec2 = np.arange(4)
display(arr)
display(vec2)
print(f"vec2.shape: {vec2.shape}")
print(f"arr.shape:  {arr.shape}")
arr * vec2

# %% [markdown]
#
# ### Rules for broadcasting:
#
# When performing an operation on `a` and `b`:
#
# - Axes (shapes) of `a` and `b` are compared from right to left
#
# - If `a` and `b` have the same length for an axis, they are compatible
#
# - If either `a` or `b` has length 1 for an axis, it is conceputally repeated
#   along this axis to fit the other array
#
# - If `a` and `b` have different lengths along an axis and neither has length 1
#   they are incompatible
#
# - The array with lower rank is treated as if it has rank 1 for the missing
#   axes, the missing axes are appended on the left


# %%
def ones(shape):
    return np.ones(shape, dtype=np.int32)


# %%
def tensor(shape):
    from functools import reduce
    from operator import mul

    size = reduce(mul, shape, 1)
    return np.arange(1, size + 1).reshape(*shape)


# %%
tensor((2, 3))

# %%
tensor((1, 3))

# %%
tensor((2, 1))

# %%
ones((1, 3)) + tensor((2, 1))

# %%
np.concatenate([tensor((2, 1))] * 3, axis=1)

# %%
ones((1, 3))

# %%
np.concatenate([ones((1, 3))] * 2, axis=0)

# %%
ones((1, 3)) + tensor((2, 1))

# %%
tensor((1, 3)) + ones((2, 1))

# %%
tensor((1, 3)) + tensor((2, 1))

# %%
tensor((2, 3, 4))

# %%
tensor((2, 3, 1))

# %%
tensor((2, 1, 4))

# %%
ones((2, 3, 1)) + tensor((2, 1, 4))

# %%
ones((2, 3, 1))

# %%
np.concatenate([ones((2, 3, 1))] * 4, axis=2)

# %%
tensor((2, 1, 4))

# %%
np.concatenate([tensor((2, 1, 4))] * 3, axis=1)

# %%
ones((2, 3, 1)) + tensor((2, 1, 4))

# %%
tensor((2, 3, 1)) + ones((2, 1, 4))

# %%
tensor((2, 3, 1)) + tensor((2, 1, 4))

# %%
tensor((3, 1)) + tensor((2, 1, 4))

# %%
tensor((3, 1))

# %%
tmp1 = np.concatenate([tensor((3, 1))] * 4, axis=1)
print("Shape:", tmp1.shape)
tmp1

# %%
tmp2 = tmp1.reshape(1, 3, 4)
print("Shape:", tmp2.shape)
tmp2

# %%
tmp3 = np.concatenate([tmp2] * 2, axis=0)
print("Shape:", tmp3.shape)
tmp3

# %%
tensor((2, 1, 4))

# %%
tmp4 = np.concatenate([tensor((2, 1, 4))] * 3, axis=1)
print("Shape:", tmp4.shape)
tmp4

# %%
display(tmp3)
display(tmp4)

# %%
tmp3 + tmp4

# %%
