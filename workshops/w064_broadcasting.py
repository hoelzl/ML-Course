# %% [markdown]
#
# # Workshop: NumPy Universal Operations and Broadcasting
#
# In this workshop we'll explore universal operations and broadcasting in more
# depth. Some of the operations used in this workshop were not presented in the
# lectures, you have to look into [the NumPy
# documentation](https://numpy.org/doc/stable/reference/ufuncs.html) to discover
# them.

# %%
import numpy as np

# %%
arr1 = np.arange(1, 25).reshape(2, 3, 4)
lst1 = [2, 3, 5, 7]

# %% [markdown]
#
# ## Universal Operations
#
# Compute arrays `arr2` and `arr3` that contain the elements of `arr1` and
# `lst1` squared, respectively.

# %% [markdown]
#
# Compute the product of `arr1` and `lst1`. Before evaluating your solution: try
# to determine the shape of the result. How is the shape of the result
# determined? Do you need an universal function or can you perform the
# multiplication as just a normal product?

# %% [markdown]
#
# Write a function `may_consume_alcohol(ages)` that takes a list or
# 1-dimensional array of ages and returns an array containing the values `"no"`
# if the corresponding index in the input array is less than 18, `"maybe"` if
# the value is above 18 but lower than 21 and `"yes"` if the value is at least
# 21.
#
# For example `may_consume_alcohol([15, 20, 30, 21, 20, 17, 18])` returns an
# array containing `['no', 'maybe', 'yes', 'yes', 'maybe', 'no', 'maybe']`.

# %% [markdown]
#
# Write a function `double_or_half(values)` that takes a list or 1-dimensional
# array of numbers and returns a vector of the same length containing `v * 2` if
# the corresponding  member of `values` is odd and `v // 2` if the corresponding
# member is even.
#
# For example, `double_or_half([0, 1, 2, 5, 10, 99])` should return a vector
# containing the values `[0, 2, 1, 10, 5, 198]`.
#
# *Hint:* Check the documentation for the `choose` function.

# %%
