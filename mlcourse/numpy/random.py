# %%
import numpy as np

# %%
rng = np.random.default_rng()
rng.random()

# %%
rng = np.random.default_rng(42)
rng.random()

# %%
rng.random()

# %%
rng.random(size=3)

# %%
rng.random(size=(3, 2))

# %%
rng.integers(2, 4)

# %%
rng.integers(low=0, high=10, endpoint=True, size=(3, 4))

# %%
arr = np.arange(15)
arr

# %%
rng.shuffle(arr)

# %%
arr

# %%
arr = np.arange(15)

# %%
rng.permutation(arr)

# %%
arr

# %%
arr = np.array([range(0, 10), range(10, 20), range(20, 30)])
arr

# %%
rng.permutation(arr)

# %%
rng.permutation(arr, axis=1)

# %%
rng.permuted(arr)

# %%
arr

# %%
rng.permuted(arr, out=arr)

# %%
arr

# %%
arr = np.arange(9)

# %%
rng.choice(arr, size=3)

# %%
rng.choice(arr, size=(2, 3, 4))

# %%
rng.choice(arr, size=(3, 3), replace=False)

# %%
rng.choice(arr, size=100, replace=False)

# %%
rng.normal(size=(3, 5))

# %%
# rng.normal??

# %%
