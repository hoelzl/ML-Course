# %%
import numpy as np

# %%
v1 = np.array([3, 2, 4])
v2 = np.array([8, 9, 7])

# %%
v1

# %%
type(v1)

# %%
v1.dtype

# %%
v3 = np.array([1.0, 2.0, 3.0])

# %%
v3.dtype

# %%
v1 + v2

# %%
lst1 = [3, 2, 4]
lst2 = [8, 9, 7]

# %%
def vector_sum(v1, v2):
    assert len(v1) == len(v2)
    result = [0] * len(v1)
    for i in range(len(v1)):
        result[i] = v1[i] + v2[i]
    return result

# %%
vector_sum(lst1, lst2)

# %%
# %%timeit
vector_sum(lst1, lst2)

# %%
# %%timeit
v1 = v2

# %%
v1 * v2

# %%
v1.dot(v2)

# %%
v1.sum()

# %%
v1.mean()

# %%
v1.std()

# %%
v1.max()

# %%
v1.min()

# %%
v1

# %%
v1.argmax()

# %%
v1.argmin()

# %%
m1 = np.array([[1, 2, 3], [4, 5, 6]])

# %%
m1

# %%
m2 = np.array([[1, 0], [0, 1], [2, 3]])

# %%
m2

# %%
m1 + m1

# %%
m2 + m2

# %%
m1 + m2

# %%
v1.shape

# %%
m1.shape

# %%
m2.shape

# %%
m1

# %%
m1.T

# %%
m2

# %%
m2.T

# %%
m1.T + m2

# %%
m1 + m2.T

# %%
m1 * m2

# %%
m1.T * m2

# %%
m1.dot(m2)

# %%
m1 @ m2

# %%
v1 @ v2

# %%
