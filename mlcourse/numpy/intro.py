# %%
vector1 = [3, 2, 4]
vector2 = [8, 9, 7]

# %%
vector1 + vector2

# %%
def vector_sum(v1, v2):
    assert len(v1) == len(v2)
    result = [0] * len(v1)
    for i in range(len(v1)):
        result[i] = v1[i] + v2[i]
    return result


# %%
vector_sum(vector1, vector2)

# %%
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

# %%
ndarray = [matrix, matrix, matrix]

# %%
ndarray
# %%
vector1[1]

# %%
matrix[1][0]

# %%
ndarray[1][0][2]

# %%
