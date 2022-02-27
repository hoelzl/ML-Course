# %% [markdown]
#
# # Pandas Type `Series`
#
# A series represents a sequence of values, similar to a Python list. Elements
# of their series can be retrieved by their numerical index, but in addition a
# series can have a semantically meaningful index (e.g., for time series).
#
# Internally, a Pandas series is backed by a numpy array, therefore most of the
# numpy operations are applicable to series as well.
#
# In addition it is easy (and cheap) to convert series to numpy.

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# ## Creation
#
# ### From Lists

# %%
pd.Series(data=[10, 20, 30, 40])


# %%
pd.Series(["a", "b", "c"])

# %% [markdown]
# ### From Lists with Index

# %%
pd.Series(data=[1, 2, 3, 4], index=["w", "x", "y", "z"])

# %% [markdown]
# ### From NumPy Arrays

# %%
arr = np.arange(5)
indices = "a b c d e".split()


# %%
pd.Series(data=arr)


# %%
pd.Series(arr, index=indices)


# %%
rng = np.random.default_rng(42)
data_vec = rng.normal(size=1000)
data = pd.Series(data=data_vec)


# %%
data


# %%
data.head()


# %%
data.tail()

# %% [markdown]
# ### From Dictionary

# %%
pd.Series(data={"Ice Cream": 2.49, "Cake": 4.99, "Fudge": 7.99})

# %% [markdown]
# ## Indices and Operations

# %%
food1 = pd.Series({"Ice Cream": 2.49, "Cake": 4.99, "Fudge": 7.99})
food2 = pd.Series({"Cake": 4.99, "Ice Cream": 3.99, "Pie": 3.49, "Cheese": 1.99})


# %%
food1


# %%
food1.index


# %%
food1.size


# %%
food1.sum()


# %%
food1.mean()


# %%
food1.name


# %%
food1.name = "Deserts"


# %%
food1.name


# %%
food1


# %%
food1.plot.bar(legend=True)


# %%
data.plot.hist(legend=False)


# %%
food1["Cake"]


# %%
food1.loc["Cake"]


# %%
# Error!
# food1["Pie"]


# %%
food1.argmin()


# %%
food1[0]


# %%
food1.iloc[0]


# %%
confusing = pd.Series(data=np.linspace(0, 5, 11), index=np.arange(-5, 6))
confusing


# %%
confusing[0]


# %%
confusing.loc[0]


# %%
confusing.iloc[0]


# %%
food_sum = food1 + food2
food_sum


# %%
food1 + 0.5


# %%
food1


# %%
def discount(price):
    return price * 0.9


food1.apply(discount)


# %%
food1


# %%
food1.apply(lambda price: price * 0.9)


# %%
food1.append(pd.Series({"Chocolate": 3.99}))


# %%
food1


# %%
all_food = food1.append(food2)


# %%
all_food

# %% [markdown]
# ### Mehrfach vorkommende Index-Werte

# %%
all_food.index


# %%
all_food.is_unique


# %%
food1.is_unique


# %%
all_food["Cake"]


# %%
type(all_food["Cake"])


# %%
all_food["Pie"]


# %%
type(all_food["Pie"])


# %%
all_food.groupby(all_food.index).max()

# %% [markdown]
# ### Sorted and unsorted Indices

# %%
all_food.index.is_monotonic_increasing


# %%
sorted_food = all_food.sort_index()


# %%
sorted_food


# %%
sorted_food.index.is_monotonic_increasing


# %%
all_food.sort_values()


# %%
all_food.sort_values().is_monotonic_increasing


# %%
all_food[["Pie", "Cake"]]


# %%
all_food


# %%
all_food[1:3]


# %%
# all_food['Cake':'Fudge']


# %%
sorted_food["Cake":"Fudge"]

# %% [markdown]
#
# **Important:** The upper value of the slice, `"Fudge"` is contained in the
# result!

# %% [markdown]
# ## Missing Values

# %%
food = food1 + food2


# %%
food.isna()


# %%
food.isna().sum()


# %%
food.dropna()

# %%
