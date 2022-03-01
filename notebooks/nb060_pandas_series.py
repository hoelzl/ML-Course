# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3.8.8 64-bit ('cam')
#     language: python
#     name: python388jvsc74a57bd0acafb728b15233fa3654ff8b422c21865df0ca42ea3b74670e1f2f098ebd61ca
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/python-logo-notext.svg"
#      style="display:block;margin:auto;width:10%"/>
# <h1 style="text-align:center;">Python: Pandas Series</h1>
# <h2 style="text-align:center;">Coding Akademie München GmbH</h2>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <div style="text-align:center;">Allaithy Raed</div>

# %% [markdown] slideshow={"slide_type": "slide"}
#
# # Der Typ `Series`
#
# Der Pandas Typ `Series` repräsentiert eine Folge von Werten, die ähnlich wie eine Python Liste numerisch indiziert werden kann, gleichzeitig aber auch einen semantisch sinnvollerern Index haben kann, z.B. Daten für Zeitreihen.
#
# Intern wird ein `Series`-Objekt durch ein NumPy Array realisiert, daher sind die meisten Operationen von NumPy Arrays auch auf Pandas-`Series`-Objekte anwendbar.

# %%
import numpy as np
import pandas as pd

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Erzeugen von Serien
#
# ### Aus Listen

# %%
pd.Series(data=[10, 20, 30, 40])

# %%
pd.Series(['a', 'b', 'c'])

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Aus Listen mit Index

# %%
pd.Series(data=[1, 2, 3, 4], index=['w', 'x', 'y', 'z'])

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Aus NumPy Arrays

# %%
arr = np.arange(5)
indices = 'a b c d e'.split()

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

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Aus Dictionary

# %%
pd.Series(data={'Ice Cream':2.49, 'Cake': 4.99, 'Fudge': 7.99})

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Indizes und Operationen

# %%
food1 = pd.Series({'Ice Cream':2.49, 'Cake': 4.99, 'Fudge': 7.99})
food2 = pd.Series({'Cake': 4.99, 'Ice Cream':3.99, 'Pie': 3.49, 'Cheese': 1.99})

# %%
food1

# %% slideshow={"slide_type": "subslide"}
food1.index

# %%
food1.size

# %%
food1.sum()

# %%
food1.mean()

# %% slideshow={"slide_type": "subslide"}
food1.name

# %%
food1.name = 'Deserts'

# %% slideshow={"slide_type": "-"}
food1.name

# %%
food1

# %% slideshow={"slide_type": "subslide"}
food1.plot.bar(legend=True);

# %%
data.plot.hist(legend=True);

# %% slideshow={"slide_type": "subslide"}
food1['Cake']

# %%
food1.loc['Cake']

# %%
# Error!
# food1['Pie']

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

# %% slideshow={"slide_type": "subslide"}
food_sum = food1 + food2
food_sum

# %% slideshow={"slide_type": "subslide"}
food1 + 0.5

# %%
food1


# %% slideshow={"slide_type": "subslide"}
def discount(price):
    return price * 0.9

food1.apply(discount)

# %%
food1

# %% slideshow={"slide_type": "subslide"}
food1.apply(lambda price: price * 0.9)

# %% slideshow={"slide_type": "subslide"}
food1.append(pd.Series({'Chocolate': 3.99}))

# %%
food1

# %% slideshow={"slide_type": "subslide"}
all_food = food1.append(food2)

# %%
all_food

# %% [markdown]
# ### Mehrfach vorkommende Index-Werte

# %% slideshow={"slide_type": "subslide"}
all_food.index

# %%
all_food.is_unique

# %%
food1.is_unique

# %% slideshow={"slide_type": "subslide"}
all_food['Cake']

# %%
type(all_food['Cake'])

# %% slideshow={"slide_type": "subslide"}
all_food['Pie']

# %%
type(all_food['Pie'])

# %% slideshow={"slide_type": "subslide"}
all_food.groupby(all_food.index).max()

# %% [markdown]
# ### Sortierte und unsortierte Indizes

# %%
all_food.index.is_monotonic_increasing

# %% slideshow={"slide_type": "subslide"}
sorted_food = all_food.sort_index()

# %%
sorted_food

# %%
sorted_food.index.is_monotonic_increasing

# %% slideshow={"slide_type": "subslide"}
all_food.sort_values()

# %%
all_food.sort_values().is_monotonic_increasing

# %% slideshow={"slide_type": "subslide"}
all_food[['Pie', 'Cake']]

# %% slideshow={"slide_type": "subslide"}
all_food

# %%
all_food[1:3]

# %% slideshow={"slide_type": "subslide"}
# all_food['Cake':'Fudge']

# %% slideshow={"slide_type": "-"}
sorted_food['Cake':'Fudge']

# %% [markdown]
# **Beachte:** Der obere Slice-Wert `'Fudge'` ist im Ergebnis enthalten!

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Nicht vorhandene Werte

# %%
food = food1 + food2

# %%
food.isna()

# %%
food.isna().sum()

# %%
food.dropna()
