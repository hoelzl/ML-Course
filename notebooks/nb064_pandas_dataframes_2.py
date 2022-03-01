# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/python-logo-notext.svg"
#      style="display:block;margin:auto;width:10%"/>
# <h1 style="text-align:center;">Python: Pandas Data Frames 2</h1>
# <h2 style="text-align:center;">Coding Akademie München GmbH</h2>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <div style="text-align:center;">Allaithy Raed</div>

# %% slideshow={"slide_type": "subslide"}
import numpy as np
import pandas as pd


# %% [markdown]
# ## Fehlende Werte

# %%
def create_data_frame_with_nans():
    return pd.DataFrame({'A': [1, 2, np.nan, np.nan, 0],
                         'B': [5, 6, 7, np.nan, 0],
                         'C': [9, 10, 11, 12, 0],
                         'D': [13, 14, 15, 16, 0],
                         'E': [np.nan, 18, 19, 20, 0]})


# %% slideshow={"slide_type": "subslide"}
df = create_data_frame_with_nans()

# %%
df

# %% slideshow={"slide_type": "subslide"}
df.isna()

# %%
df.count()

# %% slideshow={"slide_type": "subslide"}
df

# %%
df.count(axis=1)

# %% slideshow={"slide_type": "subslide"}
df.isna().sum()

# %%
df.isna().sum(axis=1)

# %% slideshow={"slide_type": "subslide"}
df

# %%
df.dropna()

# %% slideshow={"slide_type": "subslide"}
df

# %%
df.dropna(axis=1, inplace=True)

# %%
df

# %% slideshow={"slide_type": "subslide"}
df = create_data_frame_with_nans()
df

# %%
df.fillna(value=1000)

# %% slideshow={"slide_type": "subslide"}
df.fillna(value=df.mean())

# %%
df.mean()


# %%
# df.fillna(value=df.mean(), axis=1)

# %% [markdown]
# ## Gruppierung

# %%
def create_course_df():
    data = {'Course':['Python','Python','Python','Python','Java','Java','Java','C++','C++'],
            'Person':['Jack', 'Jill', 'John', 'Bill', 'Jack', 'Bill', 'Davy', 'Jack', 'Diane'],
            'Score':[97, 92, 38, 73, 81, 52, 62, 86, 98]}
    return pd.DataFrame(data)


# %%
df = create_course_df()
df

# %% slideshow={"slide_type": "subslide"}
df.groupby('Course')

# %%
df_by_course = df.groupby('Course')

# %% slideshow={"slide_type": "subslide"}
df_by_course.count()

# %%
df_by_course['Person'].count()

# %% slideshow={"slide_type": "subslide"}
df_by_course.mean()

# %% slideshow={"slide_type": "-"}
df_by_course.std()

# %% slideshow={"slide_type": "subslide"}
df_by_course.aggregate(['mean', 'std'])

# %% slideshow={"slide_type": "subslide"}
df_by_course.aggregate(['min', 'max'])

# %%
df_by_course['Score'].aggregate(['min', 'max', 'mean', 'std'])

# %% slideshow={"slide_type": "subslide"}
df.groupby('Person').mean()

# %% slideshow={"slide_type": "subslide"}
df_by_course.describe()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Nochmal Operationen

# %%
df = create_course_df()
df

# %% slideshow={"slide_type": "subslide"}
df.columns

# %%
df.index

# %% slideshow={"slide_type": "subslide"}
df.sort_values(by='Course')

# %% slideshow={"slide_type": "subslide"}
df['Course'].values

# %%
df['Person'].values

# %% slideshow={"slide_type": "subslide"}
df['Course'].unique()

# %%
df['Person'].unique()

# %%
df['Person'].nunique()

# %%
df['Person'].value_counts()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Selektion

# %% slideshow={"slide_type": "-"}
df[df['Score'] > 80]

# %% slideshow={"slide_type": "subslide"}
df[(df['Score'] > 60) & (df['Score'] <= 80)]

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Transformation von Daten

# %%
df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns = ['A', 'B'])

# %%
df

# %% slideshow={"slide_type": "subslide"}
df.apply(np.square)

# %% slideshow={"slide_type": "subslide"}
df.apply(np.sum)

# %%
df.apply(np.sum, axis=1)

# %% slideshow={"slide_type": "subslide"}
df.apply(lambda n: [np.sum(n), np.mean(n)], axis=1)

# %%
df.apply(lambda n: [np.sum(n), np.mean(n)], axis=1, result_type='expand')

# %% [markdown]
# Elementweises Anwenden einer Funktion:

# %%
df.applymap(lambda x: f"Value: {x}")

# %%
