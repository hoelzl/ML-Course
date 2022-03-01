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
# <h1 style="text-align:center;">Python: Pandas Data Frames 1</h1>
# <h2 style="text-align:center;">Coding Akademie München GmbH</h2>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <div style="text-align:center;">Allaithy Raed</div>

# %% [markdown] slideshow={"slide_type": "slide"}
# # Data Frames
#
# Data Frames sind die am häufigsten verwendete Datenstruktur von Pandas.
#
# Sie ermöglichen das bequeme Einlesen, Verarbeiten und Speichern von Daten.
#
# Konzeptionell besteht ein Data Frame aus mehreren `Series`-Instanzen, die einen gemeinsamen Index haben.

# %% slideshow={"slide_type": "-"}
import numpy as np
import pandas as pd


# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Erzeugen eines Data Frames

# %% [markdown]
# ### Aus einem NumPy Array

# %%
def create_data_frame():
    rng = np.random.default_rng(42)
    array = rng.normal(size=(5, 4), scale=5.0)
    index = 'A B C D E'.split()
    columns = 'w x y z'.split()
    return pd.DataFrame(array, index=index, columns=columns)


# %% slideshow={"slide_type": "subslide"}
df = create_data_frame()
df

# %%
type(df)

# %% [markdown]
# ### Aus einer CSV-Datei

# %%
df_csv = pd.read_csv("example_data.csv")

# %%
df_csv

# %%
df_csv = pd.read_csv("example_data.csv", index_col=0)

# %%
df_csv

# %% [markdown]
# ### Aus einer Excel Datei

# %%
df_excel = pd.read_excel("excel_data.xlsx", index_col=0)

# %%
df_excel

# %%
df_excel2 = pd.read_excel("excel_other_sheet.xlsx", index_col=0)

# %%
df_excel2

# %%
df_excel2 = pd.read_excel("excel_other_sheet.xlsx", index_col=0, sheet_name='Another Sheet')

# %%
df_excel2.head()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Andere Formate:

# %%
pd.read_clipboard
pd.read_html
pd.read_json
pd.read_pickle
pd.read_sql; # Verwendet SQLAlchemy um auf eine Datenbank zuzugreifen
# usw.

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Indizes und Operationen

# %%
df_csv.head()

# %%
df_csv.tail()

# %% slideshow={"slide_type": "subslide"}
df = create_data_frame()
df['w']

# %%
type(df['w'])

# %%
# Sollte nicht verwendet werden...
df.w

# %% slideshow={"slide_type": "subslide"}
df[['w', 'y']]

# %%
df.index

# %%
df.index.is_monotonic_increasing

# %%
df.size

# %%
df.ndim

# %%
df.shape

# %% [markdown] slideshow={"slide_type": "slide"}
# ### Erzeugen, Umbenennen und Löschen von Spalten

# %%
df = create_data_frame()
df['Summe aus w und y'] = df['w'] + df['y']

# %%
df

# %% slideshow={"slide_type": "subslide"}
df.rename(columns={'Summe aus w und y': 'w + y'})

# %%
df

# %% slideshow={"slide_type": "subslide"}
df.rename(columns={'Summe aus w und y': 'w + y'}, index={'E': 'Z'}, inplace=True)

# %%
df

# %% slideshow={"slide_type": "subslide"}
type(df['y'])

# %%
del df['y']

# %%
df

# %% slideshow={"slide_type": "subslide"}
df.drop('A')

# %%
df

# %% slideshow={"slide_type": "subslide"}
df.drop('B', inplace=True)

# %%
df

# %% slideshow={"slide_type": "subslide"}
df.drop('z', axis=1)

# %% slideshow={"slide_type": "-"}
df

# %% slideshow={"slide_type": "subslide"}
df.drop('z', axis=1, inplace=True)

# %%
df

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Auswahl

# %%
df = create_data_frame()
df

# %% slideshow={"slide_type": "subslide"}
df['w']

# %% slideshow={"slide_type": "subslide"}
# Fehler
# df['A']

# %% slideshow={"slide_type": "-"}
df.loc['B']

# %%
type(df.loc['B'])

# %% slideshow={"slide_type": "subslide"}
df

# %% slideshow={"slide_type": "-"}
df.iloc[1]

# %% slideshow={"slide_type": "subslide"}
df.loc[['A', 'C']]

# %%
df.loc[['A', 'C'], ['x', 'y']]

# %%
df.loc['B', 'z']

# %% slideshow={"slide_type": "subslide"}
df.iloc[[1, 2], [0, 3]]

# %%
df.iloc[0, 0]

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Bedingte Auswahl

# %%
df = create_data_frame()
df

# %%
df > 0

# %%
df[df > 0]

# %% slideshow={"slide_type": "subslide"}
df['w'] > 0

# %%
df[df['w'] > 0]

# %% slideshow={"slide_type": "subslide"}
df[df['w'] > 0][['x', 'y']]

# %% slideshow={"slide_type": "subslide"}
df[(df['w'] > 0) & (df['x'] < 0)]

# %% [markdown] slideshow={"slide_type": "slide"}
# # Information über Data Frames

# %%
df = pd.DataFrame(array, index=index, columns=columns)
df['txt'] = 'a b c d e'.split()
df.iloc[1, 1] = np.nan
df

# %%
df.describe()

# %% slideshow={"slide_type": "subslide"}
df.info()

# %% slideshow={"slide_type": "subslide"}
df.dtypes

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Data Frame Index

# %%
df = create_data_frame()
df['txt'] = 'a b c d e'.split()
df

# %%
df.reset_index()

# %% slideshow={"slide_type": "subslide"}
df

# %%
df.reset_index(inplace=True)

# %%
df

# %% slideshow={"slide_type": "subslide"}
df.rename(columns={'index': 'old_index'}, inplace=True)

# %%
df

# %% slideshow={"slide_type": "subslide"}
df.set_index('txt')

# %%
df

# %% slideshow={"slide_type": "subslide"}
df.set_index('txt', inplace=True)
df

# %%
df.set_index('old_index', inplace=True)
df

# %% slideshow={"slide_type": "subslide"}
df.info()

# %%
df.index

# %% slideshow={"slide_type": "subslide"}
df.index.name = None

# %%
df

# %%
