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
# <h1 style="text-align:center;">Python: Pandas Visualisierung</h1>
# <h2 style="text-align:center;">Coding Akademie München GmbH</h2>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <div style="text-align:center;">Allaithy Raed</div>

# %% [markdown] slideshow={"slide_type": "slide"}
# # Pandas Visualisierung
#
# Pandas bietet Visualisierungs-Funktionalität, die auf Matplotlib aufbaut und an Data Frames angepasst ist.
#
# Dokumentation zur Pandas `plot`-Funktion findet man  [hier](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html).

# %%
import numpy as np
import pandas as pd

# %%
df1 = pd.read_csv('df1a.csv', index_col=0)
df1_large = pd.read_csv('df1b.csv', index_col=0)
df2 = pd.read_csv('df2a.csv', index_col=0).abs()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Plot Arten
#
# Pandas bietet folgende Plot Arten:
#
# | Typ       | Beschreibung
# |:---------:|:-------------------------------|
# | `line`    | line plot (default)
# | `bar`     | vertical bar plot
# | `barh`    | horizontal bar plot
# | `hist`    | histogram
# | `box`     | boxplot
# | `kde`     | Kernel Density Estimation plot
# | `density` | same as ‘kde’
# | `area`    | area plot
# | `pie`     | pie plot
# | `scatter` | scatter plot
# | `hexbin`  | hexbin plot.
#
# Plots können entweder durch `df.plot.bar()` oder durch `df.plot(kind='bar')` erzeugt werden.

# %% slideshow={"slide_type": "subslide"}
df1.head()

# %%
df2.head()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Line Plots

# %%
df2.plot.line(y='Col 0', figsize=(12, 1.2));

# %%
df2.plot(kind='line', y='Col 0', figsize=(12, 1.2));

# %%
df2.plot(y='Col 0', figsize=(12, 1.2));

# %% slideshow={"slide_type": ""}
df2.plot(figsize=(12, 5));

# %% slideshow={"slide_type": "subslide"}
df2.plot(figsize=(12, 5), lw=3, grid=True);

# %% slideshow={"slide_type": "subslide"}
ax = df2.plot(figsize=(12, 5), lw=3, grid=True, title="Important Data")
ax.set(xlabel="The important x value", ylabel="Important y values")
ax.autoscale(axis='x', tight=True)
# ax.legend(loc=2)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Area Plots

# %%
df2.plot.area(figsize=(12, 4));

# %% slideshow={"slide_type": "subslide"}
ax = df2.plot.area(figsize=(12,6), alpha=0.2)
ax.plot(df2, lw=3)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Bar Plots

# %% slideshow={"slide_type": ""}
df2.plot.bar(figsize=(12, 6));

# %% slideshow={"slide_type": "subslide"}
df2.plot.bar(stacked=True,grid=True, figsize=(12, 6));

# %% slideshow={"slide_type": "subslide"}
df2.plot.barh(stacked=True, figsize=(12,6));

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Histograms

# %% slideshow={"slide_type": ""}
df1['Col 0'].plot.hist(figsize=(12, 6));

# %% slideshow={"slide_type": "subslide"}
ax = df1['Col 0'].plot.hist(edgecolor='k', grid=True, figsize=(12, 6))
ax.autoscale(enable=True, axis='x', tight=True)

# %% slideshow={"slide_type": "subslide"}
ax = df1['Col 0'].plot.hist(bins=50, edgecolor='k', figsize=(12, 6))

# %% slideshow={"slide_type": "subslide"}
ax = df1['Col 1'].plot.hist(bins=50, edgecolor='k', figsize=(12, 6))

# %% slideshow={"slide_type": "subslide"}
df1_large['Col 0'].plot.hist(bins=50, edgecolor='k', figsize=(12, 6))

# %%
df1.plot.hist(bins=25, alpha=0.2);

# %% slideshow={"slide_type": "subslide"}
df1['Col 0'].hist()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Scatter Plots

# %%
df_scatter = df1[:2000]

# %%
df_scatter.plot.scatter(x='Col 0', y='Col 1', figsize=(12, 6));

# %% slideshow={"slide_type": "subslide"}
df_scatter.plot.scatter(x='Col 0', y='Col 1', c='Col 2', figsize=(12, 6));

# %% slideshow={"slide_type": "subslide"}
df_scatter.plot.scatter(x='Col 0', y='Col 1',
                        c='Col 2', cmap='viridis',
                        figsize=(12, 6));

# %%
df_scatter.plot.scatter(x='Col 0', y='Col 1',
                        c='Col 2', cmap='coolwarm',
                        figsize=(12, 6),
                        s=df_scatter['Col 2'].abs() * 3,
                        alpha=0.6);

# %% [markdown]
# ## Hexagonaler Bin Plot

# %%
df_scatter.plot.hexbin(x='Col 0', y='Col 1',
                       cmap='coolwarm',
                       gridsize=20,
                       figsize=(8, 6));

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Box and Whisker Diagramme

# %%
df2.plot.box();

# %% slideshow={"slide_type": "subslide"}
df2['Group'] = 'a a b c b b b a c c a b'.split()

# %%
df2[['Col 0', 'Group']].boxplot(by='Group')

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Kernel Density Estimation Plot

# %%
df1['Col 0'].plot.kde(figsize=(12, 5))

# %%
df1[0:20].plot.kde(figsize=(12, 5));

# %%
