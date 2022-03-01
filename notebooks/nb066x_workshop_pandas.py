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

# %% [markdown]
# Importieren Sie Pandas und lesen Sie die Datei `population_by_county.csv` in einen Data Frame `population`

# %%
import pandas as pd

# %%
population = pd.read_csv('population_by_county.csv')

# %% [markdown]
# Zeigen Sie die ersten Zeilen des `population` Data Frames an.

# %%
population.head()

# %% [markdown]
# Bestimmen Sie die Spaltennamen von `population`.

# %%
population.columns

# %% [markdown]
# Welche Staten werden in der Tabelle erfasst? Wie viele Staten sind das?

# %%
population['State'].unique()

# %%
population['State'].nunique()

# %% [markdown]
# Welches sind die 10 häufigsten County-Namen in diesem Datensatz?

# %%
population['County'].value_counts()[0:10]

# %% [markdown]
# Welches sind die 10 bevölkerungsreichsten Staaten nach der Schätzung 2017?
#
# *Hinweis:* Verwenden Sie Hilfsvariablen um Zwischenergebnisse auszuwerten. Mit `Shift-Tab` können Sie sehen, welche Parameter die verwendeten Funktionen akzeptieren.

# %%
population_per_state = population.groupby('State').sum()['2017PopEstimate']

# %%
population_per_state.sort_values(ascending=False)[0:10]

# %%
# Alternativ:
population_per_state.sort_values()[-10:]

# %% [markdown]
# Wie viele Staaten haben nach der Schätzung 2017 mehr als 1 Million Einwohner?

# %%
population_per_state[population_per_state > 1_000_000].count()

# %% [markdown]
# Wie viele Counties haben das Wort "County" in ihrem Namen?

# %%
population['County'].apply(lambda n: 'County' in n).sum()

# %% [markdown]
# Berechnen Sie, als Vorbereitung für die nächste Frage, die proportionale Änderung (d.h., 1.0 entspricht keiner Änderung) zwischen der Volkszählung von 2010 und der Schätzung von 2017 für alle Counties.

# %%
(population['2017PopEstimate'] - population['2010Census']) / population['2010Census']

# %% [markdown]
# Fügen Sie eine Spalte `PercentChange` zu `population` hinzu, die die prozentuale Änderung zwischen der Volkszählung von 2010 und der Schätzung von 2017 angibt. (D.h. 100 entspricht keiner Änderung)

# %%
population['PercentChange'] = 100 * (population['2017PopEstimate'] - population['2010Census']) / population['2010Census']

# %%
population.head()

# %% [markdown]
# Welche 10 Staaten haben die höchste prozentuelle Änderung?
#
# *Hinweis:* Speichern Sie die pro Staat gruppierte Bevölkerung in einer neuen Tabelle und verwenden Sie diese zur Berechnung.

# %%
pop_by_state = population.groupby('State').sum()

# %%
del pop_by_state['PercentChange']

# %%
pop_by_state.head()

# %%
pop_by_state['PercentChange'] = 100 * (pop_by_state['2017PopEstimate'] - pop_by_state['2010Census']) / pop_by_state['2010Census']

# %%
pop_by_state.head()

# %%
pop_by_state.sort_values('PercentChange', ascending=False)[0:10]

# %%
