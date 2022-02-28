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
# # Erzeugen von NumPy Arrays

# %% pycharm={"name": "#%%\n"}
import numpy as np

# %% [markdown]
# Erzeugen Sie folgende NumPy Arrays. Verwenden Sie `np.array()` nur dann, wenn bisher keine andere Möglichkeit besprochen wurde das Array zu erzeugen.

# %% [markdown]
# ```python
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# ```

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# ```python
# array([0.  , 1.25, 2.5 , 3.75, 5.  ])
# ```

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# ```python
# array([ 1,  3, 12, 92])
# ```

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# ```python
# array([[1, 2, 3, 4],
#        [5, 6, 7, 8]])
# ```

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# ```python
# array([[[0, 2, 4, 6],
#         [1, 3, 5, 7]],
#
#        [[1, 2, 3, 4],
#         [5, 6, 7, 8]],
#
#        [[9, 8, 7, 6],
#         [5, 4, 3, 2]]])
# ```

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# Ein $2\times 8$ Array, das gleichverteilte Zufallszahlen in $[0, 1)$ enthält.

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# ```python
# array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])
# ```

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# Einen Vektor der Länge 5, der standard-normalverteilte Zahlen enthält.

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# Ein $3 \times 4$ Array, das normalverteilte Zahlen mit Mittelwert $5$ und Standardabweichung $0.5$ enthält.

# %%

# %% [markdown]
# # Gleichungssysteme
#
# Lösen Sie folgendes Gleichungssystem:
#
# $x_1 - x_2 + 2x_3 = 6$
#
# $2x_1 + 3x_2 + 2x_3 = 8$
#
# $3x_1 + 2x_2 + x_3 = 8$

# %%

# %% [markdown] pycharm={"name": "#%%\n"}
# # Erzeugen von NumPy Arrays 2
#
# Erzeugen Sie das folgende NumPy Array:
#
# ```python
# array([[ 0,  2,  4,  6],
#        [ 8, 10, 12, 14],
#        [16, 18, 20, 22],
#        [24, 26, 28, 30],
#        [32, 34, 36, 38]])
# ```

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# Erzeugen Sie das folgende NumPy Array:
#
# ```python
# array([[10, 19, 28, 37, 46, 55],
#        [13, 22, 31, 40, 49, 58],
#        [16, 25, 34, 43, 52, 61]])
# ```

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# # Regression mit KNN
#
# Sampeln Sie 150 Werte der Funktion $cos(x) + 0.2x$ im Intervall $[0, 12]$, mit normalverteiltem Fehler mit `size`$=0.5$.  Führen Sie eine Regression mit dem KNN-Verfahren durch. Verwenden Sie dabei sowohl 'uniform' als auch 'distance' als `weight` und experimentieren Sie mit verschiedenen Werten von $k$. Plotten Sie die Regressionskurve und berechnen Sie den Mean Square Error.

# %%

# %% [markdown] slideshow={"slide_type": "slide"}
# # Bilderkennung mit KNNs
#
# Da der MNIST-Datensatz für viele moderne Algorithmen zu einfach ist, wurde der Fashion-MNIST entwickelt, der die gleiche Struktur wie MNIST hat (60000 Trainingsbilder und 10000 Testbilder der Größe $32 \times 32$ Pixel in Graustufen. Im Gegensatz zu MNIST sind bei Fashion-MNIST aber 10 Kategorien von Kleidungsstücken zu Klassifizieren.
#
# Fashion MNIST ist in Tensorflow als `tensorflow.keras.datasets.fashion_mnist` verfügbar.
#
# Verwenden Sie KNN um einen Klassifizierer für Fashion MNIST zu erstellen. Evaluieren Sie Accuracy, Precision und Recall. Wie sind die Werte im Vergleich zu MNIST?

# %%

# %% [markdown]
# # Bildverarbeitung mit Ensembles und Entscheidungsbäumen
#
# Verwenden Sie verschiedene Ensemble-Techniken und Endscheidungsbaum-Varianten um das im letzten Workshop eingeführte Bilderkennungsproblem zu lösen.

# %%

# %% [markdown] pycharm={"name": "#%%\n"}
# # Extrema
#
# Erzeugen Sie einen Vektor der Länge 100 mit Zufallswerten, die im Intervall
# $[10, 20)$ gleichverteilt sind. 
#
# Berechnen Sie Minimum und Maximum der im Vektor
# enthaltenen Werte sowie die Positionen von Minimum und Maximum.

# %% pycharm={"name": "#%%\n"}

# %%

# %%

# %%

# %%

# %% [markdown]
# # Mittelwert
#
# Erzeugen Sie ein $6 \times 8$ Array, das standardverteilten Zahlen mit
# Mittelwert $2$ und Standardabweichung $1$ enthält.
#
#

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# Berechnen Sie den Mittelwert aller darin vorkommenden Werte.
#

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# Berechnen Sie die zeilen- und spaltenweisen Mittelwerte.

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# Berechnen Sie den Mittelwert aller vorkommenden Werte ohne Verwendung der
# Methode `mean()`.

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# Berechnen Sie die zeilen- und spaltenweisen Mittelwerte ohne Verwendung der
# Methode `mean()`.

# %% pycharm={"name": "#%%\n"}

# %% [markdown]
# # Roulette
#
# Analysieren Sie die Gewinnerwartung eines Spielers in folgender vereinfachter Form eines Roulettespiels mittels Monte Carlo Simulation:
#
# - Der Kessel ist in 36 Zahlen unterteilt.
# - Der Spieler wählt eine der Zahlen 1 bis 36 und wettet 1 Euro.
# - Fällt die Kugel auf die gewählte Zahl, so erhält der Spieler seinen Einsatz plus 35 Euro zurück.
# - Andernfalls verliert der Spieler seinen Einsatz.
#
# Schreiben Sie eine Version der Simulation mit `for`-Schleife in Python und testen Sie die Performance dieser Version vor und nach Kompilierung mit Numba. Schreiben Sie dann eine vektorisierte Version und testen Sie deren Performance in beiden Fällen.
#
# *Hinweise:* 
# - Die NumPy Bibliothek enthält eine Funktion `np.random.randint(low, high, size=None)`, mit der Sie ein Array mit Shape `size` erzeugen können, das gleichverteilte Zufallszahlen zwischen `low` (inklusive) und `high` (exklusive) enthält.
# - Wird `np.random.randint()` mit nur zwei Argumenten aufgerufen, so gibt es eine einzige Zahl zurück.
# - Die NumPy Bibliothek enthält eine Funtion `np.random.binomial(n, p, size=None)`, mit der Sie binomialverteilte Zufallszahlen erzeugen können.

# %%
