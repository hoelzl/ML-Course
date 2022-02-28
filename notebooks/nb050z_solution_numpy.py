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
# Erzeugen Sie folgende NumPy Arrays:

# %% [markdown]
# ```python
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# ```

# %% pycharm={"name": "#%%\n"}
np.arange(10)

# %% [markdown]
# ```python
# array([0.  , 1.25, 2.5 , 3.75, 5.  ])
# ```

# %% pycharm={"name": "#%%\n"}
np.linspace(0,5,5)

# %% [markdown]
# ```python
# array([ 1,  3, 12, 92])
# ```

# %% pycharm={"name": "#%%\n"}
np.array([1, 3, 12, 92])

# %% [markdown]
# ```python
# array([[1, 2, 3, 4],
#        [5, 6, 7, 8]])
# ```

# %% pycharm={"name": "#%%\n"}
np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

# %%
np.array([np.arange(1, 5), np.arange(5, 9)])

# %%
np.arange(1, 9).reshape((2, 4))

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
np.array([[[0, 2, 4, 6], [1, 3, 5, 7]],
          [[1, 2, 3, 4], [5, 6, 7, 8]],
          [[9, 8, 7, 6], [5, 4, 3, 2]]])

# %%
np.array([[range(0, 7, 2), range(1, 8, 2)],
          [range(1, 5), range(5, 9)],
          [range(9, 5, -1), range(5, 1, -1)]])

# %% [markdown]
# Ein $2\times 8$ Array, das gleichverteilte Zufallszahlen in $[0, 1)$ enthält.

# %% pycharm={"name": "#%%\n"}
np.random.rand(2, 8)

# %% [markdown]
# ```python
# array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]])
# ```

# %% pycharm={"name": "#%%\n"}
np.eye(3)

# %% [markdown]
# Einen Vektor der Länge 5, der standard-normalverteilte Zahlen enthält.

# %% pycharm={"name": "#%%\n"}
np.random.standard_normal(5)

# %% [markdown]
# Ein $3 \times 4$ Array, das normalverteilte Zahlen mit Mittelwert $5$ und Standardabweichung $0.5$ enthält.

# %% pycharm={"name": "#%%\n"}
np.random.normal(5, 0.5, (3, 4))

# %% [markdown] pycharm={"name": "#%%\n"}
# ## Gleichungssysteme
#
# Lösen Sie folgendes Gleichungssystem:
#
# $x_1 - x_2 + 2x_3 = 6$
#
# $2x_1 + 3x_2 + 2x_3 = 8$
#
# $3x_1 + 2x_2 + x_3 = 8$

# %%
a = np.array([[1., -1., 2.],
              [2.,  3., 2.],
              [3.,  2., 1.]])
b = np.array([6., 8., 8.])

# %%
import scipy.linalg
lu = scipy.linalg.lu_factor(a)

# %%
x = scipy.linalg.lu_solve(lu, b)

# %%
x

# %%
a.dot(x)

# %% [markdown]
# Einfacher:

# %%
np.linalg.solve(a, b)

# %% [markdown] pycharm={"name": "#%% md\n"}
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
np.arange(0, 40, 2).reshape(5, 4)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Erzeugen Sie das folgende NumPy Array:
#
# ```python
# array([[10, 19, 28, 37, 46, 55],
#        [13, 22, 31, 40, 49, 58],
#        [16, 25, 34, 43, 52, 61]])
# ```

# %% pycharm={"name": "#%%\n"}
np.arange(10, 64, 3).reshape(3, 6, order="F")

# %%
np.arange(10, 64, 3).reshape((6, 3)).T

# %% [markdown]
# # Regression mit KNN
#
# Sampeln Sie 150 Werte der Funktion $cos(x) + 0.2x$ im Intervall $[0, 12]$, mit normalverteiltem Fehler mit `size`$=0.5$.  Führen Sie eine Regression mit dem KNN-Verfahren durch. Verwenden Sie dabei sowohl 'uniform' als auch 'distance' als `weight` und experimentieren Sie mit verschiedenen Werten von $k$. Plotten Sie die Regressionskurve und berechnen Sie den Mean Square Error.

# %% slideshow={"slide_type": "subslide"}
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# %% slideshow={"slide_type": "-"}
np.random.seed(12)
NUM_SAMPLES = 50
MAX_X = 12
RANDOM_SCALE=0.5

# %% slideshow={"slide_type": "subslide"}
X = np.sort(MAX_X * np.random.random(NUM_SAMPLES))
X = X.reshape(-1, 1)
X[:3]


# %%
def fun(x):
    return np.cos(x) + 0.2 * x


# %% slideshow={"slide_type": "subslide"}
y = fun(X) + np.random.normal(size=X.shape, scale=RANDOM_SCALE)
y = y.reshape(-1)
y[:3]

# %%
n_neighbors = 5
knn = KNeighborsRegressor(n_neighbors, weights='uniform')
knn.fit(X, y)

# %% slideshow={"slide_type": "subslide"}
Xs = np.linspace(0, MAX_X, 500).reshape(-1, 1)
ys = knn.predict(Xs)
ys[:3]

# %% slideshow={"slide_type": "subslide"}
true_ys = fun(Xs)
true_ys[:3]

# %%
mean_squared_error(true_ys, ys)


# %% slideshow={"slide_type": "subslide"}
def plot_prediction(ys):
    plt.figure(figsize=(15,6))
    plt.scatter(X, y, color='orange', label='samples')
    plt.plot(Xs, ys, color='blue', label='predictions')
    plt.plot(Xs, true_ys, color='goldenrod', label='true_values')
    plt.legend()
plot_prediction(ys);

# %%
knn_dist = KNeighborsRegressor(n_neighbors, weights='distance')
knn_dist.fit(X, y)

# %%
Xs_dist = np.linspace(0, MAX_X, 500).reshape(-1, 1)
ys_dist = knn_dist.predict(Xs_dist)
ys_dist[:3]

# %%
mean_squared_error(true_ys, ys_dist)

# %%
plt.figure(figsize=(15,6))
plt.scatter(X, y, color='orange', label='samples')
plt.plot(Xs_dist, ys_dist, color='blue', label='predictions')
plt.legend()

# %% [markdown] slideshow={"slide_type": "slide"}
# # Bilderkennung mit KNNs
#
# Da der MNIST-Datensatz für viele moderne Algorithmen zu einfach ist, wurde der Fashion-MNIST entwickelt, der die gleiche Struktur wie MNIST hat (60000 Trainingsbilder und 10000 Testbilder der Größe $32 \times 32$ Pixel in Graustufen. Im Gegensatz zu MNIST sind bei Fashion-MNIST aber 10 Kategorien von Kleidungsstücken zu Klassifizieren.
#
# Fashion MNIST ist in Tensorflow als `tensorflow.keras.datasets.fashion_mnist` verfügbar.
#
# Verwenden Sie KNN um einen Klassifizierer für Fashion MNIST zu erstellen. Evaluieren Sie Accuracy, Precision und Recall. Wie sind die Werte im Vergleich zu MNIST?

# %% pycharm={"name": "#%%\n"}
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# %% pycharm={"name": "#%%\n"}
(X_train_in, y_train), (X_test_in, y_test) = fashion_mnist.load_data()


# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
def plot_random_datapoint():
    sample = np.random.randint(0, X_train_in.shape[0])

    plt.figure(figsize = (10,10))
    mnist_img = X_train_in[sample]
    plt.imshow(mnist_img, cmap="Greys")

    # Get the `Axes` instance on the current figure
    ax = plt.gca()

    plt.tick_params(
        axis='both', which='major', bottom=True, left=True,
        labelbottom=False, labelleft=False)

    plt.tick_params(
        axis='both', which='minor', bottom=False, left=False,
        labelbottom=True, labelleft=True)

    ax.set_xticks(np.arange(-.5, 28, 1))
    ax.set_yticks(np.arange(-.5, 28, 1))

    ax.set_xticks(np.arange(0, 28, 1), minor=True);
    ax.set_xticklabels([str(i) for i in np.arange(0, 28, 1)], minor=True);
    ax.set_yticks(np.arange(0, 28, 1), minor=True);
    ax.set_yticklabels([str(i) for i in np.arange(0, 28, 1)], minor=True);

    ax.grid(color='black', linestyle='-', linewidth=1.5)
    plt.colorbar(fraction=0.046, pad=0.04, ticks=[0,32,64,96,128,160,192,224,255])


# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
plot_random_datapoint()


# %% pycharm={"name": "#%%\n"}
def preprocess_data(data):
    return data.reshape(-1, 28 * 28)



# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
X_train = preprocess_data(X_train_in)
X_test = preprocess_data(X_test_in)


# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
def shuffle(X, y):
    Xs = np.column_stack((X, y))
    np.random.shuffle(Xs)
    return Xs[:, :-1], Xs[:, -1]


# %%
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

# %%
X_train, y_train = X_train[:5000], y_train[:5000]
X_test, y_test = X_test[:1000], y_test[:1000]

# %% pycharm={"name": "#%%\n"}
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# %%
n_neighbors = 5
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)

# %%
y_pred = knn.predict(X_test)
y_pred[:10], y_test[:10]

# %%
(accuracy_score(y_test, y_pred),
 precision_score(y_test, y_pred, average='macro'),
 recall_score(y_test, y_pred, average='macro'))

# %% [markdown]
# # Bildverarbeitung mit Ensembles und Entscheidungsbäumen
#
# Verwenden Sie verschiedene Ensemble-Techniken und Endscheidungsbaum-Varianten um das im letzten Workshop eingeführte Bilderkennungsproblem zu lösen.

# %%

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Extrema
#
# Erzeugen Sie einen Vektor der Länge 100 mit Zufallswerten, die im Intervall
# $[10, 20)$ gleichverteilt sind. 
#
# Berechnen Sie Minimum und Maximum der im Vektor
# enthaltenen Werte sowie die Positionen von Minimum und Maximum.

# %% pycharm={"name": "#%%\n"}
vec = np.random.random(100) * 10 + 10
vec[:10]

# %% pycharm={"name": "#%%\n"}
vec.min()

# %% pycharm={"name": "#%%\n"}
vec.argmin()

# %% pycharm={"name": "#%%\n"}
vec.max()

# %% pycharm={"name": "#%%\n"}
vec.argmax()

# %% [markdown] pycharm={"name": "#%% md\n"}
# # Mittelwert
#
# Erzeugen Sie ein $6 \times 8$ Array, das standardverteilten Zahlen mit
# Mittelwert $2$ und Standardabweichung $1$ enthält.
#
#

# %% pycharm={"name": "#%%\n"}
my_array = np.random.normal(2.0, 1.0, (6, 8))

# %% [markdown] pycharm={"name": "#%% md\n"}
# Berechnen Sie den Mittelwert aller darin vorkommenden Werte.
#

# %% pycharm={"name": "#%%\n"}
my_array.mean()

# %% [markdown] pycharm={"name": "#%% md\n"}
# Berechnen Sie die zeilen- und spaltenweisen Mittelwerte.

# %% pycharm={"name": "#%%\n"}
my_array.mean(axis=0)

# %% pycharm={"name": "#%%\n"}
my_array.mean(axis=1)

# %% [markdown] pycharm={"name": "#%% md\n"}
# Berechnen Sie den Mittelwert aller vorkommenden Werte ohne Verwendung der
# Methode `mean()`.

# %% pycharm={"name": "#%%\n"}
mean = my_array.sum() / my_array.size
mean

# %% [markdown] pycharm={"name": "#%% md\n"}
# Berechnen Sie die zeilen- und spaltenweisen Mittelwerte ohne Verwendung der
# Methode `mean()`.

# %% pycharm={"name": "#%%\n"}
my_array.sum(axis=0) / my_array.shape[0]

# %% pycharm={"name": "#%%\n"}
my_array.sum(axis=1) / my_array.shape[1]

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
import numpy as np


# %%
def roulette1(n):
    # Wir können davon ausgehen, dass der Spieler immer auf die 1 wettet
    money_spent = 0
    money_won = 0
    for i in range(n):
        money_spent += 1
        if np.random.randint(1, 37) == 1:
            money_won += 36
    return (money_won - money_spent) / n


# %%
def test_roulette(roulette):
    np.random.seed(123)
    for n in [1000, 100_000, 1_000_000]:
        # %time print(f"Gewinnerwartung ist {100 * roulette(n):.1f}% ({n} Versuche)")


# %%
test_roulette(roulette1)

# %%
import numba
roulette1_nb = numba.jit(roulette1)

# %%
test_roulette(roulette1_nb)


# %%
def roulette2(n):
    money_spent = np.ones(n)
    money_won = np.random.binomial(1, 1.0/36.0, n) * 36
    return (money_won - money_spent).sum() / n


# %%
test_roulette(roulette2)

# %%
roulette2_nb = numba.jit(roulette2)

# %%
test_roulette(roulette2_nb)


# %%
def roulette3(n):
    money_spent = n
    money_won = np.random.binomial(n, 1.0/36.0) * 36
    return (money_won - money_spent) / n


# %%
test_roulette(roulette3)

# %%
roulette3_nb = numba.jit(roulette3)

# %%
test_roulette(roulette3_nb)

# %%
roulette3(100_000_000)

# %%
