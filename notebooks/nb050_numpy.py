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
# <h1 style="text-align:center;">Python: NumPy</h1>
# <h2 style="text-align:center;">Coding Akademie München GmbH</h2>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>

# %% [markdown] slideshow={"slide_type": "slide"}
# # Listen als Vektoren und Matrizen
#
# Wir können Python Listen verwenden um Vektoren darzustellen:

# %% pycharm={"name": "#%%\n"}
vector1 = [3, 2, 4]
vector2 = [8, 9, 7]


# %% [markdown]
# Es wäre dann möglich, Vektoroperationen auf derartigen Listen zu implementieren:

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
def vector_sum(v1, v2):
    assert len(v1) == len(v2)
    result = [0] * len(v1)
    for i in range(len(v1)):
        result[i] = v1[i] + v2[i]
    return result


# %% pycharm={"name": "#%%\n"}
vector_sum(vector1, vector2)

# %% [markdown] slideshow={"slide_type": "subslide"}
# Matrizen könnten dann als "Listen von Listen" dargestellt werden:

# %% pycharm={"name": "#%%\n"}
matrix = [[1, 2, 3],
          [2, 3, 4],
          [3, 4, 5]]

# %% [markdown] slideshow={"slide_type": "subslide"}
# Diese Implementierungsvariante hat jedoch einige Nachteile:
# - Performanz
#     - Speicher
#     - Geschwindigkeit
#     - Parallelisierbarkeit
# - Interface
#     - Zu allgemein
#     - `*`, `+` auf Listen entspricht nicht den Erwartungen
#     - ...
# - ...

# %% [markdown] slideshow={"slide_type": "slide"}
# # NumPy
#
# NumPy ist eine Bibliothek, die einen Datentyp für $n$-dimensionale Tensoren (`ndarray`) sowie effiziente Operationen darauf bereitstellt.
# - Vektoren
# - Matrizen
# - Grundoperationen für Lineare Algebra
# - Tensoren für Deep Learning
#
# Fast alle anderen mathematischen und Data-Science-orientierten Bibliotheken für Python bauen auf NumPy auf (Pandas, SciPy, Statsmodels, TensorFlow, ...).

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Überblick

# %% pycharm={"name": "#%%\n"}
import numpy as np

# %% pycharm={"name": "#%%\n"}
v1 = np.array([3, 2, 4])
v2 = np.array([8, 9, 7])

# %% pycharm={"name": "#%%\n"}
type(v1)

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": ""}
v1.dtype

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
v1 + v2

# %% pycharm={"name": "#%%\n"}
v1 * v2 # Elementweises (Hadamard) Produkt

# %% pycharm={"name": "#%%\n"}
v1.dot(v2)

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
v1.sum()

# %% pycharm={"name": "#%%\n"}
v1.mean()

# %% pycharm={"name": "#%%\n"}
v1.max()

# %% pycharm={"name": "#%%\n"}
v1.argmax(), v1[v1.argmax()]

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
m1 = np.array([[1, 2, 3],
               [4, 5, 6]])
m2 = np.array([[1, 0],
               [0, 1],
               [2, 3]])

 # %%
 # m1 + m2

# %%
m1.T

# %% pycharm={"name": "#%%\n"}
m1.T + m2

# %% pycharm={"name": "#%%\n"}
m1.dot(m2)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Erzeugen von NumPy Arrays
#
# ### Aus Python Listen
#
# Durch geschachtelte Listen lassen sich Vektoren, Matrizen und Tensoren erzeugen:

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
vector = np.array([1, 2, 3, 4])
vector

# %% pycharm={"name": "#%%\n"}
vector.shape

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
matrix = np.array([[1, 2, 3], [4, 5, 6]])
matrix

# %% pycharm={"name": "#%%\n"}
matrix.shape

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
tensor = np.array([[[1, 2], [3, 4]],
                   [[5, 6], [7, 8]]])
tensor

# %% pycharm={"name": "#%%\n"}
tensor.shape

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Als Intervall bzw. Folge

# %% pycharm={"name": "#%%\n"}
np.arange(10)

# %% pycharm={"name": "#%%\n"}
np.arange(10.0)

# %% pycharm={"name": "#%%\n"}
np.arange(2, 10)

# %% pycharm={"name": "#%%\n"}
np.arange(3., 23., 5.)

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
np.linspace(0, 10, 5)

# %% pycharm={"name": "#%%\n"}
np.linspace(0.1, 1, 10)

# %% pycharm={"name": "#%%\n"}
np.arange(0.1, 1.1, 0.1)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Konstant 0 oder 1

# %% pycharm={"name": "#%%\n"}
np.zeros(3)

# %% pycharm={"name": "#%%\n"}
np.zeros((3,))

# %% pycharm={"name": "#%%\n"}
np.zeros((3, 3))

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
np.ones(2)

# %% pycharm={"name": "#%%\n"}
np.ones((4, 5))

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Als Identitätsmatrix

# %% pycharm={"name": "#%%\n"}
np.eye(4)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Aus Zufallsverteilung
#
# Numpy bietet eine große Anzahl von möglichen [Generatoren und Verteilungen](https://docs.scipy.org/doc/numpy/reference/random/index.html) zum Erzeugen von Vektoren und Arrays mit zufälligen Elementen.

# %% [markdown]
# #### Setzen des Seed-Wertes

# %% pycharm={"name": "#%%\n"}
np.random.seed(101)

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Gleichverteilt in [0, 1)

# %% pycharm={"name": "#%%\n"}
# Kompatibilität mit Matlab
np.random.seed(101)
np.random.rand(10)

# %% pycharm={"name": "#%%\n"}
np.random.rand(4, 5)

# %% pycharm={"name": "#%%\n"}
# Fehler
# np.random.rand((4, 5))

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
np.random.seed(101)
np.random.random(10)

# %% pycharm={"name": "#%%\n"}
np.random.random((4, 5))

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Normalverteilte Zufallszahlen

# %% pycharm={"name": "#%%\n"}
# Kompatibilität mit Matlab
np.random.seed(101)
np.random.randn(10)

# %% pycharm={"name": "#%%\n"}
np.random.randn(4, 5)

# %% pycharm={"name": "#%%\n"}
# Fehler
# np.random.randn((4, 5))

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
np.random.seed(101)
np.random.standard_normal(10)

# %% pycharm={"name": "#%%\n"}
np.random.standard_normal((4, 5))

# %% pycharm={"name": "#%%\n"}
np.random.seed(101)
np.random.normal(10.0, 1.0, 10)

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
np.random.normal(0.0, 1.0, (4, 5))

# %% pycharm={"name": "#%%\n"}
np.random.normal(10.0, 0.2, (2, 5))

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Multivariate Normalverteilung
#

# %%
means = np.array([0.0, 2.0, 1.0])
cov = np.array([[2.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 2.0]])
np.random.multivariate_normal(means, cov, (3,))

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Andere Verteilungen

# %% pycharm={"name": "#%%\n"}
np.random.binomial(10, 0.2, 88)

# %% pycharm={"name": "#%%\n"}
np.random.multinomial(20, [1/6.0] * 6, 10)

# %% [markdown]
# Die [Dokumentation](https://docs.scipy.org/doc/numpy/reference/random/generator.html) enthält eine Liste aller Verteilungen und ihrer Parameter.

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Mini-Workshop
#
# - Notebook `050x-Workshop NumPy`
# - Abschnitt "Erzeugen von NumPy Arrays"
#

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Exkurs: Lösen von Gleichungssystemen
#
# Wie können wir das folgende Gleichungssystem mit NumPy darstellen und lösen:
#
# $$
# 2x_0 + x_1 + x_2 = 4\\
# x_0 + 3x_1 + 2x_2 = 5\\
# x_0 = 6
# $$

# %% slideshow={"slide_type": "subslide"}
a = np.array([[2., 1., 1.],
              [1., 3., 2.],
              [1., 0., 0.]])
b = np.array([4., 5., 6.])

# %%
x = np.linalg.solve(a, b)
x

# %% slideshow={"slide_type": "subslide"}
# Test:
a.dot(x), b

# %% [markdown]
# SciPy bietet spezielle Lösungsverfahren wie LU-Faktorisierung, Cholesky-Faktorisierung, etc. an.

# %% slideshow={"slide_type": "-"}
import scipy.linalg as linalg
lu = linalg.lu_factor(a)

# %%
lu

# %% slideshow={"slide_type": "subslide"}
x = linalg.lu_solve(lu, b)

# %%
x

# %%
a.dot(x)

# %% slideshow={"slide_type": "subslide"}
# Hermite'sche Matrix, positiv definit
a = np.array([[10., -1., 2., 0.],
             [-1., 11., -1., 3.],
             [2., -1., 10., -1.],
             [0., 3., -1., 8.]])
b= np.array([6., 25., -11., 15.])

# %% slideshow={"slide_type": "subslide"}
cholesky = linalg.cholesky(a)

# %%
cholesky

# %% slideshow={"slide_type": "subslide"}
cholesky.T.conj().dot(cholesky)

# %% slideshow={"slide_type": "subslide"}
y = np.linalg.solve(cholesky.T.conj(), b)

# %%
x = np.linalg.solve(cholesky, y)

# %%
x

# %%
a.dot(x)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Mini-Workshop
#
# - Notebook `050x-Workshop NumPy`
# - Abschnitt "Gleichungssysteme"

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Attribute von Arrays

# %% pycharm={"name": "#%%\n"}
int_array = np.arange(36)
float_array = np.arange(36.0)

# %% pycharm={"name": "#%%\n"}
int_array.dtype

# %% pycharm={"name": "#%%\n"}
float_array.dtype

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
int_array.shape

# %% pycharm={"name": "#%%\n"}
int_array.size

# %% pycharm={"name": "#%%\n"}
int_array.itemsize

# %% pycharm={"name": "#%%\n"}
float_array.itemsize

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
np.info(int_array)

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
np.info(float_array)

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Ändern von Shape und Größe

# %% pycharm={"name": "#%%\n"}
float_array.shape

# %% pycharm={"name": "#%%\n"}
float_matrix = float_array.reshape((6, 6))

# %% pycharm={"name": "#%%\n"}
float_matrix

# %% pycharm={"name": "#%%\n"}
float_matrix.shape

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
float_array.shape

# %% pycharm={"name": "#%%\n"}
float_array.reshape(3, 12)

# %% pycharm={"name": "#%%\n"}
# Fehler
# float_array.reshape(4, 8)

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
float_array.reshape((4, 9), order='F')

# %%
float_array.reshape((9, 4)).T

# %% slideshow={"slide_type": "subslide"}
np.resize(float_array, (4, 8))

# %%
float_array.shape

# %% slideshow={"slide_type": "subslide"}
np.resize(float_array, (8, 10))

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Mini-Workshop
#
# - Notebook `050x-NumPy`
# - Abschnitt "Erzeugen von NumPy Arrays 2"
#

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Broadcasting von Operationen
#
# Viele Operationen mit Skalaren werden Elementweise auf NumPy Arrays angewendet:

# %% pycharm={"name": "#%%\n"}
arr = np.arange(8)
arr

# %% pycharm={"name": "#%%\n"}
arr + 5

# %% pycharm={"name": "#%%\n"}
arr * 2

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
arr ** 2

# %% pycharm={"name": "#%%\n"}
2 ** arr

# %% pycharm={"name": "#%%\n"}
arr > 5

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Minimum, Maximum, Summe, ...

# %% pycharm={"name": "#%%\n"}
np.random.seed(101)
vec = np.random.rand(10)
vec

# %% pycharm={"name": "#%%\n"}
vec.max()

# %% pycharm={"name": "#%%\n"}
vec.argmax()

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
vec.min()

# %% pycharm={"name": "#%%\n"}
vec.argmin()

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
np.random.seed(101)
arr = np.random.rand(2, 5)
arr

# %% pycharm={"name": "#%%\n"}
arr.max()

# %% pycharm={"name": "#%%\n"}
arr.argmax()

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
arr.min()

# %% pycharm={"name": "#%%\n"}
arr.argmin()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Mini-Workshop
#
# - Notebook `050x-NumPy`
# - Abschnitt "Extrema"
#

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
arr.reshape(arr.size)[arr.argmin()]

# %% pycharm={"name": "#%%\n"}
arr[np.unravel_index(arr.argmin(), arr.shape)]

# %% slideshow={"slide_type": "subslide"}
arr

# %%
arr.sum()

# %%
arr.sum(axis=0)

# %%
arr.sum(axis=1)

# %% slideshow={"slide_type": "subslide"}
arr.mean()

# %%
arr.mean(axis=0)

# %%
arr.mean(axis=1)


# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Mini-Workshop
#
# - Notebook `050x-NumPy`
# - Abschnitt "Mittelwert"
#

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Exkurs: Einfache Monte Carlo Simulation
#
# Mit der folgenden Monte Carlo Simulation kann eine Approximation von $\pi$ berechnet werden.
#
# Die Grundidee ist zu berechnen, welcher Anteil an zufällig gezogenen Paaren aus Zahlen $(x, y)$, mit $x, y \sim SV[0, 1)$  (d.h., unabhängig und stetig auf $[0, 1)$ verteilt) eine $\ell^2$ Norm kleiner als 1 hat. Diese Zahl ist eine
# Approximation von $\pi/4$.
#
# Die folgende naive Implementiertung is in (fast) reinem Python geschrieben und verwendet NumPy nur zur Berechnung der Zufallszahlen.

# %% slideshow={"slide_type": "subslide"}
def mc_pi_1(n):
    num_in_circle = 0
    for i in range(n):
        xy = np.random.random(2)
        if (xy ** 2).sum() < 1:
            num_in_circle += 1
    return num_in_circle * 4 / n


# %% slideshow={"slide_type": "subslide"}
def test(mc_pi):
    np.random.seed(64)
    for n in [100, 10_000, 100_000, 1_000_000]:
        # %time print(f"𝜋 ≈ {mc_pi(n)} ({n} iterations).")
        pass


# %%
test(mc_pi_1)

# %% [markdown]
# Durch Just-in-Time Übersetzung mit Numba kann die Performance erheblich gesteigert werden:

# %%
import numba
mc_pi_1_nb = numba.jit(mc_pi_1)

# %%
test(mc_pi_1_nb)


# %% [markdown]
# Die folgende Implementierung verwendet die Vektorisierungs-Features von NumPy:

# %% slideshow={"slide_type": "subslide"}
def mc_pi_2(n):
    x = np.random.random(n)
    y = np.random.random(n)
    return ((x ** 2 + y ** 2) < 1).sum() * 4 / n


# %% slideshow={"slide_type": "subslide"}
test(mc_pi_2)

# %%
# # %time mc_pi_2(100_000_000)

# %% [markdown]
# Auch bei dieser Version können mit Numba Performance-Steigerungen erzielt werden, aber in deutlich geringerem Ausmaß:

# %%
mc_pi_2_nb = numba.jit(mc_pi_2)

# %%
test(mc_pi_2_nb)

# %%
# # %time mc_pi_2_nb(100_000_000)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Mini-Workshop
#
# - Notebook `050x-NumPy`
# - Abschnitt "Roulette"
#

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Indizieren von NumPy Arrays

# %% pycharm={"name": "#%%\n"}
vec = np.arange(10)

# %% pycharm={"name": "#%%\n"}
vec

# %% pycharm={"name": "#%%\n"}
vec[3]

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
vec[3:8]

# %% pycharm={"name": "#%%\n"}
vec[-1]

# %% slideshow={"slide_type": "subslide"}
arr = np.arange(24).reshape(4, 6)

# %% pycharm={"name": "#%%\n"}
arr

# %% pycharm={"name": "#%%\n"}
arr[1]

# %% pycharm={"name": "#%%\n"}
arr[1][2]

# %% pycharm={"name": "#%%\n"}
arr[1, 2]

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
arr

# %% pycharm={"name": "#%%\n"}
arr[1:3]

# %% pycharm={"name": "#%%\n"}
arr[1:3][2:4]

# %% pycharm={"name": "#%%\n"}
arr[1:3, 2:4]

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
arr[:, 2:4]

# %% pycharm={"name": "#%%\n"}
# Vorsicht!
arr[: 2:4]

# %% pycharm={"name": "#%%\n"}
arr[:, 1:6:2]

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Broadcasting auf Slices
#
# In NumPy Arrays werden Operationen oftmals auf Elemente (oder Unterarrays) "gebroadcastet":

# %% pycharm={"name": "#%%\n"}
arr = np.ones((3, 3))

# %% pycharm={"name": "#%%\n"}
arr[1:, 1:] = 2.0

# %% pycharm={"name": "#%%\n"}
arr

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
lst = [1, 2, 3]
vec = np.array([1, 2, 3])

# %% pycharm={"name": "#%%\n"}
lst[:] = [99]

# %% pycharm={"name": "#%%\n"}
vec[:] = [99]

# %% pycharm={"name": "#%%\n"}
lst

# %% pycharm={"name": "#%%\n"}
vec

# %% pycharm={"name": "#%%\n"}
vec[:] = 11
vec

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Vorsicht beim `lst[:]` Idiom! 

# %% pycharm={"name": "#%%\n"}
lst1 = list(range(10))
lst2 = lst1[:]
vec1 = np.arange(10)
vec2 = vec1[:]

# %% pycharm={"name": "#%%\n"}
lst1[:] = [22] * 10
lst1

# %% pycharm={"name": "#%%\n"}
lst2

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
vec1[:] = 22
vec1

# %% pycharm={"name": "#%%\n"}
vec2

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
vec1 = np.arange(10)
vec2 = vec1.copy()

# %% pycharm={"name": "#%%\n"}
vec1[:] = 22
vec1

# %% pycharm={"name": "#%%\n"}
vec2

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Bedingte Selektion
#
# NumPy Arrays können als Index auch ein NumPy Array von Boole'schen Werten erhalten, das den gleichen Shape hat wie das Array.
#
# Dadurch werden die Elemente selektiert, an deren Position der Boole'sche Vektor den Wert `True` hat und als Vektor zurückgegeben.

# %% pycharm={"name": "#%%\n"}
vec = np.arange(8)
bool_vec = (vec % 2 == 0)

# %% pycharm={"name": "#%%\n"}
vec[bool_vec]

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
arr = np.arange(8).reshape(2, 4)
bool_arr = (arr % 2 == 0)
bool_arr

# %% pycharm={"name": "#%%\n"}
arr[bool_arr]

# %% pycharm={"name": "#%%\n"}
# Fehler!
# arr[bool_vec]

# %% pycharm={"name": "#%%\n"} slideshow={"slide_type": "subslide"}
vec[vec % 2 > 0]

# %% pycharm={"name": "#%%\n"}
arr[arr < 5]

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Boole'sche Operationen auf NumPy Arrays

# %% pycharm={"name": "#%%\n"}
bool_vec

# %% pycharm={"name": "#%%\n"}
neg_vec = np.logical_not(bool_vec)

# %% pycharm={"name": "#%%\n"}
bool_vec & neg_vec

# %% pycharm={"name": "#%%\n"}
bool_vec | neg_vec

# %% [markdown] pycharm={"name": "#%%\n"} slideshow={"slide_type": "slide"}
# ## Universelle NumPy Operationen
#
# NumPy bietet viele "universelle" Funktionen an, die auf NumPy Arrays, Listen und Zahlen angewendet werden können:

# %%
vec1 = np.random.randn(5)
vec2 = np.random.randn(5)
list1 = list(vec1)
list2 = list(vec2)

# %%
vec1

# %%
list1

# %% slideshow={"slide_type": "subslide"}
np.sin(vec1)

# %%
np.sin(list1)

# %%
import math
np.sin(math.pi)

# %% slideshow={"slide_type": "subslide"}
np.sum(vec1)

# %%
np.sum(list1)

# %% slideshow={"slide_type": "subslide"}
np.mean(vec1)

# %%
np.median(vec1)

# %%
np.std(vec1)

# %% slideshow={"slide_type": "subslide"}
np.greater(vec1, vec2)

# %%
np.greater(list1, list2)

# %%
np.greater(vec1, list2)

# %% slideshow={"slide_type": "-"}
np.maximum(vec1, vec2)

# %%
np.maximum(list1, list2)

# %%
np.maximum(list1, vec2)

# %% [markdown]
# Eine vollständige Liste sowie weitere Dokumentation findet man [hier](https://docs.scipy.org/doc/numpy/reference/ufuncs.html).

# %%
