# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

# %%
make_moons(n_samples=10, noise=0.0)


# %%
def scatterplot(data, ax=None):
    sns.scatterplot(x=data[0][:, 0], y=data[0][:, 1], hue=data[1], ax=ax)


# %%
scatterplot(make_moons(noise=0.0))

# %%
scatterplot(make_moons(noise=0.5))

# %%
fig, axs = plt.subplots(ncols=6, figsize=(20, 4))
for i, noise in enumerate(np.linspace(0, 0.3, 6)):
    scatterplot(make_moons(noise=noise), ax=axs[i])
    axs[i].set_title(f"Noise: {noise}")

# %%
# %%
fig, axs = plt.subplots(ncols=6, figsize=(20, 4))
for i, noise in enumerate(np.linspace(0, 0.3, 6)):
    scatterplot(make_circles(noise=noise, factor=0.5), ax=axs[i])
    axs[i].set_title(f"Noise: {noise}")

# %%
x, y = make_classification(
    n_features=2,
    n_informative=1,
    n_redundant=0,
    n_classes=2,
    n_clusters_per_class=1,
    scale=(0.01, 100),
    shift=10,
    random_state=42,
)

# %%
scatterplot((x, y))

# %%
scaler = StandardScaler()
scaler.fit(x)
x_t = scaler.transform(x)

# %%
scatterplot((x_t, y))


# %%
def create_datasets(noise=0.15):
    return [
        make_moons(n_samples=250, noise=noise, random_state=101),
        make_circles(n_samples=250, noise=noise, factor=0.5, random_state=102),
        make_classification(
            n_samples=250,
            n_features=2,
            n_redundant=0,
            n_informative=1,
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=103,
        ),
        make_classification(
            n_samples=250,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=104,
        ),
    ]


# %%
fig, axs = plt.subplots(ncols=4, figsize=(16, 4))
for i, ds in enumerate(create_datasets()):
    scatterplot(ds, ax=axs[i])

# %%
np.arange(-1, 1.1, 0.5)

# %%
x_plot, y_plot = np.meshgrid(np.arange(-1, 1.1, 0.5), np.arange(-2, 2.1, 1))
x_plot, y_plot

# %%
x_plot.ravel()

# %%
x_plot.reshape(-1)

# %%
np.c_[x_plot.ravel(), y_plot.ravel()]


# %%
def compute_plot_grid(x, dist):
    arr_min, arr_max = np.min(x, axis=0) - 0.5, np.max(x, axis=0) + 0.55
    return np.meshgrid(
        np.arange(arr_min[0], arr_max[0], dist), np.arange(arr_min[1], arr_max[1], 1)
    )


# %%
compute_plot_grid(np.arange(6).reshape(3, 2), 1)


# %%
def compute_plot_grid_coords(x, dist):
    coord_x, coord_y = compute_plot_grid(x, dist)
    return np.c_[coord_x.ravel(), coord_y.ravel()]


# %%
compute_plot_grid_coords(np.arange(6).reshape(3, 2), 1)

# %%
x, y = make_moons(n_samples=250, noise=0.15, random_state=101)
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=42
)

# %%
scatterplot((x, y))

# %%
lr_clf = LogisticRegression()
lr_clf.fit(x_train, y_train)
y_lr_pred = lr_clf.predict(x_test)

# %%
scatterplot((x_test, y_lr_pred))

# %%
lr_clf.score(x_test, y_test)

# %%
print(accuracy_score(y_test, y_lr_pred))
print(precision_score(y_test, y_lr_pred))
print(recall_score(y_test, y_lr_pred))
print(f1_score(y_test, y_lr_pred))


# %%
