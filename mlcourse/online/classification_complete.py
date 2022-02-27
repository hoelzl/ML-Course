# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

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
scatterplot(make_circles())

# %%
scatterplot(make_circles(noise=0.5, factor=0.5))

# %%
fig, axs = plt.subplots(ncols=6, figsize=(20, 4))
for i, noise in enumerate(np.linspace(0, 0.3, 6)):
    scatterplot(make_circles(noise=noise, factor=0.5), ax=axs[i])
    axs[i].set_title(f"Noise: {noise}")


# %%
scatterplot(
    make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=1,
        n_clusters_per_class=1,
        # random_state=42,
        class_sep=2.0,
        flip_y=0.1,
    )
)

# %%
x, y = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=1,
    n_clusters_per_class=1,
    scale=(0.01, 100),
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
        make_circles(n_samples=250, noise=noise, random_state=102),
        make_classification(
            n_samples=250,
            n_features=2,
            n_redundant=0,
            n_informative=1,
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=104,
        ),
        make_classification(
            n_samples=250,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=106,
        ),
    ]


# %%
fix, axs = plt.subplots(ncols=4, figsize=(16, 4))
for i, ds in enumerate(create_datasets()):
    scatterplot(ds, ax=axs[i])

# %%
x_plot, y_plot = np.meshgrid(np.arange(-1, 1.1, 0.5), np.arange(-2, 2.1, 1))
x_plot, y_plot

# %%
x_plot.ravel()

# %%
y_plot.ravel()

# %%
np.c_[x_plot.ravel(), y_plot.ravel()]


# %%
def compute_plot_grid(x, dist):
    arr_min, arr_max = np.min(x, axis=0) - 0.5, np.max(x, axis=0) + 0.55
    return np.meshgrid(
        np.arange(arr_min[0], arr_max[0], dist), np.arange(arr_min[1], arr_max[1], dist)
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
x, y = make_moons(n_samples=250, noise=0.1, random_state=42)
x = StandardScaler().fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=42
)
scatterplot((x, y))

# %%
lr_clf = LogisticRegression()
lr_clf.fit(x_train, y_train)
y_lr_pred = lr_clf.predict(x_test)

# %%
scatterplot((x_test, y_lr_pred))

# %%
score = lr_clf.score(x_test, y_test)
print(score)

# %%
print(accuracy_score(y_test, y_lr_pred))
print(precision_score(y_test, y_lr_pred))
print(recall_score(y_test, y_lr_pred))
print(f1_score(y_test, y_lr_pred))

# %%

# %%
compute_plot_grid_coords(x, 2)

# %%
lr_clf.predict(compute_plot_grid_coords(x, 2))

# %%
lr_clf.decision_function(compute_plot_grid_coords(x, 2))

# %%
lr_clf.decision_function(compute_plot_grid_coords(x, 2)) > -3  # noqa

# %%
lr_clf.predict_proba(compute_plot_grid_coords(x, 2))

# %%
lr_clf.predict_proba(compute_plot_grid_coords(x, 2))[:, 1]

# %%
grid_x, grid_y = compute_plot_grid(x, 0.02)
grid_x.shape, grid_y.shape

# %%
z = lr_clf.decision_function(compute_plot_grid_coords(x, 0.02)).reshape(grid_x.shape)


# %%
plt.contour(grid_x, grid_y, z)

# %%
plt.contourf(grid_x, grid_y, z)

# %%
plt.contourf(grid_x, grid_y, z, cmap=cm)

# %%
plt.contourf(grid_x, grid_y, z, cmap=cm_bright)

# %%
fig, ax = plt.subplots(figsize=(12, 12))
ax.contourf(grid_x, grid_y, z, cmap=cm, alpha=0.5)
ax.scatter(
    x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", alpha=1
)
ax.scatter(
    x_test[:, 0], x_test[:, 1], c=y_lr_pred, cmap=cm_bright, edgecolors="k", alpha=0.6
)


# %%
def evaluate_classifier_on_single_dataset(
    classifier_class, ds, ax, scale=False, *args, **kwargs
):
    x, y = ds
    if scale:
        x = StandardScaler(*args, **kwargs).fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.5, random_state=42
    )

    clf = classifier_class(*args, **kwargs)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    score = lr_clf.score(x_test, y_test)
    print("Score:    ", score)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:   ", recall_score(y_test, y_pred))
    print("F1:       ", f1_score(y_test, y_pred))
    print()

    grid_x, grid_y = compute_plot_grid(x, 0.02)

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(compute_plot_grid_coords(x, 0.02)).reshape(
            grid_x.shape
        )
    else:
        z = clf.predict_proba(compute_plot_grid_coords(x, 0.02))[:, 1].reshape(
            grid_x.shape
        )

    ax.contourf(grid_x, grid_y, z, cmap=cm, alpha=0.75)
    ax.scatter(
        x_train[:, 0], x_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k", alpha=1
    )
    ax.scatter(
        x_test[:, 0], x_test[:, 1], c=y_pred, cmap=cm_bright, edgecolors="k", alpha=0.6
    )


# %%
fig, ax = plt.subplots()
evaluate_classifier_on_single_dataset(
    LogisticRegression, make_moons(n_samples=250, noise=0.1, random_state=42), ax
)


# %%
def evaluate_classifier(cls, scale=False, *args, **kwargs):
    datasets = create_datasets()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    for ds, ax in zip(datasets, axes.reshape(-1)):
        evaluate_classifier_on_single_dataset(cls, ds, ax, scale=scale)


# %%
evaluate_classifier(LogisticRegression)

# %%
evaluate_classifier(KNeighborsClassifier)

# %%
evaluate_classifier(DecisionTreeClassifier)

# %%
evaluate_classifier(RandomForestClassifier)

# %%
evaluate_classifier(
    RandomForestClassifier, n_estimators=250, min_samples_leaf=3, min_samples_split=8
)

# %%
evaluate_classifier(GradientBoostingClassifier)

# %%
evaluate_classifier(AdaBoostClassifier)

# %%
