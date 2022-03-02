# %%
import matplotlib.pyplot as plt

# %%
def show_dataset(ds, nrows=8, ncols=8, figsize=(12, 12)):
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    it = iter(ds)
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.imshow(next(it)[0].squeeze(0), cmap="binary")
    plt.show()
