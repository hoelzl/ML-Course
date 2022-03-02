# %%
import matplotlib.pyplot as plt

# %%
def show_dataset(ds, nrows=8, ncols=8, figsize=(12, 14)):
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    it = iter(ds)
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            img, label = next(it)
            ax.title.set_text(f"Label: {label}")
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.imshow(img.squeeze(0), cmap="binary")
    plt.show()

# %%
