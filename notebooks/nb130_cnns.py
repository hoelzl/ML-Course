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
# <h1 style="text-align:center;">Convolutional Neural Nets</h1>
# <h2 style="text-align:center;">Dr. Matthias Hölzl</h2>

# %% [markdown] slideshow={"slide_type": "subslide"}
# # Darstellung von Bildern
#
# <img src="img/ag/Figure-21-001.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# # Filter für gelbe Pixel
# <img src="img/ag/Figure-21-002.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Funktionsweise des Gelbfilters
# <img src="img/ag/Figure-21-003.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## "Ausstanzen" der Werte
# <img src="img/ag/Figure-21-004.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Verschieben des Filters
# <img src="img/ag/Figure-21-005.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Beispiel
# <img src="img/ag/Figure-21-006.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Parallele Verarbeitung
# <img src="img/ag/Figure-21-007.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Konvolution
#
# <img src="img/ag/Figure-21-008.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Konvolution: Anker
#
# <img src="img/ag/Figure-21-009.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Konvolution: Funktionsweise
# <img src="img/ag/Figure-21-010.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Input und Gewichte haben die gleiche Größe
# <img src="img/ag/Figure-21-011.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Verschieben des Filters
# <img src="img/ag/Figure-21-013.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Beispiel
#
# <img src="img/ag/Figure-21-014.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>
#
# <br/>
# <div style="display: block; width: 30%; float: left">
#     <ul>
#         <li> Rote Felder: -1</li>
#         <li>Gelbe Felder: 1</li>
#         <li>Schwarze Felder: 0</li>
#         <li>Weiße Felder: 1</li>
#     </ul>
# </div>
#
# <div style="display: block; width: 50%; float: left;">
#     <ul>
#         <li>Minimalwert: -6</li>
#         <li>Maximalwert: 3</li>
#     </ul>
# </div>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-015.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-016.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Andere Betrachtungsweise: Zerschneiden von Bildern
# <img src="img/ag/Figure-21-017.png" style="width: 30%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Hierarchische Features
# <img src="img/ag/Figure-21-018.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-019.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-020.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-021.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-022.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-023.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # Randwerte
# <img src="img/ag/Figure-21-024.png" style="width: 30%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-025.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Verkleinerung des Resultats
#
# <img src="img/ag/Figure-21-026.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Padding
# <img src="img/ag/Figure-21-027.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # ConvNet für MNIST
# <img src="img/ag/Figure-21-048.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Performance
# <img src="img/ag/Figure-21-049.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/ag/Figure-21-050.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% slideshow={"slide_type": "slide"}
from fastai.vision.all import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%
model = nn.Sequential(
    nn.Conv2d(1, 32, 3, 1),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, 1),
    nn.MaxPool2d(2),
    nn.Dropout2d(0.25),
    nn.Flatten(1),
    nn.Linear(9216, 128),
    nn.ReLU(),
    nn.Dropout2d(0.5),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
)

# %%
transform = transforms.Compose([transforms.ToTensor()])
batch_size = 256
test_batch_size = 512
epochs = 5
learning_rate = 0.001

# %%
train_loader = DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform=transform),
                   batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    datasets.MNIST('../data', train=False, transform=transform),
                   batch_size=test_batch_size, shuffle=True)

# %%
data = DataLoaders(train_loader, test_loader)

# %%
learn = Learner(data, model, loss_func=F.nll_loss, opt_func=Adam, metrics=accuracy)

# %%
learn.lr_find()

# %%
learn.fit_one_cycle(epochs, learning_rate)

# %% [markdown] slideshow={"slide_type": "slide"}
# # Stride 1
# <img src="img/ag/Figure-21-028.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Stride 3 $\times$ 2
# <img src="img/ag/Figure-21-029.png" style="width: 20%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Stride 3 $\times$ 2
# <img src="img/ag/Figure-21-030.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Gleichförmige Strides: 2, 3
# <img src="img/ag/Figure-21-031.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Stride = Filtergröße
# <img src="img/ag/Figure-21-032.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Farbbilder: Mehrere Layer
# <img src="img/ag/Figure-21-033.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-034.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-035.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Stacks von Konvolutionen
# <img src="img/ag/Figure-21-036.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-037.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-038.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # 1D-Konvolution
# <img src="img/ag/Figure-21-039.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # 1$\times$1-Konvolution
# <img src="img/ag/Figure-21-040.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # 1$\times$1-Konvolution: Dimensionsreduktion
# <img src="img/ag/Figure-21-041.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # Padding und Upsampling (Fractional Convolution)
#
# <img src="img/ag/Figure-21-042.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Kein Padding
# <img src="img/ag/Figure-21-043.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 1 Pixel Padding
# <img src="img/ag/Figure-21-044.png" style="width: 30%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## 2 Pixel Padding
# <img src="img/ag/Figure-21-045.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Upsampling durch Konvolution
# <img src="img/ag/Figure-21-046.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-047.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # VGG 16
# <img src="img/ag/Figure-21-051.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/ag/Figure-21-052.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/ag/Figure-21-053.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-054.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-055.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-056.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Beispiel-Klassifizierung
# <img src="img/ag/Figure-21-057.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# # Klassifizierung von Bildern mit VGG16

# %%
path = untar_data(URLs.DOGS)

# %%
path.ls()

# %%
files = get_image_files(path/'images')
len(files)

# %%
files[0]


# %%
def label_func(f):
    return f[0].isupper()


# %%
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

# %%
dls.show_batch()

# %%
learn = cnn_learner(dls, vgg16_bn, metrics=error_rate)
learn.fine_tune(1)

# %%
learn.predict(files[0]), files[0]

# %%
learn.show_results()

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Visualisierung von VGG16 (Gradient Ascent)
# <img src="img/ag/Figure-21-058.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/ag/Figure-21-059.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/ag/Figure-21-060.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/ag/Figure-21-061.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/ag/Figure-21-062.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # Visualisierung (Effekt einzelner Layer)
# <img src="img/ag/Figure-21-063.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-064.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-065.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-066.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-067.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# <img src="img/ag/Figure-21-068.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "slide"}
# # Adverserial Examples
# <img src="img/ag/Figure-21-069.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-070.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown] slideshow={"slide_type": "subslide"}
# <img src="img/ag/Figure-21-071.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %%
