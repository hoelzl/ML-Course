# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # MNIST

# %%
from pickletools import optimize
import matplotlib.pyplot as plt
from pytorch_model_summary import summary
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split

from mlcourse.utils.data import show_dataset

# %%
input_size = 28 * 28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.005
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# %% slideshow={"slide_type": "subslide"}
mnist_transforms = transforms.Compose(
    [transforms.Resize((28, 28)), transforms.ToTensor()]
)

# %%
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=mnist_transforms, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=mnist_transforms, download=True
)

# %% slideshow={"slide_type": "subslide"}
it = iter(train_dataset)
tensor, label = next(it)
print(f"Batch shape: {tensor.shape}, label: {label}")
plt.imshow(tensor[0], cmap="binary")

# %% slideshow={"slide_type": "subslide"}
tensor, label = next(it)
print(f"Batch shape: {tensor.shape}, label: {label}")
plt.imshow(tensor[0], cmap="binary")


# %%
show_dataset(train_dataset)

# %%
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False
)


# %%
it = iter(train_loader)
first_batch, first_labels = next(it)
second_batch, second_labeld = next(it)
print(f"First batch:  {type(first_batch)}, {first_batch.shape}")
print(f"Second batch: {type(second_batch)}, {second_batch.shape}")


# %% slideshow={"slide_type": "subslide"}
def create_model(hidden_size):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, optimizer

# %%
mlp16, _ = create_model(16)
print(summary(mlp16, torch.zeros((100, 784)), show_input=True))
print(summary(mlp16, torch.zeros((100, 784))))


# %% slideshow={"slide_type": "subslide"}
def training_loop(
    n_epochs, optimizer, model, loss_fn, device, train_loader, print_progress=True
):
    model = model.to(device)
    all_batch_losses = []
    for epoch in range(1, n_epochs + 1):
        accumulated_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            output = model(images)
            batch_loss = loss_fn(output, labels)
            with torch.no_grad():
                accumulated_loss += batch_loss
                all_batch_losses.append(batch_loss.detach().cpu())

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                if print_progress:
                    print(
                        f"Epoch {epoch:2}/{n_epochs:2}, step {i + 1}: "
                        f"training loss = {accumulated_loss.item() / i:6.4f}"
                    )
                accumulated_loss = 0
    return all_batch_losses


# %%
def create_and_train_model(hidden_size, num_epochs=num_epochs, print_progress=True):
    model, optimizer = create_model(hidden_size)
    losses = training_loop(
        n_epochs=num_epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        train_loader=train_loader,
        print_progress=print_progress,
    )
    return model, losses


# %%
model, losses = create_and_train_model(32, num_epochs=5, print_progress=True)

# %%
plt.figure(figsize=(16, 5))
plt.plot(range(len(losses)), losses)


# %%
def evaluate_model(model):
    ground_truth = []
    predictions = []
    with torch.no_grad():
        for x, y in test_loader:
            new_predictions = model(x.reshape(-1, input_size).to(device))
            predictions.extend(new_predictions.argmax(dim=1).cpu().numpy())
            ground_truth.extend(y.numpy())
        return ground_truth, predictions


# %%
from sklearn.metrics import classification_report

print(classification_report(*evaluate_model(model)))

# %%
# model, losses = create_and_train_model(64, print_progress=False)

# pyplot.figure(figsize=(16, 5))
# pyplot.plot(range(len(losses)), losses)

# %%
# print(classification_report(*evaluate_model(model)))

# %%
# model, losses = create_and_train_model(512, print_progress=False)

# pyplot.figure(figsize=(16, 5))
# pyplot.plot(range(len(losses)), losses)

# %%
# print(classification_report(*evaluate_model(model)))

# %%

# %%
mlp_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(input_size, 32),
    nn.ReLU(),
    nn.Linear(32, num_classes),
)

# %%
mlp_classifier = NeuralNetClassifier(
    mlp_model,
    criterion=nn.CrossEntropyLoss,
    batch_size=100,
    max_epochs=2,
    lr=0.2,
    iterator_train__shuffle=True,
    train_split=predefined_split(test_dataset),
    device=device,
)

# %%
mlp_classifier.fit(train_dataset, None)

# %%
mlp_classifier.partial_fit(train_dataset, None)

# %%
y_pred_mlp = mlp_classifier.predict(test_dataset)

# %%
y_test = [y for _, y in test_dataset]

# %%
print(classification_report(y_test, y_pred_mlp))

# %%
print(confusion_matrix(y_test, y_pred_mlp))

# %%
plt.figure(figsize=(10, 8))
ax = plt.axes()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_mlp, ax=ax)

# %%
plt.figure(figsize=(14, 12))
ax = plt.axes()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_mlp, normalize="true", ax=ax)

# %%
mlp_classifier_adam = NeuralNetClassifier(
    mlp_model,
    criterion=nn.CrossEntropyLoss,
    max_epochs=2,
    batch_size=100,
    optimizer=torch.optim.Adam,
    lr=learning_rate / 10,
    iterator_train__shuffle=True,
    train_split=predefined_split(test_dataset),
    device=device,
)

# %%
mlp_classifier_adam.fit(train_dataset, None)

# %%
mlp_classifier_adam.partial_fit(train_dataset, None)

# %%
y_pred_mlp_adam = mlp_classifier_adam.predict(test_dataset)

# %%
print(classification_report(y_test, y_pred_mlp_adam))

# %% [markdown]
# ## Data Augmentation

# %%
augmented_transforms = transforms.Compose(
    [
        transforms.Resize((56, 56)),
        transforms.RandomResizedCrop(
            28, (0.6, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=30.0,
                    translate=(0.1, 0.1),
                    interpolation=InterpolationMode.BICUBIC,
                )
            ],
            0.8,
        ),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ]
)

# %%
augmented_train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=augmented_transforms, download=True
)

# %%
show_dataset(train_dataset, nrows=4, figsize=(12, 7))
show_dataset(augmented_train_dataset, nrows=4, figsize=(12, 7))

# %%
mlp_classifier.fit(augmented_train_dataset, None)

# %%
mlp_classifier.partial_fit(augmented_train_dataset, None)

# %% [markdown]
# ## Workshop Fashion MNIST
#
# Trainieren Sie ein Neuronales Netz, das Bilder aus dem Fashion MNIST Datenset
# klassifizieren kann.
#
# Ein Torch `Dataset` für Fashion MNIST kann mit der Klasse
# `torchvision.datasets.FashionMNIST` erzeugt werden.

# %%
