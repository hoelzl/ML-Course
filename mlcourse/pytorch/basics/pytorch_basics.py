# %%
import torch
import numpy as np
from torch import dtype, Tensor

from mlcourse.utils.general import pprint_indent
from mlcourse.utils.pytorch import describe

# %%
describe(torch.Tensor(2, 3))

# %%
describe(torch.rand(2, 3))

# %%
describe(torch.randn(2, 3))

# %%
describe(torch.randint(5, (2, 3)))

# %%
describe(torch.zeros(2, 3))

# %%
x = torch.ones(2, 3)
describe(x)

# %%
x.fill_(5)
describe(x)

# %%
x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
describe(x)

# %%
_npy = np.random.rand(2, 3)
describe(_npy)

# %%
x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
describe(x)
describe(x.long())

# %%
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
describe(x)
describe(x.float())

# %%
x = torch.randn(2, 3)
describe(x)

# %%
describe(torch.add(x, x))
describe(x + x)

# %%
x = torch.arange(6)
describe(x)

# %%
y = x.view(2, 3)
describe(x.reshape(2, 3))
describe(y)

# %%
describe(torch.sum(y, dim=0))
describe(torch.sum(y, dim=1))

# %%
describe(y)
describe(torch.transpose(y, 0, 1))
describe(y.transpose(0, 1))

# %%
x = torch.Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [0, 1, 2]]])
describe(x)

# %%
describe(torch.transpose(x, 0, 1))
describe(torch.transpose(x, 0, 2))

# %%
x = torch.arange(6).view(2, 3)
describe(x)
describe(x[:1, :2])

# %%
describe(x[0, 1])
print(x[0, 1].item())

# %%
indices = torch.tensor([0, 2], dtype=torch.int64)
describe(torch.index_select(x, dim=1, index=indices))

# %%
indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
describe(torch.index_select(x, dim=0, index=indices))
describe(x[indices])

# %%
indices = torch.tensor([0, 1, 0, 2], dtype=torch.int64)
describe(torch.index_select(x, dim=1, index=indices))
describe(x[:, indices])

# %%
_row_indices = torch.arange(2).long()
_col_indices = torch.LongTensor([0, 1])
describe(x[_row_indices, _col_indices])

# %%
_row_indices = torch.LongTensor([[0, 0, 0], [1, 1, 1]])
_col_indices = torch.LongTensor([0, 1, 2])
describe(x)
describe(x[_row_indices, _col_indices])

# %%
_row_indices = torch.LongTensor([0, 1])
_col_indices = torch.LongTensor([[0, 0], [1, 1], [2, 2]])
describe(x)
describe(x[_row_indices, _col_indices])

# %%
x = torch.arange(6).view(2, 3)
describe(x)

# %%
describe(torch.cat([x, x], dim=0))

# %%
describe(torch.cat([x, x], dim=1))

# %%
describe(torch.stack([x, x]))

# %%
x = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
describe(x)

# %%
y = torch.unsqueeze(x, 0)
describe(y)

# %%
z = torch.squeeze(y, 0)
describe(z)

# %%
x = torch.arange(6.0).view(2, 3)
describe(x)
y = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
describe(y)

# %%
describe(torch.mm(x, y))
describe(x @ y)

# %%
describe(torch.mm(y, x))
describe(y @ x)


# %%
# Gradients
x1 = torch.tensor([[0.0, 1.0], [2.0, 3.0]], requires_grad=True)
x2 = torch.tensor([[2.0, 1.0], [3.0, 1]], requires_grad=True)
describe(x1)
describe(x2)

# %%
y = (x1 + 2) * (x2 + 5) + 3
describe(y)

# %%
z = y.mean()
describe(z)

# %%
z.backward()

# %%
print("Gradient of x1: ", end="")
pprint_indent(x1.grad, indent=len("Gradient of x1: "))
print("Gradient of x2: ", end="")
pprint_indent(x2.grad, indent=len("Gradient of x2: "))

# print('Gradient of y:  ', end='')
# pprint_indent(y.grad, indent=len('Gradient of y:  '))

# %%
# CUDA
print(f"Is CUDA available? {torch.cuda.is_available()}.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device is '{device}'.")

# %%
x = torch.rand(3, 3).to(device)
describe(x)

# %%