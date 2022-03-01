# %%
from mlcourse.utils.general import pprint_indent
import torch

# %%
def describe(x):
    if isinstance(x, torch.Tensor):
        x_type = x.type()
    else:
        x_type = type(x)
    print(f'Type:       {x_type}')
    print(f'Shape/size: {x.shape}')
    print('Values:     ', end='')
    pprint_indent(x, indent=len('Values:     '))

