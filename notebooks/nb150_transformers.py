# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python [conda env:ml-training] *
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Encoder / Decoder
#
# <img src="img/encoder-decoder.png" style="width: 95%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# # Transformer

# %% [markdown]
# <img src="img/transformer.png" style="width: 45%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Multi-Head Attention
#
# <img src="img/multi-head-attention.png" style="width: 35%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Scaled Dot-Product Attention
#
# <img src="img/scaled-dot-prod-attention.png" style="width: 30%; margin-left: auto; margin-right: auto;"/>

# %%
# !nvidia-smi

# %%
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# %%
pretrained_weights = 'gpt2'
tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_weights)
model = GPT2LMHeadModel.from_pretrained(pretrained_weights)

# %%
ids = tokenizer.encode('This is an example of text, and')
ids

# %%
tokenizer.decode(ids)

# %%
import torch

# %%
t = torch.LongTensor(ids).unsqueeze(0)
preds = model.generate(t)

# %%
preds.shape, preds[0]

# %%
tokenizer.decode(preds[0].numpy())

# %%
preds[:, -10:]

# %%
preds2 = model.generate(preds[:, -10:])

# %%
torch.cat((preds[0, :-10], preds2[0]))

# %%
tokenizer.decode(torch.cat((preds[0, :-10], preds2[0])).numpy())

# %%
from fastai.text.all import *

# %%
path = untar_data(URLs.WIKITEXT_TINY)
path.ls()

# %%
df_train = pd.read_csv(path/'train.csv', header=None)
df_valid = pd.read_csv(path/'test.csv', header=None)
df_train.head()

# %%
all_texts = np.concatenate([df_train[0].values, df_valid[0].values])


# %%
class TransformersTokenizer(Transform):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def encodes(self, x): 
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
    def decodes(self, x):
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


# %%
splits = [range_of(df_train), list(range(len(df_train), len(all_texts)))]
tfmd_lists = TfmdLists(all_texts, TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)

# %%
tfmd_lists.train[0],tfmd_lists.valid[0]

# %%
tfmd_lists.tfms(tfmd_lists.train.items[0]).shape, tfmd_lists.tfms(tfmd_lists.valid.items[0]).shape

# %%
batch_size, seq_len = 6, 1024
dls = tfmd_lists.dataloaders(bs=batch_size, seq_len=seq_len)

# %%
dls.show_batch(max_n=2)


# %%
def tokenize(text):
    toks = tokenizer.tokenize(text)
    return tensor(tokenizer.convert_tokens_to_ids(toks))

tokenized = [tokenize(t) for t in progress_bar(all_texts)]


# %%
class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x): 
        return x if isinstance(x, Tensor) else tokenize(x)
    def decodes(self, x):
        return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


# %%
tfmd_lists = TfmdLists(tokenized, TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
dls = tfmd_lists.dataloaders(bs=batch_size, seq_len=seq_len)

# %%
dls.show_batch(max_n=2)


# %%
class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.learn.pred[0]


# %%
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()

# %%
learn.validate()

# %%
learn.lr_find()

# %%
learn.fit_one_cycle(1, 1e-4)

# %%
df_valid.head(1)

# %%
prompt = "\n = Unicorn = \n \n A unicorn is a magical creature with a rainbow tail and a horn"

# %%
prompt_ids = tokenizer.encode(prompt)
inp = tensor(prompt_ids)[None].cuda()
inp.shape

# %%
preds = learn.model.generate(inp, max_length=40, num_beams=5, temperature=1.5)

# %%
preds[0]

# %%
tokenizer.decode(preds[0].cpu().numpy())

# %%
