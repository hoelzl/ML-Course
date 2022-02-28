# %% [markdown]
# # Data Frames
#
# Data frames are the most commonly used Pandas data structure. They allow us to
# easily read, process and save table-based data
#
# Conceptually, a data frame consists of multiple series instances sharing a
# common index.

# %%
import numpy as np
import pandas as pd

from mlcourse.config import Config

# %%
config = Config()
raw_dir_path = config.data_dir_path / "raw"
raw_dir_path.absolute()

# %% [markdown]
# ## Creating Data Frames


# %% [markdown]
# ### From a NumPy Array

# %%
def create_data_frame():
    rng = np.random.default_rng(42)
    array = rng.normal(size=(5, 4), scale=5.0)
    index = "A B C D E".split()
    columns = "w x y z".split()
    return pd.DataFrame(array, index=index, columns=columns)


# %%
df = create_data_frame()
df


# %%
type(df)

# %% [markdown]
# ### From a CSV File

# %%
df_csv = pd.read_csv(raw_dir_path / "example_data.csv")


# %%
df_csv


# %%
df_csv = pd.read_csv(raw_dir_path / "example_data.csv", index_col=0)


# %%
df_csv

# %% [markdown]
# ### From an Excel File

# %%
df_excel = pd.read_excel(raw_dir_path / "excel_data.xlsx", index_col=0)


# %%
df_excel


# %%
df_excel2 = pd.read_excel(raw_dir_path / "excel_other_sheet.xlsx", index_col=0)


# %%
df_excel2


# %%
df_excel2 = pd.read_excel(
    raw_dir_path / "excel_other_sheet.xlsx",
    index_col=0,
    sheet_name="Another Sheet",
    # header=0,
    # skiprows=[1]
)


# %%
df_excel2.head()

# %% [markdown]
#
# ### Other Formats:
#
# - `pd.read_clipboard`
# - `pd.read_html`
# - `pd.read_json`
# - `pd.read_pickle`
# - `pd.read_sql` (uses SQLAlchemy to access a database)
# - ...

# %% [markdown]
# ### Indices and Operations

# %%
df_csv.head()


# %%
df_csv.tail()


# %%
df = create_data_frame()
df["w"]


# %%
type(df["w"])


# %%
# Use only interactively
df.w


# %%
df[["w", "y"]]


# %%
df.index


# %%
df.index.is_monotonic_increasing


# %%
df.size


# %%
df.ndim


# %%
df.shape

# %% [markdown]
# ### Creating, renaming, deleting columns

# %%
df = create_data_frame()
df["Sum of w and y"] = df["w"] + df["y"]


# %%
df


# %%
df.rename(columns={"Sum of w and y": "w + y"})


# %%
df


# %%
df.rename(columns={"Sum of w and y": "w + y"}, index={"E": "Z"}, inplace=True)


# %%
df


# %%
type(df["y"])


# %%
del df["y"]


# %%
df


# %%
df.drop("A")


# %%
df


# %%
df.drop("B", inplace=True)


# %%
df


# %%
df.drop("z", axis=1)


# %%
df


# %%
df.drop("z", axis=1, inplace=True)


# %%
df

# %% [markdown]
# ## Selection

# %%
df = create_data_frame()
df


# %%
df["w"]


# %%
# Error!
# df['A']


# %%
df.loc["B"]


# %%
type(df.loc["B"])


# %%
df


# %%
df.iloc[1]


# %%
df.loc[["A", "C"]]


# %%
df.loc[["A", "C"], ["x", "y"]]


# %%
df.loc["B", "z"]


# %%
df.iloc[[1, 2], [0, 3]]


# %%
df.iloc[0, 0]

# %% [markdown]
# ## Conditional Selection

# %%
df = create_data_frame()
df


# %%
df > 0  # noqa


# %%
df[df > 0]


# %%
df["w"] > 0  # noqa


# %%
df[df["w"] > 0]


# %%
df[df["w"] > 0][["x", "y"]]


# %%
df[(df["w"] > 0) & (df["x"] < 0)]

# %% [markdown]
#
# # Information about Data Frames

# %%
df = create_data_frame()
df["txt"] = "a b c d e".split()
df.iloc[1, 1] = np.nan
df


# %%
df.describe()


# %%
df.info()


# %%
df.dtypes

# %% [markdown]
# ## Data Frame Index

# %%
df = create_data_frame()
df["txt"] = "a b c d e".split()
df


# %%
df.reset_index()


# %%
df


# %%
df.reset_index(inplace=True)


# %%
df


# %%
df.rename(columns={"index": "old_index"}, inplace=True)


# %%
df


# %%
df.set_index("txt")


# %%
df


# %%
df.set_index("txt", inplace=True)
df


# %%
df.set_index("old_index", inplace=True)
df


# %%
df.info()


# %%
df.index


# %%
df.index.name = None


# %%
df


# %%
