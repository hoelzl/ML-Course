# %%
# from nbex.interactive import session, display_interactive
import pandas as pd
import seaborn as sns
from IPython.display import display

from mlcourse.config import Config

# %%
config = Config()

# %%
sns.set_theme()

# %%
csv_path = config.data_dir_path / "generated"

# %%
linear_salaries = pd.read_csv(csv_path / "linear_salaries.csv")

# %%
display(linear_salaries)

# %%
linear_salaries = pd.read_csv(csv_path / "linear_salaries.csv", index_col=0)

# %%
print(type(linear_salaries))
display(linear_salaries)

# %%
print("Length =", len(linear_salaries))
print("Columns:", linear_salaries.columns)

# %%
print("Index:")
print(linear_salaries.index)

# %%
display(linear_salaries.head(3))

# %%
display(linear_salaries.tail(3))

# %%
display(linear_salaries[:3])

# %%
display(linear_salaries[-3:])

# %%
display(linear_salaries[1])

# %%
print(type(linear_salaries["age"]))
display(linear_salaries["age"])
display(linear_salaries["age"].index)

# %%
print(type(linear_salaries.iloc[1]))
display(linear_salaries.iloc[1])
display(linear_salaries.iloc[1].index)

# %%
display(linear_salaries.iloc[:3])
display(linear_salaries.iloc[-3:])

# %%
print(type(linear_salaries[["age"]]))
display(linear_salaries[["age"]])
display(linear_salaries[["age"]].index)
display(linear_salaries[["age"]].columns)

# %%
print(type(linear_salaries[["salary", "age"]]))
display(linear_salaries[["salary", "age"]])
display(linear_salaries[["salary", "age"]].index)
display(linear_salaries[["salary", "age"]].columns)


# %%
