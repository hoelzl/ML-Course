# %%
import numpy as np
import pandas as pd

# %%
pd.Series(data=[10, 20, 30, 40])

# %%
pd.Series(["a", "b", "c"])

# %%
pd.Series(data=[1, 2, 3, 4], index=["w", "x", "y", "z"])

# %%
arr = np.arange(5)
indices = "a b c d e".split()
arr, indices

# %%
pd.Series(data=arr)

# %%
pd.Series(data=arr, index=indices)

# %%
rng = np.random.default_rng(42)
data_vec = rng.normal(size=1000)
data = pd.Series(data=data_vec)

# %%
data

# %%
data.head()

# %%
data.head(2)

# %%
type(data.head())

# %%
data.tail()

# %%
pd.Series({"Ice Cream": 2.49, "Cake": 4.99, "Fudge": 7.99})

# %%
food1 = pd.Series({"Ice Cream": 2.49, "Cake": 4.99, "Fudge": 7.99})
food2 = pd.Series({"Cake": 4.99, "Ice Cream": 3.99, "Pie": 3.49, "Cheese": 1.99})

# %%
food1

# %%
food1.index

# %%
food1.size

# %%
food1.sum()

# %%
food1.mean()

# %%
food1.name

# %%
food1.name = "Deserts"

# %%
food1

# %%
food1.name

# %%
food1.plot.bar(legend=True)

# %%
data.plot.hist()

# %%
food1

# %%
food1["Cake"]

# %%
food1[0]

# %%
food1.loc["Cake"]

# %%
food1.iloc[0]

# %%
food1.argmin()

# %%
# food1["Pie"]

# %%
confusing = pd.Series(data=np.linspace(0, 5, 11), index=np.arange(-5, 6))
confusing

# %%
confusing[0]

# %%
confusing.loc[0]

# %%
confusing.iloc[0]

# %%
confusing.loc[[True, False] * 5 + [True]]

# %%
food1

# %%
food2

# %%
food_sum = food1 + food2
food_sum

# %%
food1 + 0.5

# %%
food1


# %%
def discount(price):
    return price * 0.9


# %%
food1.apply(discount)

# %%
food1

# %%
all_food = food1.append(food2)
all_food

# %%
all_food["Cake"]

# %%
all_food["Pie"]

# %%
all_food.index

# %%
all_food.is_unique

# %%
food1.is_unique

# %%
all_food.groupby(all_food.index).mean()

# %%
all_food.index.is_monotonic_increasing

# %%
all_food.index

# %%
sorted_food = all_food.sort_index()

# %%
sorted_food

# %%
all_food.sort_values()


# %%
all_food[["Pie", "Cake"]]

# %%
all_food[["Pie"]]

# %%
all_food[1:3]

# %%
all_food.iloc[1:3]

# %%
sorted_food["Cake":"Fudge"]

# %%
food = food1 + food2
food

# %%
food.isna()

# %%
food.isna().sum()

# %%
food.dropna()

# %%
food

# %%
food.dropna(inplace=True)

# %%
food

# %%
food = food1 + food2
sum_food = food

# %%
food = food.dropna()

# %%
food

# %%
sum_food

# %%
food = food1 + food2
sum_food = food

# %%
sum_food is food  # noqa

# %%
food.dropna(inplace=True)

# %%
food

# %%
sum_food

# %%
from mlcourse.config import Config  # noqa

# %%
config = Config()
raw_dir_path = config.data_dir_path / "raw"
raw_dir_path


# %%
def create_data_frame():
    rng = np.random.default_rng(42)
    array = rng.normal(size=(5, 4), scale=5.0)
    index = "A B C D E".split()
    columns = "w x y z".split()
    return pd.DataFrame(data=array, index=index, columns=columns)


# %%
create_data_frame()

# %%
pd.DataFrame([[2, 3, 4], [5, 6, 7]])

# %%
df = create_data_frame()

# %%
type(df)

# %%
df_csv = pd.read_csv(raw_dir_path / "example_data.csv")

# %%
df_csv

# %%
df_csv = pd.read_csv(raw_dir_path / "example_data.csv", index_col=0)
df_csv

# %%
df_exel = pd.read_excel(raw_dir_path / "excel_data.xlsx", index_col=0)
df_exel

# %%
df_exel2 = pd.read_excel(
    raw_dir_path / "excel_other_sheet.xlsx",
    index_col=0,
    sheet_name="Another Sheet",
    skiprows=[1],
)
df_exel2

# %%
df = create_data_frame()
df

# %%
df["w"]

# %%
type(df["w"])

# %%
df[["w", "y"]]

# %%
df[["w"]]

# %%
df.index

# %%
df.size

# %%
df.ndim

# %%
df.shape

# %%
df = create_data_frame()

# %%
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
df = create_data_frame()

# %%
df

# %%
df["w"]

# %%
df.loc["A"]


# %%
df.iloc[0]

# %%
df.loc["B", "z"]

# %%
df.loc[["A", "C"], ["x", "y"]]

# %%
df > 0  # noqa

# %%
df[df > 0]

# %%
