import pandas as pd
df = pd.read_csv("../Dataset/Traning/Final_70_30_Dataset.csv", nrows=10000)
print("Columns:", df.columns.tolist())
df_full = pd.read_csv("../Dataset/Traning/Final_70_30_Dataset.csv", usecols=["label"])
print("Unique Labels:", df_full["label"].unique().tolist())
print("Label Counts:\n", df_full["label"].value_counts())
