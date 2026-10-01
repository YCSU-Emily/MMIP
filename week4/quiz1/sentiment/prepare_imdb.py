import os
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading IMDb dataset...")

dataset = load_dataset("imdb")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# 0 = negative, 1 = positive
train_df = train_df[["text", "label"]]
test_df = test_df[["text", "label"]]

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False,
    encoding="utf-8"
)

print("\nDataset preparation completed.")
print(f"Train samples: {len(train_df)}")
print(f"Test samples : {len(test_df)}")

print("\nTrain label distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nExample:")
print(train_df.iloc[0])
