import pandas as pd
import os

images = sorted(os.listdir("data/images"))
labels = sorted(os.listdir("data/labels"))

data_df = pd.DataFrame(columns=["inputs", "outputs"])

data_df = pd.DataFrame({
    "inputs": [f"data/images/{img}" for img in images],
    "outputs": [f"data/labels/{lbl}" for lbl in labels]
})

data_df.to_csv("data/CSVs/dataset.csv", index=False)

print(data_df)