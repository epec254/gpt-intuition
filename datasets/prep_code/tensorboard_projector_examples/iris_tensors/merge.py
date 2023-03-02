# %%

import pandas as pd

labels = pd.read_csv("iris_labels.tsv", sep="\t", header=0)

labels.head()

embedding_headers = []

for i in range(1, 5):
    embedding_headers.append(f"embedding_{i}")

print(embedding_headers)

embeddings = pd.read_csv("iris_tensors.csv", header=None, names=embedding_headers)

embeddings.head()

embeddings = embeddings.reset_index()

merged = pd.concat([labels, embeddings], axis=1)

merged.head()

print(merged)

merged.to_csv("iris_labels_merged.csv", header=True, index=False)
