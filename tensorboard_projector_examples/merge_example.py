# %%

import pandas as pd

labels = pd.read_csv("word2vec_10000_200d_labels.tsv", sep="\t", header=0)

labels.head()

embedding_headers = []

for i in range(1, 201):
    embedding_headers.append(f"embedding_{i}")

print(embedding_headers)

embeddings = pd.read_csv(
    "word2vec_10000_200d_tensors.csv", header=None, names=embedding_headers
)

embeddings.head()

merged = pd.concat([labels, embeddings], axis=1)

merged.head()

merged.to_csv("word2vec_10000_200d_merged.csv", header=True, index=False)
