# %%
import os

os.environ["OPENAI_API_KEY"] = "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"

from langchain.embeddings import OpenAIEmbeddings
import pandas as pd
import numpy as np

embeddings = OpenAIEmbeddings()

# doc_result = embeddings.embed_documents(texts)

# %%


from langchain.text_splitter import RecursiveCharacterTextSplitter


# This is a long document we can split up.
with open("./data/paul_graham_essay.txt") as f:
    paul = f.read()

# print(paul)


text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

texts = text_splitter.create_documents([paul])

print(len(texts))

all_the_embeddings = []

for text in texts:
    # print(text.page_content)

    query_result = embeddings.embed_query(text.page_content)

    print("---")

    # print(len(query_result))

    text_arr = [text.page_content, "index"]
    text_np = np.array(text_arr)
    text_df = pd.DataFrame(text_np.reshape(1, -1))

    embed_np = np.array(query_result)
    embed_df = pd.DataFrame(embed_np.reshape(1, -1))

    merged_df = pd.concat([text_df, embed_df], axis=1)

    all_the_embeddings.append(merged_df)

    # break

queries = [
    "What did the author do growing up?",
    "What did the author do after his time at Y Combinator?",
    "Who won the summer olympics in 2008?",
]

for query in queries:
    query_result = embeddings.embed_query(query)

    print("---")

    # print(len(query_result))

    text_arr = [query, "query"]
    text_np = np.array(text_arr)
    text_df = pd.DataFrame(text_np.reshape(1, -1))

    embed_np = np.array(query_result)
    embed_df = pd.DataFrame(embed_np.reshape(1, -1))

    merged_df = pd.concat([text_df, embed_df], axis=1)

    all_the_embeddings.append(merged_df)

dimensions = 1536
embedding_headers = ["text_chunk", "index"]
for i in range(1, dimensions + 1):
    embedding_headers.append(f"embedding_{i}")

outputs = pd.concat(all_the_embeddings)

outputs.set_axis(embedding_headers, axis=1, inplace=True)

print(outputs)

outputs.to_csv("paul_graham_splits_1000_200overlap.csv", header=True, index=False)


# %%
