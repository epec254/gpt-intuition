# %%

import json
import numpy as np
import pandas as pd

# Opening JSON file
f = open("index_simplevector.json")

# returns JSON object as
# a dictionary
data = json.load(f)

dimensions = 1536
embedding_headers = ["index"]
for i in range(1, dimensions + 1):
    embedding_headers.append(f"embedding_{i}")


outputs = pd.DataFrame(columns=embedding_headers)
stuffs = []

for key in data["vector_store"]["simple_vector_store_data_dict"][
    "embedding_dict"
].keys():
    print(key)

    new_arr = [key]

    embedding = data["vector_store"]["simple_vector_store_data_dict"]["embedding_dict"][
        key
    ]

    # new_arr.append(embedding)

    print(len(embedding))

    new = np.array(new_arr)
    temp = pd.DataFrame(new.reshape(1, -1))

    new2 = np.array(embedding)
    temp2 = pd.DataFrame(new2.reshape(1, -1))

    print(temp)
    print(temp2)

    df = pd.concat([temp, temp2], axis=1)

    print(df)

    df.reset_index()

    stuffs.append(df)

    # outputs = pd.concat([df, outputs], axis=0)

print(stuffs)

outputs = pd.concat(stuffs)

outputs.set_axis(embedding_headers, axis=1, inplace=True)

print(outputs)

outputs.to_csv("paul_graham.csv", header=True, index=False)
