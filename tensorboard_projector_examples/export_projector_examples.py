# %%
# Google did something really annoying.  They stored the word2vec vectors in a binary format that represents a serialized JS Float32Array array of size [number of records* tensor size]
# why this isn't a 2d array or just a CSV file is beyond me - hopefully someone got promoted off this complexity that saves 30 seconds of load ;-)

# 1. Go to projector.tensorflow.org
# 2. Open the javascript console
# 3. Paste & run the following code

# let xhr = new XMLHttpRequest();

# tensorsPath = 'https://projector.tensorflow.org/oss_data/word2vec_10000_200d_tensors.bytes'

# tensorsPath = 'https://projector.tensorflow.org/oss_data/word2vec_full_200d_tensors.bytes'

# xhr.open('GET', tensorsPath);
# xhr.responseType = 'arraybuffer';

# xhr.onload = () => {

#   data = new Float32Array(xhr.response);
#   console.log(data)
# }

# xhr.send();

# 4. Copy the output `value1, value2, value3, ...` into a txt file

# %%

import numpy as np
import pandas as pd

x = [
    -0.14177100360393524,
    0.24957600235939026,
    -0.18858399987220764,
    -0.08152230083942413,
    0.12844200432300568,
    0.5471850037574768,
    -0.19736599922180176,
    0.14226900041103363,
    -0.438946008682251,
    -0.041615698486566544,
]

FILE = "word2vec_10000_200d_tensors"

# Load the raw set of tensor values from the txt file above
tensors_raw_file = open(f"{FILE}.txt", "r")

# convert into an array
tensors_raw_array = tensors_raw_file.read().split(",")

print(f"Number of lines: {len(tensors_raw_array)}")

# number of word2vec vectors
N = 10000
# size of each vector
dim = 200

offset = 0

# Store the results in a DF temporarily
df = pd.DataFrame()

# implementation of https://github.com/tensorflow/tensorboard/blob/4a10a311af00fac32b4759db6065784c2b181a76/tensorboard/plugins/projector/vz_projector/data-provider.ts#L262

for i in range(0, N):
    # print(i)
    new = np.array(tensors_raw_array[offset : offset + dim])

    temp = pd.DataFrame(new.reshape(1, -1))

    df = pd.concat([df, temp], axis=0)

    # print(df)

    offset += dim

# print(df)

df.to_csv(f"{FILE}.csv", header=False)
