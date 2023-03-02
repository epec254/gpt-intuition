# Embedding Projector for Streamlit

## Context

Embeddings are the core of modern machine learning (including LLMs and other large transformer based models).  I believe a critical step in using LLMs (or any other model for that instance) is developing intuition around how the embedding spaces represent your data.

While OpenAI provides a [Notebook for visualizing embeddings](https://github.com/openai/openai-python/blob/main/examples/embeddings/Visualize_in_3d.ipynb), I believe more robust tools are needed.  [Tensorflow's Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin) is an underrated tool that fills this gap, but can be painful to use (need to export embeddings into a specific set of 2 files, must be loaded as a standalone application, etc).

This is a "good enough" version of a visualization tool based on Streamlit and open source uMAP and tSNE algorithms.

## Usage
I use `pipenv` for management of pip packages & virtualenvs.  You can either do `pipenv install` or `pip install -r requirements.txt`.  If you use pip directly, I suggest doing so inside a virtual environment (`pipenv` handles this for you).

Run `streamlit run embed.py` to launch the Streamlit web application.

## Configuration

The app comes configured with 2 sample sets of embeddings shamelessly stolen from Tensorflow.  To add your own:

1. Create a CSV file with
    * Optional label or other metadata column (e.g., name of the embedding, etc)
    * Optional label to use for coloring the points on the graph
    * Required a set of columns "embedding_1", "embedding_2", ... that represent the embedding itself
2. Add a dict similar to the below to `DATA_SETS` in `embed.py`.  Dimensions is required to properly load the embeddings.
```
DATA_SETS = {
    ...
    "word2vec-10k-sample": {
        "data_file": "datasets/word2vec_10000_200d_merged.csv",
        "dimensions": 200,
        "label_column": "word",
        "color_column": "",
    },
}
```

## Roadmap

- [ ] Add a way to select a single embedding and see the closet KNN within the space
- [ ] Add support for creating a [GPT-Index](https://gpt-index.readthedocs.io/) and visually seeing how it works
- [ ] Add in support for generating OpenAI-based embeddings directly
- [ ] Colors match between the graphs
- [ ] Animations!!
- [ ] Testing - aka anything more than refresh and check ;-)

## Known issues

* Sometimes, the app exits with the error `Terminating: Nested parallel kernel launch detected, the workqueue threading layer does not supported nested parallelism. Try the TBB threading layer.` - I believe this is due to the uMAP algorithm's implemention of multi-threading conflicting with Streamlit, but have not debugged the issue in depth.  Typically, restarting it a few times works.

## Credits

Code for the uMAP and tSNE implementations with Plotly courtesy of the [Plotly docs](https://plotly.com/python/t-sne-and-umap-projections/).