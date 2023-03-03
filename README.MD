# Embedding Projector for Streamlit

## Context

Embeddings are the core of modern machine learning (including LLMs and other large transformer based models).  I believe a critical step in using LLMs (or any other model for that instance) is developing intuition around how the embedding spaces represent your data.

While OpenAI provides a [Notebook for visualizing embeddings](https://github.com/openai/openai-python/blob/main/examples/embeddings/Visualize_in_3d.ipynb), I believe more robust tools are needed.  [Tensorflow's Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin) is an underrated tool that fills this gap, but can be painful to use (need to export embeddings into a specific set of 2 files, must be loaded as a standalone application, etc).

This is a "good enough" version of a visualization tool based on Streamlit and open source uMAP and tSNE algorithms.  The tool's primary goal is to make it easy for you to load your own embeddings *quickly.*

## Working demo version

Courtesy of Streamlit cloud, here is a working demo: [https://gpt-intuition-embedding-visualizer.streamlit.app/](https://gpt-intuition-embedding-visualizer.streamlit.app/)

## Usage
I use `pipenv` for management of pip packages & virtualenvs.  You can either do `pipenv install` or `pip install -r requirements.txt`.  If you use pip directly, I suggest doing so inside a virtual environment (`pipenv` handles this for you).

Run `streamlit run embed.py` to launch the Streamlit web application.

## Configuration

The app comes configured with 4 sample sets of embeddings, 2 shamelessly stolen from Tensorflow and 2 generated with `llama-index` and `langchain` from the Paul Graham essay.  

Adding your own is easy:

1. Create a CSV file in the `/data/` directory with the following columns:
    1. **Required** a set of columns "embedding_1", "embedding_2", ... that represent the embedding itself
    2. **Optional** label or other metadata column (e.g., name that each the embedding represents, etc)
    3. **Optional** label to use for coloring the points on the graph (e.g., a low dimensional bucket-esque label)
2. Add a dict in the same format as the one below to `DATA_SETS` in `embed.py`.  Dimensions is required to properly load the embeddings.
```
DATA_SETS = {
    ...
    "word2vec-10k-sample": {
        "data_file": "datasets/word2vec_10000_200d_merged.csv", # required
        "dimensions": 200, # required
        "label_column": "word", # optional
        "color_column": "", # optional
    },
}
```

## Roadmap
- [ ] Upload CSV file and visualize
- [ ] Add a way to select a single embedding and see the closet KNN within the space
- [ ] Add in support for generating OpenAI-based embeddings directly from the UX
- [ ] Automatically load any CSV file put into `/data/`
- [ ] Add support for creating a [GPT-Index](https://gpt-index.readthedocs.io/) or LangChain's chains and visually seeing how it works
- [ ] Animations!!
- [ ] Testing - aka anything more than refresh and check ;-)


## Known issues

* Sometimes, the app exits with the error `Terminating: Nested parallel kernel launch detected, the workqueue threading layer does not supported nested parallelism. Try the TBB threading layer.` - I believe this is due to the uMAP algorithm's implemention of multi-threading conflicting with Streamlit, but have not debugged the issue in depth.  Typically, restarting it a few times works.

## Credits

Code for the uMAP and tSNE implementations with Plotly courtesy of the [Plotly docs](https://plotly.com/python/t-sne-and-umap-projections/).
