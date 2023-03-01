# Embedding Projector for Streamlit

## Context

Embeddings are the core of modern machine learning (including LLMs and other large transformer based models).  I believe a critical step in using LLMs (or any other model for that instance) is developing intuition around how the embedding spaces represent your data.

While OpenAI provides a [Notebook for visualizing embeddings](https://github.com/openai/openai-python/blob/main/examples/embeddings/Visualize_in_3d.ipynb), I believe more robust tools are needed.  [Tensorflow's Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin) is an underrated tool that fills this gap, but can be painful to use (need to export embeddings into a specific set of 2 files, must be loaded as a standalone application, etc).

## Roadmap

WIP

## Setup
I use `pipenv` for management of pip packages.  You can either do `pipenv install` or `pip install -r requirements.txt`.

`streamlit run embed.py` launches the web application.
