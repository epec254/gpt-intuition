import streamlit as st
import pandas as pd
import numpy as np
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE


### ADD A NEW DATASET HERE
### The expected format is a CSV file with any number of columns.  The embedding columns MUST be named "embedding_1", "embedding_2", etc.
### Color = the column name to use for coloring the points, should be a smaller number of unique values
### Label = the column name to use for labeling the points, will be displayed on hover
DATA_SETS = {
    "iris": {
        "data_file": "tensorboard_projector_examples/iris_labels_merged.csv",
        "dimensions": 4,
        "label_column": "class",
        "color_column": "class",
    },
    "word2vec-10k-sample": {
        "data_file": "tensorboard_projector_examples/word2vec_10000_200d_merged.csv",
        "dimensions": 200,
        "label_column": "word",
        "color_column": "",
    },
}


st.set_page_config(
    page_title="Embedding Visualizer",
    page_icon="🎈",
    layout="wide",
)

st.title("📊 Embedding Visualization App")
st.header("")

# ## animatation - THIS IS NOT TESTED - IGNORE

# an attempt to replicate the animation of Tensorflow Embedding Projector
# x_eye = -1.25
# y_eye = 2
# z_eye = 0.5

# ## animate
# # x_eye = -1.25
# # y_eye = 4
# # z_eye = 4

# fig_3d.update_layout(
#     title="Animation Test",
#     width=600,
#     height=600,
#     scene_camera_eye=dict(x=x_eye, y=y_eye, z=z_eye),
#     updatemenus=[
#         dict(
#             type="buttons",
#             showactive=False,
#             y=1,
#             x=0.8,
#             xanchor="left",
#             yanchor="bottom",
#             pad=dict(t=45, r=10),
#             buttons=[
#                 dict(
#                     label="Play",
#                     method="animate",
#                     args=[
#                         None,
#                         dict(
#                             frame=dict(duration=5, redraw=True),
#                             transition=dict(duration=0),
#                             fromcurrent=True,
#                             mode="immediate",
#                         ),
#                     ],
#                 )
#             ],
#         )
#     ],
# )


# def rotate_z(x, y, z, theta):
#     w = x + 1j * y
#     return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z


# frames = []
# for t in np.arange(0, 6.26, 0.01):
#     xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
#     frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
# fig_3d.frames = frames


# ## end animation

# Compute the embeddings visualization - cache for performance


@st.cache_data
def compute_embeddings(projector_type, algorithm_params, data_set):
    # sample data - need to replace with embeddings
    # df = px.data.iris()

    dimensions = DATA_SETS[data_set]["dimensions"]
    data_file = DATA_SETS[data_set]["data_file"]

    df = pd.read_csv(data_file, header=0)

    embedding_headers = []
    for i in range(1, dimensions + 1):
        embedding_headers.append(f"embedding_{i}")

    # features = df.loc[:, "embedding_1":"embedding_200"]

    features = df[embedding_headers]

    if projector_type == ALG_UMAP:
        umap_2d = UMAP(
            n_components=2,
            init="random",
            random_state=algorithm_params["random_state"],
            n_neighbors=algorithm_params["n_neighbors"],
        )
        umap_3d = UMAP(
            n_components=3,
            init="random",
            random_state=algorithm_params["random_state"],
            n_neighbors=algorithm_params["n_neighbors"],
        )

        proj_2d = umap_2d.fit_transform(features)
        proj_3d = umap_3d.fit_transform(features)
    elif projector_type == ALG_TSNE:
        tsne = TSNE(
            n_components=3,
            perplexity=algorithm_params["perplexity"],
            random_state=algorithm_params["random_state"],
            n_iter=algorithm_params["n_iter"],
            learning_rate=algorithm_params["learning_rate"],
            n_iter_without_progress=algorithm_params["n_iter_without_progress"],
        )
        proj_3d = tsne.fit_transform(
            features,
        )
        tsne = TSNE(
            n_components=2,
            perplexity=algorithm_params["perplexity"],
            random_state=algorithm_params["random_state"],
            n_iter=algorithm_params["n_iter"],
            learning_rate=algorithm_params["learning_rate"],
            n_iter_without_progress=algorithm_params["n_iter_without_progress"],
        )
        proj_2d = tsne.fit_transform(
            features,
        )

    # add back in the labels
    proj_3d = pd.concat([df, pd.DataFrame(proj_3d)], axis=1)
    proj_2d = pd.concat([df, pd.DataFrame(proj_2d)], axis=1)

    return df, proj_2d, proj_3d


# publish to Streamlit with controls

controls, graphs = st.columns([1, 4])

ALG_TSNE = "t-SNE"
ALG_UMAP = "UMAP"

with controls:

    CurrentDataSet = st.selectbox("Dataset", options=DATA_SETS.keys())

    # TODO: Add error checking on this
    LabelColumn = st.text_input(
        "Label Column", DATA_SETS[CurrentDataSet]["label_column"]
    )

    # TODO: Add error checking on this
    ColorColumn = st.text_input(
        "Color Column", DATA_SETS[CurrentDataSet]["color_column"]
    )

    AlgorithmType = st.radio("Algorithm", (ALG_UMAP, ALG_TSNE))

    if AlgorithmType == ALG_TSNE:
        st.markdown(
            "t-SNE [configuration](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)"
        )

        Perplexity = st.slider("Perplexity", 1, 100, 30)

        RandomState = st.slider("RandomState", 1, 100, 25)

        LearningRate = st.select_slider(
            "Learning Rate", options=["auto", 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]
        )

        MaxIterations = st.slider("Max Iterations", 1, 1500, 1000)

        IterationsWithoutProgress = st.slider(
            "Iterations Without Progress", 1, 1500, 300
        )

        algorithm_params = {
            "perplexity": Perplexity,
            "random_state": RandomState,
            "learning_rate": LearningRate,
            "n_iter": MaxIterations,
            "n_iter_without_progress": IterationsWithoutProgress,
        }
    elif AlgorithmType == ALG_UMAP:
        st.markdown(
            "UMAP [configuration](https://umap-learn.readthedocs.io/en/latest/parameters.html)"
        )
        RandomState = st.slider("RandomState", 0, 100, 2)

        NumNeighbors = st.slider("Number Neighbors", 5, 50, 10)

        algorithm_params = {
            "random_state": RandomState,
            "n_neighbors": NumNeighbors,
        }

    embedding_computation_state = st.text("Computing embeddings...")

    df, proj_2d, proj_3d = compute_embeddings(
        AlgorithmType, algorithm_params, CurrentDataSet
    )

    embedding_computation_state.text("Embeddings computed!")


color_input = None if ColorColumn == "" else proj_3d[ColorColumn]

fig_2d = px.scatter(
    proj_2d,
    x=0,
    y=1,
    hover_data=[LabelColumn],
    color=color_input,
    # labels={"color": "species"}
)
fig_3d = px.scatter_3d(
    proj_3d,
    x=0,
    y=1,
    z=2,
    color=color_input,
    # labels={"color": LabelColumn},
    hover_data=[LabelColumn],
    size_max=18,
    opacity=0.7,
    # range_x=[0, 20],
    # range_y=[0, 20],
    # range_z=[0, 20],
    # log_x=True,
    # log_y=True,
    # log_z=True,
)
fig_3d.update_traces(marker_size=5)

fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))

# name = "defaul222t"
# # Default parameters which are used when `layout.scene.camera` is not provided
camera = dict(
    up=dict(x=1, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0, z=1.25)
)

fig_3d.update_layout(scene_camera=camera)

with graphs:
    st.plotly_chart(fig_3d, theme="streamlit", use_container_width=True)
    st.plotly_chart(fig_2d, theme=None, use_container_width=True)
