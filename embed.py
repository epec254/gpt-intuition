import streamlit as st
import pandas as pd
import numpy as np
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go

# sample data - need to replace with embeddings
df = px.data.iris()

features = df.loc[:, :"petal_width"]

umap_2d = UMAP(n_components=2, init="random", random_state=0)
umap_3d = UMAP(n_components=3, init="random", random_state=0)

proj_2d = umap_2d.fit_transform(features)
proj_3d = umap_3d.fit_transform(features)

fig_2d = px.scatter(proj_2d, x=0, y=1, color=df.species, labels={"color": "species"})
fig_3d = px.scatter_3d(
    proj_3d,
    x=0,
    y=1,
    z=2,
    color=df.species,
    labels={"color": "species"},
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


# ## animatation

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

# publish to Streamlit

st.plotly_chart(fig_3d, theme="streamlit", use_container_width=True)


st.plotly_chart(fig_2d, theme=None, use_container_width=True)
