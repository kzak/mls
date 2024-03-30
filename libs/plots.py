import numpy as np
import plotly.graph_objects as go


# ------------------------------
# plotly functions
# ------------------------------
def plotly_build_surface(surface_fn, **kwargs):
    # ------------------------------
    # Get parameters
    # ------------------------------
    r = kwargs.get("r", np.pi)
    colorscale = kwargs.get("colorscale", "Blues")
    opacity = kwargs.get("opacity", 0.5)

    # ------------------------------
    # Build the visualizing object
    # ------------------------------
    g = np.linspace(-r, r)
    xx1, xx2 = np.meshgrid(g, g)
    G = np.vstack([xx1.ravel(), xx2.ravel()]).T
    zz = surface_fn(G).reshape(len(g), len(g))

    return go.Surface(
        x=xx1,
        y=xx2,
        z=zz,
        colorscale=colorscale,
        showscale=False,
        opacity=opacity,
    )


def plotly_build_scatter3d(X, y, **kwargs):
    # ------------------------------
    # Get parameters
    # ------------------------------
    color = kwargs.get("color", "blue")

    # ------------------------------
    # Build the visualizing object
    # ------------------------------
    return go.Scatter3d(
        x=X[:, 0],
        y=X[:, 1],
        z=y,
        mode="markers",
        marker=dict(size=1, color=color),
    )


def plotly_true_and_prediction_surface(true_fn, prediction_fn, X, y, **kwargs):
    # ------------------------------
    # Get parameters
    # ------------------------------
    r = kwargs.get("r", np.pi)

    # ------------------------------
    # Build visualizing objects
    # ------------------------------
    true_surface = plotly_build_surface(true_fn, r=r, colorscale="Reds", opacity=0.2)
    prediction_surface = plotly_build_surface(prediction_fn, r=r, colorscale="Blues", opacity=0.5)
    scatter_3d = plotly_build_scatter3d(X, y, color="red")

    fig = go.Figure(data=[prediction_surface, true_surface, scatter_3d])

    # ------------------------------
    # Show fig
    # ------------------------------
    fig.update_layout(
        autosize=False,
        margin=dict(l=0, r=0, t=10, b=10),
        scene=dict(xaxis=dict(title="x1"), yaxis=dict(title="x2"), zaxis=dict(title="y")),
        showlegend=False,
    )

    fig.show()
