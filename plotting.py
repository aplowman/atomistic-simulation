import numpy as np
import plotly.graph_objs as go


def get_grid_trace_plotly(vectors, grid_size, grid_origin=None, line_args=None,
                          marker_args=None):
    """
    Return a list of Plotly Scatter traces which represent a grid.

    Parameters
    ----------
    vectors : ndarray of shape (2,2)
        Define the unit vectors of the grid as 2D column vectors
    grid_size : tuple of length 2
        Multiples of grid units to draw.
    grid_origin : tuple of length 2
        The position on the grid which should coincide with the origin.
    line_args : dict, optional
        Used to set the properties of the grid lines. Defaults to None, in 
        which case silver lines of width 1 are drawn.
    marker_args : dict, optional
        Used to set the properties of markers used at grid line intersection
        points. Defaults to None, in which case no marker is shown.

    Returns
    -------
    list of Plotly Scatter traces        

    """

    if grid_origin is None:
        grid_origin = (0, 0)

    if line_args is None:
        line_args = {
            'color': 'silver',
            'width': 1
        }

    # We want grid_size number of 'boxes' so grid_size + 1 number of lines:
    grid_size = (grid_size[0] + 1, grid_size[1] + 1)

    gd_lns_xx = np.array([[0, grid_size[0] - 1]] * (grid_size[1]))
    gd_lns_xy = np.array([[i, i] for i in range(grid_size[1])])

    gd_lns_yy = np.array([[0, grid_size[1] - 1]] * (grid_size[0]))
    gd_lns_yx = np.array([[i, i] for i in range(grid_size[0])])

    gd_lns_xx -= grid_origin[0]
    gd_lns_xy -= grid_origin[1]
    gd_lns_yx -= grid_origin[0]
    gd_lns_yy -= grid_origin[1]

    (gd_lns_xx_v,
     gd_lns_xy_v) = np.einsum('ij,jkm->ikm',
                              vectors,
                              np.concatenate([gd_lns_xx[np.newaxis],
                                              gd_lns_xy[np.newaxis]]))

    (gd_lns_yx_v,
     gd_lns_yy_v) = np.einsum('ij,jkm->ikm',
                              vectors,
                              np.concatenate([gd_lns_yx[np.newaxis],
                                              gd_lns_yy[np.newaxis]]))

    sct = []

    # Grid lines parallel to first vector
    for i in range(grid_size[1]):
        sct.append(
            go.Scatter(
                x=gd_lns_xx_v[i],
                y=gd_lns_xy_v[i],
                mode='lines',
                showlegend=False,
                hoverinfo='none',
                line=line_args
            ))

    # Grid lines parallel to second vector
    for i in range(grid_size[0]):
        sct.append(
            go.Scatter(
                x=gd_lns_yx_v[i],
                y=gd_lns_yy_v[i],
                mode='lines',
                showlegend=False,
                hoverinfo='none',
                line=line_args
            ))

    if marker_args is not None:

        gd_int = np.vstack(
            [np.meshgrid(*tuple(np.arange(g) for g in grid_size))]
        ).reshape(len(grid_size), -1)

        gd_int_v = np.dot(vectors, gd_int)

        sct.append(
            go.Scatter(
                x=gd_int_v[0],
                y=gd_int_v[1],
                mode='markers',
                showlegend=False,
                hoverinfo='none',
                marker=marker_args
            ))

    return sct
