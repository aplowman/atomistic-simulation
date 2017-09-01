import numpy as np
import os

# Plotly
from plotly import tools
from plotly.offline import plot, iplot, init_notebook_mode
from plotly import graph_objs as go

# Matplotlib
import matplotlib.pyplot as plt

# Bokeh
import bokeh.plotting as bok_plot


COLS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


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

    TODO:
    -   Investigate implementing in 3D. This would be good for showing the
        Bravais lattice unit cells within a CrystalBox.

    """

    if grid_origin is None:
        grid_origin = (0, 0)

    line_args_def = {
        'color': 'silver',
        'width': 1
    }
    if line_args is None:
        line_args = line_args_def
    else:
        line_args = {**line_args_def, **line_args}

    # We want grid_size number of 'boxes' so grid_size + 1 number of lines:
    grid_size = (grid_size[0] + 1, grid_size[1] + 1)

    gd_lns_xx = np.array([[0, grid_size[0] - 1]] * (grid_size[1]))
    gd_lns_xy = np.array([[i, i] for i in range(grid_size[1])])

    gd_lns_yy = np.array([[0, grid_size[1] - 1]] * (grid_size[0]))
    gd_lns_yx = np.array([[i, i] for i in range(grid_size[0])])

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

    gd_lns_xx_v = gd_lns_xx_v + grid_origin[0]
    gd_lns_xy_v = gd_lns_xy_v + grid_origin[1]
    gd_lns_yx_v = gd_lns_yx_v + grid_origin[0]
    gd_lns_yy_v = gd_lns_yy_v + grid_origin[1]

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


def get_3d_arrow_plotly(dir, origin, length, head_length=None,
                        head_radius=None, stem_args=None, n_points=100):
    """
    Get a list of Plotly traces which together represent a 3D arrow.

    Parameters
    ----------
    dir : ndarray of shape (3, )
        Direction vector along which the arrow should point.
    origin : ndarray of shape (3, )
        Origin for the base of the stem of the arrow.
    length : float or int
        Total length of the arrow from base of the stem to the tip of the arrow
        head.
    head_length : float or int, optional
        Length of the arrow head from the tip to the arrow head base. Default
        is None, in which case it will be set to 0.1 * `length`
    head_radius : float or int, optional
        Radius of the base of the arrow head. Default is None, in which case it
        will be set to 0.05 * `length`.
    stem_args : dict, optional
        Specifies the properties of the Plotly line trace used to represent the
        stem of the arrow. Use this to set e.g. `width` and `color`.
    n_points : int, optional
        Number of points to approximate the circular base of the arrow head.
        Default is 100.

    Returns
    -------
    list of Plotly traces

    """

    if head_length is None:
        head_length = length * 0.1

    if head_radius is None:
        head_radius = length * 0.05

    if stem_args is None:
        stem_args = {}

    if stem_args.get('width') is None:
        stem_args['width'] = head_radius * 10

    if stem_args.get('color') is None:
        stem_args['color'] = 'blue'

    sp = (2 * np.pi) / n_points
    θ = np.linspace(0, (2 * np.pi) - sp, n_points)
    θ_deg = np.rad2deg(θ)
    opacity = 0.5

    # First construct arrow head as pointing in the z-direction
    # with its base on (0,0,0)
    x = head_radius * np.cos(θ)
    y = head_radius * np.sin(θ)

    # Arrow head base:
    x1 = np.hstack(([0], x))
    y1 = np.hstack(([0], y))
    z1 = np.zeros(x.shape[0] + 1)
    ah_base = np.vstack([x1, y1, z1])

    # Arrow head cone:
    x2 = np.copy(x1)
    y2 = np.copy(y1)
    z2 = np.copy(z1)
    z2[0] = head_length
    ah_cone = np.vstack([x2, y2, z2])

    # Rotate arrow head so that it points in `dir`
    dir_unit = dir / np.linalg.norm(dir)
    z_unit = np.array([0, 0, 1])

    if np.allclose(z_unit, dir_unit):
        rot_ax = np.array([1, 0, 0])
        rot_an = 0

    elif np.allclose(-z_unit, dir_unit):
        rot_ax = np.array([1, 0, 0])
        rot_an = np.pi

    else:
        rot_ax = np.cross(z_unit, dir_unit)
        rot_an = vectors.col_wise_angles(
            dir_unit[:, np.newaxis], z_unit[:, np.newaxis])[0]

    rot_an_deg = np.rad2deg(rot_an)
    rot_mat = vectors.rotation_matrix(rot_ax, rot_an)[0]

    # Reorient arrow head and translate
    stick_length = length - head_length
    ah_translate = (origin + (stick_length * dir_unit))
    ah_base_dir = np.dot(rot_mat, ah_base) + ah_translate[:, np.newaxis]
    ah_cone_dir = np.dot(rot_mat, ah_cone) + ah_translate[:, np.newaxis]

    i = np.zeros(x1.shape[0] - 1, dtype=int)
    j = np.arange(1, x1.shape[0])
    k = np.roll(np.arange(1, x1.shape[0]), 1)

    data = [
        {
            'type': 'mesh3d',
            'x': ah_base_dir[0],
            'y': ah_base_dir[1],
            'z': ah_base_dir[2],
            'i': i,
            'j': j,
            'k': k,
            'hoverinfo': 'none',
            'color': 'blue',
            'opacity': opacity,
        },
        {
            'type': 'mesh3d',
            'x': ah_cone_dir[0],
            'y': ah_cone_dir[1],
            'z': ah_cone_dir[2],
            'i': i,
            'j': j,
            'k': k,
            'hoverinfo': 'none',
            'color': 'blue',
            'opacity': opacity,
        },
        {
            'type': 'scatter3d',
            'x': [origin[0], ah_translate[0]],
            'y': [origin[1], ah_translate[1]],
            'z': [origin[2], ah_translate[2]],
            'hoverinfo': 'none',
            'mode': 'lines',
            'line': stem_args,
            'projection': {
                'x': {
                    'show': False
                }
            }
        },
    ]

    return data


def get_sphere_plotly(radius, colour='blue', n=50, lighting_args=None,
                      θ_max=np.pi, φ_max=2 * np.pi, label=None,
                      wireframe=False, origin=None):
    """
    Get a surface trace representing a (segment of a) sphere.

    Parameters
    ----------
    radius : float or int
        Radius of the sphere.
    colour : str, optional
        Colour of the sphere.
    n : int, optional
        Number of segments used to draw the sphere. Default is 50.
    lighting_args : dict
        Dictionary to pass to Plotly for the lighting.
    θ_max : float
        Maximum angle to draw in the polar coordinate.
    φ_max : float
        Maximum angle to draw in the azimuthal coordinate.
    wireframe : bool
        If True, draw a wireframe sphere instead of a filled sphere.
    origin : ndarray of shape (3, 1)
        If specified, the origin for the centre of the sphere. Default is None,
        in which case it is set to (0,0,0).

    Returns
    -------
    list of single Plotly trace

    Notes
    -----
    Uses the physics convention of (r, θ, φ) being radial, polar and azimuthal
    angles, respectively.

    TODO:
    -   wireframe=True doesn't work properly.

    """

    if origin is None:
        origin = np.zeros((3, 1))

    if lighting_args is None:
        lighting_args = {
            'ambient': 0.85,
            'roughness': 0.4,
            'diffuse': 0.2,
            'specular': 0.10
        }

    θ = np.linspace(0, θ_max, n)
    φ = np.linspace(0, φ_max, n)
    x = radius * np.outer(np.cos(φ), np.sin(θ))
    y = radius * np.outer(np.sin(φ), np.sin(θ))
    z = radius * np.outer(np.ones(n), np.cos(θ))

    x += origin[0][0]
    y += origin[1][0]
    z += origin[2][0]

    data = [
        {
            'type': 'surface',
            'x': x,
            'y': y,
            'z': z,
            'surfacecolor': np.zeros_like(x),
            'cauto': False,
            'colorscale': [[0, colour], [1, colour]],
            'showscale': False,
            'contours': {
                'x': {
                    'highlight': False,
                },
                'y': {
                    'highlight': False,
                },
                'z': {
                    'highlight': False,
                },
            },
            'lighting': lighting_args,
            'hoverinfo': 'none',
        }
    ]

    if wireframe:
        data[0].update({
            'hidesurface': True
        })
        for i in ['x', 'y', 'z']:
            data[0]['contours'][i].update({
                'show': True
            })

    if label is not None:
        data[0].update({
            'hover_info': 'text',
            'text': [[label] * x.shape[0]] * x.shape[1]
        })

    return data


def get_circle_shape_plotly(radius, origin=None, fill_colour=None,
                            line_args=None, text=''):
    """
    Generate a trace and a dict which can be added to a Plotly
    `layout['shapes']` list to represent a circle with a text hover.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    origin: list of length two, optional
        Position of the circle's centre. By default, None, in which case
        set to (0,0).

    Returns
    -------
    tuple of (dict, dict)
        The first dict is the trace which holds the text information. The
        second is the shape dict which is to be added to the Plotly
        `layout['shapes']` list.

    Notes
    -----
    This generates a shape for adding to a Plotly layout, rather than a circle
    trace.

    """

    if origin is None:
        origin = [0, 0]

    shape = {
        'type': 'circle',
        'x0': origin[0] - radius,
        'x1': origin[0] + radius,
        'y0': origin[1] - radius,
        'y1': origin[1] + radius,
    }

    if fill_colour is not None:
        shape.update({
            'fillcolor': fill_colour,
        })

    if line_args is not None:
        shape.update({
            'line': line_args
        })

    trace = {
        'type': 'scatter',
        'x': [origin[0]],
        'y': [origin[1]],
        'text': [text],
        'mode': 'markers',
        'opacity': 0,
        'showlegend': False,
    }

    return (trace, shape)


def basic_plot_plotly(all_traces, save_args=None):
    """
    Parameters
    ----------
    all_traces : list of dict of (str: dict)
        The data within each list element is to be plotted in a subplot.

    """
    WIDTH_PER_SUBPLOT = 500
    save_args_def = {
        'auto_open': False
    }

    if save_args is None:
        save_args = save_args_def
    else:
        save_args = {**save_args_def, **save_args}

    common_ax_props = dict(
        linecolor='black',
        linewidth=1,
        ticks='inside',
        tickwidth=1,
        mirror='ticks'
    )

    fig = tools.make_subplots(1, len(all_traces))

    data = []
    for subplot_idx, t in enumerate(all_traces):
        for k_idx, (k, v) in enumerate(t.items()):
            data.append({
                'type': 'scatter',
                'x': v['x'],
                'y': v['y'],
                'name': k,
                'legendgroup': k,
                'showlegend': True if subplot_idx == 0 else False,
                'xaxis': 'x' + str(subplot_idx + 1),
                'yaxis': 'y' + str(subplot_idx + 1),
                'line': {
                    'color': COLS[k_idx]
                }
            })

    layout = {
        'width': WIDTH_PER_SUBPLOT * len(all_traces),
        'height': 500,
    }

    for subplot_idx in range(1, len(all_traces) + 1):
        layout.update({
            'xaxis' + str(subplot_idx): {
                **common_ax_props,
                'anchor': 'y' + str(subplot_idx),
            },
            'yaxis' + str(subplot_idx): {
                **common_ax_props,
                'anchor': 'x' + str(subplot_idx),
            },
        })

    fig['data'] = data
    fig['layout'].update(layout)
    plot(fig, **save_args)


def basic_plot_mpl(traces, filename):

    for k, v in traces.items():
        plt.plot(v['x'], v['y'], label=k)

    plt.grid(True)
    plt.savefig(filename)


def basic_plot_bokeh(x, y, filename):

    bok_plot.output_file(filename)
    p = bok_plot.figure()
    p.line(x, y)
    bok_plot.save(p)


def contour_plot_mpl(traces, filename):

    DPI = 96
    fig = plt.figure(figsize=(500 / DPI, 500 / DPI), dpi=DPI)
    ax = fig.gca()

    for k, v in traces.items():
        x, y, z = v['x'], v['y'], v['z']
        x_flat, y_flat = [np.array(i).flatten() for i in [x, y]]
        cset = ax.contourf(x, y, z, cmap=plt.get_cmap('rainbow'))
        ax.scatter(x_flat, y_flat, color='red', s=1)

        x_minmax = [np.min(x_flat), np.max(x_flat)]
        y_minmax = [np.min(y_flat), np.max(y_flat)]

        if v.get('xlabel'):
            ax.set_xlabel(v.get('xlabel'))
        if v.get('ylabel'):
            ax.set_ylabel(v.get('ylabel'))

        ax.set_xlim(x_minmax)
        ax.set_ylim(y_minmax)
        ax.set_aspect('equal')
        cbar = plt.colorbar(cset)

        if v.get('zlabel'):
            cbar.set_label(v.get('zlabel'))

    plt.savefig(filename)


def get_circle_trace_plotly(radius, origin=None, start_ang=0, stop_ang=360, degrees=True, line_args=None, fill_args=None, segment=False):
    """

    Get a Plotly trace representing a cicle (sector, segment).

    Parameters
    ----------
    radius : int of float
    origin : list of length two, optional
        Position of the centre of the circle. Set to [0, 0] if not specified.
    start_ang : int or float, optional
        Angle at which to start cicle sector, measured from positive x-axis.
        Specified in either degrees or radians depending on `degrees`. Set to 0
        if not specified.
    stop_ang : int or float, optional
        Angle at which to stop cicle sector, measured from positive x-axis.
        Specified in either degrees or radians depending on `degrees`. Set to
        360 if not specified.
    degrees : bool, optional
        If True, `start_ang` and `stop_ang` are expected in degrees, otherwise
        in radians. Set to True by default.
    line_args : dict
    fill_args : dict
        Dict with allowed keys:
        fill : str ("none" | "toself")
        fillcolor : str
            For transparency set color string as "rgba(a, b, c, d)"
    segment : bool
        If True, generate a circle segment instead of a sector. The outline of
        circle sector includes the origin, whereas the outline of a circle
        segment may not include the origin. Default is False.

    Returns
    -------
    dict
        Representing a Plotly trace

    """

    if origin is None:
        origin = [0, 0]

    if degrees:
        start_ang = np.deg2rad(start_ang)
        stop_ang = np.deg2rad(stop_ang)

    line_args_def = {}
    if line_args is None:
        line_args = line_args_def
    else:
        line_args = {**line_args_def, **line_args}

    fill_args_def = {
        'fill': 'tozerox',
    }
    if fill_args is None:
        fill_args = fill_args_def
    else:
        fill_args = {**fill_args_def, **fill_args}

    θ = np.linspace(start_ang, stop_ang, 100)
    x = radius * np.cos(θ) + origin[0]
    y = radius * np.sin(θ) + origin[1]

    if not segment and not np.isclose([abs(start_ang - stop_ang)], [2 * np.pi]):
        x = np.hstack([[origin[0]], x, [origin[0]]])
        y = np.hstack([[origin[1]], y, [origin[1]]])

    out = {
        'type': 'scatter',
        'x': x,
        'y': y,
        'hoveron': 'fills',
        'text': '({:.3f}, {:.3f})'.format(origin[0], origin[1]),
        'line': line_args,
        **fill_args,
    }

    return out


def plot_many_mpl(figs, save_dir=None):
    """
    Plot multiple figures with mutliple subplots and traces in Matplotlib.

    """

    SUBPLOT_WIDTH = 500
    SUBPLOT_HEIGHT = 400
    DPI = 96

    num_figs = len(figs)
    add_colourbar = False

    for f in figs:

        if f.get('subplot_width') is not None:
            SUBPLOT_WIDTH = f['subplot_width']
        if f.get('subplot_height') is not None:
            SUBPLOT_HEIGHT = f['subplot_height']

        num_subplots = len(f['subplots'])
        width = (SUBPLOT_WIDTH * num_subplots) / DPI
        height = (SUBPLOT_HEIGHT) / DPI
        f_i, all_ax = plt.subplots(
            1, num_subplots, figsize=(width, height), dpi=DPI)

        for s_idx, s in enumerate(f['subplots']):

            # print('plot_many_mpl: s')

            num_traces = len(s['traces'])

            try:
                ax = all_ax[s_idx]
            except:
                ax = all_ax

            for t in s['traces']:

                # print('plot_many_mpl: t: {}'.format(t))

                for sub_t in t:

                    x = sub_t['x']
                    y = sub_t['y']
                    xv, yv = x['vals'], y['vals']

                    x_arr = np.array(xv)
                    y_arr = np.array(yv)

                    # print('shape x: {}'.format(x_arr.shape))
                    # print('shape y: {}'.format(y_arr.shape))

                    plt_type = sub_t['type']
                    label = sub_t['name']

                    if sub_t.get('title') is not None:
                        label += ' ' + sub_t['title']

                    if plt_type == 'marker':
                        size = sub_t['marker'].get('size')
                        ax.scatter(xv, yv, s=size, label=label)

                    elif plt_type == 'line':
                        linewidth = sub_t['line'].get('width')
                        ax.plot(xv, yv, linewidth=linewidth, label=label)

                    elif plt_type == 'contour':

                        z = sub_t['z']
                        zv = z['vals']

                        bad_len_msg = ('Lengths do not agree for contour plot:'
                                       ' (x, y, z, row_idx, col_idx) = {}')
                        bad_shp_msg = ('Shapes do not agree for contour plot:'
                                       ' (x, y, z) = {}')

                        if sub_t.get('grid') is True:
                            # If grid is True, expect equivalenttly-shaped
                            # 2D data for x, y, z:

                            X = np.array(xv)
                            Y = np.array(yv)
                            Z = np.array(zv)

                            shp_lst = [X, Y, Z]
                            shps = [i.shape for i in shp_lst]
                            if len(set(shps)) != 1:
                                raise ValueError(bad_shp_msg.format(shps))

                        else:
                            # If grid is False, construct 2D arrays using 1D
                            # x, y, z, row_idx, col_idx and shape tuple:

                            col_idx = sub_t['col_idx']
                            row_idx = sub_t['row_idx']
                            len_lst = [xv, yv, zv, row_idx, col_idx]
                            lens = [len(i) for i in len_lst]
                            if len(set(lens)) != 1:
                                raise ValueError(bad_len_msg.format(lens))

                            Z = np.ones(sub_t['shape']) * np.nan
                            for i_idx in range(len(xv)):
                                ri = row_idx[i_idx]
                                ci = col_idx[i_idx]
                                Z[ri][ci] = zv[i_idx]

                            X = x_arr.reshape(sub_t['shape'])
                            Y = y_arr.reshape(sub_t['shape'])

                        minX, maxX = np.min(X), np.max(X)
                        minY, maxY = np.min(Y), np.max(Y)

                        cmap = plt.cm.get_cmap(sub_t['colour_map'])
                        cax = ax.contourf(X, Y, Z, cmap=cmap)
                        ax.set_aspect('equal')

                        ax.set_xlabel(x['label'])
                        ax.set_ylabel(y['label'])
                        ax.set_xlim([minX, maxX])
                        ax.set_ylim([minY, maxY])

                        cbar = f_i.colorbar(cax, ax=ax)
                        if z.get('label'):
                            cbar.set_label(z['label'])

                        if sub_t.get('show_xy'):
                            ax.scatter(x_arr, y_arr, c='black', s=2)

            ax.legend()
            ax.set_title(s['title'])

        f_i.suptitle(f['title'])

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fn = f['filename'] + '_' + f['title']
            fn_base = fn.replace(':', '_').replace(' ', '_')
            for fmt_i in f['fmt']:
                fn = fn_base + '.' + fmt_i
                path = os.path.join(save_dir, fn)
                plt.savefig(path)
