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


def plot_many_mpl_old(figs, save_dir=None):
    """
    Plot multiple figures with mutliple subplots and traces in Matplotlib.

    """

    SUBPLOT_WIDTH = 500
    SUBPLOT_HEIGHT = 400
    MAX_HORZ_SUBPLOTS = 4
    DPI = 96

    num_figs = len(figs)
    add_colourbar = False

    for f in figs:

        if f.get('subplot_width') is not None:
            SUBPLOT_WIDTH = f['subplot_width']
        if f.get('subplot_height') is not None:
            SUBPLOT_HEIGHT = f['subplot_height']

        num_subplots = len(f['subplots'])

        # Partition subplots into reasonable grid
        sp_nrows = int(np.ceil(num_subplots / MAX_HORZ_SUBPLOTS))
        sp_ncols = int(np.ceil(num_subplots / sp_nrows))

        width = (SUBPLOT_WIDTH * sp_ncols) / DPI
        height = (SUBPLOT_HEIGHT * sp_nrows) / DPI

        f_i, all_ax = plt.subplots(
            sp_nrows, sp_ncols, figsize=(width, height), dpi=DPI)

        for s_idx, s in enumerate(f['subplots']):

            sp_ridx = int(np.floor(s_idx / sp_ncols))
            sp_cidx = int(s_idx - (sp_ridx * sp_ncols))

            num_traces = len(s['traces'])

            if sp_nrows == 1 or sp_ncols == 1:
                ax_idx = sp_ridx if sp_ncols == 1 else sp_cidx
                ax = all_ax[ax_idx]
            else:
                ax = all_ax[sp_ridx][sp_cidx]

            all_labels = []

            for t in s['traces']:

                for sub_t in t:

                    x = sub_t['x']
                    y = sub_t['y']
                    xv, yv = x['vals'], y['vals']

                    x_arr = np.array(xv)
                    y_arr = np.array(yv)

                    plt_type = sub_t['type']
                    label = sub_t.get('title', None)

                    if label is not None:
                        if sub_t.get('title') is not None:
                            label += ' ' + sub_t['title']
                        all_labels.append(label)

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

                        ax.set_xlim([minX, maxX])
                        ax.set_ylim([minY, maxY])

                        cbar = f_i.colorbar(cax, ax=ax)
                        if z.get('label'):
                            cbar.set_label(z['label'])

                        if sub_t.get('show_xy'):
                            ax.scatter(x_arr, y_arr, c='black', s=2)

                    ax.set_xlabel(x['label'])
                    ax.set_ylabel(y['label'])

                    if x.get('reverse'):
                        ax.invert_xaxis()

                    if y.get('reverse'):
                        ax.invert_yaxis()

            if len(all_labels) > 0:
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
        else:
            plt.show()


def normalise_mpl_subplots_axes(all_ax, nrows, ncols):
    """
    Normalise the axes produced by plt.subplots() so it is always a 2D array.

    """

    if nrows == 1:

        if ncols == 1:
            all_ax = np.array([[all_ax]])

        else:
            all_ax = np.array([all_ax])

    elif ncols == 1:
        all_ax = np.array([[i] for i in all_ax])

    return all_ax


def encode_axis_props_mpl(props, ax):
    """
    Change axis properties into those accepted by Matplotlib.

    Parameters
    ----------
    props : dict
        Dict of axis properties which are to be mapped to MPL axis properties 
        which can be passed to `ax.set()`.
    ax : str
        `x` or `y`      
    """
    isx = False
    if ax == 'x':
        isx = True

    encoded_props = {}
    for k, v in props.items():

        if k == 'label':
            if isx:
                encoded_props.update({'xlabel': v})
            else:
                encoded_props.update({'ylabel': v})

    return encoded_props


def encode_axis_props_plotly(props):
    """Change axis properties into those accepted by Plotly."""

    PASS_PROPS = [
        'side',
        'overlaying',
        'mirror',
        'linecolor',
        'linewidth',
        'ticks',
        'showgrid',
        'zeroline',
    ]

    encoded_props = {}
    for k, v in props.items():

        if k == 'label':
            encoded_props.update({'title': v})
        elif k == 'reverse':
            encoded_props.update({'autorange': 'reversed' if v else True})
        elif k in PASS_PROPS:
            encoded_props.update({k: v})

    return encoded_props


def get_subplot_axes_mpl(base_ax, all_ax_defn, all_ax_props):
    """
    Generate a list of mpl axes objects, starting from an original axes object.

    Parameters
    ----------
    base_ax : matplotlib.axes.Axes object
        New Axes are generated from this Axes object with e.g. ax.twiny()
    all_ax_defn: list of list of str
        Each list element represents a set of axes (e.g. x and y) to appear
        on a given subplot. Inner lists represent the x- and y-axes and 
        are defined here so that properties can be assiged in the 
        `all_ax_props` dict.
    all_ax_props : dict
        Keys are strings which are listed in `all_ax_defn`. Values are 
        axis properties defined which are passed to `encode_axis_props_mpl`.

    Returns
    -------
    list of matplotlib.axes.Axes objects
        The list represents all the Axes objects for a given subplot.

    """

    subplot_ax = [base_ax]
    for ax_i_names in all_ax_defn[1:]:

        if ax_i_names[0] == all_ax_defn[0][0]:
            subplot_ax.append(base_ax.twinx())

        elif ax_i_names[1] == all_ax_defn[0][1]:
            subplot_ax.append(base_ax.twiny())

    for ax_idx, ax_i_names in enumerate(all_ax_defn):

        x_name = ax_i_names[0]
        y_name = ax_i_names[1]

        x_props = all_ax_props.get(x_name, {})
        y_props = all_ax_props.get(y_name, {})

        if x_props.get('reverse', False):
            subplot_ax[ax_idx].invert_xaxis()
            del x_props['reverse']

        if y_props.get('reverse', False):
            subplot_ax[ax_idx].invert_yaxis()
            del y_props['reverse']

        x_set_props = encode_axis_props_mpl(x_props, 'x')
        y_set_props = encode_axis_props_mpl(y_props, 'y')
        subplot_ax[ax_idx].set(**x_set_props, **y_set_props)

    return subplot_ax


def set_subplot_axes_plotly(layout, idx, all_ax_defn, all_ax_props):
    """
    Generate additional axes for a given subplot index (`idx`) in a grid of
    subplots, where the single x and y axes for each subplot have already been
    defined in `layout`.

    """

    base_xax_idx = str(idx + 1)
    base_yax_idx = str(idx + 1)
    base_xax_name = 'xaxis' + base_xax_idx
    base_yax_name = 'yaxis' + base_yax_idx
    xax = layout[base_xax_name]
    yax = layout[base_yax_name]

    num_xax = len([i for i in layout.keys() if 'xaxis' in i])
    num_yax = len([i for i in layout.keys() if 'yaxis' in i])

    all_x_defn = utils.get_col(all_ax_defn, 0)
    all_y_defn = utils.get_col(all_ax_defn, 1)

    default_base_ax = {
        'linecolor': 'black',
        'linewidth': 1,
        'mirror': True,
        'ticks': 'outside',
        'zeroline': False,
    }
    default_non_base_ax = {
        'ticks': 'outside',
        'showgrid': False,
        'zeroline': False,
    }

    xax.update({
        **encode_axis_props_plotly({
            **all_ax_props.get(all_ax_defn[0][0], {}),
            **default_base_ax,
        }),
    })
    yax.update({
        **encode_axis_props_plotly({
            **all_ax_props.get(all_ax_defn[0][1], {}),
            **default_base_ax,
        })
    })

    ax_names_map = [['x' + base_xax_idx, 'y' + base_yax_idx]]
    for ax_nm_idx, ax_i_names in enumerate(all_ax_defn[1:]):

        if ax_i_names[0] in all_x_defn[:ax_nm_idx + 1]:
            xax_i_idx = all_x_defn[:ax_nm_idx + 1].index(ax_i_names[0])
            xax_name = ax_names_map[xax_i_idx][0]

        else:
            xax_idx = str(num_xax + 1)
            xax_name = 'x' + xax_idx

            layout.update({
                'xaxis' + xax_idx: {
                    **encode_axis_props_plotly({
                        **all_ax_props.get(ax_i_names[0], {}),
                        **default_non_base_ax,
                    }),
                    'domain': layout[base_xax_name]['domain'],
                    'anchor': 'y' + str(idx + 1),
                }
            })
            num_xax += 1

        if ax_i_names[1] in all_y_defn[:ax_nm_idx + 1]:
            yax_i_idx = all_y_defn[:ax_nm_idx + 1].index(ax_i_names[1])
            yax_name = ax_names_map[yax_i_idx][1]

        else:
            yax_idx = str(num_yax + 1)
            yax_name = 'y' + yax_idx
            layout.update({
                'yaxis' + yax_idx: {
                    **encode_axis_props_plotly({
                        **all_ax_props.get(ax_i_names[1], {}),
                        **default_non_base_ax,
                    }),
                    'domain': layout[base_yax_name]['domain'],
                    'anchor': 'x' + str(idx + 1),
                    'side': 'right',
                    'overlaying': 'y' + base_yax_idx,
                }
            })
            num_yax += 1

        ax_names_map.append([xax_name, yax_name])

    return ax_names_map


def make_plotly_grid(nrows, ncols, subplot_width, subplot_height, titles):
    """Define an x- and y-axis for each in a grid of subplots."""

    sep_x = 80
    sep_y = 50
    subplot_title_sep = 20

    w = (subplot_width + (sep_x * 2)) * ncols
    h = (subplot_height + (sep_y * 2) + subplot_title_sep) * nrows

    sep_x_frac = sep_x / w
    sep_y_frac = sep_y / h
    spt_sep_frac = subplot_title_sep / h

    layout = {
        'width': w,
        'height': h,
        'annotations': [],
    }
    idx = 1
    for r in range(nrows):
        for c in range(ncols):
            xname = 'xaxis' + str(idx)
            yname = 'yaxis' + str(idx)
            xfrac = (c / ncols)
            yfrac = (r / nrows)

            ytop = (r + 1) / nrows - sep_y_frac
            xdom = [c / ncols + sep_x_frac, (c + 1) / ncols - sep_x_frac]
            ydom = [r / nrows + sep_y_frac, ytop]

            layout.update({
                xname: {
                    'anchor': 'y' + str(idx),
                    'domain': xdom,
                },
                yname: {
                    'anchor': 'x' + str(idx),
                    'domain': ydom
                }
            })

            if titles[idx - 1] is not None:

                xmid = (c + 1 / 2) / ncols
                layout['annotations'].append({
                    'xref': 'paper',
                    'yref': 'paper',
                    'y': ytop + spt_sep_frac,
                    'x': xmid,
                    'text': titles[idx - 1],
                    'showarrow': False,
                    'font': {'size': 16},
                    'xanchor': 'center',
                    'yanchor': 'bottom',
                })

            idx += 1

    return layout


def get_subplots(fig, lib='mpl'):
    """
    Later add ability to get subplots from grid etc.

    Parameters
    ----------
    fig : dict
    lib : str ('mpl' | 'plotly')

    """

    SUBPLOT_WIDTH = 300
    SUBPLOT_HEIGHT = 300
    MAX_HORZ_SUBPLOTS = 4
    DPI = 96

    if fig.get('subplot_width') is not None:
        SUBPLOT_WIDTH = fig['subplot_width']
    if fig.get('subplot_height') is not None:
        SUBPLOT_HEIGHT = fig['subplot_height']

    num_subplots = len(fig['subplots'])

    # Partition subplots into reasonable grid
    sp_nrows = int(np.ceil(num_subplots / MAX_HORZ_SUBPLOTS))
    sp_ncols = int(np.ceil(num_subplots / sp_nrows))

    if lib == 'mpl':

        width = (SUBPLOT_WIDTH * sp_ncols) / DPI
        height = (SUBPLOT_HEIGHT * sp_nrows) / DPI

        fig, all_ax = plt.subplots(
            sp_nrows, sp_ncols, figsize=(width, height), dpi=DPI)

        # Normalise all_ax so can always index with row and column indices
        all_ax = normalise_mpl_subplots_axes(all_ax, sp_nrows, sp_ncols)
        return fig, all_ax, sp_nrows, sp_ncols

    elif lib == 'plotly':

        subplot_titles = [i.get('title') for i in fig['subplots']]
        layout = make_plotly_grid(sp_nrows, sp_ncols, SUBPLOT_WIDTH,
                                  SUBPLOT_HEIGHT, subplot_titles)
        return layout, sp_nrows, sp_ncols


def set_trace_defaults(trc):

    trc_def = {
        'type': 'line',
    }

    return {**trc_def, **trc}


def plot_many_plotly(figs):
    """
    Plot one or more figures each containing multiple subplots and traces
    using Plotly.

    """

    figs = copy.deepcopy(figs)

    for f_idx, f in enumerate(figs):

        layout, nrows, ncols = get_subplots(f, lib='plotly')
        layout.update({
            'title': f['title']
        })
        data = []
        legend_names_cols = {}
        for s_idx, s in enumerate(f['subplots']):

            # Generate additional axes for a given subplot
            subplot_ax_names = set_subplot_axes_plotly(
                layout, s_idx, s['axes'], s['axes_props'])

            for t_idx, t in enumerate(s['traces']):

                t = set_trace_defaults(t)
                t_xax = subplot_ax_names[t['axes_idx']][0]
                t_yax = subplot_ax_names[t['axes_idx']][1]

                xv = t['x']['vals']
                yv = t['y']['vals']
                name = t['name'] if t.get('name', False) else None

                showlegend = False
                if name not in legend_names_cols:
                    legend_names_cols.update(
                        {name: COLS[len(legend_names_cols)]})
                    showlegend = True

                trc = {
                    'x': xv,
                    'y': yv,
                    'name': name,
                    'legendgroup': name,
                    'showlegend': showlegend,
                    'xaxis': t_xax,
                    'yaxis': t_yax,
                }

                if t['type'] == 'line':
                    trc.update({
                        'type': 'scatter',
                        'mode': 'line',
                        'line': {
                            'color': legend_names_cols[name],
                        }
                    })

                elif t['type'] == 'marker':
                    trc.update({
                        'type': 'scatter',
                        'mode': 'markers',
                        'marker': {
                            'color': legend_names_cols[name],
                        }
                    })

                data.append(trc)

        fig = go.Figure(data=data, layout=layout)
        iplot(fig)


def plot_many_mpl(figs):
    """
    Plot one or more figures each containing multiple subplots and traces using
    Matplotlib.

    """

    figs = copy.deepcopy(figs)

    for f_idx, f in enumerate(figs):

        f_i, all_ax, nrows, ncols = get_subplots(f, lib='mpl')
        f_i.suptitle(f['title'])

        legend_names_cols = {}
        for s_idx, s in enumerate(f['subplots']):

            ridx, cidx = utils.get_row_col_idx(s_idx, nrows, ncols)
            ax = all_ax[ridx][cidx]

            # Generate additional axes for a given subplot
            subplot_ax = get_subplot_axes_mpl(ax, s['axes'], s['axes_props'])

            if s.get('title', False):
                subplot_ax[0].set_title(s['title'])

            for t_idx, t in enumerate(s['traces']):

                t = set_trace_defaults(t)
                t_ax = subplot_ax[t['axes_idx']]

                xv = t['x']['vals']
                yv = t['y']['vals']
                name = t.get('name')

                if name not in legend_names_cols:
                    legend_names_cols.update(
                        {name: COLS[len(legend_names_cols)]})

                if t['type'] == 'line':
                    t_ax.plot(xv, yv, c=legend_names_cols[name], label=name)

                elif t['type'] == 'marker':
                    t_ax.scatter(
                        x=xv, y=yv, c=legend_names_cols[name], label=name)

                t_ax.legend()

        plt.tight_layout()
        plt.show()
