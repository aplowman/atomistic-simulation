import numpy as np
import fractions
from atsim import vectors, plotting
import copy
from plotly import graph_objs as go
from plotly.offline import plot, iplot, init_notebook_mode
from atsim.utils import transpose_list


def check_centrosymmetry(points, centre, periodic_box=None):
    """
    Determine if a set of points exhibit centrosymmetry about a centre.

    Parameters
    ----------
    points : ndarray of shape (3, N)
        Array of column vectors representing a set of points in 3D space to
        test for centrosymmetry.
    centre : ndarray of shape (3, 1)
        Position in space representing the candidate inversion centre of the
        set of points `points`.
    periodic_box : ndarray of shape (3, 3), optional
        Array of column vectors representing the edge vectors of a
        parallelopiped. If this is specified, the points are assumed to be
        periodic in this box.

    Returns
    -------
    bool
        True if the set of points have an inversion centre at `centre`.
        Otherwise, False.

    Notes
    -----
    Algorithm proceeds as follows:
    If `periodic_box` not specified:
        1. Invert `points` through `centre`
        2. Test if inverted points and original points are the same
    If `periodic_box` is specified:
        1. Change `points` to fractional coordinates of `box`.
        2. Wrap all points inside box and wrap coordinate of 1 to 0.
        3. Change `centre` to fractional coordinates of `box.
        4. Invert fractional, wrapped points through centre.
        5. Wrap all inverted points inside box and wrap coordinate of 1 to 0.
        6. Test if inverted points and original points are the same


    """

    if periodic_box is None:
        # Invert points:
        p_inv = (2 * centre) - points

    else:
        box_inv = np.linalg.inv(periodic_box)
        p_frac = np.dot(box_inv, points)
        p_frac -= np.floor(p_frac)
        cen_frac = np.dot(box_inv, centre)

        # Invert points:
        p_inv = (2 * cen_frac) - p_frac
        p_inv -= np.floor(p_inv)
        points = p_frac

    # Round to ensure correct sorting
    points = np.round(points, decimals=10)
    p_inv = np.round(p_inv, decimals=10)

    srt_idx = np.lexsort((points[2], points[1], points[0]))
    srt_inv_idx = np.lexsort((p_inv[2], p_inv[1], p_inv[0]))
    p_sort = points[:, srt_idx]
    p_inv_sort = p_inv[:, srt_inv_idx]

    return np.allclose(p_sort, p_inv_sort)


def get_box_corners(box, origin=None, tolerance=1E-10):
    """
    Get all 8 corners of parallelopipeds, each defined by three edge vectors.

    Parameters
    ----------
    box : ndarray of shape (N, 3, 3) or (3, 3)
        Array defining N parallelopipeds, each as three 3D column vectors which
        define the edges of the parallelopipeds.
    origin : ndarray of shape (3, N), optional
        Array defining the N origins of N parallelopipeds as 3D column vectors.

    Returns
    -------
    ndarray of shape (N, 3, 8)
        Returns 8 3D column vectors for each input parallelopiped.

    Examples
    --------
    >>> a = np.random.randint(-1, 4, (2, 3, 3))
    >>> a
    [[[ 0  3  1]
      [ 2 -1 -1]
      [ 1  2  0]]

     [[ 0  0  3]
      [ 1  2  0]
      [-1  1 -1]]]
    >>> geometry.get_box_corners(a)
    array([[[ 0.,  0.,  3.,  1.,  3.,  1.,  4.,  4.],
            [ 0.,  2., -1., -1.,  1.,  1., -2.,  0.],
            [ 0.,  1.,  2.,  0.,  3.,  1.,  2.,  3.]],

           [[ 0.,  0.,  0.,  3.,  0.,  3.,  3.,  3.],
            [ 0.,  1.,  2.,  0.,  3.,  1.,  2.,  3.],
            [ 0., -1.,  1., -1.,  0., -2.,  0., -1.]]])

    """

    if box.ndim == 2:
        box = box[np.newaxis]

    N = box.shape[0]

    if origin is None:
        origin = np.zeros((3, N), dtype=box.dtype)

    corners = np.zeros((N, 3, 8), dtype=box.dtype)
    corners[:, :, 1] = box[:, :, 0]
    corners[:, :, 2] = box[:, :, 1]
    corners[:, :, 3] = box[:, :, 2]
    corners[:, :, 4] = box[:, :, 0] + box[:, :, 1]
    corners[:, :, 5] = box[:, :, 0] + box[:, :, 2]
    corners[:, :, 6] = box[:, :, 1] + box[:, :, 2]
    corners[:, :, 7] = box[:, :, 0] + box[:, :, 1] + box[:, :, 2]

    corners += origin.T[:, :, np.newaxis]

    return corners


def get_box_xyz(box, origin=None, faces=False):
    """
    Get coordinates of paths which trace the edges of parallelopipeds
    defined by edge vectors and origins. Useful for plotting parallelopipeds.

    Parameters
    ----------
    box : ndarray of shape (N, 3, 3) or (3, 3)
        Array defining N parallelopipeds, each as three 3D column vectors which
        define the edges of the parallelopipeds.
    origin : ndarray of shape (3, N) or (3,)
        Array defining the N origins of N parallelopipeds as 3D column vectors.
    faces : bool, optional
        If False, returns an array of shape (N, 3, 30) where the coordinates of
        a path tracing the edges of each of N parallelopipeds are returned as
        column 30 vectors.

        If True, returns a dict where the coordinates for
        each face is a key value pair. Keys are like `face01a`, where the
        numbers refer to the column indices of the vectors in the plane of the
        face to plot, the `a` faces intersect the origin and the `b` faces are
        parallel to the `a` faces. Values are arrays of shape (N, 3, 5), which
        define the coordinates of a given face as five 3D column vectors for
        each of the N input parallelopipeds.

    Returns
    -------
    ndarray of shape (N, 3, 30) or dict of str : ndarray of shape (N, 3, 5)
    (see `faces` parameter).

    """

    if box.ndim == 2:
        box = box[np.newaxis]

    N = box.shape[0]

    if origin is None:
        origin = np.zeros((3, N), dtype=box.dtype)

    elif origin.ndim == 1:
        origin = origin[:, np.newaxis]

    if origin.shape[1] != box.shape[0]:
        raise ValueError('If `origin` is specified, there must be an origin '
                         'specified for each box.')

    c = get_box_corners(box, origin=origin)

    face01a = c[:, :, [0, 1, 4, 2, 0]]
    face01b = c[:, :, [3, 5, 7, 6, 3]]
    face02a = c[:, :, [0, 1, 5, 3, 0]]
    face02b = c[:, :, [2, 4, 7, 6, 2]]
    face12a = c[:, :, [0, 2, 6, 3, 0]]
    face12b = c[:, :, [1, 4, 7, 5, 1]]

    coords = [face01a, face01b, face02a, face02b, face12a, face12b]

    if not faces:
        xyz = np.concatenate(coords, axis=2)

    else:
        faceNames = ['face01a', 'face01b', 'face02a',
                     'face02b', 'face12a', 'face12b']
        xyz = dict(zip(faceNames, coords))

    return xyz


def get_bounding_box(box, bound_vecs=None, padding=0):
    """
    Find bounding boxes around parallelopipeds.

    Parameters
    ----------
    box : ndarray of shape (N, 3, 3)
        Array defining N parallelograms, each specified by three 3D-column
        vectors.
    bound_vecs : ndarray of shape (3, 3), optional
        Array defining the vectors of which the computed bounding box edge
        vectors should be integer multiples. Default is identity matrix of
        shape (3, 3).
    padding : int
        Integer number of additional `bound_vecs` to include in the bounding
        box in each direction as padding around the box. Note that as currently
        implemented, this adds a total of (2 * padding) bound vecs magnitude to
        each of the bounding box edge vectors.

    Returns
    -------
    dict of (str : ndarray)
        `bound_box` is an ndarray of shape (N, 3, 3) defining bounding box edge
        vectors as three 3D-column vectors for each input parallelogram.

        `bound_box_origin` is an ndarray of shape (3, N) defining the origins
        of the bounding boxes as 3D-column vectors.

        `bound_box_bv` is an ndarray of shape (3, N) defining as 3D-column
        vectors the multiples of bound_vecs which form the bounding box.

        `bound_box_origin_bv` is an ndarray with shape (3, N) defining as
        3D-column vectors the origins of the bouding boxes in the `bound_vecs`
        basis.

    TODO:
    -   Allow compute non-integer bounding box (i.e. bounding box just
        determined by directions of `bound_vecs`, not magnitudes.)

    """

    if bound_vecs is None:
        bound_vecs = np.eye(3)

    # Transformation matrix to `bound_vecs` basis:
    bound_vecs_inv = np.linalg.inv(bound_vecs)

    corners = get_box_corners(box)
    corners_bound = bound_vecs_inv @ corners

    tol = 1e-12
    mins = vectors.snap_arr_to_val(
        np.min(corners_bound, axis=2)[:, :, np.newaxis], 0, tol)
    maxs = vectors.snap_arr_to_val(
        np.max(corners_bound, axis=2)[:, :, np.newaxis], 0, tol)

    mins_floor = np.floor(mins) - padding
    maxs_ceil = np.ceil(maxs)

    bound_box_origin = np.concatenate(bound_vecs @ mins_floor, axis=1)
    bound_box_bv = np.concatenate(
        (maxs_ceil - mins_floor + padding).astype(int), axis=1)
    bound_box = vectors.snap_arr_to_val(
        bound_box_bv.T[:, np.newaxis] * bound_vecs[np.newaxis], 0, tol)
    bound_box_origin_bv = np.concatenate(mins_floor.astype(int), axis=1)

    out = {
        'bound_box': bound_box,
        'bound_box_origin': bound_box_origin,
        'bound_box_bv': bound_box_bv,
        'bound_box_origin_bv': bound_box_origin_bv
    }

    return out


def get_box_centre(box, origin=None):
    """
    Find the centre of a parallelepiped.

    Parameters
    ----------
    box : ndarray of shape (3, 3)
        Array of edge vectors defining a parallelopiped.
    origin : ndarray of shape (3, 1)
        Origin of the parallelepiped.

    Returns
    -------
    ndarray of shape (3, N)

    """

    return get_box_corners(box, origin=origin).mean(2).T


class Grid(object):
    """
    Class to represent a 2D nested grid structure formed in 3D space.

    Attributes
    ----------
    grids : list of dicts

    Methods
    -------
    plot
    get_grid_points

    TODO:
    -   Fix issues for nesting depths greater than 1
    -   Support the offset grid key

    """

    def __init__(self, edge_vecs, grid_spec=None, grid_list=None):

        # Validation
        if ((grid_spec is None and grid_list is None) or
                (grid_spec is not None and grid_list is not None)):
            raise ValueError('Specify exactly one of `grid_spec` or '
                             '`grid_list`')

        if grid_list is not None:

            self.edge_vecs = edge_vecs
            self.grids = grid_list

        else:

            grid_spec = copy.deepcopy(grid_spec)

            # First form base grid parent grid spec:
            A = edge_vecs[:, 0:1]
            B = edge_vecs[:, 1:2]
            θ = vectors.col_wise_angles(A, B)[0]
            A_mag = np.linalg.norm(A)
            B_mag = np.linalg.norm(B)
            α = A_mag * np.array([[1, 0]]).T
            β = B_mag * np.array([[1, 0]]).T
            β = vectors.rotate_2D(β, θ)
            edge_vecs = np.hstack([α, β])

            parent_gs = {
                'size': (1, 1),
                'edge_vecs': edge_vecs,
                'origin_std': np.zeros((2, 1)),
                'origin_frac': np.zeros((2, 1)),
                'start': (0, 0),
                'step': (1, 1),
                'stop': (1, 1),
                'par_frac': np.array([1, 1]),
            }

            # We hard code parent_start/stop for the base grid.
            if (grid_spec.get('parent_start') is not None
                    or grid_spec.get('parent_stop') is not None):
                raise ValueError(
                    '`parent_start` and `parent_stop` are not allowed keys in the'
                    ' base grid spec.')

            grid_spec.update({
                'parent_start': (0, 0),
                'parent_stop': (1, 1),
            })

            self.edge_vecs = edge_vecs
            self.grids = self._compute_grid_points(
                edge_vecs, grid_spec, parent_gs)
            self._remove_duplicate_points()

    def to_jsonable(self):
        """
        Generate a dict representation of the Grid object's attributes which
        can be encoded as JSON with json.dump().

        """
        grids_list = copy.deepcopy(self.grids)
        for gd_idx, gd in enumerate(grids_list):
            for k, v in gd.items():
                if isinstance(v, np.ndarray):
                    grids_list[gd_idx][k] = v.tolist()

        grid_json = {
            'grids': grids_list,
            'edge_vecs': self.edge_vecs.tolist()
        }

        return grid_json

    @classmethod
    def from_jsonable(cls, grid_json):
        """
        Generate a Grid object from a dict representation of the Grid object's
        attributes which have been decoded from a JSON object.

        """

        ARR_ATT = [
            'grid_points_frac',
            'grid_points_std',
            'origin_std',
            'unit_cell',
        ]

        TUP_ATT = [
            'parent_start',
            'parent_stop',
            'size',
            'start',
            'step',
            'stop',
        ]

        grid_list = []
        for gd_idx, gd in enumerate(grid_json['grids']):
            grid_dict = {}
            for k, v in gd.items():

                if k in ARR_ATT:
                    v = np.array(v)
                if k in TUP_ATT:
                    v = tuple(v)

                grid_dict.update({k: v})

            grid_list.append(grid_dict)

        params = {
            'edge_vecs': np.array(grid_json['edge_vecs']),
            'grid_list': grid_list
        }
        return cls(**params)

    def get_grid_points(self):
        """
        Get all grid points from all grids and subgrids.

        Returns
        -------
        dict
            Keys:
                points_std : ndarray
                points_frac : ndarray
                points_frac_obj : list of list of Fraction
                grid_idx_nested : list of lists
                grid_idx_flat : ndarray
                point_idx : ndarray

        """

        points_std = []
        points_frac = []
        points_num_den = []
        grid_idx_nested = []
        grid_idx_flat = []
        point_idx = []
        row_idx = []
        col_idx = []

        for gd_idx, gd in enumerate(self.grids):

            if gd['use_points']:

                pnts_s = gd['grid_points_std']
                pnts_f = gd['grid_points_frac']
                pnts_t = gd['grid_points_num_den']
                num_pnts = pnts_s.shape[1]

                points_std.append(pnts_s)
                points_frac.append(pnts_f)
                points_num_den.extend(pnts_t)
                grid_idx_nested.extend([gd['idx']] * num_pnts)
                grid_idx_flat.extend([gd_idx] * num_pnts)
                point_idx.extend(range(num_pnts))

                row_idx.extend(gd['row_idx'])
                col_idx.extend(gd['col_idx'])

        points_std = np.hstack(points_std)
        points_frac = np.hstack(points_frac)

        points_frac_obj = []
        for i in points_frac:
            sublist = []
            for j in i:
                sublist.append(fractions.Fraction(j).limit_denominator())
            points_frac_obj.append(sublist)

        out = {
            'points_std': points_std,
            'points_frac': points_frac,
            'points_frac_obj': points_frac_obj,
            'points_num_den': points_num_den,
            'grid_idx_nested': grid_idx_nested,
            'grid_idx_flat': np.array(grid_idx_flat),
            'point_idx': np.array(point_idx),
            'row_idx': row_idx,
            'col_idx': col_idx,
        }
        return out

    def visualise(self, show_iplot=True, save=False, save_args=None):
        """Generate a plot showing grid points and lines."""

        cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        data = []

        for gd_idx, gd in enumerate(self.grids):

            gd_tr = plotting.get_grid_trace_plotly(
                gd['unit_cell'],
                gd['size'],
                grid_origin=gd['origin_std'].flatten(),
                line_args={'color': cols[gd_idx]})

            data.extend(gd_tr)

            if gd['use_points']:
                pnts = gd['grid_points_std']

                spec_str = (' @ ({0}:{2}, {1}:{3}) => ({10}x{11}), ({4}:{6}:'
                            '{8}, {5}:{7}:{9})')
                spec_fmt = (*gd['parent_start'], *gd['parent_stop'],
                            *gd['start'], *gd['step'], *gd['stop'],
                            *gd['size'])
                label = str(gd['idx']) + spec_str.format(*spec_fmt)

                data.append({
                    'type': 'scatter',
                    'x': pnts[0],
                    'y': pnts[1],
                    'mode': 'markers',
                    'marker': {
                        'color': cols[gd_idx],
                    },
                    'name': label,
                    'legendgroup': label,
                })

        layout = {
            'width': 1000,
            'height': 800,
            'xaxis': {
                'showgrid': False,
                'title': 'x',
                'scaleanchor': 'y',
            },
            'yaxis': {
                'showgrid': False,
                'title': 'y',
            },

        }

        fig = go.Figure(data=data, layout=layout)

        if show_iplot:
            init_notebook_mode()
            iplot(fig)

        if save:
            save_args_def = {
                'filename': 'grid.html',
                'auto_open': False,
            }
            if save_args is None:
                save_args = save_args_def
            else:
                save_args = {**save_args_def, **save_args}

            plot(fig, **save_args)

    def _remove_duplicate_points(self):
        """
        Remove duplicate grid points, keeping points which are more-nested.

        """

        p = self.get_grid_points()
        points_std = p['points_std']
        gi_nested = p['grid_idx_nested']
        gi_flat = p['grid_idx_flat']
        point_idx = p['point_idx']

        # Find which grid point are equivalent
        eq_gp = vectors.get_equal_indices(points_std.T)[0]

        # For each set of equivalent grid points find the grid point which
        # is most-nested, i.e. that which has the longest length grid idx,
        # and get the local indices of the grid points which are to be
        # removed from each grid.
        del_idx = [[] for _ in range(len(self.grids))]
        for k, v in eq_gp.items():
            i = [k] + v
            most_nest_idx = 0
            for j_idx, j in enumerate(i):
                if len(gi_nested[j]) > len(gi_nested[i[most_nest_idx]]):
                    most_nest_idx = j_idx

            i.pop(most_nest_idx)
            for k in i:
                del_idx[gi_flat[k]].append(point_idx[k])

        # Remove duplicate point from each grid
        for gd_idx, (gd, di) in enumerate(zip(self.grids, del_idx)):

            pnts_std = gd['grid_points_std']
            pnts_frac = gd['grid_points_frac']
            pnts_num_den = gd['grid_points_num_den']
            row_idx = gd['row_idx']
            col_idx = gd['col_idx']
            num_pnts = pnts_std.shape[1]

            msk = np.ones(num_pnts, dtype=int)
            msk[di] = False

            w = np.where(msk)[0]
            unique_pnts_std = pnts_std[:, w]
            unique_pnts_frac = pnts_frac[:, w]
            unique_pnts_tup = np.array(pnts_num_den)[:, w].tolist()
            unique_ri = row_idx[w]
            unique_ci = col_idx[w]

            self.grids[gd_idx]['grid_points_std'] = unique_pnts_std
            self.grids[gd_idx]['grid_points_frac'] = unique_pnts_frac
            self.grids[gd_idx]['grid_points_num_den'] = unique_pnts_tup
            self.grids[gd_idx]['row_idx'] = unique_ri
            self.grids[gd_idx]['col_idx'] = unique_ci

    def _compute_grid_points(self, global_edge_vecs, grid_spec,
                             parent_grid_spec, depth=0, idx=[0]):

        # Convenience:
        gs = grid_spec
        pg = parent_grid_spec

        if depth > 1:
            raise NotImplementedError(
                'Higher order nesting is not yet supported.')

        if gs.get('offset') is not None:
            raise NotImplementedError(
                'Grid point offset is not yet supported.')

        if gs.get('parent_start') is None or gs.get('parent_stop') is None:
            raise ValueError('Subgrids specs must have keys: `parent_start` '
                             'and `parent_stop`.')

        valid_specs = [
            'size',
            'max_sep',
            'start',
            'step',
            'stop',
            'use_points',
            'sub_grids',
            'offset',
            'parent_start',
            'parent_stop',
        ]

        for k, v in grid_spec.items():
            if k not in valid_specs:
                raise ValueError('Invalid grid specification: "{}". Allowed '
                                 'keys are: {}'.format(k, valid_specs))

        # Get grid edge vecs:
        par_start = gs['parent_start']
        par_stop = gs['parent_stop']

        pg_sz = np.array(pg['size'])
        pg_ev = pg['edge_vecs']
        pg_pf = pg['par_frac']
        pg_start = np.array(pg['start'])
        pg_stop = np.array(pg['stop'])
        pg_step = np.array(pg['step'])
        pg_origin = pg['origin_frac']

        par_frac = (np.array(par_stop) - np.array(par_start)) / \
            np.array(((pg_stop - pg_start) / pg_step) + pg_start)

        par_frac_tup = [((par_stop[i] - par_start[i]) * pg_step[i],
                         pg_stop[i]) for i in [0, 1]]

        edge_vecs = par_frac * pg_ev

        # Get grid origin:
        origin_frac = (np.array(par_start) / np.array(pg_sz)).reshape((2, 1))
        origin_frac_tup = [(par_start[i], pg_sz[i]) for i in [0, 1]]
        origin_std = np.dot(global_edge_vecs, origin_frac)

        # Get grid points:
        gs_sz = gs.get('size')
        gs_ms = gs.get('max_sep')

        if (gs_sz is not None and gs_ms is not None) or (gs_sz is None and gs_ms is None):
            raise ValueError('Specify exactly one of `size` and `max_sep`.')

        if gs_ms is not None:
            gs_sz = (int(np.ceil(edge_vecs[0][0] / gs_ms[0])),
                     int(np.ceil(edge_vecs[1][1] / gs_ms[1])))

        # Add defaults:
        if gs.get('step') is None:
            gs['step'] = (1, 1)

        if gs.get('start') is None:
            gs['start'] = (0, 0)

        if gs.get('stop') is None:
            gs['stop'] = gs_sz

        # Validation:
        if any([sp > sz for sp, sz in zip(gs.get('stop'), gs_sz)]):
            raise ValueError('Stop value cannot be larger than size value.')

        gs_msh = np.meshgrid(*tuple(np.arange(g + 1) for g in gs_sz))

        # Slice according to start, stop and step:
        x_slice = slice(gs['start'][0], gs['stop'][0] + 1, gs['step'][0])
        y_slice = slice(gs['start'][1], gs['stop'][1] + 1, gs['step'][1])
        gs_msh = [m[:, x_slice] for m in gs_msh]
        gs_msh = [m[y_slice] for m in gs_msh]

        gs_msh_frac = np.array([gm / s for gm, s in zip(gs_msh, gs_sz)])
        gs_msh_frac *= pg_pf.reshape((2, 1, 1)) * par_frac.reshape((2, 1, 1))
        gs_msh_frac += origin_frac.reshape((2, 1, 1))
        gs_msh_std = np.einsum('ij,jkm->ikm', global_edge_vecs, gs_msh_frac)

        nrows = gs_msh[0].shape[0]
        ncols = gs_msh[0].shape[1]
        grd_shape = (nrows, ncols)
        col_idx, row_idx = np.meshgrid(np.arange(ncols), np.arange(nrows))
        row_idx = row_idx.reshape((-1))
        col_idx = col_idx.reshape((-1))

        gs_points_num = np.vstack(gs_msh).reshape(2, -1)
        gs_points_den = np.array(gs_sz).reshape((2, 1))

        gs_points_num_2 = np.array([gs_points_num[0] * par_frac_tup[0][0],
                                    gs_points_num[1] * par_frac_tup[1][0]])

        gs_points_den_2 = np.array([gs_points_den[0] * par_frac_tup[0][1],
                                    gs_points_den[1] * par_frac_tup[1][1]])

        gs_points_num_3 = np.array([gs_points_num_2[0] * origin_frac_tup[0][1],
                                    gs_points_num_2[1] * origin_frac_tup[1][1]])

        gs_points_den_3 = np.array([gs_points_den_2[0] * origin_frac_tup[0][1],
                                    gs_points_den_2[1] * origin_frac_tup[1][1]])

        origin_frac_tup = [(origin_frac_tup[0][0] * gs_points_den_2[0][0],
                            origin_frac_tup[0][1] * gs_points_den_2[0][0]),
                           (origin_frac_tup[1][0] * gs_points_den_2[1][0],
                            origin_frac_tup[1][1] * gs_points_den_2[1][0])]

        gs_points_num_den = []
        for i_idx, i in enumerate(gs_points_num_3.T):
            gs_points_num_den.append(((i[0] + origin_frac_tup[0][0],
                                       gs_points_den_3[0, 0]),
                                      (i[1] + origin_frac_tup[1][0],
                                       gs_points_den_3[1, 0])))

        gs_points_num_den = transpose_list(gs_points_num_den)

        gs_points_frac = (gs_points_num / gs_points_den) * \
            pg_pf.reshape((2, 1)) * par_frac.reshape((2, 1)) + origin_frac
        gs_points_std = np.dot(global_edge_vecs, gs_points_frac)

        unit_cell = edge_vecs / gs_sz

        grid_spec.update({
            'edge_vecs': edge_vecs,
            'unit_cell': unit_cell,
            'origin_std': origin_std,
            'origin_frac': origin_frac,
            'gs_points_num_den': gs_points_num_den,
            'grid_points': gs_msh_std,
            'par_frac': par_frac,
        })

        if gs_ms is not None:
            grid_spec.update({
                'size': gs_sz
            })

        all_grids = [
            {
                'unit_cell': unit_cell,
                'origin_std': origin_std,
                'grid_points_std': gs_points_std,
                'grid_points_frac': gs_points_frac,
                'grid_points_num_den': gs_points_num_den,
                'size': gs_sz,
                'row_idx': row_idx,
                'col_idx': col_idx,
                'start': gs['start'],
                'step': gs['step'],
                'stop': gs['stop'],
                'shape': grd_shape,
                'parent_start': par_start,
                'parent_stop': par_stop,
                'idx': idx,
                'use_points': True,
            }
        ]

        sub_grids = gs.get('sub_grids')
        use_points = gs.get('use_points')
        if sub_grids is not None:

            if use_points is None:
                use_points = False
            all_grids[0].update({'use_points': use_points})

            all_subgrids = []
            for sg_idx, sg in enumerate(sub_grids):

                gds = self._compute_grid_points(global_edge_vecs, sg,
                                                grid_spec, depth=depth + 1,
                                                idx=idx + [sg_idx])
                all_subgrids.extend(gds)

            all_grids.extend(all_subgrids)

        else:
            if use_points is not None and not use_points:
                raise ValueError('Setting `use_points` to `False` is not '
                                 'allowed for grids which do not contain '
                                 'subgrids.')

        return all_grids
