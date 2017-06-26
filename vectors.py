import numpy as np
from numpy import linalg as la

from memory_profiler import profile


def row_wise_dot(a, b):
    """ Compute the dot product between rows of a and rows of b."""

    return np.einsum('ij,ij->i', a, b)


def col_wise_dot(a, b):
    """ Compute the dot product between columns of a and columns of b."""

    return np.einsum('ij,ij->j', a, b)


def row_wise_cos(a, b):
    """ Given two arrays of row vectors a and b, find the pairwise cosines between a and b."""

    cosθ = row_wise_dot(a, b) / (la.norm(a, axis=1) * la.norm(b, axis=1))
    return cosθ


def col_wise_cos(a, b):
    """ Given two arrays of column vectors a and b, find the pairwise cosines between a and b."""

    cosθ = col_wise_dot(a, b) / (la.norm(a, axis=0) * la.norm(b, axis=0))
    return cosθ


def col_wise_sin(a, b):
    """ Given two arrays of column vectors a and b, find the pairwise sines between a and b."""

    sinθ = la.norm(np.cross(a, b, axis=0), axis=0) / \
        (la.norm(a, axis=0) * la.norm(b, axis=0))
    return sinθ


def col_wise_angles(a, b, degrees=False):
    """ a, b are 3 x N arrays whose columns represented vectors to find the angles betweeen."""

    A = la.norm(np.cross(a, b, axis=0), axis=0)
    B = col_wise_dot(a, b)
    angles = np.arctan2(A, B)

    if degrees:
        angles = np.rad2deg(angles)

    return angles


def snap_arr_to_val(arr, val, tol):
    """ Returns a copy of `arr` where elements in `arr` which are close
        to `val` are set to be exactly `val` if within tolerance `tol`.

    """

    out = np.copy(arr)
    out[abs(arr - val) < tol] = val

    return out


def is_arr_int(arr):
    """ Finds if all the elements in an array are integers. Returns bool."""

    return np.all(np.equal(np.mod(arr, 1), 0))


def project_vec_to_plane(vec, plane_normal):
    """
        Returns the vector that is the projection of `vec` onto the plane given by
        `plane_normal`.

        `vec` is an array of column vectors
        `plane_normal` is a single column vector

    """

    p = vec - (np.einsum('ij, ik->j', vec, plane_normal) /
               la.norm(plane_normal)**2) * plane_normal

    return p


def rotation_matrix(axes, angles, degrees=False):
    """
    Generates pre-multiplication rotation matrices for given axes and angles.

    Parameters
    ----------
    axes : ndarray
        Array of shape (N, 3) or (3,), which if N is 1 or shape is (3,), will
        be tiled to the size (M, 3). Otherwise, N must be equal to M (for M,
        see `angles`).
    angles : ndarray or float or int
        Array of shape (M,) or a number which will be converted to an array
        with M = 1.
    degrees : bool (optional)
        If True, `angles` interpreted as degrees.

    Returns
    -------
    ndarray of shape (N or M, 3, 3).

    Notes
    -----
    Computed using the Rodrigues' rotation formula.

    Examples
    --------

    Find the rotation matrix for a single axis and angle:

    >>> rotation_matrix(np.array([0,0,1]), np.pi/4)
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]]])

    Find the rotation matrix for a single axis and angle:

    >>> rotation_matrix(np.array([[0,0,1]]), np.array([np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]]])

    Find the rotation matrices for different angles about the same axis:

    >>> rotation_matrix(np.array([[0,0,1]]), np.array([np.pi/4, -np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]],

           [[ 0.70710678,  0.70710678,  0.        ],
            [-0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]]])

    Find the rotation matrices about different axes by the same angle:

    >>> rotation_matrix(np.array([[0,0,1], [0,1,0]]), np.array([np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]],

           [[ 0.70710678,  0.        ,  0.70710678],
            [ 0.        ,  1.        ,  0.        ],
            [-0.70710678,  0.        ,  0.70710678]]])

    Find the rotation matrices about different axes and angles:

    >>> rotation_matrix(
        np.array([[0,0,1], [0,1,0]]), np.array([np.pi/4, -np.pi/4]))
    array([[[ 0.70710678, -0.70710678,  0.        ],
            [ 0.70710678,  0.70710678,  0.        ],
            [ 0.        ,  0.        ,  1.        ]],

           [[ 0.70710678,  0.        , -0.70710678],
            [ 0.        ,  1.        ,  0.        ],
            [ 0.70710678,  0.        ,  0.70710678]]])

    """

    # Check dimensions

    if axes.ndim == 1:
        axes = axes[np.newaxis]

    angles_err_msg = '`angles` must be a number or array of shape (M,).'

    if isinstance(angles, np.ndarray):
        if angles.ndim != 1:
            raise ValueError(angles_err_msg)

    else:
        try:
            angles = np.array([angles])

        except ValueError:
            print(angles_err_msg)

    if axes.shape[0] == angles.shape[0]:
        n = axes.shape[0]
    else:
        if axes.shape[0] == 1:
            n = angles.shape[0]
        elif angles.shape[0] == 1:
            n = axes.shape[0]
        else:
            raise ValueError(
                'Incompatible dimensions: the first dimension of `axes` or'
                '`angles` must be one otherwise the first dimensions of `axes`'
                'and `angles` must be equal.')

    # Convert to radians if necessary
    if degrees:
        angle = np.deg2rad(angle)

    # Normalise axes to unit vectors:
    axes = axes / np.linalg.norm(axes, axis=1)[:, np.newaxis]

    cross_prod_mat = np.zeros((n, 3, 3))
    cross_prod_mat[:, 0, 1] = -axes[:, 2]
    cross_prod_mat[:, 0, 2] = axes[:, 1]
    cross_prod_mat[:, 1, 0] = axes[:, 2]
    cross_prod_mat[:, 1, 2] = -axes[:, 0]
    cross_prod_mat[:, 2, 0] = -axes[:, 1]
    cross_prod_mat[:, 2, 1] = axes[:, 0]

    rot_mats = np.tile(np.eye(3), (n, 1, 1)) + (
        np.sin(angles)[:, np.newaxis, np.newaxis] * cross_prod_mat) + (
            (1 - np.cos(angles)[:, np.newaxis, np.newaxis]) * (
                cross_prod_mat @ cross_prod_mat))

    return rot_mats


def get_equal_indices(arr, scale_factors=None):
    """
    Return the indices along the first dimension of an array which index equal
    sub-arrays.

    Parameters
    ----------
    arr : ndarray
        Array of any shape whose elements along its first dimension are
        compared for equality.
    scale_factors : list of float or list of int, optional
        Multiplicative factors to use when checking for equality between
        subarrays. Each factor is checked independently.

    Returns
    -------
    tuple of dict of int: list of int
        Each tuple item corresponds to a scale factor for which each dict maps
        a subarray index to a list of equivalent subarray indices given that
        scale factor. Length of returned tuple is equal to length of
        `scale_factors` or 1 if `scale_factors` is not specified.

    Notes
    -----
    If we have a scale factor `s` which returns {a: [b, c, ...]}, then the
    inverse scale factor `1/s` will return {b: [a], c: [a], ...}.


    Examples
    --------

    1D examples:

    >>> a = np.array([5, 1, 4, 6, 1, 8, 2, 7, 4, 7])
    >>> get_equal_indices(a)
    ({1: [4], 2: [8], 7: [9]},)

    >>> a = np.array([1, -1, -1, 2])
    >>> get_equal_indices(a, scale_factors=[1, -1, -2, -0.5])
    ({1: [2]}, {0: [1, 2]}, {1: [3], 2: [3]}, {3: [1, 2]})

    2D example:

    >>> a = np.array([[1., 2.], [3., 4.], [-0.4, -0.8]])
    >>> get_equal_indices(a, scale_factors=[-0.4])
    ({0: [2]},)

    """

    if scale_factors is None:
        scale_factors = [1]

    a_dims = len(arr.shape)
    arr_B = arr[:, np.newaxis]

    sf_shape = tuple([len(scale_factors)] + [1] * (a_dims + 1))
    sf = np.array(scale_factors).reshape(sf_shape)

    bc = np.broadcast_arrays(arr, arr_B, sf)
    c = np.isclose(bc[0], bc[1] * bc[2])

    if a_dims > 1:
        c = np.all(c, axis=tuple(range(3, a_dims + 2)))

    out = ()
    for c_sub in c:

        w2 = np.where(c_sub)
        d = {}
        skip_idx = []

        for i in set(w2[0]):

            if i not in skip_idx:

                row_idx = np.where(w2[0] == i)[0]
                same_idx = list(w2[1][row_idx])

                if i in same_idx:

                    if len(row_idx) == 1:
                        continue

                    elif len(row_idx) > 1:
                        same_idx.remove(i)

                d.update({i: same_idx})
                skip_idx += same_idx

        out += (d,)

    return out


# @profile
def find_unique_int_vecs(s):
    """
    Find non-collinear integer vectors within an origin-centered cube of given
    size.

    The zero vector is excluded.

    Parameters
    ----------
    s : int
        Size of half the cube edge, such that vectors have maximum components
        |s|. Must be: 0 < s <= 127.

    Returns
    -------
    ndarray
        Array of column vectors.

    Examples
    --------
    >>> find_unique_int_vecs(1)
    [[ 0  0  1]
     [ 0  1  0]
     [ 0  1  1]
     [ 0  1 -1]
     [ 1 -1  0]
     [ 1 -1  1]
     [ 1 -1 -1]
     [ 1  0  0]
     [ 1  0  1]
     [ 1  0 -1]
     [ 1  1  0]
     [ 1  1  1]
     [ 1  1 -1]]

    """

    if s > 127 or s < 1:
        # Would need to change `s_i` dtype to np.int16 or beyond for s > 127:
        raise ValueError('Seach size less than 1 or greater than 127 not'
                         'supported.')

    s_i = np.zeros((2 * s) + 1, dtype=np.int8)
    s_i[1::2] = np.arange(1, s + 1)
    s_i[2::2] = -np.arange(1, s + 1)

    a = np.vstack(np.meshgrid(s_i, s_i, s_i, copy=False)).reshape((3, -1)).T
    a[:, [0, 1]] = a[:, [1, 0]]

    # print('s_i: {}\n'.format(s_i))

    # print('a.shape: {}'.format(a.shape))
    # print('a.nbytes: {}'.format(a.nbytes))
    # print('a.itemsize: {}'.format(a.itemsize))
    # print('a: \n{}\n'.format(a))

    # Remove the zero vector
    a = a[1:]

    # Use cross product to find which vectors are collinear
    c = np.cross(a, a[:, np.newaxis])
    # print('c.shape: {}'.format(c.shape))
    # print('c.nbytes: {}'.format(c.nbytes))
    # print('c.itemsize: {}'.format(c.itemsize))
    # print('c: \n{}\n'.format(c))

    w = np.where(np.all(c == 0, axis=-1).T)

    all_remove_idx = []

    # Get the indices of collinear vectors
    for i in set(w[0]):

        col_idx = np.where(w[0] == i)[0]

        if len(col_idx) != 1:
            all_remove_idx.extend(w[1][col_idx[1:]])

    all_remove_idx = list(set(all_remove_idx))

    # Remove collinear vectors
    a = np.delete(a, all_remove_idx, axis=0)
    a = a[np.lexsort((a[:, 1], a[:, 0]))]

    return a


def find_parallel_vectors(vec_a, vec_b=None, print_progress=False):
    """
    Find which vectors in one array are (anti-)parallel to those in another.

    Parameters
    ----------
    vec_a : ndarray of shape (3, N)
        First array of column vectors.
    vec_b : ndarray of shape (3, M), optional
        Second array of column vectors. Defaults to None, in which case the
        vectors in `vec_a` are compared with one another.
    print_progress : bool, optional
        If True, print a progress percentage, default is False.

    Returns
    -------
    dict of int : list of int
        Each key is an index of `vec_a`. Each values is a list of indices
        which index `vec_b` vectors that are parallel or anti-parallel to the
        `vec_a` vector indexed by the key.

    """

    parallel_idx = {}
    skip_idx = []
    optimise = False
    num_iter = vec_a.shape[1]

    if vec_b is None:
        vec_b = vec_a
        optimise = True

    for v_i_idx, v_i in enumerate(vec_a.T[:, np.newaxis]):

        if v_i_idx in skip_idx:
            continue

        if print_progress:
            print('{:6.2f}% complete'.format(
                100 * v_i_idx / num_iter), end='\r')

        start_idx = v_i_idx + 1 if optimise else 0
        xx = np.cross(vec_b[:, start_idx:], v_i, axisa=0, axisc=0)
        aa = np.all(np.isclose(xx, 0), axis=0)
        ww = np.where(aa)[0] + start_idx

        if len(ww) > 1 or (optimise and len(ww) > 0):

            ww_l = list(ww)
            parallel_idx.update({v_i_idx: ww_l})

            if optimise:
                skip_idx.extend(ww_l)

    if print_progress:
        print('100.00% complete')

    return parallel_idx
