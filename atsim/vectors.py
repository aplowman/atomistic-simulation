import numpy as np
from numpy import linalg as la
from itertools import combinations


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
        angles = np.deg2rad(angles)

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
    arr : ndarray or list
        Array or list of any shape whose elements along its first dimension are
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

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

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


def find_unique_int_vecs(s):
    """
    Find non-collinear integer vectors within an origin-centered cube of given
    size.

    The zero vector is excluded.

    Parameters
    ----------
    s : int
        Size of half the cube edge, such that vectors have maximum component
        |s|.

    Returns
    -------
    ndarray
        Array of row vectors.

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

    s_i = np.zeros((2 * s) + 1, dtype=int)
    s_i[1::2] = np.arange(1, s + 1)
    s_i[2::2] = -np.arange(1, s + 1)

    a = np.vstack(np.meshgrid(s_i, s_i, s_i)).reshape((3, -1)).T
    a[:, [0, 1]] = a[:, [1, 0]]

    # Remove the zero vector
    a = a[1:]

    # Use cross product to find which vectors are collinear
    c = np.cross(a, a[:, np.newaxis])
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


def get_vec_distances(vecs):
    """
    Find the Euclidean distances between all unique column vector
    pairs.

    Parameters
    ----------
    vecs : ndarray of shape (3, N)

    Returns
    -------
    ndarray of shape (N, )

    """

    # Get indices of unique pairs:
    idx = np.array(list(combinations(range(vecs.shape[1]), 2))).T
    b = vecs.T[idx]
    dist = np.sqrt(np.sum((b[0] - b[1])**2, axis=1))

    return dist


def rotate_2D(vec, angle):
    """Rotate a 2D column vector."""

    c = np.cos(angle)
    s = np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    vec_r = np.dot(r, vec)

    return vec_r


def find_eq_row(arr):
    """
    Get the indices of equal rows in a 2D array.

    Returns
    -------
    list of list
        Each inner list represents the row indices which correspond to an identical row.

    """
    if arr.ndim != 2:
        raise ValueError('Array must have 2 dimensions.')
    eq_idx = get_equal_indices(arr)[0]
    return [[k] + v for k, v in eq_idx.items()]


def find_eq_col(arr):
    """
    Get the indices of equal columns in a 2D array.

    Returns
    -------
    list of list
        Each inner list represents the column indices which correspond to an identical column.

    """
    if arr.ndim != 2:
        raise ValueError('Array must have 2 dimensions.')
    eq_idx = get_equal_indices(arr.T)[0]
    return [[k] + v for k, v in eq_idx.items()]


def num_equal_rows(arr):
    """
    Find the number of equal rows in a 2D array.

    Returns
    -------
    tuple
        Each element represents the number of times a unique row is repeated

    """
    return [len(i) for i in find_eq_row(arr)]


def num_equal_cols(arr):
    """
    Find the number of equal columns in a 2D array.

    Returns
    -------
    tuple
        Each element represents the number of times a unique column is repeated

    """
    return [len(i) for i in find_eq_col(arr)]
