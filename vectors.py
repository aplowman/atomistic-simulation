import numpy as np
from numpy import linalg as la


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


def rotation_matrix(axis, angle, degrees=False):
    """
        Generates the rotation matrix to act on column vectors by pre-multiplication
        for a given rotation axis and angle. `axis` is a 1D array of length 3 or a 1 x 3
        or 3 x 1 array. `angle` is the rotation angle in radians (or degrees if degrees
        is True)

    """

    # Remove extra dimensions
    axis = axis.squeeze()

    # Convert to radians if necessary
    if degrees:
        angle = np.deg2rad(angle)

    # Normalise axis to a unit vector:
    axis_unit = axis / la.norm(axis)

    # Find the rotation matrix for a rotation about `axis` by `angle`
    cross_prod_mat = np.zeros((3, 3))
    cross_prod_mat[0][1] = -axis_unit[2]
    cross_prod_mat[0][2] = axis_unit[1]
    cross_prod_mat[1][0] = axis_unit[2]
    cross_prod_mat[1][2] = -axis_unit[0]
    cross_prod_mat[2][0] = -axis_unit[1]
    cross_prod_mat[2][1] = axis_unit[0]

    rot_mat = np.eye(3) + (np.sin(angle) * cross_prod_mat) + (
        (1 - np.cos(angle)) * np.dot(cross_prod_mat, cross_prod_mat))

    return rot_mat


def get_equal_indices(arr, lone_elems=False, scale_factors=None):
    """
    Return the indices along the first dimension of an array which index equal
    sub-arrays.

    Parameters
    ----------
    arr : ndarray
        Array of any shape whose elements along its first dimension are 
        compared for equality.
    lone_elems : bool
        If True, the returned list may include single-element lists 
        representing elements which are not repeated.
    scale_factors : list of float or list of int
        Subarrays which are equal up to any multiplicative factor in this list
        are determined to be equal.

    Returns
    -------
    list of list of int
        A list of lists, each of which contains indices of `arr` which index
        equal subarrays. Each sublist is ordered.

    Examples
    --------

    1D example:

    >>> a = np.random.randint(0, 9, (10))
    [5 1 4 6 1 8 2 7 4 7] # random
    >>> get_equal_indices(a, lone_elems=False)
    [[1, 4], [2, 8], [7, 9]]
    >>> get_equal_indices(a, lone_elems=True)
    [[0], [1, 4], [2, 8], [3], [5], [6], [7, 9]]

    2D example:

    >>> a = np.array([[1., 2.], [3., 4.], [-0.4, -0.8]])
    >>> get_equal_indices(a, scale_factors=[-0.4])
    [[0, 2]]

    """

    if scale_factors is None:
        scale_factors = [1]

    else:
        scale_factors.append(1)

    a_dims = len(arr.shape)
    arr_B = arr[:, np.newaxis]

    sf_shape = tuple([len(scale_factors)] + [1] * (a_dims + 1))
    sf = np.array(scale_factors).reshape(sf_shape)

    bc = np.broadcast_arrays(arr, arr_B, sf)
    c = np.isclose(bc[0] * bc[2], bc[1])

    if a_dims > 1:
        c = np.all(c, axis=tuple(range(3, a_dims + 2)))

    # Check over scale factors:
    c_sf = np.any(c, axis=0)

    # If non-unit scale factors used, array won't be symmetric,
    # so coerce into a symmetric array:
    c_sf_t = np.logical_or(c_sf, c_sf.T)
    w = np.where(c_sf_t)
    all_same_idx = []

    for i in set(w[0]):

        row_idx = np.where(w[0] == i)[0]

        if len(row_idx) == 1 and not lone_elems:
            continue

        same_idx = list(w[1][row_idx])

        if same_idx not in all_same_idx:
            all_same_idx.append(same_idx)

    all_same_idx = [list(i) for i in all_same_idx]

    return all_same_idx


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
