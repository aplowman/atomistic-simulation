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

    sinθ = la.norm(np.cross(a, b, axis=0), axis=0) / (la.norm(a, axis=0) * la.norm(b, axis=0))
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
    out[abs(arr-val) < tol] = val

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

    p = vec - (np.einsum('ij, ik->j', vec, plane_normal) / la.norm(plane_normal)**2) * plane_normal

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
