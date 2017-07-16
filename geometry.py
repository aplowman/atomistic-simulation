import numpy as np
import vectors


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

    # Compare inverted points to original points
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
