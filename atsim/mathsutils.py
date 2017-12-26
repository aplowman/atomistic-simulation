import numpy as np


def double_sigmoid(x, a, b, width):

    low_x = width * (1 / 4)
    high_x = width * (3 / 4)

    x_low_idx = (x < low_x)
    x_mid_idx = (x >= low_x) & (x < high_x)
    x_high_idx = (x >= high_x)

    x1 = x[x_low_idx]
    x2 = x[x_mid_idx]
    x3 = x[x_high_idx]

    y = np.empty_like(x)

    y[x_low_idx] = sigmoid(x1, a, b, 0, 0.5)
    y[x_mid_idx] = sigmoid(x2, a, b, width / 2, -0.5)
    y[x_high_idx] = sigmoid(x3, a, b, width, -1.5)

    return y


def single_sigmoid(x, a, b, width):

    return sigmoid(x, a, b, width / 2, 0)


def sigmoid(x, a, b, c, d):
    """
        `a` scales curve along y
        `b` sharpness of S bend
        `c` displaces curve along x
        `d` displaces curve along y

    """

    e_arg = -b * (x - c)
    f = np.empty_like(e_arg)

    lo_idx = np.where(e_arg < -50)[0]
    hi_idx = np.where(e_arg < +50)[0]

    lim_lo_idx = e_arg < -50
    lim_hi_idx = e_arg > 50
    lim_idx = np.logical_or(lim_lo_idx, lim_hi_idx)
    not_lim_idx = np.logical_not(lim_idx)

    f[not_lim_idx] = 1 / (1 + np.exp(e_arg[not_lim_idx]))
    f[lim_lo_idx] = 1
    f[lim_hi_idx] = 0

    return a * (f - d)


def fit_quad(x, y):
    """
    Fit x and y data to a quadratic and get the minimum point

    Returns
    -------
    tuple
        poly1d, min_x, min_y

    """

    # Validation
    if len(x) != len(y):
        raise ValueError('x and y must have the same length')
    if len(x) < 3:
        raise ValueError('x and y must be of at least length 3.')

    z = np.polyfit(x, y, 2)
    p1d = np.poly1d(z)
    dpdx = np.polyder(p1d)
    min_x = -dpdx[0] / dpdx[1]
    min_y = p1d(min_x)

    return p1d, min_x, min_y


def linear(x, m=1, c=0):
    return m * x + c


def get_cell_parameters(cell_vectors, degrees=False):
    """
    Find the cell parameters a, b, c, α, β, γ from cell column vectors.

    Parameters
    ----------
    cell_vectors : ndarray of shape (3, 3)
        Array of three column vectors representing the edge vectors
        of a cell.
    degrees : bool, optional
        If True, return the angle parameters α, β and γ in degrees. If
        False, return in radians.

    Returns
    -------
    dict of (str : float)
        Cell parameters.

    """

    if cell_vectors.shape != (3, 3):
        raise ValueError('`cell_vectors` must be an ndarray of shape '
                         '(3, 3) whose columns represent cell edge '
                         'vectors.')

    a, b, c = np.linalg.norm(cell_vectors, axis=0)

    α = np.arccos(np.dot(cell_vectors[:, 1],
                         cell_vectors[:, 2]) / (b * c))
    β = np.arccos(np.dot(cell_vectors[:, 0],
                         cell_vectors[:, 2]) / (a * c))
    γ = np.arccos(np.dot(cell_vectors[:, 0],
                         cell_vectors[:, 1]) / (a * b))

    if degrees:
        α, β, γ = np.rad2deg([α, β, γ])

    ret = {
        'a': a,
        'b': b,
        'c': c,
        'α': α,
        'β': β,
        'γ': γ,
    }

    return ret
