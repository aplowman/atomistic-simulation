import random
import time
import numpy as np
import copy
import collections
import subprocess
import dropbox
import fnmatch
import os
import posixpath
import ntpath
from atsim import readwrite

"""
TODO:
-   Add/tidy up docs for these utility functions. May not need them all for the
    cases where the function body is a single line.
"""


def get_first_not_none(a):
    """ Given a list `a` return the first value that is not None, else return None if all values are None."""

    try:
        return next(item for item in a if item is not None)

    except StopIteration:
        return None


def get_all_not_none(a):
    """ Given a list `a` return all elements that are not None."""

    return [i for i in a if i is not None]


def check_all_equal(x):
    """ Checks if all elements in a sequence (list, tuple or string) are equal."""

    if len(x) > 0:
        return x.count(x[0]) == len(x)
    else:
        return True


def check_all_unique(x):
    """ Checks if all elements in a sequence (list, tuple or string) are unique."""

    return len(set(x)) == len(x)


def check_elems_equal_to(a, b, ignore_none=True):
    """ Given a list of numbers `a`, and a dict `b` whose keys are some indices of `a`, check that for each list index in `b`,
        the element is equal to the value in `b`

        Returns bool True if all list elements indexed in `b` are equal to the values in `b`, else returns False

    """
    for k, v in b.items():
        if (ignore_none and a[k] is not None) or (not ignore_none):
            if a[k] != v:
                return False

    return True


def check_elems_not_equal_to(a, b, ignore_none=True):
    """ Given a list of numbers `a`, and a dict `b` whose keys are some indices of `a`, check that for each list index in `b`,
        the element is not equal to the value in `b`

        Returns bool True if all list elements indexed in `b` are not equal to the values in `b`, else returns False

    """
    for k, v in b.items():
        if (ignore_none and a[k] is not None) or (not ignore_none):
            if a[k] == v:
                return False

    return True


def check_equal_elems(a, b):
    """ Given a list `a` and a list `b` whose elements are lists of indices of `a`, check that for each sublist of indices in `b`, the elements in `a` are equal.

        Returns bool True if the elements of `a` indexed by each sublist in `b` are equal to each other, else returns False.

    """

    for equal_idices in b:

        a_sublist = [a[x] for x in equal_idices if a[x] is not None]

        if not check_all_equal(a_sublist):
            return False

    return True


def check_unique_elems(a, b):
    """ Given a list `a` and a list `b` whose elements are lists of indices of `a`, check that for each sublist of indices in `b`, the elements in `a` are unique.

        Returns bool True if the elements of `a` indexed by each sublist in `b` are unique, else returns False.

    """

    for equal_idices in b:

        a_sublist = [a[x] for x in equal_idices if a[x] is not None]

        if not check_all_unique(a_sublist):
            return False

    return True


def repeat_elems_idx(a):
    """ Given a list of lists `a`, return a dict where each key is each unique
        element in `a`, and each value is a list of sublist indices of that element.

        E.g. a = [[1,1,2], [2,3], [4]] => {1:[0], 2:[0,1], 3:[1], 4:[2]}

    """

    indices = {}

    for sblist_idx, sblist in enumerate(a):

        for i in sblist:

            if indices.get(i) is not None:

                if sblist_idx not in indices[i]:
                    indices[i].append(sblist_idx)
            else:
                indices.update({i: [sblist_idx]})

    return indices


def validate_numeric_params(params, equal_to={}, not_equal_to={}, equal_groups=[], unique_groups=[], defaults={}):
    """ Validates a list of numeric parameters according to four types of rules:

        1. `equal_to` is a dictionary whose keys are `params` indices and values are the values
            which those params must be equal to.
        2. `not_equal_to` is a dictionary whose keys are `params` indices and values are the values
            which those params must not be equal to.
        3. `equal_groups` is a list of lists, where each sublist is a group of `params` indices which must all
            index the same value.
        4. `unique_groups` is a list of lists, where each sublist is a group of `params` indices which must all
            index different values.

        Some of `params` may be None, in which case values are assigned by index according to the dictionary `defaults`,
        unless this conflicts with a previously assigned parameter.

        Returns a list of the same length as `params` whose elements are validated `params`.

        Raises ValueError if any validation step fails.

        To-do:
            * Better ValueError messages.
            * maybe change name to validate_param_group() since I think it
              should work for strings as well.

    """

    # If the same index appears in more than one sublist of `equal_groups`, combine those sublists
    for i in list(repeat_elems_idx(equal_groups).values()):
        if len(i) > 1:
            merge_sublists(equal_groups, i)

    # Check equal_to, not_equal_to, equal_groups and unique_groups conditions, with supplied parameters, ignoring any None params

    if not check_elems_equal_to(params, equal_to):
        raise ValueError('Validation failure.')

    if not check_elems_not_equal_to(params, not_equal_to):
        raise ValueError('Validation failure.')

    if not check_equal_elems(params, equal_groups):
        raise ValueError('Validation failure.')

    if not check_unique_elems(params, unique_groups):
        raise ValueError('Validation failure.')

    validated_params = list(params)  # copy the list

    # Assign values to parameters assigned to None according to rules and defaults
    if len(defaults) > 0:

        defaults_list = [None] * len(params)
        for d_idx, d in defaults.items():
            defaults_list[d_idx] = d

        # Verify specified defaults conform to specified validation rules!
        try:
            validate_numeric_params(
                defaults_list, equal_to=equal_to, not_equal_to=not_equal_to,
                equal_groups=equal_groups, unique_groups=unique_groups)

        except ValueError:
            raise ValueError('Specified defaults do not conform to specified'
                             'validation rules.')

    # Assign params that are assigned to None, if possible
    for params_idx in range(len(params)):

        if validated_params[params_idx] is not None:
            continue

        equal_val_indices = [
            el for group in equal_groups for el in group if params_idx in group and el != params_idx]
        equal_group_val = get_first_not_none(
            [validated_params[i] for i in range(len(params)) if i in equal_val_indices])

        new_val = None

        # if this parameter is in an equal group, then must set it to that value
        if equal_group_val is not None:
            new_val = equal_group_val

        # if this parameter is not in an equal group, check which values is must not be:
        else:
            unique_val_indices = [
                el for group in unique_groups for el in group if params_idx in group and el != params_idx]
            unique_group_vals = get_all_not_none(
                [validated_params[i] for i in range(len(params)) if i in unique_val_indices])

            if not_equal_to.get(params_idx):
                unique_group_vals.append(not_equal_to[params_idx])

            # if the parameter has a default which is not a value that is forbidden, set it to the default.
            # we need to recheck if the default is allowed since previous parameters may have changed
            if defaults.get(params_idx):
                if defaults.get(params_idx) not in unique_group_vals:
                    new_val = defaults[params_idx]

        validated_params[params_idx] = new_val

    return validated_params


def get_date_time_stamp(split=False, num_only=False):
    """
    Get a string representing the datetime plus a five digit random number.

    """
    num = '{:05d}'.format(random.randint(1, 1E5))
    date_time = time.strftime('%Y-%m-%d-%H%M')

    if split:
        return date_time, num
    else:
        return date_time + '_' + num if num_only == False else num


def nest_lists(my_list):
    """
        `a` is a list of `N` sublists.

        E.g.
        my_list = [
            [1,2],
            [3,4,5],
            [6,7]
        ]

        returns a list of lists of length `N` such that all combinations of elements from sublists in
        `a` are found
        E.g
        out = [
            [1, 3, 6],
            [1, 3, 7],
            [1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 3, 6],
            [2, 3, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7]
        ]

    """

    N = len(my_list)
    sub_len = [len(i) for i in my_list]

    products = np.array([1] * (N + 1))
    for i in range(len(my_list) - 1, -1, -1):
        products[:i + 1] *= len(my_list[i])

    out = [[None for x in range(N)] for y in range(products[0])]

    for row_idx, row in enumerate(out):

        for col_idx, col in enumerate(row):

            num_repeats = products[col_idx + 1]
            sub_list_idx = int(row_idx / num_repeats) % len(my_list[col_idx])
            out[row_idx][col_idx] = copy.deepcopy(
                my_list[col_idx][sub_list_idx])

    return out


def combine_list_of_dicts(a):

    a = copy.deepcopy(a)

    for i in range(1, len(a)):
        update_dict(a[0], a[i])

    return a[0]


def update_dict(d, u):
    """ Update an arbitrarily-nested dict."""

    for k, v in u.items():
        if isinstance(d, collections.Mapping):
            if isinstance(v, collections.Mapping):
                r = update_dict(d.get(k, {}), v)
                d[k] = r
            else:
                d[k] = u[k]
        else:
            d = {k: u[k]}

    return d


def transpose_list(a):
    return [list(x) for x in zip(*a)]


def parse_as_int_arr(arr):
    """Take a list or array and return an int array"""

    if isinstance(arr, list):
        arr = np.array(arr)

    if isinstance(arr, np.ndarray):

        if np.allclose(np.mod(arr, 1), 0):
            arr = arr.astype(int)

        else:
            raise ValueError('`arr` cannot be parsed as an int array.')

    else:
        raise ValueError('`arr` is not a list or array.')

    return arr


def confirm(prompt=None, resp=False):
    """
    Prompts for yes or no response from the user, returning True for yes and
    False for no.

    Parameters
    ----------
    prompt : str
        The prompt to show the user. Default is 'Confirm'
    resp : bool
        The default response if the user types `Enter`.

    Returns
    -------
    bool

    """

    pos = ['y', 'Y']
    neg = ['n', 'N']

    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '{} [{}]|{}: '.format(prompt, 'y', 'n')
    else:
        prompt = '{} [{}]|{}: '.format(prompt, 'n', 'y')

    while True:

        ans = input(prompt)

        if not ans:
            return resp

        if ans not in (pos + neg):
            print('Please enter y or n.')
            continue

        if ans in pos:
            return True
        elif ans in neg:
            return False


def dir_exists_remote(host, dir_path):
    comp_proc = subprocess.run(
        ['bash', '-c', 'ssh {} "[ -d {} ]"'.format(host, dir_path)])

    if comp_proc.returncode == 0:
        return True
    elif comp_proc.returncode == 1:
        return False


def rsync_remote(src, host, dst, exclude=None, include=None):
    """
    Execute an rsync command to a remote host.

    Parameters
    ----------
    exclude : list
        List of strings to pass to --exclude option. Cannot be used with
        `include`.
    include : list
        List of strings to pass to --include option. If this is specified,
        only matching paths with by copied and all subdirectories will be
        traversed. Cannot be used with `exclude`.
    """

    # Validation
    if exclude is not None and include is not None:
        raise ValueError('Cannot specify both `include` and `exclude`.')

    in_ex_str = ''
    if exclude is not None:
        in_ex_str = ''.join([' --exclude="{}"'.format(i) for i in exclude])

    elif include is not None:
        in_ex_str = ''.join([' --include="{}"'.format(i) for i in include])
        in_ex_str += ' --include="*/" --exclude="*"'

    rsync_cmd = ('rsync -az --chmod=Du=rwx,Dgo=rx,Fu=rw,Fog=r{}'
                 ' "{}" {}:{}').format(in_ex_str, src, host, dst)
    subprocess.run(['bash', '-c', rsync_cmd])


def get_idx_arr(arr):
    """
    Return an array which indexes all elements in a given array.

    Parameters
    ----------
    arr : ndarray of dimension N or list

    Returns
    -------
    ndarray of shape (N.ndim, N.size)

    """

    if isinstance(arr, list):
        arr = np.array(arr)

    idx = np.indices(arr.shape)
    flt = arr.flatten()

    idx_i = [idx[i].flatten() for i in range(idx.shape[0])]
    all_idx = np.vstack(idx_i)

    return all_idx


def format_time(secs):
    """Format a time in seconds into days, hours, minutes and seconds."""

    d = secs / (24 * 60 * 60)
    h = (d - np.floor(d)) * 24
    m = (h - np.floor(h)) * 60
    s = np.round((m - np.floor(m)) * 60, decimals=0)
    t = [int(np.floor(i)) for i in [d, h, m]]
    time_strs = ['days', 'hrs', 'mins']
    time_fmt = ''.join(['{} {} '.format(i, j)
                        for i, j in zip(t, time_strs) if i > 0])
    time_fmt += '{:.0f} sec'.format(s)

    return time_fmt


def trim_common_nones(a, b, ret_idx=False):
    """
    Trim two equal-length lists from both ends by removing common None values.

    Parameters
    ----------
    a : list of length N
    b : list of length N

    """

    if len(a) != len(b):
        raise ValueError('Lengths of lists `a` ({}) and `b` ({}) must be '
                         'equal.'.format(len(a), len(b)))

    a_none_idx = [i_idx for i_idx, i in enumerate(a) if i is None]
    b_none_idx = [i_idx for i_idx, i in enumerate(b) if i is None]

    # Remove common `None`s from start of lists:
    trim_idx = []

    if len(a_none_idx) == 0 or len(b_none_idx) == 0:
        return

    a_n = a_none_idx[0]
    b_n = b_none_idx[0]
    left_idx = 0
    c = 0
    while a_n == b_n and a_n == left_idx:

        trim_idx.append(a_n)
        if c == len(a_none_idx) - 1 or c == len(b_none_idx) - 1:
            break

        c += 1
        left_idx += 1
        a_n = a_none_idx[c]
        b_n = b_none_idx[c]

    a_n = a_none_idx[-1]
    b_n = b_none_idx[-1]
    right_idx = len(a) - 1
    c = 0
    while a_n == b_n and a_n == right_idx:

        trim_idx.append(a_n)
        if c == len(a_none_idx) - 1 or c == len(b_none_idx) - 1:
            break

        c += 1
        right_idx -= 1
        a_n = a_none_idx[-1 - c]
        b_n = b_none_idx[-1 - c]

    a[:] = [i for i_idx, i in enumerate(a) if i_idx not in trim_idx]
    b[:] = [i for i_idx, i in enumerate(b) if i_idx not in trim_idx]

    if ret_idx:
        return trim_idx


def dict_from_list(lst, conditions, false_keys=None, ret_index=False):
    """
    Get the first dict from a list of dict given one or more matching
    key-values.

    Parameters
    ----------
    lst : list
    conditions : dict
        To return a dict from the list, keys and values specified here must
        exist in the list dict.
    false_keys : list, optional
        Dicts which have keys listed here will not be returned.
    ret_index : bool, optional
        If True, return a tuple (element_index, element) else return element.

    """

    if false_keys is None:
        false_keys = []

    for el_idx, el in enumerate(lst):

        for cnd_key, cnd_val in conditions.items():

            v = el.get(cnd_key)

            if v is not None:

                if any([isinstance(i, np.ndarray) for i in [v, cnd_val]]):

                    v_arr = np.array(v)
                    cnd_val_arr = np.array(cnd_val)

                    if v_arr.shape != cnd_val_arr.shape:
                        condition_match = False
                        break

                    if np.allclose(v, cnd_val):
                        condition_match = True

                    else:
                        condition_match = False
                        break

                elif v == cnd_val:
                    condition_match = True
                else:
                    condition_match = False
                    break
            else:
                condition_match = False
                break

        if condition_match:

            skip = False
            for fkey in false_keys:
                if fkey in el:
                    skip = True
                    break
            if skip:
                break

            if ret_index:
                return (el_idx, el)
            else:
                return el

    if ret_index:
        return (None, None)
    else:
        return None


def get_bash_path(path, end_path_sep=False):
    """Get the path in a posix style, e.g. for using with bash commands in
    Windows Subsystem for Linux.

    This replaces drives letters specified like "C:\foo" with
    "/mnt/c/foo".

    Parameters
    ----------
    end_path_sep : bool, optional
        Specifies whether the returned path should end in path separator.

        Default is False.
    """

    drv, pst_drv = os.path.splitdrive(path)
    path_bash = posixpath.sep.join(
        [posixpath.sep + 'mnt', drv[0].lower()] +
        pst_drv.strip(ntpath.sep).split(ntpath.sep))

    if end_path_sep:
        path_bash += posixpath.sep

    return path_bash


def get_col(a, col_idx):
    """Return a column in a list of lists"""
    return [row[col_idx] for row in a]


def get_col_none(a, col_idx):
    b = []
    for row_idx, row in enumerate(a):
        try:
            b.append(row[col_idx])
        except:
            b.append(None)
    return b


def index_lst(lst, idx, not_idx=False):
    """Return indexed elements of a list."""
    if not_idx:
        return [i for i_idx, i in enumerate(lst) if i_idx not in idx]
    else:
        return [i for i_idx, i in enumerate(lst) if i_idx in idx]


def arguments(print_args=True):
    """Returns tuple containing dictionary of calling function's
        named arguments and a list of calling function's unnamed
        positional arguments.
    """
    from inspect import getargvalues, stack
    posname, kwname, args = getargvalues(stack()[1][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(args.pop(kwname, []))
    if print_args:
        print('args: \n{}\n'.format(readwrite.format_dict(args)))
        print('posargs: \n{}\n'.format(readwrite.format_list(posargs)))
    return args, posargs


def prt(obj, name):
    if isinstance(obj, np.ndarray):
        print('{} {} {}: \n{}\n'.format(name, obj.shape, obj.dtype, obj))
    else:
        print('{}: \n{}\n'.format(name, obj))


def flatten_dict_keys(d, base_k=None, delim='.'):

    flat_d = {}
    for k, v in d.items():

        if base_k:
            new_k = base_k + delim + k
        else:
            new_k = k

        if isinstance(v, dict):
            flat_d.update(get_flat_dict(v, new_k))

        else:
            flat_d.update({new_k: v})

    return flat_d


def unflatten_dict_keys(d, delim='.'):

    unflat_d = {}
    for k, v, in d.items():

        if delim not in k:
            unflat_d.update({k: v})

        else:

            k_split = k.split(delim)
            sub_d = unflat_d

            for ks_idx in range(len(k_split)):

                ks = k_split[ks_idx]
                prev_sub_d = sub_d
                sub_d = sub_d.get(ks)

                if sub_d is None:

                    if ks_idx == len(k_split) - 1:
                        prev_sub_d.update({ks: v})
                        break

                    else:
                        prev_sub_d.update({ks: {}})
                        sub_d = prev_sub_d[ks]

    return unflat_d


def nan_to_none(arr):
    """
    Convert a Numpy array to a (nested) list with np.nan replaced by None

    Notes
    -----
    The inverse can be done by specifying the dtype: np.array(lst, dtype=float)
    """

    none_idx = np.where(np.isnan(arr))
    arr = arr.astype(object)
    arr[none_idx] = None
    return arr.tolist()


def get_unique_idx(a):
    unique = []
    unique_idx = []
    for ai_idx, ai in enumerate(a):
        if ai in unique:
            unique_idx[unique.index(ai)].append(ai_idx)
        elif None in ai:
            continue
        else:
            unique.append(ai)
            unique_idx.append([ai_idx])
    return unique, unique_idx


def get_row_col_idx(idx, nrows, ncols):
    """
    For a grid defined by number of rows and columns, get the row and column indices 
    from a single index which increments first columns and then rows.

    """
    ridx = int(np.floor(idx / ncols))
    cidx = int(idx - (ridx * ncols))
    return ridx, cidx


def get_key_max(lst, key):
    """Get the maximum value of a key in a list of dicts"""
    maxv = None
    for d_idx, d in enumerate(lst):
        if d_idx == 0:
            maxv = d[key]
        elif d[key] > maxv:
            maxv = d[key]
    return maxv


def combination_idx(*seq):
    """
    Find the indices of unique combinations of elements in equal-length
    ordered sequences.

    Parameters
    ----------
    seq : one or more sequences
        All sequences must be of the same length. These may be lists, tuples, 
        strings or ndarrays etc.

    Returns
    -------
    tuple of (list of ndarray, ndarray)
        The list is the unique combinatons (as Numpy object arrays) and the 
        second is the indices for a given combindation.

    """

    # Validation
    seq_len = -1
    msg = 'All sequences must have the same length.'
    for s in seq:
        if seq_len == -1:
            seq_len = len(s)
        else:
            if len(s) != seq_len:
                raise ValueError(msg)

    combined_str = np.vstack(seq)
    combined_obj = np.vstack([np.array(i, dtype=object) for i in seq])

    u, uind, uinv = np.unique(combined_str, axis=1,
                              return_index=True, return_inverse=True)

    ret_k = []
    ret_idx = []
    for i in range(u.shape[1]):

        ret_k.append(combined_obj[:, uind[i]])
        ret_idx.append(np.where(uinv == i)[0])

    return ret_k, ret_idx


def check_indices(seq, seq_idx):
    """
    Given a sequence (e.g. list, tuple, ndarray) which is indexed by another,
    check the indices are sensible.

    Parameters
    ----------
    seq : sequence
    seq_idx : sequence of int

    """

    # Check: minimum index is greater than zero
    if min(seq_idx) < 0:
        raise IndexError('Found index < 0.')

    # Check maximum index is equal to length of sequence - 1
    if max(seq_idx) > len(seq) - 1:
        raise IndexError('Found index larger than seqence length.')


def to_col_vec(a, dim=3):
    a = np.array(a).squeeze().reshape((dim, 1))
    return a
