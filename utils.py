import random
import time
import numpy as np
import copy
import collections
import subprocess

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
            out[row_idx][col_idx] = my_list[col_idx][sub_list_idx]

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


def rsync_remote(src, host, dst):
    rsync_cmd = ('rsync -az --chmod=Du=rwx,Dgo=rx,Fu=rw,Fog=r'
                 ' {} {}:{}').format(src, host, dst)
    subprocess.run(['bash', '-c', rsync_cmd])
