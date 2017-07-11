import os
import re
import pickle


def find_files_in_dir(dir_path, match_regex, recursive=False):
    """ 
    Return paths (relative to `dir_path`) of file names in `dir_path` which 
    match the regular expression given by `match_regex`. If `recursive` is
    True, search all subdirectories.

    Returns a list of strings.

    """

    matched_files = []

    for idx, (root, _, files) in enumerate(os.walk(dir_path)):

        for f in files:

            if re.search(match_regex, f):

                if idx == 0:
                    matched_path = f

                else:
                    matched_path = os.path.join(root.split(os.path.sep)[-1], f)

                matched_files.append(matched_path)

        if not recursive:
            break

    return matched_files


def write_pickle(obj, file_path):
    """ Write `obj` to a pickle file at `file_path`. """

    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def read_pickle(file_path):
    """ Return object in pickle file given by `file_path`. """

    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj


def format_args_check(**kwargs):
    """
    Check types of parameters used in `format_arr`, 'format_list' and 
    'format_dict' functions.

    """

    if 'depth' in kwargs and not isinstance(kwargs['depth'], int):
        raise ValueError('`depth` must be an integer.')

    if 'indent' in kwargs and not isinstance(kwargs['indent'], str):
        raise ValueError('`indent` must be a string.')

    if 'col_delim' in kwargs and not isinstance(kwargs['col_delim'], str):
        raise ValueError('`col_delim` must be a string.')

    if 'row_delim' in kwargs and not isinstance(kwargs['row_delim'], str):
        raise ValueError('`row_delim` must be a string.')

    if 'dim_delim' in kwargs and not isinstance(kwargs['dim_delim'], str):
        raise ValueError('`dim_delim` must be a string.')

    if 'format_spec' in kwargs and not isinstance(kwargs['format_spec'], str):
        raise ValueError('`format_spec` must be a string.')

    if 'assign' in kwargs:

        if not isinstance(kwargs['assign'], str):
            raise ValueError('`assign` must be a string.')


def format_arr(arr, depth=0, indent='\t', col_delim='\t', row_delim='\n',
               dim_delim='\n', format_spec='{}'):
    """
    Get a string representation of a Numpy array, formatted with indents.

    Parameters
    ----------
    arr : ndarray
        Array of any shape to format as a string.
    depth : int, optional
        The indent depth at which to begin the formatting.
    indent : str, optional
        The string used as the indent. The string which indents each line of
        the array is equal to (`indent` * `depth`).
    col_delim : str, optional
        String to delimit columns (the innermost dimension of the array).
        Default is tab character, \t.
    row_delim : str, optional
        String to delimit rows (the second-innermost dimension of the array).
        Defautl is newline character, \n.
    dim_delim : str, optional
        String to delimit outer dimensions. Default is newline character, \n.
    format_spec : str, optional
        Format specifier for the array.

    Returns
    -------
    str

    """

    format_args_check(depth=depth, indent=indent, col_delim=col_delim,
                      row_delim=row_delim, dim_delim=dim_delim,
                      format_spec=format_spec)

    if not isinstance(arr, np.ndarray):
        raise ValueError('Cannot format as array, object is '
                         'not an array: type is {}'.format(type(arr)))

    out = ''
    dim_seps = ''
    d = arr.ndim
    if d == 1:
        out += (indent * depth)
        for col in arr:
            out += format_spec.format(col) + col_delim
        out += row_delim

    else:
        if d > 2:
            dim_seps = dim_delim * (d - 2)
        out = dim_seps.join(
            [format_arr(i, depth, indent, col_delim,
                        row_delim, dim_delim, format_spec) for i in arr])

    return out


def format_list(lst, depth=0, indent='\t', assign='=', arr_kw=None):
    """
    Get a string representation of a nested list, formatted with indents.

    Parameters
    ----------
    lst : list
        List to format as a string. The list may contain other nested lists,
        nested dicts and Numpy arrays.
    depth : int, optional
        The indent depth at which to begin the formatting.
    indent : str, optional
        The string used as the indent. The string which indents each line is
        equal to (`indent` * `depth`).
    assign : str, optional
        The string used to represent the assignment operator.
    arr_kw : dict, optional
        Array-specific keyword arguments to be passed to `format_arr`. (See 
        `format_arr`)

    Returns
    -------
    str

    """

    if arr_kw is None:
        arr_kw = {}

    format_args_check(depth=depth, ident=indent, assign=assign)

    # Disallow some escape character in `assign` string:
    assign = assign.replace('\n', r'\n').replace('\r', r'\r')

    out = ''
    for elem in lst:
        if isinstance(elem, dict):
            out += (indent * depth) + '{\n' + \
                format_dict(elem, depth + 1, indent, assign, arr_kw) + \
                (indent * depth) + '}\n\n'

        elif isinstance(elem, list):
            out += (indent * depth) + '[\n' + \
                format_list(elem, depth + 1, indent, assign, arr_kw) + \
                (indent * depth) + ']\n\n'

        elif isinstance(elem, np.ndarray):
            out += (indent * depth) + '*[\n' + \
                format_arr(elem, depth + 1, indent, **arr_kw) + \
                (indent * depth) + ']\n\n'

        elif isinstance(elem, (int, float, str)):
            out += (indent * depth) + '{}\n'.format(elem)

        else:
            out += (indent * depth) + '{!r}\n'.format(elem)

    return out


def format_dict(d, depth=0, indent='\t', assign='=', arr_kw=None):
    """
    Get a string representation of a nested dict, formatted with indents.

    Parameters
    ----------
    d : dict
        Dict to format as a string. The dict may contain other nested dicts,
        nested lists and Numpy arrays.
    depth : int, optional
        The indent depth at which to begin the formatting
    indent : str, optional
        The string used as the indent. The string which indents each line is
        equal to (`indent` * `depth`).
    assign : str, optional
        The string used to represent the assignment operator.
    arr_kw : dict, optional
        Array-specific keyword arguments to be passed to `format_arr`. (See 
        `format_arr`)

    Returns
    -------
    str

    """

    if arr_kw is None:
        arr_kw = {}

    format_args_check(depth=depth, indent=indent, assign=assign)

    # Disallow some escape character in `assign` string:
    assign = assign.replace('\n', r'\n').replace('\r', r'\r')

    out = ''
    for k, v in sorted(d.items()):

        if isinstance(v, dict):
            out += (indent * depth) + '{} '.format(k) + assign + ' {\n' + \
                format_dict(v, depth + 1, indent, assign, arr_kw) + \
                (indent * depth) + '}\n\n'

        elif isinstance(v, list):
            out += (indent * depth) + '{} '.format(k) + assign + ' [\n' + \
                format_list(v, depth + 1, indent, assign, arr_kw) + \
                (indent * depth) + ']\n\n'

        elif isinstance(v, np.ndarray):
            out += (indent * depth) + '{} '.format(k) + assign + ' *[\n' + \
                format_arr(v, depth + 1, indent, **arr_kw) + \
                (indent * depth) + ']\n\n'

        elif isinstance(v, (int, float, str)):
            out += (indent * depth) + '{} '.format(k) + \
                assign + ' {}\n'.format(v)

        else:
            out += (indent * depth) + '{} '.format(k) + \
                assign + ' {!r}\n'.format(v)

    return out
