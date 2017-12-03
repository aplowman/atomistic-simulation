import os
import re
import pickle
import numpy as np
import fnmatch
import shutil
import filecmp
import json


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
                    pth = os.path.relpath(root, dir_path).split(os.path.sep)
                    matched_path = os.path.join(*pth, f)

                matched_files.append(matched_path)

        if not recursive:
            break

    return matched_files


def find_files_in_dir_glob(dir_path, match_glob, recursive=False):
    """ 
    Return paths (relative to `dir_path`) of file names in `dir_path` which 
    match glob pattern `match_glob`. If `recursive` is True, search all
    subdirectories.

    Returns a list of strings.

    """

    matched_files = []

    for idx, (root, _, files) in enumerate(os.walk(dir_path)):

        for f in files:

            if fnmatch.fnmatch(f, match_glob):

                if idx == 0:
                    matched_path = f

                else:
                    pth = os.path.relpath(root, dir_path).split(os.path.sep)
                    matched_path = os.path.join(*pth, f)

                matched_files.append(matched_path)

        if not recursive:
            break

    return matched_files


def make_int_dir(path, zero_pad=2):
    """
    Find directories in a path whose names can be parsed as integers and make a
    new directory with a larger integer name.

    Parameters
    ----------
    path : str
        The directory in which to make a new integer-parsable directory.
    zero_pad : int, optional
        The zero-padded width of the new directory name.

    Returns
    -------
    str

    """

    # Check for existing directories:
    cur = [i for i in os.listdir(path)
           if os.path.isdir(os.path.join(path, i))]

    # Keep only integer-parsable directories:
    cur_int = []
    for i in cur:
        try:
            cur_int.append(int(i))
        except:
            pass

    # Make new directory
    new_int = np.max(cur_int) + 1 if len(cur_int) > 0 else 1
    new_dir = ('{{:0{}'.format(zero_pad) + 'd}').format(new_int)
    new_path = os.path.join(path, new_dir)
    os.makedirs(new_path)

    return new_dir


def factor_common_files(path, common_files):
    """
    Search for identical files (name and content) in a directory, extract out
    to a common path, and write a mapping JSON file.

    Parameters
    ----------
    path : str
        Directory in which to search for common files and in which a new
        directory `common_files` will be generated containing factored out
        files.
    common_files : list of str
        File names to include in the search, according to Unix shell-style
        wildcards.

    See Also
    --------
    `unfactor_common_files`

    """

    exclude = ['common_files']

    # First generate a dict containing the directories of files
    # which match `common_files`
    cmn_fls = {}
    for (root, drs, fls) in os.walk(path, topdown=True):

        drs[:] = [d for d in drs if d not in exclude]
        pth = os.path.relpath(root, path).split(os.path.sep)

        for f in fls:
            if any([fnmatch.fnmatch(f, c) for c in common_files]):
                if f in cmn_fls:
                    cmn_fls[f].append(pth)
                else:
                    cmn_fls.update({f: [pth]})

    map_dir_path = os.path.join(path, 'common_files')
    os.makedirs(map_dir_path, exist_ok=True)

    # Now check which are identical and extract out:
    cmn_map = {}
    first_idx = None
    for k, v in cmn_fls.items():

        x = range(len(v))
        skip_idx = []

        for i in x:

            if i in skip_idx:
                continue

            same_idx = [v[i]]

            for j in x[i + 1:]:

                if j in skip_idx:
                    continue

                pth_1 = os.path.join(path, *v[i], k)
                pth_2 = os.path.join(path, *v[j], k)

                if filecmp.cmp(pth_1, pth_2):
                    same_idx.append(v[j])
                    skip_idx.append(j)

            if len(same_idx) > 1:
                new_dir = make_int_dir(map_dir_path, zero_pad=2)
                if first_idx is None:
                    first_idx = new_dir
                cmn_map.update({new_dir: {'filename': k, 'dirs': same_idx}})

                # Factor out common files
                src_fl_path = pth_1
                dst_fl_path = os.path.join(map_dir_path, new_dir, k)
                shutil.copy(src_fl_path, dst_fl_path)

                # Remove common files which have been factored out:
                for s in same_idx:
                    rm_path = os.path.normpath(os.path.join(path, *s, k))
                    os.remove(rm_path)

    if len(cmn_map) > 0:
        last_idx = int(first_idx) + len(cmn_map) - 1
        map_id = '{}-{:02}'.format(first_idx, last_idx)
        map_fn = os.path.join(map_dir_path, map_id + '_map.json')
        with open(map_fn, 'w') as mf:
            json.dump(cmn_map, mf, indent=2, sort_keys=True)


def unfactor_common_files(path):
    """To be used in conjunction with `factor_common_files`."""

    map_dir_path = os.path.join(path, 'common_files')
    if not os.path.isdir(map_dir_path):
        raise ValueError('Cannot find map directory path.')

    map_fls = find_files_in_dir(map_dir_path, '.map\.json')

    for m in map_fls:

        with open(os.path.join(map_dir_path, m), 'r') as mf:
            cmn_map = json.load(mf)

        for k, v in cmn_map.items():

            src_path = os.path.join(map_dir_path, k, v['filename'])

            for p in v['dirs']:
                dst_path = os.path.join(path, *p)
                shutil.copy(src_path, dst_path)

            # Remove maps dir:
            shutil.rmtree(os.path.join(map_dir_path, k))

        # Remove JSON map file:
        os.remove(os.path.join(map_dir_path, m))

    # Remove common_files dir
    os.rmdir(map_dir_path)


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

    if 'format_spec' in kwargs and not isinstance(kwargs['format_spec'],
                                                  (str, list)):
        raise ValueError('`format_spec` must be a string or list of strings.')

    if 'assign' in kwargs:

        if not isinstance(kwargs['assign'], str):
            raise ValueError('`assign` must be a string.')


def format_arr(arr, depth=0, indent='\t', col_delim='\t', row_delim='\n',
               dim_delim='\n', format_spec='{}'):
    """
    Get a string representation of a Numpy array, formatted with indents.

    Parameters
    ----------
    arr : ndarray or list of ndarray
        Array of any shape to format as a string, or list of arrays whose
        shapes match except for the final dimension, in which case the arrays
        will be formatted horizontally next to each other.
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
    format_spec : str or list of str, optional
        Format specifier for the array or a list of format specifiers, one for 
        each array listed in `arr`.

    Returns
    -------
    str

    """

    # Validation:
    format_args_check(depth=depth, indent=indent, col_delim=col_delim,
                      row_delim=row_delim, dim_delim=dim_delim,
                      format_spec=format_spec)

    if isinstance(arr, np.ndarray):
        arr = [arr]

    out_shape = list(set([i.shape[:-1] for i in arr]))

    if len(out_shape) > 1:
        raise ValueError('Array shapes must be identical apart from the '
                         'innermost dimension.')

    if not isinstance(arr, (list, np.ndarray)):
        raise ValueError('Cannot format as array, object is '
                         'not an array or list of arrays: type is {}'.format(
                             type(arr)))

    if isinstance(format_spec, str):
        format_spec = [format_spec] * len(arr)

    elif isinstance(format_spec, list):

        fs_err_msg = ('`format_spec` must be a string or list of N strings '
                      'where N is the number of arrays specified in `arr`.')

        if not all([isinstance(i, str)
                    for i in format_spec]) or len(format_spec) != len(arr):
            raise ValueError(fs_err_msg)

    arr_list = arr
    out = ''
    dim_seps = ''
    d = arr_list[0].ndim

    if d == 1:
        out += (indent * depth)

        for sa_idx, sub_arr in enumerate(arr_list):
            for col_idx, col in enumerate(sub_arr):
                out += format_spec[sa_idx].format(col)
                if (col_idx < len(sub_arr) - 1):
                    out += col_delim

        out += row_delim

    else:

        if d > 2:
            dim_seps = dim_delim * (d - 2)

        sub_arr = []
        for i in range(out_shape[0][0]):

            sub_arr_lst = []
            for j in arr_list:
                sub_arr_lst.append(j[i])

            sub_arr.append(format_arr(sub_arr_lst, depth, indent, col_delim,
                                      row_delim, dim_delim, format_spec))

        out = dim_seps.join(sub_arr)

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
                (indent * depth) + '}\n'

        elif isinstance(elem, list):
            out += (indent * depth) + '[\n' + \
                format_list(elem, depth + 1, indent, assign, arr_kw) + \
                (indent * depth) + ']\n'

        elif isinstance(elem, np.ndarray):
            out += (indent * depth) + '*[\n' + \
                format_arr(elem, depth + 1, indent, **arr_kw) + \
                (indent * depth) + ']\n'

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
                (indent * depth) + '}\n'

        elif isinstance(v, list):
            out += (indent * depth) + '{} '.format(k) + assign + ' [\n' + \
                format_list(v, depth + 1, indent, assign, arr_kw) + \
                (indent * depth) + ']\n'

        elif isinstance(v, np.ndarray):
            out += (indent * depth) + '{} '.format(k) + assign + ' *[\n' + \
                format_arr(v, depth + 1, indent, **arr_kw) + \
                (indent * depth) + ']\n'

        elif isinstance(v, (int, float, str)):
            out += (indent * depth) + '{} '.format(k) + \
                assign + ' {}\n'.format(v)

        else:
            out += (indent * depth) + '{} '.format(k) + \
                assign + ' {!r}\n'.format(v)

    return out


def write_list_file(path, my_list):
    """Write a list to file with one element per line."""

    with open(path, 'w', newline='') as f:

        for idx, l in enumerate(my_list):

            if idx != 0:
                f.write('\n')

            f.write(l)

        f.write('\n')


def delete_line(file_path, search_str):
    """
    Delete from file a line which contains a search string.

    TODO:
    -   Match string with regex.

    """

    new_file_path = file_path + '_new'

    with open(file_path, 'r', encoding='utf-8', newline='') as orig_file:

        with open(new_file_path, 'w', encoding='utf-8', newline='') as new_file:

            for ln in orig_file:

                if search_str not in ln:
                    new_file.write(ln)

    os.remove(file_path)
    os.rename(new_file_path, file_path)


def replace_in_file(file_path, search_str, replace_str):
    """
    Replace a search string in a file with a new string.

    TODO:
    -   Match string with regex.
    """

    with open(file_path, 'r', encoding='utf-8', newline='') as orig_file:
        file_data = orig_file.read()

    file_data = file_data.replace(search_str, replace_str)

    with open(file_path, 'w', encoding='utf-8', newline='') as new_file:
        new_file.write(file_data)


def add_line(file_path, line_idx, line):
    """
    Add a line to a file at a specified line index, pushing subsequent lines
    down by one.

    Parameters
    ---------
    file_path : str
    line_idx : int
    line : str

    Returns
    -------
    None

    """

    # Validation
    if not isinstance(line_idx, int):
        raise ValueError('`line_idx` must be an integer.')

    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        lns = f.readlines()

    lns.insert(line_idx, line + '\n')

    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        lns = ''.join(lns)
        f.write(lns)
