import os
import re

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