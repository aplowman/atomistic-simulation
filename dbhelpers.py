import dropbox


def get_dropbox(key):
    return dropbox.Dropbox(key)


def check_dropbox_file_exist(dbx, dropbox_path):
    try:
        dbx.files_get_metadata(dropbox_path)
        return True
    except:
        return False


def download_dropbox_file(dbx, dropbox_path, local_path):
    dbx.files_download_to_file(local_path, dropbox_path)


def upload_dropbox_file(dbx, local_path, dropbox_path, overwrite=False,
                        autorename=False):
    """
    Parameters
    ----------
    dbx: Dropbox
    local_path : str
        Path of file on local computer to upload to dropbox.
    dropbox_path : str
        Directory on dropbox to upload the file to.
    overwrite : bool
        If True, the file overwrites an existing file with the same name.
    autorename : bool
        If True, rename the file if there is a conflict.

    """

    if overwrite:
        mode = dropbox.dropbox.files.WriteMode('overwrite', None)
    else:
        mode = dropbox.dropbox.files.WriteMode('add', None)

    with open(local_path, mode='rb') as f:

        dbx.files_upload(f.read(), dropbox_path,
                         mode=mode, autorename=autorename)


def upload_dropbox_dir(dbx, local_path, dropbox_path, overwrite=False,
                       autorename=False, include=None, exclude=None):
    """
    Parameters
    ----------
    dbx: Dropbox
    local_path : str
        Path of file on local computer to upload to dropbox.
    dropbox_path : str
        Directory on dropbox to upload the file to.
    overwrite : bool
        If True, the file overwrites an existing file with the same name.
    autorename : bool
        If True, rename the file if there is a conflict.
    include : list, optional
        List of file names to include, matched with `fnmatch`. If specified,
        only matched file names will be uploaded. Either specify `include` or
        `exclude` but not both.
    exclude : list, optional
        List of file names to exclude, matched with `fnmatch`. Either specify 
        `include` or `exclude` but not both.

    Notes
    -----
    Does not upload empty directories.

    """

    # Validation
    if include is not None and exclude is not None:
        raise ValueError('Either specify `include` or `exclude` but not both.')

    for root, dirs, files in os.walk(local_path):

        for fn in files:

            up_file = False

            if include is not None:
                if any([fnmatch.fnmatch(fn, i) for i in include]):
                    up_file = True

            elif exclude is not None:
                if not any([fnmatch.fnmatch(fn, i) for i in exclude]):
                    up_file = True

            else:
                up_file = True

            if up_file:

                local_path_ps = posixpath.join(*local_path.split(os.path.sep))
                db_path_ps = posixpath.join(
                    *dropbox_path.split(os.path.sep))

                root_ps = posixpath.join(*root.split(os.path.sep))
                fn_local_path = posixpath.join(root_ps, fn)
                rel_path = posixpath.relpath(fn_local_path, local_path_ps)
                fn_db_path = '/' + posixpath.join(db_path_ps, rel_path)

                upload_dropbox_file(dbx, fn_local_path, fn_db_path,
                                    overwrite=overwrite, autorename=autorename)
