import sys
import os
from copy import deepcopy
from distutils.dir_util import copy_tree
import shutil
import subprocess

import atsim.dbhelpers as dbh
from atsim.readwrite import read_pickle, write_pickle, find_files_in_dir_glob, factor_common_files
from atsim import SET_UP_PATH, SCRIPTS_PATH
from atsim.simsio import castep, lammps


def search_database_by_session_id(database, s_id):

    base_opt = None
    for k, v in database.items():

        set_up_opts = v.get('set_up')
        if set_up_opts is not None:
            db_sid = set_up_opts['session_id']
        else:
            db_sid = v['session_id']

        if db_sid == s_id:
            base_opt = v
            break

    return base_opt


def check_errors(sms_path, src_path, skip_idx=None):
    """Some basic checks for errors in sim series output files."""

    # Open the sims pickle, get list of AtomisticSimulation objects:
    sms = read_pickle(sms_path)
    base_opt = sms['base_options']
    all_sms = sms['all_sims']
    method = base_opt['method']

    error_paths = []
    s_count = 0
    for s_idx, sim_i in enumerate(all_sms):

        print('s_idx: {}'.format(s_idx))

        if skip_idx is not None and s_idx in skip_idx:
            continue

        s_count += 1
        srs_paths = []
        srs_id = sim_i.options.get('series_id')
        if srs_id is not None:
            for srs_id_lst in srs_id:
                srs_paths.append('_'.join([i['path'] for i in srs_id_lst]))

        calc_path = os.path.join(src_path, 'calcs', *srs_paths)

        if method == 'castep':
            out = castep.read_castep_output(calc_path)

        elif method == 'lammps':
            out = lammps.read_lammps_output(calc_path)

        if len(out['errors']) > 0:
            error_paths.extend(srs_paths)

    return error_paths


def move_offline_files(s_id, src_path, offline_files):

    arch_dir = offline_files['path']
    fl_types = offline_files.get('file_types') or offline_files.get('match')

    fls_paths = []
    for t in fl_types:
        fls_paths.extend(find_files_in_dir_glob(src_path, t, recursive=True))

    print('Offline files dir: {}'.format(arch_dir))
    print('Offline files types: {}'.format(fl_types))

    fls_paths = []
    for t in fl_types:
        fls_paths.extend(find_files_in_dir_glob(src_path, t, recursive=True))

    print('Offline files paths: {}'.format(fls_paths))

    if len(fls_paths) > 0:

        # Generate the offline files directories
        s_path = os.path.join(arch_dir, s_id)
        os.makedirs(s_path, exist_ok=True)

        for fp in fls_paths:
            dr, fl = os.path.split(fp)
            os.makedirs(os.path.join(s_path, dr), exist_ok=True)

            # Move the offline files to the offline dirs
            fl_src = os.path.join(src_path, fp)
            fl_dst = os.path.join(s_path, fp)
            print('Moving file from {} to {}'.format(fl_src, fl_dst))
            shutil.move(fl_src, fl_dst)

        print('{} offline files moved.'.format(len(fls_paths)))

    else:
        print('No offline files found.')


def main(opt, s_id):

    if opt['database']['dropbox']:
        # Download database file:
        dbx = dbh.get_dropbox()
        tmp_db_path = os.path.join(SET_UP_PATH, 'temp_db')
        db_path = opt['database']['path']
        db_exists = dbh.check_dropbox_file_exist(dbx, db_path)
        if not db_exists:
            raise ValueError('Cannot find database on Dropbox. Exiting.')
        dbh.download_dropbox_file(dbx, db_path, tmp_db_path)
        db = read_pickle(tmp_db_path)

    else:
        db = read_pickle(opt['database']['path'])

    # Find the base options for this sid:
    base_opt = search_database_by_session_id(db, s_id)

    # For compatibility:
    if base_opt.get('set_up'):
        base_opt = {**base_opt, **base_opt['set_up']}

    src_path = os.path.join(base_opt['scratch']['path'], s_id)
    dst_path = os.path.join(base_opt['archive']['path'], s_id)
    sms_path = os.path.join(src_path, 'sims.pickle')

    print('session_id: {}'.format(s_id))
    print('Source path: {}'.format(src_path))
    print('Destination path: {}'.format(dst_path))

    error_paths = check_errors(sms_path, src_path, opt.get('skip_idx'))
    if len(error_paths) > 0:
        raise ValueError('Errors found! Exiting process.py.')

    off_fls = base_opt['scratch']['offline_files']
    move_offline_files(s_id, src_path, off_fls)

    if not os.path.isdir(src_path):
        raise ValueError('Source path is not a directory: {}'.format(src_path))

    arch_opt = base_opt['archive']
    is_dropbox = arch_opt.get('dropbox')
    exclude = opt.get('exclude')

    cpy_msg = 'remote Dropbox' if is_dropbox else 'local computer'
    print('Copying completed sims to archive (on {}).'.format(cpy_msg))
    print('From path: {}'.format(src_path))
    print('To path: {}'.format(dst_path))

    if is_dropbox is True:
        dbh.upload_dropbox_dir(dbx, src_path, dst_path, exclude=exclude)
    else:
        # If Archive is not on Dropbox, assume it is on the scratch machine
        # i.e. the one from which this script (process.py) is run.
        copy_tree(src_path, dst_path)

    print('Archive copying finished.')
