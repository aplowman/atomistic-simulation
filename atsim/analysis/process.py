import sys
import os
from copy import deepcopy
from distutils.dir_util import copy_tree
import shutil
import subprocess

import atsim.dbhelpers as dbh
from atsim.readwrite import read_pickle, write_pickle, find_files_in_dir_glob, factor_common_files
from atsim import simsio, SET_UP_PATH, SCRIPTS_PATH


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
            out = simsio.castep.read_castep_output(calc_path)

        elif method == 'lammps':
            out = simsio.lammps.read_lammps_output(calc_path)

        if len(out['errors']) > 0:
            error_paths.extend(srs_paths)

    return error_paths


def modernise_pickle(sms_path):
    """
    References
    ----------
    [1] http://mywiki.wooledge.org/glob

    """

    modernise = False

    # Copy the first sims options into a top level 'base_options':
    sms = read_pickle(sms_path)
    bo = deepcopy(sms['all_sims'][0].options)

    # Make sure offline_files are glob [1] matches with asterisk:
    off_fls = bo['set_up']['scratch'].get('offline_files')
    if off_fls is not None:
        off_fls_types = off_fls['file_types']
        for ft_idx, ft in enumerate(off_fls_types):
            if '*' not in ft:
                modernise = True
                print('Adding wildcard to offline file types.')
                off_fls_types[ft_idx] = '*' + ft

    # Strip single quotes from set_up->common_files
    cmn_fls = bo['set_up'].get('common_files')
    if cmn_fls is not None:
        for cf_idx, cf in enumerate(cmn_fls):
            if "'" in cf:
                modernise = True
                print('Removing single quotes from `common_files`.')
                bo['set_up']['common_files'][cf_idx] = cf.strip("'")
    else:
        cmn_fls = [
            '*.cell',
            '*.param',
            '*.usp',
            '*-out.cell',
            '*.bib'
        ]
        bo['set_up']['common_files'] = cmn_fls
        modernise = True

    # Change series to a list of lists
    new_series = None
    if bo.get('series') is not None and len(bo.get('series')) > 0:
        print('series is non-None: {}, len: {}'.format(bo.get('series'),
                                                       len(bo.get('series'))))
        if len(bo['series']) == 1 and isinstance(bo['series'][0], dict):
            print('Changing series to a list of lists in `base_options`.')
            new_series = [
                [bo['series'][0]]
            ]
            modernise = True
            bo['series'] = new_series

        if bo.get('series_id') is not None:
            modernise = True
            del bo['series_id']

    sms['base_options'] = bo

    # Change series_id
    sim_0_opt = sms['all_sims'][0].options
    sim_0_sid = sim_0_opt.get('series_id')
    if sim_0_sid is not None and isinstance(sim_0_sid, dict):

        modernise = True
        # For each sim, change series_id to a list of lists
        for s_idx, s in enumerate(sms['all_sims']):

            sms['all_sims'][s_idx].options['set_up']['common_files'] = cmn_fls

            if off_fls is not None:
                sms['all_sims'][s_idx].options['set_up']['scratch']['offline_files'] = off_fls

            if new_series is not None:
                sms['all_sims'][s_idx].options['series'] = new_series

            if len(s.options['series_id']) == 1:
                new_series_id = []
                for k, v in s.options['series_id'].items():
                    new_series_id.append([
                        {
                            'name': k,
                            **v,
                        }
                    ])
                sms['all_sims'][s_idx].options['series_id'] = new_series_id

    if modernise:
        print('Modernising `sims.pickle`.')
        # Rename original sims.pickle:
        print('Renaming old `sims.pickle`.')
        home_path, sms_fn = os.path.split(sms_path)
        sms_path_old = os.path.join(home_path, sms_fn + '_old')
        os.rename(sms_path, sms_path_old)

        # Write the modernised sims.pickle back to disk:
        print('Writing new `sims.pickle`.')
        write_pickle(sms, sms_path)

    else:
        print('Modernisation of `sims.pickle` is not neccessary.')


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


def main(s_id):

    # Download database file:
    dbx = dbh.get_dropbox()
    tmp_db_path = os.path.join(SET_UP_PATH, 'temp_db')
    db_path = '/calcs/db.pickle'
    db_exists = dbh.check_dropbox_file_exist(dbx, db_path)
    if not db_exists:
        raise ValueError('Cannot find database on Dropbox. Exiting.')
    dbh.download_dropbox_file(dbx, db_path, tmp_db_path)

    # Open database file:
    db = read_pickle(tmp_db_path)

    # Find the base options for this sid:
    base_opt = search_database_by_session_id(db, s_id)

    src_path = os.path.join(base_opt['set_up']['scratch']['path'], s_id)
    dst_path = os.path.join(base_opt['set_up']['archive']['path'], s_id)
    sms_path = os.path.join(src_path, 'sims.pickle')

    print('s_id: {}'.format(s_id))
    print('src_path: {}'.format(src_path))
    print('dst_path: {}'.format(dst_path))

    # Modernise sims.pickle:
    # Temporarily #
    if not os.path.isfile(os.path.join(src_path, 'sims.pickle_old')):
        modernise_pickle(sms_path)

    error_paths = check_errors(sms_path, src_path)
    if len(error_paths) > 0:
        raise ValueError('Errors found! Exiting process.py.')

    # Get base options from the modernised pickle:
    sms = read_pickle(sms_path)
    base_opt = sms['base_options']
    off_fls = base_opt['set_up']['scratch']['offline_files']
    move_offline_files(s_id, src_path, off_fls)

    com_fls = base_opt['set_up']['common_files']
    print('Factoring common files turned OFF due to BUG!')
    # print('Factoring common files: {}'.format(com_fls))
    # factor_common_files(src_path, com_fls)

    if not os.path.isdir(src_path):
        raise ValueError('Source path is not a directory: {}'.format(src_path))

    arch_opt = base_opt['set_up']['archive']
    is_dropbox = arch_opt.get('dropbox')

    if is_dropbox is True:
        print('Uploading completed sims to dropbox...')
        dbh.upload_dropbox_dir(dbx, src_path, dst_path)
    else:
        # If Archive is not on Dropbox, assume it is on the scratch machine
        # i.e. the one from which this script (process.py) is run.
        print('Copying completed sims to archive...')
        print('From path: {}'.format(src_path))
        print('To path: {}'.format(dst_path))
        copy_tree(src_path, dst_path)


if __name__ == '__main__':
    main(sys.argv[1])