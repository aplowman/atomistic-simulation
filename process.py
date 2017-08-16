import dbhelpers as dbh
import sys
import os
from copy import deepcopy
from set_up.secret import DB_KEY
from readwrite import read_pickle, write_pickle, find_files_in_dir_glob, factor_common_files
import shutil
import subprocess

SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
SU_PATH = os.path.join(SCRIPTS_PATH, 'set_up')


def modernise_pickle(sms_path):
    """
    References
    ----------
    [1] http://mywiki.wooledge.org/glob

    """
    print('Modernising `sims.pickle`.')

    # Copy the first sims options into a top level 'base_options':
    sms = read_pickle(sms_path)
    bo = deepcopy(sms['all_sims'][0].options)

    # Make sure offline_files are glob [1] matches with asterisk:
    off_fls = bo['set_up']['scratch'].get('offline_files')
    if off_fls is not None:
        off_fls_types = off_fls['file_types']
        for ft_idx, ft in enumerate(off_fls_types):
            if '*' not in ft:
                print('Adding wildcard to offline file types.')
                off_fls_types[ft_idx] = '*' + ft

    # Strip single quotes from set_up->common_files
    cmn_fls = bo['set_up'].get('common_files')
    if cmn_fls is not None:
        for cf_idx, cf in enumerate(cmn_fls):
            if "'" in cf:
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

    # Change series to a list of lists
    new_series = None
    if bo.get('series') is not None:
        if len(bo['series']) == 1 and isinstance(bo['series'][0], dict):
            print('Changing series to a list of lists in `base_options`.')
            new_series = [
                [bo['series'][0]]
            ]
            bo['series'] = new_series

        del bo['series_id']
    sms['base_options'] = bo

    # Change series_id
    sim_0_opt = sms['all_sims'][0].options
    sim_0_sid = sim_0_opt.get('series_id')
    if sim_0_sid is not None and isinstance(sim_0_sid, dict):

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

    # Rename original sims.pickle:
    print('Renaming old `sims.pickle`.')
    home_path, sms_fn = os.path.split(sms_path)
    sms_path_old = os.path.join(home_path, sms_fn + '_old')
    os.rename(sms_path, sms_path_old)

    # Write the modernised sims.pickle back to disk:
    print('Writing new `sims.pickle`.')
    write_pickle(sms, sms_path)


def move_offline_files(s_id, offline_files):

    arch_dir = offline_files['path']
    fl_types = offline_files['file_types']
    src_path = os.path.join(SCRATCH_DIR, s_id)

    fls_paths = []
    for t in fl_types:
        fls_paths.extend(find_files_in_dir_glob(src_path, t, recursive=True))

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
    dbx = dbh.get_dropbox(DB_KEY)
    tmp_db_path = os.path.join(SU_PATH, 'temp_db')
    db_path = '/calcs/db.pickle'
    db_exists = dbh.check_dropbox_file_exist(dbx, db_path)
    if not db_exists:
        raise ValueError('Cannot find database on Dropbox. Exiting.')
    dbh.download_dropbox_file(dbx, db_path, tmp_db_path)

    # Open database file:
    db = read_pickle(tmp_db_path)

    # Find the base options for this sid:
    for k, v in db.items():
        if v['set_up']['session_id'] == s_id:
            base_opt = v
            break

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

    off_fls = base_opt['set_up']['scratch']['offline_files']
    move_offline_files(s_id, off_fls)

    com_fls = base_opt['set_up']['common_files']
    print('Factoring common files turned OFF due to BUG!')
    # print('Factoring common files: {}'.format(com_fls))
    # factor_common_files(src_path, com_fls)

    if not os.path.isdir(src_path):
        raise ValueError('Source path is not a directory: {}'.format(src_path))

    print('Uploading to dropbox...')
    dbh.upload_dropbox_dir(dbx, src_path, dst_path)


if __name__ == '__main__':
    main(sys.argv[1])
