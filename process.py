import dbhelpers as dbh
import sys
import os
from copy import deepcopy
import dbkey

SCRATCH_DIR = '/mnt/lustre/mbdxqap3/dated'
ARCHIVE_DIR = '/calcs/'
DB_KEY = dbkey.DB_KEY


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
        for cf_idx, cf in cmn_fls:
            if "'" in cf:
                print('Removing single quotes from `common_files`.')
                bo['set_up']['common_files'][cf_idx] = cf.strip("'")
    else:
        cmn_fls = [
            '*.cell',
            '*.param',
            '*.usp',
            '*-out.cell',
        ]
        bo['set_up']['common_files'] = cmn_fls

    # Change series to a list of lists
    new_series = None
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
    if isinstance(sim_0_opt['series_id'], dict):

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


def archive_files(s_id, offline_files):

    arch_dir = offline_files['path']
    fl_types = offline_files['file_types']
    print('Offline file path: {}'.format(arch_dir))
    print('Offline file types: {}'.format(fl_types))

    src_path = os.path.join(SCRATCH_DIR, s_id)

    fls_paths = []
    for t in fl_types:
        fls_paths.extend(find_files_in_dir_glob(src_path, recursive=True))

    print('fls_paths: {}'.format(fls_paths))


def main(s_id):

    src_path = os.path.join(SCRATCH_DIR, s_id)
    dst_path = os.path.join(ARCHIVE_DIR, s_id)
    sms_path = os.path.join(src_path, 'sims.pickle')

    print('s_id: {}'.format(s_id))
    print('scratch_dir: {}'.format(SCRATCH_DIR))
    print('src_path: {}'.format(src_path))
    print('dst_path: {}'.format(dst_path))

    # Modernise sims.pickle:
    # Temporarily #
    if not os.path.isfile(os.path.join(src_path, 'sims.pickle_old')):
        modernise_pickle(sms_path)

    # Open up the base options
    sms = read_pickle(sms_path)
    base_opt = sms['base_options']
    off_fls = base_opt['set_up']['scratch']['offline_files']

    archive_files(s_id, off_fls)

    # dbx = dbh.get_dropbox(DB_KEY)

    # if not os.path.isdir(src_path):
    #     raise ValueError('Source path is not a directory: {}'.format(src_path))

    # print('Uploading to dropbox...')
    # dbh.upload_dropbox_dir(dbx, src_path, dst_path)


if __name__ == '__main__':
    # main(sys.argv[1])
    tst_pick = read_pickle(
        '/mnt/lustre/mbdxqap3/dated/2017-07-18-0205_00775/sims.pickle')
    print('tst_pick: {}'.format(tst_pick))
    exit()
