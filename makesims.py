import os
import numpy as np
import shutil
import utils
import dbhelpers
import readwrite
from readwrite import replace_in_file, delete_line
import atomistic
import simsio
from bravais import BravaisLattice
from crystal import CrystalStructure
import copy
import shutil
import subprocess
import posixpath
import ntpath
import time
import warnings
from set_up.opt import OPT
from set_up.setup_profiles import HOME_PATH
import geometry
import fractions
from sys import stdout

SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
REF_PATH = os.path.join(SCRIPTS_PATH, 'ref')
SU_PATH = os.path.join(SCRIPTS_PATH, 'set_up')


class Archive(object):
    """
    Class to represent the area on a machine where simulations are archived.

    Attributes
    ----------
    path : str
    dropbox : bool

    """

    def __init__(self, session_id, path, dropbox=False, dropbox_key=None, scratch=None):

        self.dropbox = dropbox
        self.dropbox_key = dropbox_key

        # If Archive is not on dropbox, assume it is on the scratch machine
        # (i.e. instantiate Archive with scratch.os_name)
        if dropbox:
            self.remote = True
            self.host = None
            self.os_name = 'posix'

        else:
            if scratch.remote:
                self.remote = True
                self.host = scratch.host
                self.os_name = scratch.os_name

            else:
                self.remote = False
                self.host = None
                self.os_name = os.name

        if self.os_name == 'nt':
            path_mod = ntpath
        elif self.os_name == 'posix':
            path_mod = posixpath

        self.path = path_mod.join(path, session_id)


class Stage(object):
    """
    Class to represent the area on the local machine in which simulations
    input files are generated.

    Attributes
    ----------
    path : str
        Directory of the staging area, including session_id.
    os_name : str
        Either 'nt` (Windows) or `posix` (Unix-like, MacOS)

    """

    def __init__(self, path, session_id):
        """Constructor method to generate a Stage object."""
        path = path.rstrip(os.sep)
        self.path = os.path.join(path, session_id)
        self.os_name = os.name

    def get_path(self, *add_path):
        """Get the path of a directory inside the stage area."""

        return os.path.join(self.path, *add_path)

    def get_bash_path(self, end_path_sep=False):
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

        if self.os_name == 'posix':
            return self.path

        elif self.os_name == 'nt':
            drv, pst_drv = os.path.splitdrive(self.path)
            stage_path_bash = posixpath.sep.join(
                [posixpath.sep + 'mnt', drv[0].lower()] +
                pst_drv.strip(ntpath.sep).split(ntpath.sep))

            if end_path_sep:
                stage_path_bash += posixpath.sep

            return stage_path_bash

    def copy_to_scratch(self, scratch):
        """
        Copy simulation directory to Scratch

        Parameters
        ----------
        scratch : Scratch

        """

        scratch_dir_exists = 'Directory already exists on scratch. Aborting.'
        copy_msg = 'Copying simulations to {} scratch.'.format(
            'remote' if scratch.remote else 'local'
        )
        # Don't copy html plots
        rsync_ex = ['*.html', 'plots']

        if scratch.remote:

            if self.os_name == 'nt' and scratch.os_name == 'posix':

                if utils.dir_exists_remote(scratch.host, scratch.path):
                    raise ValueError(scratch_dir_exists)

                print(copy_msg)
                bash_path = self.get_bash_path(end_path_sep=True)
                utils.rsync_remote(bash_path, scratch.host, scratch.path,
                                   exclude=rsync_ex)

            else:
                raise NotImplementedError('Unsupported remote transfer.')

        else:

            if self.os_name == 'nt' and scratch.os_name == 'nt':

                if os.path.isdir(scratch.path):
                    raise ValueError(scratch_dir_exists)

                print(copy_msg)
                shutil.copytree(self.path, scratch.path)

            elif self.os_name == 'posix' and scratch.os_name == 'posix':
                # Use rsync/scp
                raise NotImplementedError('Unsupported local transfer.')

    def copy_to_archive(self, archive):
        """
        """

        archive_dir_exists = 'Directory already exists on archive. Aborting.'
        copy_msg = 'Copying plots to {}{} archive.'.format(
            'remote' if archive.remote else 'local',
            ' (Dropbox)' if archive.dropbox else ''
        )
        if archive.remote:

            # Only copy plots:
            inc_filter = ['*.html']

            if archive.dropbox:

                dbx = dbhelpers.get_dropbox(archive.dropbox_key)
                db_path = archive.path
                print(copy_msg)
                dbhelpers.upload_dropbox_dir(dbx, self.path, db_path,
                                             include=inc_filter)

            else:

                # Connect to remote archive host for copying stuff
                if utils.dir_exists_remote(archive.host, archive.path):
                    raise ValueError(archive_dir_exists)

                print(copy_msg)
                bash_path = self.get_bash_path(end_path_sep=True)
                utils.rsync_remote(bash_path, archive.host, archive.path,
                                   include=inc_filter)

        else:

            pass
            # Don't need to pre-copy stuff to Archive since everything
            # went to scratch in this case and will thus be copied to
            # Archive when process.py is run.

    def submit_on_scratch(self, scratch):
        """
        Submit simulations on Scratch.

        Parameters
        ----------
        scratch : Scratch

        """

        no_dir_msg = 'Directory does not exist on scratch. Aborting.'

        if scratch.remote:

            if self.os_name == 'nt' and scratch.os_name == 'posix':

                if not utils.dir_exists_remote(scratch.host, scratch.path):
                    raise ValueError(no_dir_msg)

                print('Submitting simulations on scratch...')
                comp_proc = subprocess.run(
                    ['bash',
                     '-c',
                     'ssh {} "cd {} && qsub jobscript.sh"'.format(
                         scratch.host, scratch.path)])
            else:
                raise NotImplementedError('Unsupported remote transfer.')

        else:

            if self.os_name == 'nt' and scratch.os_name == 'nt':
                if not os.path.isdir(scratch.path):
                    raise ValueError(no_dir_msg)

                js_path = os.path.join(scratch.path, 'jobscript.bat')
                # Run batch file in a new console window:
                subprocess.Popen(js_path,
                                 creationflags=subprocess.CREATE_NEW_CONSOLE)

            elif self.os_name == 'posix' and scratch.os_name == 'posix':
                # Use rsync/scp
                raise NotImplementedError('Unsupported.')


class Scratch(object):
    """
    Class to represent the area on a machine in which simulations are to be
    run.

    Attributes
    ----------
    path : str
        Directory of the scratch area, including session_id.
    remote : bool
        If True, the scratch area is on a remote machine. If False, it is on
        the current machine.
    os_name : str
        Either 'nt` (Windows) or `posix` (Unix-like, MacOS)
    offline_files : dict, optional
        Stores information regarding how to deal with particular files which
        shouldn't be moved over a network after simulations have completed. A
        location on the scratch machine can be specifed, to which a subset of
        calculation output files will be moved after simulations. These files
        will not be moved over the network to some other location. It is a dict
        with the following keys:
            path : str
                The location on the scratch machine in which to copy files
                which match the list elements in `file_types`.
            file_types : list of str
                File types to match, which will be moved to `path`.
    host : str, optional
        Only applicable if `remote` is True. Default is None.

    """

    def __init__(self, path, remote, session_id, num_cores,  os_name=None,
                 host=None, offline_files=None, parallel_env=None, sge=False,
                 job_array=False, job_name=None, selective_submission=False,
                 module_load=None):
        """Constructor method to generate a Scratch object."""

        # Validation
        if remote and os_name is None:
            raise ValueError('If `remote` is True, must specify `os_name`.')

        if not remote:
            os_name = os.name

        self.os_name = os_name
        if self.os_name == 'nt':
            path_mod = ntpath
        elif self.os_name == 'posix':
            path_mod = posixpath
        self.path_sep = path_mod.sep
        path = path.rstrip(self.path_sep)
        self.path = self.path_sep.join([path, session_id])
        self.remote = remote
        self.host = host
        self.offline_files = offline_files
        self.num_cores = num_cores
        self.parallel_env = parallel_env
        self.sge = sge
        self.job_array = job_array
        self.selective_submission = selective_submission
        self.job_name = job_name
        self.module_load = module_load

    def get_path(self, *add_path):
        """Get the path of a directory inside the scratch area."""

        return self.path_sep.join([self.path] + list(add_path))


def prepare_series_update(series_spec, atomistic_structure):
    """
    Return a list of dicts representing each element in a simulation series.

    Parameters
    ----------
    series_spec : dict
        Specification of the series.
    atomistic_structure : AtomisticStructure

    """

    # Validation
    allowed_sn = [
        'kpoint',
        'cut_off_energy',
        'smearing_width',
        'gb_size',
        'box_lat',
        'nextra_bands',
        'geom_energy_tol',
        'geom_stress_tol',
        'relative_shift',
        'gamma_surface',
        'boundary_vac',
    ]
    if series_spec.get('name') not in allowed_sn:
        raise NotImplementedError('Series name: {} not understood.'.format(sn))

    common_series_info = {}

    # Some series generate other series: e.g. gamma_surface should generate a
    # relative_shift series.

    if series_spec['name'] == 'gamma_surface':

        edge_vecs = atomistic_structure.boundary_vecs
        grid = geometry.Grid(edge_vecs, series_spec.get('grid_spec'))
        ggp = grid.get_grid_points()
        rel_shifts = ggp['points_frac'].T
        rel_shifts_tup = ggp['points_tup']
        grid_idx = ggp['grid_idx_flat']
        row_idx = ggp['row_idx']
        col_idx = ggp['col_idx']
        point_idx = ggp['point_idx']
        common_series_info.update({
            'gamma_surface': {
                **grid.to_jsonable(),
                'preview': series_spec['preview']
            },
        })
        series_spec = {
            'name': 'relative_shift',
            'vals': rel_shifts,
            'vals_tup': rel_shifts_tup,
            'as_fractions': True,
            'extra_update': {
                'grid_idx': grid_idx,
                'row_idx': row_idx,
                'col_idx': col_idx,
                'point_idx': point_idx,
            }
        }

    # Convenience
    ss = series_spec
    sn = ss['name']
    start = ss.get('start')
    step = ss.get('step')
    stop = ss.get('stop')
    vals = ss.get('vals')
    exclude = ss.get('exclude')

    params_none = [i is None for i in [start, step, stop]]
    if any(params_none) and not all(params_none):
        raise ValueError('Must specify all of `start`, `step` and `stop` '
                         'if one is specified, for series name: {}'.format(sn))

    if vals is not None and start is not None:
        raise ValueError('Either specify (`start`, `step` and `stop`) or '
                         '`vals`, but not both, for series name: {}'.format(
                             sn))

    # If start, step and stop are provided, generate a set of vals from these:
    if vals is None and start is not None:
        diff = start - stop if start > stop else stop - start
        num = int(np.round((diff + step) / step))
        vals = np.linspace(start, stop, num=num)

        if exclude is not None:
            # TODO: parse single line arrays as 1D arrays in dict-parser
            exclude = exclude[0, :]
            # TODO: Maybe need to round both vals and exclude here:
            vals = np.setdiff1d(vals, exclude)

    # Additional processing of series values
    if sn == 'kpoint':

        # Get the MP grid from the supercell and modify vals to remove those
        # which generate duplicate MP grids.
        unique_grids = {}
        for v in vals:

            v = float(v)
            print('v: {}'.format(v))
            kpt_gd = tuple(atomistic_structure.get_kpoint_grid(v))

            if unique_grids.get(kpt_gd) is None:
                unique_grids.update({kpt_gd: [v]})
            else:
                unique_grids[kpt_gd].append(v)

        unique_vals = []
        for k, v in unique_grids.items():
            unique_vals.append(sorted(v)[0])

        vals = sorted(unique_vals)
        print(vals)

    out = []

    # Maybe refactor this logic later:

    if sn == 'kpoint':

        for v in vals:

            v = float(v)  # TODO: parse data types in option file
            out.append({
                'castep': {'cell': {'kpoint_mp_spacing': '{:.3f}'.format(v)}},
                'series_id': {'name': sn, 'val': v, 'path': '{:.3f}'.format(v)}
            })

    elif sn == 'cut_off_energy':

        for v in vals:

            v = float(v)  # TODO: parse data types in option file
            out.append({
                'castep': {'param': {sn: '{:.0f}'.format(v)}},
                'series_id': {'name': sn, 'val': v, 'path': '{:.0f}'.format(v)}
            })

    elif sn == 'smearing_width':

        for v in vals:

            v = float(v)  # TODO: parse data types in option file
            out.append({
                'castep': {'param': {sn: '{:.2f}'.format(v)}},
                'series_id': {'name': sn, 'val': v, 'path': '{:.2f}'.format(v)}
            })

    elif sn == 'nextra_bands':

        for v in vals:

            v = int(v)  # TODO: parse data types in option file
            out.append({
                'castep': {'param': {sn: '{:d}'.format(v)}},
                'series_id': {'name': sn, 'val': v, 'path': '{:d}'.format(v)}
            })

    elif sn == 'geom_energy_tol':

        for v in vals:

            v = float(v)  # TODO: parse data types in option file
            out.append({
                'castep': {'param': {sn: '{:.1e}'.format(v)}},
                'series_id': {'name': sn, 'val': v, 'path': '{:.1e}'.format(v)}
            })

    elif sn == 'geom_stress_tol':

        for v in vals:

            v = float(v)  # TODO: parse data types in option file
            out.append({
                'castep': {'param': {sn: '{:.1e}'.format(v)}},
                'series_id': {'name': sn, 'val': v, 'path': '{:.1e}'.format(v)}
            })

    elif sn == 'gb_size':

        for v in vals:

            out.append({
                'base_structure': {sn: v},
                'series_id': {'name': sn, 'val': v, 'path': '{}_{}_{}'.format(*v)}
            })

    elif sn == 'box_lat':

        for v in vals:

            out.append({
                'base_structure': {sn: v},
                'series_id': {'name': sn, 'val': v,
                              'path': '{}_{}_{}-{}_{}_{}-{}_{}_{}'.format(
                                  *v.flatten())}
            })

    elif sn == 'relative_shift':

        num_digts = len(str(len(vals)))
        pad_fmt = '{{:0{}d}}'.format(num_digts)
        v_tup = ss.get('vals_tup')
        for v_idx, v in enumerate(vals):

            if ss.get('as_fractions') is True:
                v_t = v_tup[v_idx]
                v_str = '_'.join(['{}({})'.format(i[0], i[1]) for i in v_t])

            else:
                v_str = '{}_{}'.format(*v)

            out.append({
                'base_structure': {'relative_shift_args': {'shift': v}},
                'series_id': {'name': sn, 'val': v,
                              'path': (pad_fmt + '__').format(v_idx) + v_str}
            })

    elif sn == 'boundary_vac':

        for v in vals:

            out.append({
                'base_structure': {'boundary_vac_args': {'vac_thickness': v}},
                'series_id': {'name': sn, 'val': v,
                              'path': '{:.2f}'.format(v)}
            })

    extra_update = ss.get('extra_update')
    if extra_update is not None:
        for up_idx, up in enumerate(out):
            eu = {}
            for k, v in extra_update.items():
                eu.update({k: np.asscalar(v[up_idx])})
            out[up_idx]['series_id'].update(eu)

    return out, common_series_info


def prepare_all_series_updates(all_series_spec, atomistic_structure):
    """
    Return a list of dicts representing each element in a simulation series.

    Parameters
    ----------
    all_series_spec : dict
    atomistic_structure : AtomisticStructure
        This is the base AtomisticStructure object used to form the simulation
        series. This is only needed in some cases, where some additional
        processing occurs on the series values which depends on the base
        structure. For example, in generating a kpoint series from a specified
        list of kpoint spacings, it is useful to remove kpoint spacing values
        which produce duplicate kpoint MP grids.

    """

    # Replace each series dict with a list of update dicts:
    common_series_info = {}
    srs_update = []
    for i in all_series_spec:
        upds = []
        for j in i:
            upd, csi = prepare_series_update(j, atomistic_structure)
            upds.append(upd)
            common_series_info.update(csi)
        srs_update.append(upds)

    # Combine parallel series:
    for s_idx, s in enumerate(srs_update):

        l = [len(i) for i in s]
        if len(set(l)) > 1:
            raise ValueError('Length of parallel series must be identical. '
                             'Lengths are: {}'.format(l))

        s_t = utils.transpose_list(s)
        for i_idx, i in enumerate(s_t):

            new_sid = [copy.deepcopy(s_t[i_idx][0]['series_id'])]

            for j_idx, j in enumerate(i[1:]):

                next_sid = copy.deepcopy(s_t[i_idx][j_idx + 1]['series_id'])
                new_sid.append(next_sid)

                s_t[i_idx][0] = utils.update_dict(
                    s_t[i_idx][0], s_t[i_idx][j_idx + 1])

            s_t[i_idx] = s_t[i_idx][0]
            s_t[i_idx]['series_id'] = new_sid

        srs_update[s_idx] = s_t

    # Nest series:
    srs_update_nest = utils.nest_lists(srs_update)

    # Combine dicts into single update dict for each series element:
    all_updates = []
    for i in srs_update_nest:
        all_sids = []
        for j in i:
            for k, v in j.items():
                if k == 'series_id':
                    all_sids.append(v)

        m = utils.combine_list_of_dicts(i)
        m['series_id'] = all_sids
        all_updates.append(m)

    return all_updates, common_series_info


def process_castep_opt(castep_opt, sym_ops=None):
    """

    """

    if castep_opt.get('checkpoint') is True:
        if castep_opt.get('backup_interval') is not None:
            castep_opt['param'].update(
                {'backup_interval': castep_opt['backup_interval']})

    else:
        castep_opt['param'].update({'write_checkpoint': 'none'})

    castep_opt.pop('backup_interval', None)
    castep_opt.pop('checkpoint', None)

    castep_opt['sym_ops'] = None
    if castep_opt['find_inv_sym']:

        if castep_opt['cell'].get('symmetry_generate') is True:
            raise ValueError('Cannot find inversion symmetry if '
                             '`symmetry_generate` is `True`.')

        sym_rots = sym_ops['rotations']
        sym_trans = sym_ops['translations']
        inv_sym_rot = -np.eye(3, dtype=int)
        inv_sym_idx = np.where(np.all(sym_rots == inv_sym_rot, axis=(1, 2)))[0]

        if len(inv_sym_idx) == 0:
            raise ValueError('The bicrystal does not have inversion symmetry.')
        if len(inv_sym_idx) > 1:
            raise ValueError('Multiple inversion sym ops found!.')

        inv_sym_trans = sym_trans[inv_sym_idx[0]]

        castep_opt['sym_ops'] = [
            np.vstack([np.eye(3), np.zeros((3,))]),
            np.vstack([inv_sym_rot, inv_sym_trans])
        ]


def process_lammps_opt(lammps_opt, structure):
    """
    """

    # Set full potential path
    fn = lammps_opt['potential_file']
    lammps_opt['potential_path'] = os.path.join(REF_PATH, 'potentials', fn)
    lammps_opt.pop('potential_file')

    # Set the potential species
    sp = structure.all_species

    if len(sp) > 1:
        raise NotImplementedError('Writing the potential command in '
                                  'multi-species LAMMPS input files has not '
                                  'yet been implemented.')

    lammps_opt['potential_species'] = sp[0]


def process_constraints(opt, structure):
    """
    Process constraint options, so they are ready to be passed to the methods
    which write input files.

    For atom constraints, convert `none` to None, and convert `all` to an index
    array of all atoms.

    """

    cll_cnst = opt['constraints']['cell']
    cll_cnst_def = {
        'fix_angles': 'none',
        'fix_lengths': 'none',
        'angles_equal': 'none',
        'lengths_equal': 'none'
    }
    cll_cnst = {**cll_cnst_def, **cll_cnst}
    atm_cnst = opt['constraints']['atom']
    atm_cnst_def = {
        'fix_xy_idx': 'none',
        'fix_xyz_idx': 'none'
    }
    atm_cnst = {**atm_cnst_def, **atm_cnst}

    for fx in ['fix_xy_idx', 'fix_xyz_idx']:
        if atm_cnst[fx] == 'none':
            atm_cnst[fx] = None
        elif atm_cnst[fx] == 'all':
            atm_cnst[fx] = np.arange(structure.atom_sites.shape[1])
        else:
            # atom constraints are parsed as 2D arrays (pending TODO of
            # dict-parser) (want 1D arrays)
            atm_cnst[fx] = atm_cnst[fx][:, 0]

    opt['constraints']['atom'] = atm_cnst

    for fx in ['fix_angles', 'fix_lengths', 'angles_equal', 'lengths_equal']:
        if cll_cnst[fx] == 'none':
            cll_cnst[fx] = None
        if cll_cnst[fx] == 'all':
            cll_cnst[fx] = 'abc'

    opt['constraints']['cell'] = cll_cnst

    if opt['method'] == 'castep':

        task = opt['castep']['param']['task']

        if task == 'SinglePoint':
            opt['constraints']['cell'].pop('cell_angles_equal', None)
            opt['constraints']['cell'].pop('cell_lengths_equal', None)
            opt['constraints']['cell'].pop('fix_cell_angles', None)
            opt['constraints']['cell'].pop('fix_cell_lengths', None)
            opt['constraints']['atom'].pop('fix_xy_idx', None)
            opt['constraints']['atom'].pop('fix_xyz_idx', None)


def append_db(opt):

    # Add to database
    su = opt['set_up']

    if su['database'].get('dropbox'):

        dpbx_key = su['database'].get('dropbox_key')
        if dpbx_key is not None:

            dbx = dbhelpers.get_dropbox(dpbx_key)

            # Download database file:
            tmp_db_path = os.path.join(SU_PATH, 'temp_db')
            dpbx_path = su['database']['path']

            # Check if db file exists, if not prompt to create:
            db_exists = dbhelpers.check_dropbox_file_exist(dbx, dpbx_path)

            if db_exists:
                dbhelpers.download_dropbox_file(dbx, dpbx_path, tmp_db_path)
            else:
                db_create = utils.confirm('Database file does not exist. '
                                          'Create it?')
                if db_create:
                    readwrite.write_pickle({}, tmp_db_path)
                else:
                    warnings.warn('This simulaton has not been added to a '
                                  'database')

            if db_exists or db_create:

                # Modify database file:
                db_file = readwrite.read_pickle(tmp_db_path)
                db_file.update({su['time_stamp']: opt})
                readwrite.write_pickle(db_file, tmp_db_path)
                dbhelpers.upload_dropbox_file(dbx, tmp_db_path, dpbx_path,
                                              overwrite=True)

    else:

        db_path = su['database']['path']
        db_exists = os.path.isfile(db_path)

        if db_exists:
            db = readwrite.read_pickle(db_path)
        else:
            db_dir, db_fname = os.path.split(db_path)
            os.makedirs(db_dir, exist_ok=True)
            db_create = utils.confirm('Database file does not exist. '
                                      'Create it?')
            if db_create:
                db = {}
            else:
                warnings.warn('This simulaton has not been added to a '
                              'database')

        if db_exists or db_create:

            # Modify database file:
            db.update({su['time_stamp']: opt})
            readwrite.write_pickle(db, db_path)


def main():
    """
    Read the options file and generate a simulation (series).

    TODO:
    -   Validation:
        -   check all ints in cs_idx resolve in crystal_structures
    -   Allow datatype parsing on list elements so specifying crystal structure
        index when forming struct_opt is cleaner.
    -   Move more code into separate functions (e.g. dropbox database stuff)
    -   Can possible update database without writing a temp file.

    """

    struct_lookup = {
        'BulkCrystal': atomistic.BulkCrystal,
        'CSLBicrystal': atomistic.CSLBicrystal,
        'CSLBulkCrystal': atomistic.CSLBulkCrystal,
        'CSLSurfaceCrystal': atomistic.CSLSurfaceCrystal
    }
    srs_is_struct = {
        'kpoint': False,
        'cut_off_energy': False,
        'smearing_width': False,
        'gb_size': True,
        'box_lat': True,
        'nextra_bands': False,
        'relative_shift': True,
    }
    log = []

    # Read the options file
    opt = OPT

    # Convenience
    su = opt['set_up']

    # Modify options dictionary to include additional info

    # Get unique representation of this series:
    ts = time.time()
    su['time_stamp'] = ts

    s_date, s_num = utils.get_date_time_stamp(split=True)
    s_id = s_date + '_' + s_num
    su['session_id'] = s_id
    su['scratch']['job_name'] = "j_" + s_num

    stage_path = su['stage_path']
    sub_dirs = su.get('sub_dirs')
    if sub_dirs is not None:
        stage_path = os.path.join(stage_path, *sub_dirs)
        su['scratch']['path'] = os.path.join(su['scratch']['path'], *sub_dirs)
        su['archive']['path'] = os.path.join(su['archive']['path'], *sub_dirs)

    stage = Stage(session_id=s_id, path=stage_path)
    scratch = Scratch(session_id=s_id, **su['scratch'])
    archive_opts = su['archive']
    database_opts = su['database']

    is_dropbox = [i.get('dropbox') for i in [archive_opts, database_opts]]
    if any(is_dropbox):
        try:
            db_key = su['dropbox_key']
            if is_dropbox[0]:
                archive_opts.update({'dropbox_key': db_key})
            if is_dropbox[1]:
                database_opts.update({'dropbox_key': db_key})

        except:
            print('To use dropbox for Archive or database location, a dropbox '
                  'key must be specified as opiton: `dropbox_key`.')

    archive = Archive(session_id=s_id, scratch=scratch, **archive_opts)
    su['database'] = database_opts

    # log.append('Making stage directory at: {}.'.format(stage_path))
    os.makedirs(stage.path, exist_ok=False)

    # Generate CrystalStructure objects:
    log.append('Generating CrystalStructure objects.')
    cs = []
    for cs_opt in opt['crystal_structures']:
        if 'from_file' in cs_opt:
            path = cs_opt['from_file']['path']
            cs.append(CrystalStructure.from_file(path,
                                                 **cs_opt['from_file']['lattice']))
        else:
            brav_lat = BravaisLattice(**cs_opt['lattice'])
            cs.append(CrystalStructure(brav_lat, cs_opt['motif']))

    # Generate base structure
    log.append('Generating base AtomisticStructure object.')
    struct_opt = {}
    base_as_opt = opt['base_structure']
    for k, v in base_as_opt.items():
        if k == 'type' or k == 'import' or k == 'sigma':
            continue
        elif k == 'crystal_idx':
            if base_as_opt['type'] == 'CSLSurfaceCrystal':
                struct_opt.update({'surface_idx': v})
        elif k == 'cs_idx':
            struct_opt.update({'crystal_structure': cs[v]})
        else:
            struct_opt.update({k: v})

    base_as = struct_lookup[base_as_opt['type']](**struct_opt)

    in_struct = base_as_opt.get('import')
    if in_struct is not None:

        # Get atom sites/supercell from output of previous calculation

        imp_sid = in_struct['sid']
        imp_pick_pth = os.path.join(HOME_PATH, imp_sid, 'sims.pickle')
        imp_pick = readwrite.read_pickle(imp_pick_pth)
        imp_method = imp_pick['base_options']['method']
        imp_sim = imp_pick['all_sims'][0]  # For now just the first sim
        imp_rel_idx = in_struct['relax_idx']
        imp_atom_basis = in_struct['atom_basis']

        if imp_method == 'castep':
            raise NotImplementedError('Have not sorted CASTEP import yet.')

        elif imp_method == 'lammps':
            imp_sup = imp_sim.results['supercell'][imp_rel_idx]
            imp_atoms = imp_sim.results['atoms'][imp_rel_idx]
            imp_atoms_frac = np.dot(np.linalg.inv(imp_sup), imp_atoms)

        new_supercell = base_as.supercell
        if imp_atom_basis == 'fractional':
            # Import fractional atom coordinates into new supercell
            new_atoms = np.dot(new_supercell, imp_atoms_frac)

        elif imp_atom_basis == 'cart':
            # Import atoms and supercell
            new_supercell = imp_sup
            new_atoms = imp_atoms

        # For now, it's all Zr...
        base_as = atomistic.AtomisticStructure(
            new_atoms,
            new_supercell,
            all_species=['Zr'],
            all_species_idx=np.zeros((new_atoms.shape[1],)))

    # Visualise base AtomisticStructure:
    save_args = {
        'filename': stage.get_path('base_structure.html'),
        'auto_open': False
    }

    # Save current options dict
    opt_p_str_path = stage.get_path('opt_processed.txt')
    with open(opt_p_str_path, mode='w', encoding='utf-8') as f:
        f.write(readwrite.format_dict(opt))

    # Get series definitions:
    srs_df = opt.get('series')
    is_srs = srs_df is not None and len(srs_df) > 0

    is_struct = []
    if is_srs:
        for i in srs_df:
            if isinstance(i, dict):
                is_struct.append(srs_is_struct.get(i.get('name')))

            elif isinstance(i, list):
                is_struct.append(any([
                    srs_is_struct.get(j.get('name')) for j in i]))

    # Get the last series depth index which affects the structure:
    lst_struct_idx = -1
    if True in is_struct:
        lst_struct_idx = [idx for idx, i in enumerate(is_struct) if i][-1]

    # If only one simulation, plot the structure:
    if lst_struct_idx == -1:
        base_as.visualise(show_iplot=False, save=True,
                          save_args=save_args, proj_2d=True)

    # Prepare series update data:
    all_upd = [{}]
    all_scratch_paths = []
    all_sims = []
    csi = {}
    if is_srs:
        all_upd, csi = prepare_all_series_updates(srs_df, base_as)

        if 'gamma_surface' in csi:
            # Plot gamma surface grid:
            save_args = {'filename': stage.get_path('grid.html')}
            geometry.Grid.from_jsonable(csi['gamma_surface']).visualise(
                show_iplot=False, save=True, save_args=save_args)

            if csi['gamma_surface'].get('preview'):
                if not utils.confirm('Check gamma surface grid now. '
                                     'Continue?'):
                    print('Exiting.')
                    return

                    # Generate simulation series:
    for upd_idx, upd in enumerate(all_upd):

        stdout.write('Making sim: {} of {}\r'.format(upd_idx + 1, num_sims))
        stdout.flush()

        # Update options:
        srs_opt = copy.deepcopy(opt)
        utils.update_dict(srs_opt, upd)

        # Generate AtomisticStructure:
        log.append('Generating series AtomisticStructure object.')
        srs_struct_opt = {}
        srs_as_opt = srs_opt['base_structure']
        for k, v in srs_as_opt.items():
            if k == 'type' or k == 'import' or k == 'sigma':
                continue
            elif k == 'crystal_idx':
                if srs_as_opt['type'] == 'CSLSurfaceCrystal':
                    srs_struct_opt.update({'surface_idx': v})
            elif k == 'cs_idx':
                srs_struct_opt.update({'crystal_structure': cs[v]})
            else:
                srs_struct_opt.update({k: v})

        srs_as = struct_lookup[srs_as_opt['type']](**srs_struct_opt)

        # Form the directory path for this sim
        # and find out which series affect the structure (for plotting purposes):
        srs_path = []
        if is_srs:
            for sid in upd['series_id']:
                srs_path.append('_'.join([i['path'] for i in sid]))
        else:
            srs_path.append('')

        stage_srs_path = stage.get_path('calcs', *srs_path)
        scratch_srs_path = scratch.get_path('calcs', *srs_path)

        all_scratch_paths.append(scratch_srs_path)
        srs_opt['set_up']['stage_series_path'] = stage_srs_path
        srs_opt['set_up']['scratch_srs_path'] = scratch_srs_path

        if lst_struct_idx > -1:

            plt_path_lst = [stage.path, 'calcs'] + \
                srs_path[:lst_struct_idx + 1] + ['plots']
            plt_path = os.path.join(*plt_path_lst)

            if not os.path.isdir(plt_path):
                os.makedirs(plt_path)
                save_args = {
                    'filename': os.path.join(plt_path, 'structure.html'),
                    'auto_open': False
                }
                srs_as.visualise(show_iplot=False, save=True,
                                 save_args=save_args, proj_2d=True)

        # Process constraints options
        process_constraints(srs_opt, srs_as)

        # Process CASTEP options
        if srs_opt['method'] == 'castep':
            sym_ops = None
            if srs_opt['castep']['find_inv_sym']:
                sym_ops = srs_as.get_sym_ops()
            process_castep_opt(srs_opt['castep'], sym_ops=sym_ops)

        elif srs_opt['method'] == 'lammps':
            process_lammps_opt(srs_opt['lammps'], srs_as)

        # Generate AtomisticSim:
        asim = atomistic.AtomisticSimulation(srs_as, srs_opt)
        asim.write_input_files()
        all_sims.append(asim)

    # Save all sims as pickle file:
    pick_path = stage.get_path('sims.pickle')
    pick = {
        'all_sims': all_sims,
        'base_options': opt,
        'common_series_info': csi,
    }
    readwrite.write_pickle(pick, pick_path)

    # Write jobscript
    js_params = {
        'path': stage.path,
        'calc_paths': all_scratch_paths,
        'method': opt['method'],
        'num_cores': scratch.num_cores,
        'scratch_os': scratch.os_name,
        'scratch_path': scratch.path,
        'sge': scratch.sge,
        'job_array': scratch.job_array,
        'job_name': scratch.job_name,
        'parallel_env': scratch.parallel_env,
        'selective_submission': scratch.selective_submission,
        'module_load': scratch.module_load
    }

    if opt['method'] == 'castep':
        seedname = opt['castep'].get('seedname')
        if seedname:
            js_params.update({'seedname': seedname})

    simsio.write_jobscript(**js_params)

    # Now prompt the user to check the calculation has been set up correctly
    # in the staging area:
    print('Simulation series generated here: {}'.format(stage.path))
    if utils.confirm('Copy to scratch?'):
        stage.copy_to_scratch(scratch)
        if su['append_db']:
            print('Adding options to database.')
            append_db(opt)
        if utils.confirm('Submit on scratch?'):
            stage.submit_on_scratch(scratch)
        else:
            print('Did not submit.')
        if su['upload_plots']:
            stage.copy_to_archive(archive)
    else:
        print('Exiting.')
        return

    series_msg = ' series.' if len(all_scratch_paths) > 0 else '.'
    print('Finished setting up simulation{}'
          ' session_id: {}'.format(series_msg, s_id))


if __name__ == '__main__':
    main()
