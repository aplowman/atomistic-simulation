import os
import numpy as np
import shutil
import fractions
import copy
import shutil
import subprocess
import posixpath
import ntpath
import time
import warnings
from sys import stdout
from pathlib import Path
from atsim import utils, dbhelpers, readwrite, geometry, OPT_FILE_NAMES
from atsim import readwrite
from atsim import SERIES_NAMES, SET_UP_PATH, REF_PATH, SCRIPTS_PATH
from atsim.readwrite import replace_in_file, delete_line, add_line
from atsim.simulation.sim import AtomisticSimulation
from atsim.structure.bravais import BravaisLattice
from atsim.structure.crystal import CrystalStructure
from atsim.structure import atomistic
from atsim.structure.atomistic import AtomisticStructureException
from atsim.structure import bicrystal

JS_TEMPLATE_DIR = os.path.join(SCRIPTS_PATH, 'set_up', 'jobscript_templates')
STRUCT_LOOKUP = {
    'bulk': atomistic.BulkCrystal,
    'csl_bicrystal': bicrystal.csl_bicrystal_from_parameters,
    'csl_bulk_bicrystal': bicrystal.csl_bulk_bicrystal_from_parameters,
    'csl_surface_bicrystal': bicrystal.csl_surface_bicrystal_from_parameters,
    'csl_bicrystal_from_structure': bicrystal.csl_bicrystal_from_structure,
}


def write_jobscript(path, calc_paths, method, num_cores, sge, job_array,
                    scratch_os=None, scratch_path=None, parallel_env=None,
                    selective_submission=False, module_load=None,
                    job_name=None, seedname=None):
    """
    Write a jobscript file whose execution runs calculation input files.

    Parameters
    ----------
    path : str
        Directory in which to save the generated jobscript file.
    calc_paths : list of str
        Directories in which calculations are to be run.
    method : str
        Either 'castep' or 'lammps'
    num_cores : int
        Number of processor cores to use for the calculations.
    sge : bool
        If True, jobscript is generated for the SGE batch scheduler. If False,
        jobscript is generated to immediately run the calculations.
    job_array : bool
        Only applicable if `sge` is True. If True, calculations are submitted
        as an SGE job array. If False, calculations are submitted in one go. If
        the number of calculations is one (i.e. len(`calc_paths`) == 1), this
        will be set to False. Setting this to False can be handy for many small
        calculations which won't take a long time to complete. If submitted as
        a job array, a significant fraction of the total completion time may be
        queuing.
    scratch_os : str, optional
        Either 'nt' (Windows) or 'posix' (Unix-like, MacOS). The operating
        system on which the jobscript file will be executed. Default is to
        query to system on which this script is invoked i.e. `os.name`.
    scratch_path : str, optional
        Directory in which the jobscript is to be executed. Specify this path
        if the jobscript will be executed in a location different to the
        directory `path`, in which it is generated. By default, this is set to
        the same string as `path`.
    parallel_env : str, optional
        The SGE parallel environment on which to submit the calculations. Only
        applicable in `num_cores` > 1. Default is None.
    selective_submission : bool, optional
        Only applicable if `sge` is True. If True, the SGE task id flag `-t`
        [1] will be excluded from the jobscript file and instead this flag will
        be expected as a command line argument when executing the jobscript:
        e.g. "qsub jobscript.sh -t 1-10:2". Default is False.
    module_load : str, optional
        A string representing the path to a module to load within the
        jobscript. If specified, the statement "module load `module_load`" will
        be added to the top of the jobscript.
    job_name : str, optional
        Only applicable if `sge` is True. Default is None.
    seedname : str, optional
        Must be set if `method` is 'castep'.

    Returns
    -------
    None

    References
    ----------
    [1] http://gridscheduler.sourceforge.net/htmlman/htmlman1/qsub.html

    TODO:
    -   Add option for specifying parallel environment type for sge jobscripts

    """

    # General validation:

    if method == 'castep' and seedname is None:
        raise ValueError('`seedname` must be specified for CASTEP jobscripts.')

    if num_cores > 1 and parallel_env is None:
        raise ValueError('`parallel_env` must be set if `num_cores` > 1.')

    num_calcs = len(calc_paths)

    if num_cores <= 0:
        raise ValueError('Num cores not valid.')
    elif num_cores == 1:
        multi_type = 'serial'
    else:
        multi_type = 'parallel'

    if sge:
        sge_str = 'sge'
    else:
        sge_str = 'no_sge'

    if num_calcs == 1:
        job_array = False

    if job_array:
        job_arr_str = 'job_array'
    else:
        job_arr_str = 'single_job'

    if scratch_path is None:
        scratch_path = path

    if scratch_os is None:
        scratch_os = os.name

    if scratch_os == 'nt':
        scratch_path_sep = '\\'
    elif scratch_os == 'posix':
        scratch_path_sep = '/'

    # Get the template file name:
    tmp_fn = method + '_' + sge_str + '_' + \
        multi_type + '_' + scratch_os + '_' + job_arr_str + '.txt'

    # Get the template file path:
    tmp_path = os.path.join(JS_TEMPLATE_DIR, tmp_fn)

    # Write text file with all calc paths
    dirlist_fn = 'dir_list.txt'
    dir_list_path_stage = os.path.join(path, dirlist_fn)
    dir_list_path_scratch = scratch_path_sep.join([scratch_path, dirlist_fn])
    readwrite.write_list_file(dir_list_path_stage, calc_paths)

    if scratch_os == 'posix':
        js_ext = 'sh'
    elif scratch_os == 'nt':
        js_ext = 'bat'

    js_name = 'jobscript.' + js_ext

    # Copy template file to path
    js_path = os.path.join(path, js_name)
    shutil.copy(tmp_path, js_path)

    # Add module load to jobscript:
    if module_load is not None:
        add_line(js_path, 1, '')
        add_line(js_path, 2, 'module load {}'.format(module_load))

    # Make replacements in template file:
    replace_in_file(js_path, '<replace_with_dir_list>', dir_list_path_scratch)

    if multi_type == 'parallel':
        replace_in_file(js_path, '<replace_with_num_cores>', str(num_cores))
        replace_in_file(js_path, '<replace_with_pe>', parallel_env)

    if sge:
        if job_name is not None:
            replace_in_file(js_path, '<replace_with_job_name>', job_name)
        else:
            delete_line(js_path, '<replace_with_job_name>')

        if selective_submission:
            delete_line(js_path, '#$ -t')
        else:
            replace_in_file(js_path, '<replace_with_job_index_range>',
                            '1-' + str(num_calcs))

    if method == 'castep':
        replace_in_file(js_path, '<replace_with_seed_name>', seedname)

    # For `method` == 'lammps', `sge` == True, `job_array` == False, we need
    # a helper jobscript, called by the SGE jobscript:
    if method == 'lammps' and sge and not job_array:

        if multi_type != 'serial' or scratch_os != 'posix':
            raise NotImplementedError('Jobscript parameters not supported.')

        help_tmp_path = os.path.join(
            JS_TEMPLATE_DIR, 'lammps_no_sge_serial_posix_single_job.txt')

        help_js_path = os.path.join(path, 'lammps_single_job.sh')
        shutil.copy(help_tmp_path, help_js_path)
        replace_in_file(help_js_path, '<replace_with_dir_list>', dir_list_path)

    # Make a directory for job-related output. E.g. .o and .e files from CSF.
    os.makedirs(os.path.join(path, 'output'))


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
            new_path = self.path

        elif self.os_name == 'nt':
            drv, pst_drv = os.path.splitdrive(self.path)
            new_path = posixpath.sep.join(
                [posixpath.sep + 'mnt', drv[0].lower()] +
                pst_drv.strip(ntpath.sep).split(ntpath.sep))

        if end_path_sep:
            new_path += posixpath.sep

        return new_path

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

            if utils.dir_exists_remote(scratch.host, scratch.path):
                raise ValueError(scratch_dir_exists)

            print(copy_msg)
            bash_path = self.get_bash_path(end_path_sep=True)
            utils.rsync_remote(bash_path, scratch.host, scratch.path,
                               exclude=rsync_ex)

        else:

            if os.path.isdir(scratch.path):
                raise ValueError(scratch_dir_exists)

            print(copy_msg)
            shutil.copytree(self.path, scratch.path)

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

                dbx = dbhelpers.get_dropbox()
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

            if not utils.dir_exists_remote(scratch.host, scratch.path):
                raise ValueError(no_dir_msg)

            print('Submitting simulations on scratch...')
            comp_proc = subprocess.run(
                ['bash',
                    '-c',
                    'ssh {} "cd {} && qsub jobscript.sh"'.format(
                        scratch.host, scratch.path)])

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
                js_path = os.path.join(scratch.path, 'jobscript.sh')
                os.chmod(js_path, 0o744)
                subprocess.Popen(js_path, shell=True)


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


def prepare_series_update(series_spec, common_series_info, atomistic_structure):
    """
    Return a list of dicts representing each element in a simulation series.

    Parameters
    ----------
    series_spec : dict
        Specification of the series.
    atomistic_structure : AtomisticStructure

    """

    # Validation
    allowed_sn = SERIES_NAMES
    if series_spec.get('name') not in allowed_sn:
        raise NotImplementedError(
            'Series name: {} not understood.'.format(series_spec.get('name')))

    # Some series generate other series: e.g. gamma_surface should generate a
    # relative_shift series.

    if series_spec['name'] == 'gamma_surface':

        edge_vecs = atomistic_structure.boundary_vecs
        grid = geometry.Grid(edge_vecs, series_spec.get('grid_spec'))
        ggp = grid.get_grid_points()
        rel_shifts = ggp['points_frac'].T
        rel_shifts_num_den = ggp['points_num_den']
        grid_idx = ggp['grid_idx_flat']
        point_idx = ggp['point_idx']
        common_series_info.append({
            'series_name': 'gamma_surface',
            **grid.to_jsonable(),
            'preview': series_spec['preview']
        })
        series_spec = {
            'name': 'relative_shift',
            'vals': rel_shifts,
            'vals_num_den': rel_shifts_num_den,
            'as_fractions': True,
            'extra_update': {
                'grid_idx': grid_idx,
                'point_idx': point_idx,
                'csi_idx': [len(common_series_info) - 1] * len(rel_shifts),
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
            kpt_gd = tuple(atomistic_structure.get_kpoint_grid(v))

            if unique_grids.get(kpt_gd) is None:
                unique_grids.update({kpt_gd: [v]})
            else:
                unique_grids[kpt_gd].append(v)

        unique_vals = []
        for k, v in unique_grids.items():
            unique_vals.append(sorted(v)[0])

        vals = sorted(unique_vals)
        print('Found unique kpoint spacings: {}'.format(vals))

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
        v_num_den = ss.get('vals_num_den')
        for v_idx, v in enumerate(vals):

            if ss.get('as_fractions') is True:
                v_nd = utils.get_col(v_num_den, v_idx)
                v_str = '_'.join(['{}({})'.format(i[0], i[1]) for i in v_nd])

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

    elif sn == 'boundary_vac_flat':

        for v in vals:

            out.append({
                'base_structure': {'boundary_vac_flat_args': {'vac_thickness': v}},
                'series_id': {'name': sn, 'val': v,
                              'path': '{:.2f}'.format(v)}
            })

    extra_update = ss.get('extra_update')
    if extra_update is not None:
        for up_idx, up in enumerate(out):
            eu = {}
            for k, v in extra_update.items():
                vu = v[up_idx]
                if isinstance(vu, np.generic):
                    vu = np.asscalar(vu)
                eu.update({k: vu})
            out[up_idx]['series_id'].update(eu)

    return out


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

    Notes
    -----
    Consider two parallel series A, B and a nested series C, so 
    `all_series_spec` looks like this:
    [
        [A_defn, B_defn], [C_defn],
    ]
    where X_defn is a dict which when passed to `prepare_series_update`
    returns a list of updates dicts, each denoted Xi.

    Firstly, replace definition dict with list of update dicts:
    [
        [ [A1, A2], [B1, B2] ], [C1, C2, C3]
    ]

    Next, combine parallel series update dicts:
    [
        [A1B1, A2B2], [C1, C2, C3]
    ]

    Next, nest the series:
    [
        [A1B1, C1], [A1B1, C2], [A1B1, C3], [A2B2, C1], [A2B2, C2], [A2B2, C3],
    ]

    Finally, for each simulation we combine update dicts into a single update
    dict, resulting in a flat list of dicts:
    [
        A1B1C1, A1B1C2, A1B1C4, A2B2C1, A2B2C2, A2B2C3,
    ]

    Alternatively, consider the D series to be a "lookup series" where the C
    series elements are found from a lookup based on the parent series element:
    [
        [A_defn], [B_defn, C_defn], [D_lookup_defn]
    ]
    As before, firstly replace definition dict with list of update dicts (and 
    remove the lookup defn):
    [
        [ [A1, A2] ], [ [B1, B2], [C1, C2] ],
    ]

    And combine parallel series update dicts and move lookup out of list:
    [
        [A1, A2], [B1C1, B2C2],
    ]

    Nest series:
    [
        [A1, B1C1], [A1, B2C2], [A2, B1C1], [A2, B2C2],
    ]

    Combine update dicts up to lookup series:
    [
        A1B1C1, A1B2C2, A2B1C1, A2B2C2,
    ]

    Now loop through each parent_val in the lookup definition and see if there
    if a matching update dict.

    For each element in the lookup parent series, generate a child series if it
    matches a parent specified in the lookup dict. Note that the child series 
    lengths need not be the same for each parent series element.
    [
        [[A1B1C1], [D1, D2]], [[A1B2C2], [D3]], [[A2B1C1], []], [[A2B2C2], []]
    ]

    Nest, and extend:
    [
        [A1B1C1, D1], [A1B1C1, D2], [A1B2C2, D3], [A2B1C1], [], [A2B2C2], [],
    ]

    Combine:
    [
        A1B1C1, A1B1C2, A2B2C1
    ]
    """

    # Replace each series dict with a list of update dicts:
    common_series_info = []
    srs_update = []
    lookup_idx = None
    for i_idx, i in enumerate(all_series_spec):

        if lookup_idx is not None and i_idx > lookup_idx:
            raise NotImplementedError('Lookup series must be the final (most-'
                                      'nested) series.')

        upds = []
        for j_idx, j in enumerate(i):

            if j['name'] == 'lookup':
                lookup_defn = j

                # Validation:
                if lookup_idx is not None:
                    raise NotImplementedError('Cannot specify multiple lookup '
                                              'series')

                if len(all_series_spec[i_idx]) > 1:
                    raise NotImplementedError('Lookup series cannot have '
                                              'parallel series.')
                if i_idx == 0:
                    raise ValueError('Lookup series cannot be a top-level '
                                     'series; there must be a parent series.')

                lookup_idx = i_idx
                break

            upd = prepare_series_update(
                j, common_series_info, atomistic_structure)

            upds.append(upd)

        if len(upds) > 0:
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

    if lookup_idx is None:
        return all_updates, common_series_info

    all_updates_lkup = []
    for i in all_updates:

        # Check for a match in series lookup:

        par_srs = lookup_defn['src']['parent_series']
        par_vals = lookup_defn['src']['parent_val']

        srs_id = i['series_id']
        # print('Parent series id: {}'.format(srs_id))

        # Loop through series ids
        for sid in srs_id:

            # Loop through parent vals of lookup series
            for pv_idx, pvs in enumerate(par_vals):

                for psn, pv in zip(par_srs, pvs):

                    m = utils.dict_from_list(sid, {'name': psn, 'val': pv})

                    if m is not None:
                        chd_srs = lookup_defn['src']['child_series'][pv_idx]
                        new_srs = prepare_series_update(
                            chd_srs, common_series_info, atomistic_structure)
                        all_updates_lkup.append([[i], new_srs])

    # Nest:
    all_updates_lkup_nest = []
    for i in all_updates_lkup:
        all_updates_lkup_nest.extend(utils.nest_lists(i))

    # Merge update dicts:
    for i_idx, i in enumerate(all_updates_lkup_nest):
        lkup_srs_id = all_updates_lkup_nest[i_idx][1]['series_id']
        all_updates_lkup_nest[i_idx][0]['series_id'].append([lkup_srs_id])
        del i[1]['series_id']

        all_updates_lkup_nest[i_idx] = utils.combine_list_of_dicts(i)

    return all_updates_lkup_nest, common_series_info


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

    geom_opt_str = ['GEOMETRYOPTIMISATION', 'GEOMETRYOPTIMIZATION']
    if castep_opt['param']['task'].upper() not in geom_opt_str:
        castep_opt['param'].pop('geom_max_iter', None)
        castep_opt['param'].pop('geom_method', None)


def process_lammps_opt(lammps_opt, structure, stage_path, scratch_path):
    """
    """

    for k, v in lammps_opt['potential_files'].items():

        pot_path = os.path.join(REF_PATH, 'potentials', v)
        pot_path_stage = os.path.join(stage_path, v)
        pot_path_scratch = os.path.join(scratch_path, v)

        try:
            shutil.copy(pot_path, pot_path_stage)
        except:
            raise ValueError(
                'Check potential file: "{}" exists.'.format(pot_path))

        for ln_idx, ln in enumerate(lammps_opt['interactions']):
            if k in ln:
                pot_path_scratch = '"' + pot_path_scratch + '"'
                lammps_opt['interactions'][ln_idx] = ln.replace(
                    k, pot_path_scratch)

    del lammps_opt['potential_files']

    charges_dict = lammps_opt.get('charges')
    if charges_dict is not None:
        charges = []
        for sp in structure.all_species:
            try:
                charges.append(charges_dict[sp])
            except:
                raise ValueError('Cannot find charge specification for species: '
                                 ' {}'.format(sp))

        lammps_opt['charges'] = charges


def process_constraints(opt, structure):
    """
    Process constraint options, so they are ready to be passed to the methods
    which write input files.

    For atom constraints, convert `none` to None, and convert `all` to an index
    array of all atoms. Indexing starts from 1!

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
        'fix_xz_idx': 'none',
        'fix_yz_idx': 'none',
        'fix_xyz_idx': 'none'
    }
    atm_cnst = {**atm_cnst_def, **atm_cnst}

    valid_atom_cnst = {}
    for k, v in atm_cnst.items():
        if isinstance(v, list):

            valid_atom_cnst.update({k: np.array(v)})
        elif isinstance(v, str):
            if v.upper() == 'NONE':
                valid_atom_cnst.update({k: None})
            elif v.upper() == 'ALL':
                valid_atom_cnst.update(
                    {k: np.arange(structure.atom_sites.shape[1]) + 1})

    opt['constraints']['atom'] = valid_atom_cnst

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
            opt['constraints']['atom'].pop('fix_xz_idx', None)
            opt['constraints']['atom'].pop('fix_yz_idx', None)
            opt['constraints']['atom'].pop('fix_xyz_idx', None)


def append_db(opt):

    if opt['database'].get('dropbox'):

        dbx = dbhelpers.get_dropbox()

        # Download database file:
        tmp_db_path = os.path.join(SET_UP_PATH, 'temp_db')
        dpbx_path = opt['database']['path']

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
            db_file.update({opt['time_stamp']: opt})
            readwrite.write_pickle(db_file, tmp_db_path)
            dbhelpers.upload_dropbox_file(dbx, tmp_db_path, dpbx_path,
                                          overwrite=True)

    else:

        db_path = opt['database']['path']
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
            db.update({opt['time_stamp']: opt})
            readwrite.write_pickle(db, db_path)


def make_crystal_structures(cs_opt):
    cs = []
    for cs_defn in cs_opt:

        cs_params = {}

        if 'path' in cs_defn:

            cs_params.update({
                'path': cs_defn['path'],
                **cs_defn['lattice'],
            })
            cs.append(CrystalStructure.from_file(**cs_params))

        else:
            cs_params.update({
                'bravais_lattice': BravaisLattice(**cs_defn['lattice']),
                'motif': cs_defn['motif'],
            })
            cs.append(CrystalStructure(**cs_params))

    return cs


def make_base_structure(bs_opt, crystal_structures):

    struct_opt = {}
    for k, v in bs_opt.items():
        if k == 'type' or k == 'import' or k == 'sigma':
            continue
        elif k == 'crystal_idx':
            if bs_opt['type'] == 'CSLSurfaceCrystal':
                struct_opt.update({'surface_idx': v})
        elif k == 'cs_idx':
            struct_opt.update({'crystal_structure': crystal_structures[v]})
        else:
            struct_opt.update({k: v})

    base_as = STRUCT_LOOKUP[bs_opt['type']](**struct_opt)

    in_struct = bs_opt.get('import')
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

    return base_as


def make_series_structure(ss_opt, crystal_structures):

    srs_struct_opt = {}
    for k, v in ss_opt.items():
        if k == 'type' or k == 'import' or k == 'sigma':
            continue
        elif k == 'crystal_idx':
            if ss_opt['type'] == 'CSLSurfaceCrystal':
                srs_struct_opt.update({'surface_idx': v})
        elif k == 'cs_idx':
            srs_struct_opt.update({'crystal_structure': crystal_structures[v]})
        else:
            srs_struct_opt.update({k: v})

    return STRUCT_LOOKUP[ss_opt['type']](**srs_struct_opt)


def plot_gamma_surface_grids(opt, common_series_info, stage):

    g_preview = []
    cnd = {'series_name': 'gamma_surface'}
    g_idx, gamma = utils.dict_from_list(
        common_series_info, cnd, ret_index=True)

    if gamma is not None and opt['make_plots']:

        if gamma.get('preview') is not None:
            g_preview.append(gamma['preview'])

        gamma_count = 0
        while gamma is not None:

            g_fn = stage.get_path('γ_surface_{}.html'.format(gamma_count))
            vis_args = {
                'show_iplot': False,
                'save': True,
                'save_args': {
                    'filename': g_fn
                }
            }
            geometry.Grid.from_jsonable(gamma).visualise(**vis_args)

            csi_sub = [i for i_idx, i in enumerate(common_series_info)
                       if i_idx != g_idx]
            g_idx, gamma = utils.dict_from_list(csi_sub, cnd, ret_index=True)
            gamma_count += 1

        if any(g_preview):
            g_prv_msg = 'Check gamma surface grid(s) now. Continue?'
            if not utils.confirm(g_prv_msg):
                print('Exiting.')
                return False

    return True


def main(opt):
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

    srs_is_struct = {
        'kpoint': False,
        'cut_off_energy': False,
        'smearing_width': False,
        'gb_size': True,
        'box_lat': True,
        'nextra_bands': False,
        'relative_shift': True,
        'boundary_vac': True,
        'boundary_vac_flat': True,
    }
    log = []

    # TEMP
    """
    keys in opt:
        ['archive', 
        'upload_plots', 
        'crystal_structures', 
        'offline_files', 
        'scratch', 
        'subdirs', 
        'constraints', 
        'castep', 
        'series', 
        'method', 
        'base_structure', 
        'append_db', 
        'database', 
        'make_plots',
        'stage']

    """
    ####

    # Get unique representation of this series:
    opt['time_stamp'] = time.time()

    s_date, s_num = utils.get_date_time_stamp(split=True)
    s_id = s_date + '_' + s_num
    opt['session_id'] = s_id
    opt['scratch']['job_name'] = 'j_' + s_num

    # stage_path = su['stage_path']
    stage_path = opt['stage']['path']
    sub_dirs = opt['sub_dirs']
    stage_path = os.path.join(stage_path, *sub_dirs)
    opt['scratch']['path'] = os.path.join(opt['scratch']['path'], *sub_dirs)
    opt['archive']['path'] = os.path.join(opt['archive']['path'], *sub_dirs)

    stage = Stage(session_id=s_id, path=stage_path)
    scratch = Scratch(session_id=s_id, **opt['scratch'])
    archive = Archive(session_id=s_id, scratch=scratch, **opt['archive'])

    # log.append('Making stage directory at: {}.'.format(stage_path))
    os.makedirs(stage.path, exist_ok=False)

    crys_structs = None
    if opt.get('crystal_structures') is not None:
        crys_structs = make_crystal_structures(opt['crystal_structures'])
    base_as = make_base_structure(opt['base_structure'], crys_structs)

    # Copy makesims options file
    opt_path = stage.get_path(OPT_FILE_NAMES['makesims'])
    shutil.copy(os.path.join(
        SET_UP_PATH, OPT_FILE_NAMES['makesims']), opt_path)

    # Save current options dict
    opt_p_str_path = stage.get_path('opt_processed.txt')
    with open(opt_p_str_path, mode='w', encoding='utf-8') as f:
        f.write(readwrite.format_dict(opt))

    # Get series definitions:
    srs_df = opt.get('series')
    is_srs = srs_df is not None and srs_df != [[]]

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
        if opt['make_plots']:
            save_args = {
                'filename': stage.get_path('base_structure.html'),
                'auto_open': False
            }
            base_as.visualise(show_iplot=False, save=True,
                              save_args=save_args, proj_2d=True)

    # Prepare series update data:
    all_upd = [{}]
    all_scratch_paths = []
    all_sims = []
    csi = []
    if is_srs:
        all_upd, csi = prepare_all_series_updates(srs_df, base_as)

        # Plot gamma surface grids:
        if not plot_gamma_surface_grids(opt, csi, stage):
            return

    # Generate simulation series:
    skipped_sims = []
    num_sims = len(all_upd)
    for upd_idx, upd in enumerate(all_upd):

        stdout.write('Making sim: {} of {}\r'.format(upd_idx + 1, num_sims))
        stdout.flush()

        srs_opt = copy.deepcopy(opt)
        utils.update_dict(srs_opt, upd)
        try:
            srs_as = make_base_structure(
                srs_opt['base_structure'], crys_structs)
        except AtomisticStructureException as e:
            skipped_sims.append(upd_idx)
            continue

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
        srs_opt['stage_series_path'] = stage_srs_path
        srs_opt['scratch_srs_path'] = scratch_srs_path

        if lst_struct_idx > -1 and opt['make_plots']:

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
            process_lammps_opt(srs_opt['lammps'],
                               srs_as, stage.path, scratch.path)

        # Generate AtomisticSim:
        asim = AtomisticSimulation(srs_as, srs_opt)
        asim.write_input_files()
        all_sims.append(asim)

    print('Completed making {} sim(s). {} sims were skipped: {}.'.format(
        num_sims, len(skipped_sims), skipped_sims))

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

    write_jobscript(**js_params)

    # Now prompt the user to check the calculation has been set up correctly
    # in the staging area:
    print('Simulation series generated here: {}'.format(stage.path))
    if utils.confirm('Copy to scratch?'):
        stage.copy_to_scratch(scratch)
        if opt['append_db']:
            print('Adding options to database.')
            append_db(opt)
        if utils.confirm('Submit on scratch?'):
            stage.submit_on_scratch(scratch)
        else:
            print('Did not submit.')
        if opt['upload_plots']:
            stage.copy_to_archive(archive)
    else:
        print('Exiting.')
        return

    series_msg = ' series.' if len(all_scratch_paths) > 0 else '.'
    print('Finished setting up simulation{}'
          ' session_id: {}'.format(series_msg, s_id))


if __name__ == '__main__':
    main(OPT)
