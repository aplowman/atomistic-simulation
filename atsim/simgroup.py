"""Module containing a class to represent a group of simulations defined by one
or more simulation sequences.

"""
import os
import copy
import yaml
import json
import shutil
import numpy as np
from atsim import utils
from atsim import opt_parser
from atsim.sequence import SimSequence
from atsim.structure.crystal import CrystalStructure
from atsim.structure.bravais import BravaisLattice
from atsim.utils import (get_date_time_stamp, nest, merge, prt, dict_from_list,
                         mut_exc_args, set_nested_dict, get_recursive, update_dict)

from atsim import SET_UP_PATH, OPT_FILE_NAMES, SIM_CLASS_MAP, SCRIPTS_PATH, BaseUpdate
from atsim.simulation.sim import AtomisticSimulation
from atsim.resources import Stage, Scratch, Archive, ResourceConnection
from atsim.readwrite import replace_in_file, delete_line, add_line, write_list_file, format_dict, format_list

JS_TEMPLATE_DIR = os.path.join(SCRIPTS_PATH, 'set_up', 'jobscript_templates')


def write_jobscript(path, calc_paths, method, num_cores, is_sge, job_array, executable,
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
    is_sge : bool
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
    executable : str
        Executable command.
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

    if is_sge:
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
    write_list_file(dir_list_path_stage, calc_paths)

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
        delete_line(js_path, '#$ -V')
        replace_in_file(js_path, '#!/bin/bash', '#!/bin/bash --login')

    # Make replacements in template file:
    replace_in_file(js_path, '<replace_with_dir_list>', dir_list_path_scratch)

    if multi_type == 'parallel':
        replace_in_file(js_path, '<replace_with_num_cores>', str(num_cores))
        replace_in_file(js_path, '<replace_with_pe>', parallel_env)

    if is_sge:
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

    replace_in_file(js_path, '<replace_with_executable>', executable)

    # For `method` == 'lammps', `sge` == True, `job_array` == False, we need
    # a helper jobscript, called by the SGE jobscript:
    if method == 'lammps' and is_sge and not job_array:

        if multi_type != 'serial' or scratch_os != 'posix':
            raise NotImplementedError('Jobscript parameters not supported.')

        help_tmp_path = os.path.join(
            JS_TEMPLATE_DIR, 'lammps_no_sge_serial_posix_single_job.txt')

        help_js_path = os.path.join(path, 'lammps_single_job.sh')
        shutil.copy(help_tmp_path, help_js_path)
        replace_in_file(help_js_path, '<replace_with_dir_list>',
                        dir_list_path_scratch)

    # Make a directory for job-related output. E.g. .o and .e files from CSF.
    os.makedirs(os.path.join(path, 'output'))


class SimGroup(object):
    """Class to represent a group of related simulations.

    Things we'd want to do with a SimGroup:
    -   Set up directories on stage:
            - write sim input files
            - write jobscripts
            - write json serialisation of self
    -   Write input files for one or more simulation runs on stage/scratch
    -   Submit runs
    -   Add results to one or more runs
    -   Save as a JSON file
    -   Load from a JSON file (for adding results, or runs)


    """

    # Default path formatting options:
    path_options_def = {
        'parallel_sims_join': '_+_',
        'sim_numbers': True,
        'sim_numbers_join': '__',
        'sequence_names': False,
        'subdirs': [],
        'run_fmt': '{0}',
        'calcs_path': 'calcs',
        'human_id': '%Y-%m-%d-%H%M_%%r%%r%%r%%r%%r',
    }

    machine_id = None
    software_lookup = None

    @classmethod
    def set_machine_id(cls, machine_id):
        """Set resource ID of current machine."""
        cls.machine_id = machine_id

    @classmethod
    def load_software_lookup(cls):
        soft_path = os.path.join(SET_UP_PATH, OPT_FILE_NAMES['software'])
        with open(soft_path) as soft_fs:
            cls.software_lookup = yaml.safe_load(soft_fs)

    def __init__(self, opts_path=None, opts_spec_path=None, state=None):
        """Initialise a SimGroup object.

        If generating a new SimGroup, use parameters `opts_path` and
        `opts_spec_path`. If loading from a saved state (e.g. from JSON), use
        parameter `state`.

        """

        mut_exc_args(
            {'opts_path': opts_path, 'opts_spec_path': opts_spec_path},
            {'state': state}
        )

        if state:
            self.sequences = state['sequences']
            self.sim_updates = state['sim_updates']
            self.sims = state['sims']
            self.run_opt = state['run_opt']
            self.software_name = state['software_name']
            self.path_options = state['path_options']
            self.options_unparsed = state['options_unparsed']
            self.options = state['options']
            self.hid = state['hid']
            self.job_name = state['job_name']

        else:

            with open(opts_spec_path, 'r') as opts_spec_path_fp:
                opts_spec = yaml.safe_load(opts_spec_path_fp)['makesims']
                opt_parser.validate_opt_spec(opts_spec)

            with open(opts_path, 'r') as opts_path_fp:
                opts_unparsed = yaml.safe_load(opts_path_fp)

            opts_parsed = opt_parser.parse_opt(opts_unparsed, opts_spec)
            sequences = opts_parsed.pop('sequences')
            run_opt = opts_parsed.pop('run')
            path_options = opts_parsed.pop('path_options', {})

            self.options_unparsed = opts_unparsed
            self.options = opts_parsed

            seq_fn = os.path.join(SET_UP_PATH, OPT_FILE_NAMES['sequences'])
            SimSequence.load_sequence_definitions(seq_fn)

            self.sequences = [SimSequence(i) for i in sequences]
            self.sim_updates = self._get_sim_updates()
            self.sims = None

            # Process run group options:
            self.software_name = None
            self.run_opt = self._parse_run_group_opt(run_opt)

            path_options = {**SimGroup.path_options_def, **path_options}
            sub_dirs = [utils.parse_times(i)[0]
                        for i in path_options['sub_dirs']]
            hid, hid_num = utils.parse_times(path_options['human_id'])
            path_options.update({'sub_dirs': sub_dirs})

            self.path_options = path_options
            self.hid = hid
            self.job_name = 'j_' + hid_num

        # For state or new:
        add_path = self.path_options['sub_dirs'] + [self.hid]
        self.stage = Stage(self.run_opt['stage_id'], add_path)
        self.scratch = Scratch(self.run_opt['scratch_id'], add_path)
        self.archive = Archive(self.run_opt['archive_id'], add_path)

        # prt(format_list(self.sim_updates), 'self.sim_updates')
        # exit()

        # prt(self.stage.__dict__, 'self.stage')
        # prt(self.scratch.__dict__, 'self.scratch')

    def _parse_run_group_opt(self, run_opt):

        for rg_idx, _ in enumerate(run_opt['groups']):

            run_group = run_opt['groups'][rg_idx]

            # Load software entry
            soft_id = run_group['software_id']
            software = dict_from_list(
                SimGroup.software_lookup['software'],
                {'id': soft_id}
            )
            # Load software name:
            software_name = dict_from_list(
                SimGroup.software_lookup['software_name'],
                {'id': software['software_name_id']}
            )['name']

            software['name'] = software_name

            if self.software_name is None:
                self.software_name = software_name

            elif self.software_name != software_name:
                raise ValueError('All run groups must use the same software '
                                 '(name)!')

            run_group['software'] = software

            rg_sim_idx = run_group['sim_idx']
            sim_idx_msg = ('Run group `sim_idx` must be either "all" or a '
                           'list of integers that index the sims.')

            if rg_sim_idx == 'all':
                run_group['sim_idx'] = list(range(self.num_sims))

            elif isinstance(rg_sim_idx, list):
                if min(rg_sim_idx) < 0 or max(rg_sim_idx) > (self.num_sims - 1):
                    raise ValueError(sim_idx_msg)

            else:
                raise ValueError(sim_idx_msg)

        return run_opt

    def _get_merged_updates(self):
        """Merge updates from 'parallel' sequences (those with the same `nest_idx`)"""

        # Get the updates for each sequence:
        seq_upds = []
        for seq in self.sequences:
            seq_upds.append(seq.updates)

        # Merge parallel sequences (those with same `nest_idx`):
        seq_upds_mergd = {}
        for idx, seq_i in enumerate(self.sequences):
            nest_idx = seq_i.nest_idx

            if nest_idx in seq_upds_mergd:
                mergd = merge(seq_upds_mergd[nest_idx], seq_upds[idx])
                seq_upds_mergd[nest_idx] = mergd

            else:
                seq_upds_mergd.update({
                    nest_idx: seq_upds[idx]
                })

        # Sort by `nest_idx`
        merged_updates = [val for _, val in sorted(seq_upds_mergd.items())]
        return merged_updates

    def _get_sim_updates(self):

        grp_upd = nest(*self._get_merged_updates())

        grp_upd_flat = []
        for upd_lst in grp_upd:
            upd_lst_flat = [j for i in upd_lst for j in i]
            grp_upd_flat.append(upd_lst_flat)

        return grp_upd_flat

    @property
    def num_sims(self):
        """Return the number of simulations in this group."""
        return len(self.sim_updates)

    def _apply_sim_update(self, base_dict, update):
        """Apply a BaseUpdate namedtuple to a dict.

        Parameters
        ----------
        base_dict : dict
            dict to which an update is applied.
        update : BaseUpdate
            BaseUpdate namedtuple with fields: `address`, `val`, `val_seq_type`
            and `mode`.

        """

        if update.mode == 'replace':
            upd_val = update.val

        elif update.mode == 'append':
            upd_val = get_recursive(base_dict, update.address, [])
            upd_val.append(update.val)

        upd_dict = set_nested_dict(update.address, upd_val)
        new_base_dict = update_dict(base_dict, upd_dict)

        return new_base_dict

    def get_sim_options(self, sim_idx=None):
        """Get options parameterising a given simulation belonging to this group."""

        sim_opt = copy.deepcopy(self.options)
        if sim_idx is not None:
            for upd in self.sim_updates[sim_idx]:
                sim_opt = self._apply_sim_update(sim_opt, upd)

        return sim_opt

    @property
    def sim_class(self):
        """Get the Simulation class associated with this group."""
        return SIM_CLASS_MAP[self.software_name]

    def get_sim(self, sim_idx=None):
        """Get a Simulation object belonging to this group."""

        sim_opt = self.get_sim_options(sim_idx)

        # prt(sim_opt, 'sim_opt')
        sim = self.sim_class(sim_opt)

        return sim

    @property
    def sequence_lengths(self):

        all_nest = [i.nest_idx for i in self.sequences]
        _, uniq_idx = np.unique(all_nest, return_index=True)
        ret = tuple([self.sequences[i].num_vals for i in uniq_idx])

        return ret

    def get_path_labels(self, sim_idx):

        num_elems = self.sequence_lengths
        ind = [sim_idx]
        for j in range(len(num_elems) - 1):
            prod_range = num_elems[::-1][:j + 1]
            prod = np.product(prod_range)
            ind.append(prod * int(ind[-1] / prod))

        return ind[::-1]

    def to_jsonable(self):
        """Generate a dict representation that can be JSON serialised."""

        # JSONify sim_updates
        sim_updates_js = []
        for i in self.sim_updates:
            i_js = []
            for base_update in i:

                base_upd_js = {
                    'address': base_update.address,
                    'val_seq_type': base_update.val_seq_type,
                    'mode': base_update.mode,
                }

                base_upd_val = base_update.val
                if base_update.val_seq_type == 'array':
                    base_upd_val = base_upd_val.tolist()

                base_upd_js.update({
                    'val': base_upd_val
                })

                i_js.append(base_upd_js)

            sim_updates_js.append(i_js)

        ret = {
            'sequences': [i.to_jsonable() for i in self.sequences],
            'sim_updates': sim_updates_js,
            'sims': [i.to_jsonable() for i in self.sims],
            'run_opt': self.run_opt,
            # software_name not strictly necessary, should re-parse this on import
            'software_name': self.software_name,
            'path_options': self.path_options,
            'options_unparsed': self.options_unparsed,
            'hid': self.hid,
            'job_name': self.job_name,
        }
        return ret

    @classmethod
    def from_jsonable(cls, state, opts_spec_path):
        """Instantiate from a JSONable dict."""

        sim_updates_ntv = []
        for i in state['sim_updates']:

            i_js = []

            for base_update in i:

                base_upd_val = base_update['val']

                if base_update['val_seq_type'] == 'array':
                    base_upd_val = np.array(base_upd_val)

                elif base_update['val_seq_type'] == 'tuple':
                    base_upd_val = tuple(base_upd_val)

                base_upd_ntv = BaseUpdate(
                    base_update['address'],
                    base_upd_val,
                    base_update['val_seq_type'],
                    base_update['mode'],
                )

                i_js.append(base_upd_ntv)

            sim_updates_ntv.append(i_js)

        # Get correct Simulation class from software method:
        sim_class = SIM_CLASS_MAP[state['software_name']]
        sims_native = [sim_class.from_jsonable(i) for i in state['sims']]

        # Parse options:
        with open(opts_spec_path, 'r') as opts_spec_path_fp:
            opts_spec = yaml.safe_load(opts_spec_path_fp)['makesims']

        opts_unparsed = state['options_unparsed']
        opts_parsed = opt_parser.parse_opt(opts_unparsed, opts_spec)

        opts_parsed.pop('sequences')
        opts_parsed.pop('run')
        opts_parsed.pop('path_options', {})

        seq_fn = os.path.join(SET_UP_PATH, OPT_FILE_NAMES['sequences'])
        SimSequence.load_sequence_definitions(seq_fn)

        state.update({
            'sequences': [SimSequence.from_jsonable(i) for i in state['sequences']],
            'sim_updates': sim_updates_ntv,
            'sims': sims_native,
            'run_opt': state['run_opt'],
            'software_name': state['software_name'],
            'path_options': state['path_options'],
            'options_unparsed': opts_unparsed,
            'options': opts_parsed,
            'hid': state['hid'],
            'job_name': state['job_name'],
        })

        sim_group = cls(state=state)

        # Finish restoring the state of the simulations
        for sim_idx in range(sim_group.num_sims):
            sim_opt = sim_group.get_sim_options(sim_idx)
            sim_opt.pop('structure')
            sim_group.sims[sim_idx].options = sim_opt

        return sim_group

    def save_state(self):
        """Write a JSON file representing this SimGroup object."""
        path = self.stage.path.joinpath('sim_group.json')
        with path.open('w') as sim_group_fp:
            json.dump(self.to_jsonable(), sim_group_fp, indent=2)

    @classmethod
    def load_state(cls, path, opts_spec_path):
        """Load a JSON file representing a SimGroup object."""

        with open(path, 'r') as sim_group_fp:
            sg_json = json.load(sim_group_fp)

        return cls.from_jsonable(sg_json, opts_spec_path)

    def write_initial_runs(self):
        """Populate the sims attribute and write input files on stage."""

        if self.sims is not None:
            raise ValueError('Simulations have already been generated for this'
                             ' SimGroup object.')

        # Check this machine is the stage machine
        if self.stage.machine_id != SimGroup.machine_id:
            raise ValueError('This machine does not have the same ID as that '
                             'of the Stage associated with this SimGroup.')

        stage_path = self.stage.path
        scratch_path = self.scratch.path
        stage_path.mkdir(parents=True)

        self.sim_class.copy_reference_data(
            self.options['params'][self.software_name], stage_path, scratch_path
        )

        # Generate sims:
        self.sims = [self.get_sim(i) for i in range(self.num_sims)]

        # Loop through each requested run group:
        for rg_idx, run_group in enumerate(self.run_opt['groups']):

            # Loop over each simulation in this run group
            sim_paths_stage = []
            sim_paths_scratch = []
            for sim_idx in run_group['sim_idx']:

                run_path = self.get_run_path(sim_idx, rg_idx)
                sm_pth_stg = stage_path.joinpath(*run_path)
                sm_pth_sct = scratch_path.joinpath(*run_path)
                sim_paths_stage.append(sm_pth_stg)
                sim_paths_scratch.append(str(sm_pth_sct))

                # Write simulation input files:
                self.sims[sim_idx].write_input_files(str(sm_pth_stg))
                self.sims[sim_idx].runs.append({
                    'run_group_id': rg_idx,
                    'run_state': 'on_stage',
                    'software_id': run_group['software_id'],
                    'result': None,
                    'num_cores': run_group['num_cores'],
                })

            # Write supporting files: jobscript, dirlist, options records
            rg_path = ['run_groups', str(rg_idx)]
            rg_path_stage = stage_path.joinpath(*rg_path)
            rg_path_scratch = scratch_path.joinpath(*rg_path)
            rg_path_stage.mkdir(parents=True)

            js_params = {
                'path': str(rg_path_stage),
                'calc_paths': sim_paths_scratch,
                'method': self.software_name,
                'num_cores': run_group['num_cores'],
                'is_sge': run_group['is_sge'],
                'job_array': False,
                'selective_submission': False,
                'scratch_os': self.scratch.os_type,
                'scratch_path': str(rg_path_scratch),
                'parallel_env': run_group['software']['parallel_env'],
                'module_load': run_group['software']['module_load'],
                'job_name': self.job_name,
                'seedname': 'sim',
                'executable': run_group['software']['executable'],
            }

            if run_group['is_sge']:
                js_params.update({
                    'job_array': run_group['sge']['job_array'],
                    'selective_submission': run_group['sge']['selective_submission'],
                })
            write_jobscript(**js_params)

    def submit_run_groups(self, run_group_idx):
        """Submit one or more run groups on Scratch. Can be invoked either on
        Stage (for submitting initial runs) or Scratch.

        Parameters
        ----------
        run_group_idx : list
            The indices of the run groups to submit.

        """

        conn = ResourceConnection(self.get_machine_resource()[0], self.scratch)
        sub_msg = ('Submitting run group: {}')
        for rg_idx in run_group_idx:

            run_group = self.run_opt['groups'][rg_idx]

            # CD to the run group directory and submit the jobscript:
            print(sub_msg.format(rg_idx))

            js_ext = 'sh' if self.scratch.os_type == 'posix' else 'bat'
            js_fn = 'jobscript.{}'.format(js_ext)
            rg_path = ('run_groups', str(rg_idx), js_fn)
            js_path = self.scratch.path.joinpath(*rg_path)

            if run_group['is_sge']:
                cmd = 'qsub {}'.format(js_path)
            else:
                cmd = str(js_path)

            conn.run_command(cmd)

    def copy_to_scratch(self):
        """Copy group from Stage to Scratch."""
        self.check_is_stage_machine()
        conn = self.get_stage_to_scratch_conn()
        conn.copy()

    def get_machine_resource(self):
        """Returns a list of Resource (Stage and/or Scratch) this machine
        corresponds to.

        Returns
        -------
        list of Resource
            If this machine is the Stage, the Stage resource will be the first
            element in the returned list.

        """

        ret = []
        if SimGroup.machine_id == self.stage.machine_id:
            ret.append(self.stage)

        if SimGroup.machine_id == self.scratch.machine_id:
            ret.append(self.scratch)

        return ret

    def check_is_stage_machine(self):
        """Check the current machine is the Stage machine for this group."""

        if not isinstance(self.get_machine_resource()[0], Stage):
            raise ValueError('This machine does not have the same ID as that '
                             'of the Stage associated with this SimGroup.')

    def get_stage_to_scratch_conn(self):
        """Get a ResourceConnection between Stage and Scratch."""
        self.check_is_stage_machine()
        return ResourceConnection(self.stage, self.scratch)

    def add_runs(self, opt):
        """Add additional run to one or more simulation on scratch."""

        # Need to check current machine is Scratch machine (config.yml)

    def get_path(self, resource):
        """Get the absolute path of the directory representing this SimGroup,
        on Stage, Scratch or Archive.
        """
        if resource == 'stage':
            resource_path = self.stage.path
        elif resource == 'scratch':
            resource_path = self.scratch.path
        elif resource == 'archive':
            resource_path = self.archive.path

        return [resource_path] + self.path_options['sub_dirs'] + [self.hid]

    def get_sim_path(self, sim_idx, ):
        """Get the path to a simulation directory.

        Parameters
        ----------
        sim_idx : int
            Index of simulation within the group.

        """

        sim_opt = self.get_sim_options(sim_idx)
        seq_id = sim_opt['sequence_id']

        path = []
        nest_idx = seq_id['nest_idx'][0] - 1

        for sid_idx in range(len(seq_id['paths'])):

            # prt(sid_idx, 'sid_idx')

            add_path = seq_id['paths'][sid_idx]
            # add_path = sid['path']

            if self.path_options['sequence_names']:
                # add_path = sid['name'] + '_' + add_path
                add_path = seq_id['names'][sid_idx] + '_' + add_path

            if seq_id['nest_idx'][sid_idx] == nest_idx:
                path[-1] += self.path_options['parallel_sims_join'] + add_path

            # if sid['nest_idx'] == nest_idx:
            #     path[-1] += self.path_options['parallel_sims_join'] + add_path
            else:
                path.append(add_path)

            nest_idx = seq_id['nest_idx'][sid_idx]

        if self.path_options['sim_numbers']:
            path_labs = self.get_path_labels(sim_idx)
            path = [str(i) + self.path_options['sim_numbers_join'] + j
                    for i, j in zip(path_labs, path)]

        path = [self.path_options['calcs_path']] + path

        return path

    def get_run_path(self, sim_idx, run_idx):
        """Get the path to a simulation run directory."""

        sim_path = self.get_sim_path(sim_idx)
        run_path = sim_path + [self.path_options['run_fmt'].format(run_idx)]

        return run_path
