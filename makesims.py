import os
import numpy as np
import shutil
import dict_parser
import utils
import atomistic
from bravais import BravaisLattice
from crystal import CrystalStructure
import copy


SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
REF_PATH = os.path.join(SCRIPTS_PATH, 'ref')
SU_PATH = os.path.join(SCRIPTS_PATH, 'set_up')


def prepare_series_update(series_spec, atomistic_structure):
    """
    Return a list of dicts representing each element in a simulation series.

    Parameters
    ----------
    series_spec : dict
        Specification of the series.
    atomistic_structure : AtomisticStructure

    """

    # Convenience
    ss = series_spec
    sn = ss['name']
    start = ss.get('start')
    step = ss.get('step')
    stop = ss.get('stop')
    vals = ss.get('vals')

    # Validation

    allowed_sn = [
        'kpoint',
        'cut_off_energy',
        'smearing_width'
    ]

    if sn not in allowed_sn:
        raise NotImplementedError('Series name: {} not understood.'.format(sn))

    params_none = [i is None for i in [start, step, stop]]
    if any(params_none) and not all(params_none):
        raise ValueError('Must specify all of `start`, `step` and `stop` '
                         'if one is specified, for series name: {}'.format(sn))

    if vals is not None and start is not None:
        raise ValueError('Either specify (`start`, `step` and `stop`) or '
                         '`vals`, but not both, for series name: {}'.format(
                             sn))

    # If start, step and stop are provided, generate a set of vals from these:
    if vals is None:
        diff = start - stop if start > stop else stop - start
        num = (diff + step) / step
        vals = np.linspace(start, stop, num=num)
    else:
        # TODO: parse data types in option file
        vals = [float(v) for v in vals]

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

    out = []

    # Maybe refactor this logic later:

    if sn == 'kpoint':

        for v in vals:

            v = float(v)  # TODO: parse data types in option file
            out.append({
                'castep': {
                    'cell': {'kpoint_mp_spacing': '{:.3f}'.format(v)}},
                'series_id': {
                    'kpoint': {'val': v, 'path': '{:.3f}'.format(v)}}
            })

    elif sn == 'cut_off_energy':

        for v in vals:

            v = float(v)  # TODO: parse data types in option file
            out.append({
                'castep': {
                    'param': {'cut_off_energy': '{:.0f}'.format(v)}},
                'series_id': {
                    'cut_off_energy': {'val': v, 'path': '{:.0f}'.format(v)}}
            })

    elif sn == 'smearing_width':

        for v in vals:

            v = float(v)  # TODO: parse data types in option file
            out.append({
                'castep': {
                    'param': {'smearing_width': '{:.2f}'.format(v)}},
                'series_id': {
                    'smearing_width': {'val': v, 'path': '{:.2f}'.format(v)}}
            })

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

    """

    srs_update = []
    for i in all_series_spec:

        if isinstance(i, dict):
            srs_update.append(prepare_series_update(i, atomistic_structure))

        elif isinstance(i, list):
            sub_up = []
            for j in i:
                sub_up.append(prepare_series_update(j, atomistic_structure))

            srs_update.append(sub_up)
    """
        Given update data for nested series passed in this form: [data_1, [data_2, data_3]], where
        data_i are lists of dicts, each element in the list represents a series to be
        nested and each element in a sublist represents a set of parallel series which
        must have the same length, combine the series to form and return a list of
        the necessary combinations of the update data, such that each element in the return list
        corresponds to a single simulation.
    
    """

    # Now combine update data for each series into a flat list of dicts, where
    # each dict represents the update data for a single simulation series
    # element.
    su_flat = []
    for i in srs_update:

        if isinstance(i[0], dict):
            su_flat.append(i)

        elif isinstance(i[0], list):

            if len(set([len(j) for j in i])) > 1:
                raise ValueError('Parallel series must have the same length.')

            si_sub = []
            for j in utils.transpose_list(i):
                si_sub.extend([utils.combine_list_of_dicts(j)])

            su_flat.append(si_sub)

    all_updates_lst = utils.nest_lists(su_flat)
    all_updates = [utils.combine_list_of_dicts(i) for i in all_updates_lst]
    return all_updates


def process_castep_opt(opt):
    """

    """

    if opt.get('checkpoint') is True:
        if opt.get('backup_interval') is not None:
            opt['param'].update(
                {'backup_interval': opt['backup_interval']})

    else:
        opt['param'].update({'write_checkpoint': 'none'})

    opt.pop('backup_interval', None)
    opt.pop('checkpoint', None)

    task = opt['param']['task']

    if task == 'SinglePoint':
        opt['cell_constraints'].pop('cell_angles_equal', None)
        opt['cell_constraints'].pop('cell_lengths_equal', None)
        opt['cell_constraints'].pop('fix_cell_angles', None)
        opt['cell_constraints'].pop('fix_cell_lengths', None)
        opt['atom_constraints'].pop('fix_xy_idx', None)
        opt['atom_constraints'].pop('fix_xyz_idx', None)

    elif task == 'GeometryOptimisation':

        # atom constraints are parsed as 2D arrays (pending todo of
        # dict-parser) (want 1D arrays)

        for k, v in opt['atom_constraints'].items():
            if isinstance(v, np.ndarray):
                opt['atom_constraints'][k] = v[:, 0]


def main():
    """
    Read the options file and generate a simulation (series).

    TODO:
    -   Validation:
        -   check all ints in cs_idx resolve in crystal_structures
    -   Allow datatype parsing on list elements so specifying crystal structure
        index when forming struct_opt is cleaner.
    -   Allow dict_parser to parse other files so don't need csl_lookup (can
        have a file in /ref: csl_hex_[0001].txt, which can provide csl_vecs for
        a given sigma value.)
    -   Also allow dict_parser to have variables so crystal structure can be
        reference as a variable instead of an index: instead of in opt.txt:
        base_structure --> cs_idx = [0] we could have base_structure 
        --> crystal_structure = <crystal_structures>[0] where 
        crystal_structures is also defined in the opt.txt

    """

    struct_lookup = {
        'BulkCrystal': atomistic.BulkCrystal,
        'CSLBicrystal': atomistic.CSLBicrystal,
        'CSLBulkCrystal': atomistic.CSLBulkCrystal,
        'CSLSurfaceCrystal': atomistic.CSLSurfaceCrystal
    }
    csl_lookup = {
        7: [
            np.array([
                [3, 2, 0],
                [1, 3, 0],
                [0, 0, 1]
            ]),
            np.array([
                [2, 3, 0],
                [-1, 2, 0],
                [0, 0, 1]
            ]),
        ],
        13: [
            np.array([
                [4, 3, 0],
                [1, 4, 0],
                [0, 0, 1]
            ]),
            np.array([
                [3, 4, 0],
                [-1, 3, 0],
                [0, 0, 1]
            ]),
        ],
        19: [
            np.array([
                [5, 2, 0],
                [3, 5, 0],
                [0, 0, 1]
            ]),
            np.array([
                [5, 3, 0],
                [2, 5, 0],
                [0, 0, 1]
            ]),
        ],
        31: [
            np.array([
                [6, -1, 0],
                [1, 5, 0],
                [0, 0, 1]
            ]),
            np.array([
                [5, 1, 0],
                [-1, 6, 0],
                [0, 0, 1]
            ]),
        ],
    }
    log = []

    # Read the options file
    log.append('Reading log file.')
    opt_path = os.path.join(SU_PATH, 'opt.txt')
    opt = dict_parser.parse_dict_file(opt_path)

    # Modify options dictionary to include additional info

    s_date, s_num = utils.get_date_time_stamp(split=True)
    s_id = s_date + '_' + s_num
    opt['set_up']['session_id'] = s_id
    opt['set_up']['job_name'] = "j_" + s_num

    scratch_os = opt['set_up']['scratch_os']
    stage_os = os.name

    if scratch_os == 'nt':
        scratch_path_sep = '\\'
    elif scratch_os == 'posix':
        scratch_path_sep = '/'

    stage_path = os.path.join(opt['set_up']['stage_path'], s_id)
    print('stage_path: {}'.format(stage_path))

    scratch_path = scratch_path_sep.join([opt['set_up']['scratch_path'], s_id])
    print('scratch_path: {}'.format(scratch_path))

    log.append('Making stage directory at: {}.'.format(stage_path))
    os.makedirs(stage_path, exist_ok=False)

    # Generate CrystalStructure objects:
    log.append('Generating CrystalStructure objects.')
    cs = []
    for cs_opt in opt['crystal_structures']:
        brav_lat = BravaisLattice(**cs_opt['lattice'])
        cs.append(CrystalStructure(brav_lat, cs_opt['motif']))

    # Generate base structure
    log.append('Generating base AtomisticStructure object.')
    struct_opt = {}
    base_as_opt = opt['base_structure']
    for k, v in base_as_opt.items():
        if k == 'type':
            continue
        elif k == 'cs_idx':
            struct_opt.update({'crystal_structure': cs[v[0][0]]})
        elif k == 'sigma':
            struct_opt.update({'csl_vecs': csl_lookup[v]})
        else:
            struct_opt.update({k: v})

    base_as = struct_lookup[base_as_opt['type']](**struct_opt)

    # Visualise base AtomisticStructure:
    save_args = {
        'filename': os.path.join(stage_path, 'base_structure.html'),
        'auto_open': False
    }
    base_as.visualise(show_iplot=False, save=True, save_args=save_args)

    # Save original options file
    opt_src_path = os.path.join(SU_PATH, 'opt.txt')
    opt_dst_path = os.path.join(stage_path, 'opt_in.txt')
    shutil.copy(opt_src_path, opt_dst_path)

    # Save current options dict
    opt_p_str_path = os.path.join(stage_path, 'opt_processed.txt')
    with open(opt_p_str_path, mode='w', encoding='utf-8') as f:
        f.write(dict_parser.formatting.format_dict(opt))

    # Get series definitions:
    srs_df = opt['series']

    # Prepare series update data:
    all_upd = prepare_all_series_updates(srs_df, base_as)

    all_scratch_paths = []

    # Generate simulation series:
    for upd_idx, upd in enumerate(all_upd):

        # Update options:
        srs_opt = copy.deepcopy(opt)
        utils.update_dict(srs_opt, upd)

        # Generate AtomisticStructure:
        log.append('Generating series AtomisticStructure object.')
        srs_struct_opt = {}
        srs_as_opt = srs_opt['base_structure']
        for k, v in srs_as_opt.items():
            if k == 'type':
                continue
            elif k == 'cs_idx':
                srs_struct_opt.update({'crystal_structure': cs[v[0][0]]})
            elif k == 'sigma':
                srs_struct_opt.update({'csl_vecs': csl_lookup[v]})
            else:
                srs_struct_opt.update({k: v})

        srs_as = struct_lookup[srs_as_opt['type']](**srs_struct_opt)

        # Form the directory path for this sim:
        srs_path = []
        for i in srs_df:

            if isinstance(i, dict):
                srs_path.append(upd['series_id'][i['name']]['path'])

            elif isinstance(i, list):
                srs_path.append(
                    '_'.join([upd['series_id'][j['name']]['path'] for j in i]))

        stage_srs_path = os.path.join(stage_path, 'calcs', *srs_path)
        scratch_srs_path = scratch_path_sep.join(
            [scratch_path, 'calcs', *srs_path])

        all_scratch_paths.append(scratch_srs_path)
        srs_opt['set_up']['stage_series_path'] = stage_srs_path
        srs_opt['set_up']['scratch_srs_path'] = scratch_srs_path

        # print('stage_srs_path: {}'.format(stage_srs_path))
        # print('scratch_srs_path: {}'.format(scratch_srs_path))

        # Process CASTEP options
        if srs_opt['method'] == 'castep':
            process_castep_opt(srs_opt['castep'])

        # Generate AtomisticSim:
        asim = atomistic.AtomisticSimulation(srs_as, srs_opt)
        asim.write_input_files()


if __name__ == '__main__':
    main()
