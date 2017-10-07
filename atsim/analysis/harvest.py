import os
import numpy as np
import shutil
import json
import copy
from pathlib import Path
from atsim.readwrite import read_pickle, write_pickle, format_list, format_dict
from atsim import utils, plotting, vectors, SCRIPTS_PATH, REF_PATH, OPT_FILE_NAMES
from atsim.simsio import castep, lammps
from atsim.analysis import compute_funcs
from atsim.analysis.compute_funcs import get_depends, SINGLE_COMPUTE_LOOKUP, MULTI_COMPUTE_LOOKUP
from atsim.utils import dict_from_list, get_unique_idx


# List of multi computes which require the common series info list:
REQUIRES_CSI = [
    'gamma_surface_info'
]
# Allowed variable types and required keys for each type
VAR_REQ_KEYS = ['type', 'name', 'id']
VAR_ALLOWED_REQUIRED = {
    'result': VAR_REQ_KEYS,
    'parameter': VAR_REQ_KEYS,
    'compute': VAR_REQ_KEYS,
    'series_id': VAR_REQ_KEYS + ['col_id'],
}

# Keys which are not sent to compute functions. So can pass arbitrary
# parameters to compute functions by specifying them in the variable dict.
VAR_STD_KEYS = VAR_REQ_KEYS + ['display_name', 'idx', 'vals', ]


def read_results(sid, archive_path, skip_idx=None, overwrite=False, query_all=False):
    """
    Parameters
    ----------
    sid : str
        Simulation series ID.
    skip_idx : list, optional
        List of simulation series indices to skip. Default is None, in which
        case none of the simulations are skipped. Useful for skipping failed
        simulations.
    overwrite : bool or str ("ask"), optional
        If True, overwrite previously recorded results. If False, do not
        overwrite previously recoreded results. If "ask", query user, in which
        case `query_all` controls whether user is asked for each simulation
        in the series or just the first. Default is False.
    query_all : bool, optional
        Only applies if `overwrite` is "ask". If True, user is asked whether to
        overwrite results for each simulation in the series. If False, user is
        asked for only the first simulation and the answer is rememered for
        the remaining simulations. Default is False.

    """
    sid_path = os.path.join(archive_path, sid)
    sims = read_pickle(os.path.join(sid_path, 'sims.pickle'))
    # Get options from first sim if they don't exist (legacy compatiblity)
    base_opt = sims.get('base_options', sims['all_sims'][0].options)
    method = base_opt['method']

    s_count = 0
    for s_idx, sim_i in enumerate(sims['all_sims']):

        if skip_idx is not None and s_idx in skip_idx:
            continue

        s_count += 1
        srs_paths = []
        srs_id = sim_i.options.get('series_id')
        if srs_id is not None:

            # (legacy compatibility)
            if isinstance(srs_id, dict) and len(srs_id) == 1:

                new_srs_id = []
                for k, v in srs_id.items():
                    new_srs_id.append([{'name': k, **v}])
                srs_id = new_srs_id

            if not isinstance(srs_id, dict):
                for srs_id_lst in srs_id:
                    srs_paths.append('_'.join([i['path'] for i in srs_id_lst]))

            else:
                raise ValueError('Cannot parse `series_id` option from '
                                 's_idx: {}'.format(s_idx))

        calc_path = os.path.join(sid_path, 'calcs', *srs_paths)

        if method == 'castep':
            out = castep.read_castep_output(calc_path)

        elif method == 'lammps':
            out = lammps.read_lammps_output(calc_path)

        results_exist = False
        if hasattr(sim_i, 'results'):
            results_exist = True
            if sim_i.results is None:
                results_exist = False

        if results_exist:
            if overwrite == True:
                # Overwrite without querying
                save_res = True

            elif overwrite == False:
                # Skip without querying
                continue

            elif overwrite == 'ask':
                # Ask user whether to overwrite. If `query_all` is True, user
                # is asked for each simulation, otherwise user is asked for the
                # first simulation, and the answer is applied for remaining
                # simulations.

                query_i = False
                if query_all:
                    query_i = True
                elif not query_all and s_count == 1:
                    query_i = True

                save_res = True
                if query_i:
                    save_res = False
                    msg = 'Results already collated for: {}'.format(sid)
                    if query_all:
                        msg += ' : {}'.format(s_idx)
                    msg += '. Overwrite?'

                    if utils.confirm(msg):
                        save_res = True
                    elif not query_all:
                        overwrite = False

        else:
            save_res = True

        if save_res:
            sims['all_sims'][s_idx].results = out

    pick_path = os.path.join(sid_path, 'sims.pickle')
    write_pickle(sims, pick_path)


def resolve_var_id_conflict(ordered_vars, vr_id, modify_existing=True):

    # Resolve any ID conflicts. Search for same ID in ordered_vars, rename
    # existing ordered_vars variable ID if conflict found. Conflict should
    # only exist between an ordered_vars variable added as a dependency,
    # since we check above for dupliate user-specified IDs.

    trial_id = vr_id
    id_match_idx, id_match = dict_from_list(ordered_vars, {'id': trial_id},
                                            ret_index=True)
    count = 1
    while id_match is not None:
        trial_id += '_{:d}'.format(count)
        id_match = dict_from_list(ordered_vars, {'id': trial_id})
        count += 1

    if id_match_idx is not None:
        if modify_existing:
            ordered_vars[id_match_idx]['id'] = trial_id
        else:
            return trial_id

    elif not modify_existing:
        return trial_id


def var_in_list(lst, var, ret_idx=False):
    var_cnd = {k: v for k, v in var.items() if k not in ['id']}
    return dict_from_list(lst, var_cnd, ret_index=ret_idx)


def get_reduced_depends(ordered_vars, vr, inc_id, inc_val):

    vr_type = vr['type']
    vr_name = vr['name']
    vr_name_idx = vr.get('idx')
    vr_id = vr['id']

    ov_idx, ov = var_in_list(ordered_vars, vr, ret_idx=True)

    if ov is not None:
        if inc_id:
            # If ID already exists in ordered_vals, modify in ordered_vals
            resolve_var_id_conflict(ordered_vars, vr_id)
            # If already exists, update variable ID to match user-specified ID:
            ordered_vars[ov_idx]['id'] = vr_id

    else:
        if vr_type != 'compute':
            # If ID already exists in ordered_vals, modify in ordered_vals
            if inc_id:
                resolve_var_id_conflict(ordered_vars, vr_id)
            vr_copy = copy.deepcopy(vr)
            vr_copy.update({'vals': []})
            ordered_vars.append(vr_copy)
            return ordered_vars

        else:
            cmp_kw = {k: v for k, v in vr.items() if k not in VAR_STD_KEYS}
            dpnds = get_depends(vr_name, inc_id=inc_id,
                                inc_val=inc_val, **cmp_kw)

            # print('depends: \n{}\n'.format(dpnds))

            for d_idx, d in enumerate(dpnds):

                dov = var_in_list(ordered_vars, d)
                if dov is None:

                    if inc_id:
                        if d_idx == len(dpnds) - 1:
                            # Last dependency is the variable itself, so user-
                            # specified ID takes precedence.
                            resolve_var_id_conflict(ordered_vars, vr_id)
                            d['id'] = vr_id

                        else:
                            d['id'] = resolve_var_id_conflict(
                                ordered_vars, d['id'], modify_existing=False)

                    ordered_vars.append(copy.deepcopy(d))
    return ordered_vars


def collate_results(res_opt, skip_idx=None, debug=False):
    """
    Save a JSON file containing the results of one of more simulation series.

    Idea is to build a dict (saved as a JSON file) which has results from
    simulations in flat lists.

    """

    rs_date, rs_num = utils.get_date_time_stamp(split=True)
    rs_id = rs_date + '_' + rs_num
    if debug:
        rs_id = '0000-00-00-0000_00000'

    def append_series_items(series_items, series_id, num_series, sim_idx, srs_names):

        out = {
            'path': []
        }
        for i in series_id:
            path = []
            for j in i:
                srs_idx = srs_names.index(j['name'])
                for k, v in j.items():

                    if k == 'path':
                        path.append(v)
                        continue
                    if k not in out:
                        out.update({k: [None] * num_series})
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    out[k][srs_idx] = v

            path_join = '_'.join([str(i) for i in path])
            out['path'].append(path_join)

        for k, v in out.items():
            if k in series_items:
                series_items[k].extend([v])
            else:
                blank = [None] * num_series
                series_items.update({k: [blank] * sm_idx + [v]})

        for k, v in series_items.items():
            if k not in out:
                blank = [None] * num_series
                series_items[k].append(blank)

        return series_items

    computes = []
    add_vars = []

    # Make a list of all variable ids
    var_ids = []

    # Variables ordered such that dependenices are listed before
    ordered_vars = []

    # Loop though variables: do validation
    for vr_idx, vr in enumerate(res_opt['variables']):

        vr_type = vr['type']
        vr_name = vr['name']
        vr_name_idx = vr.get('idx')
        vr_id = vr['id']

        # Check type is allowed:
        if vr_type not in VAR_ALLOWED_REQUIRED:
            raise ValueError('"{}" is not an allowed variable type: {}'.format(
                vr_type, VAR_ALLOWED_REQUIRED.keys()))

        # Check all required keys are given:
        for rk in VAR_ALLOWED_REQUIRED[vr_type]:
            if rk not in vr:
                rk_error = 'Variable #{} must have key: {}'.format(vr_idx, rk)
                raise ValueError(rk_error)

        # Check `id` is not repeated
        if vr_id not in var_ids:
            var_ids.append(vr_id)
        else:
            raise ValueError('Variable #{} id is not unique.'.format(vr_idx))

        ordered_vars = get_reduced_depends(
            ordered_vars, vr, inc_id=True, inc_val=True)

    # Start building output dict, which will be saved as a JSON file:
    out = {
        'session_id': [],
        'session_id_idx': [],
        'idx': [],
        'series_name': [],
        'variables': ordered_vars,
        'rid': rs_id,
        'output_path': res_opt['output']['path'],
    }

    # Get a list of lists of sims:
    all_sims = []
    for sid in res_opt['sid']:
        path = os.path.join(res_opt['archive']['path'], sid)
        pick_path = os.path.join(path, 'sims.pickle')
        pick = read_pickle(pick_path)
        all_sims.append(pick['all_sims'])

    # Get a flat list of series names for this sim series and get all sims:
    all_srs_name = []
    for series_sims in all_sims:
        sm_0 = series_sims[0]
        sm_0_opt = sm_0.options

        srs_id = sm_0_opt.get('series_id')
        if srs_id is not None:

            # (legacy compatibility)
            if isinstance(srs_id, dict) and len(srs_id) == 1:

                new_srs_id = []
                for k, v in srs_id.items():
                    new_srs_id.append([{'name': k, **v}])
                srs_id = new_srs_id

            for series_id_list in srs_id:
                for series_id_sublist in series_id_list:
                    nm = series_id_sublist['name']
                    if nm not in all_srs_name:
                        all_srs_name.append(nm)

    # Need better logic later to avoid doing this:
    if 'gamma_surface' in all_srs_name:
        all_srs_name[all_srs_name.index('gamma_surface')] = 'relative_shift'
    out['series_name'] = all_srs_name

    # Collect common series info list for each simulation series:
    all_csi = []

    # Loop through each simulation series to append vals to `result`,
    # `parameter` and single `compute` variable types:
    all_ids = {}
    all_sim_idx = 0
    for sid_idx, sid in enumerate(res_opt['sid']):

        skips = skip_idx[sid_idx]
        path = os.path.join(res_opt['archive']['path'], sid)

        # Open the pickle file associated with this simulation series:
        pick_path = os.path.join(path, 'sims.pickle')
        pick = read_pickle(pick_path)
        sims = pick['all_sims']
        # Get options from first sim if they don't exist (legacy compatiblity)
        base_opt = pick.get('base_options', sims[0].options)
        all_csi.append(pick.get('common_series_info'))

        # Loop through each simulation for this series
        for sm_idx, sm in enumerate(sims):

            if sm_idx in skips:
                continue

            if sid in out['session_id']:
                sid_idx = out['session_id'].index(sid)
            else:
                out['session_id'].append(sid)
                sid_idx = len(out['session_id']) - 1

            out['session_id_idx'].append(sid_idx)
            out['idx'].append(sm_idx)

            srs_id = sm.options.get('series_id')
            if srs_id is not None:

                # (legacy compatibility)
                if isinstance(srs_id, dict) and len(srs_id) == 1:

                    new_srs_id = []
                    for k, v in srs_id.items():
                        new_srs_id.append([{'name': k, **v}])
                    srs_id = new_srs_id

            if srs_id is None:
                srs_id = [[]]

            all_ids = append_series_items(all_ids,
                                          srs_id,
                                          len(all_srs_name),
                                          all_sim_idx,
                                          all_srs_name)

            # Loop through requested variables:
            for vr_idx, vr in enumerate(out['variables']):

                vr_name = vr['name']
                vr_type = vr['type']
                args = {k: v for k, v in vr.items() if k not in VAR_STD_KEYS}

                if vr_type not in ['result', 'parameter', 'compute']:
                    continue

                if vr_type == 'result':
                    val = sm.results[vr_name]
                elif vr_type == 'parameter':
                    val = sm.options[vr_name]
                elif vr_type == 'compute':
                    func_name = SINGLE_COMPUTE_LOOKUP.get(vr_name)
                    if func_name is not None:
                        val = func_name(out, sm, all_sim_idx, **args)
                    else:
                        # Must be a multi compute
                        continue

                all_sub_idx = vr.get('idx')
                if all_sub_idx is not None:
                    for sub_idx in all_sub_idx:
                        if vr_type == 'parameter':
                            try:
                                val = val[sub_idx]
                            except KeyError:
                                val = vr.get('default')
                                break

                        else:
                            val = val[sub_idx]

                # To ensure the data is JSON compatible:
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                elif isinstance(val, np.generic):
                    val = np.asscalar(val)

                out['variables'][vr_idx]['vals'].append(val)

            all_sim_idx += 1

    all_ids = {k: v for k, v in all_ids.items() if k != 'name'}
    out['series_id'] = all_ids

    all_vrs = out['variables']
    # Now calculate variables which are multi `compute`s and `series_id`s:
    for vr_idx, vr in enumerate(all_vrs):

        vr_type = vr['type']
        vr_name = vr['name']

        if vr_type == 'series_id':
            cid = all_srs_name.index(vr['col_id'])
            vals = utils.get_col(all_ids[vr_name], cid)
            if vr.get('col_idx') is not None:
                vals = utils.get_col_none(vals, vr['col_idx'])
            all_vrs[vr_idx]['vals'] = vals

        elif vr_type == 'compute' and SINGLE_COMPUTE_LOOKUP.get(vr_name) is None:
            func = MULTI_COMPUTE_LOOKUP[vr_name]
            req_vars_defn = get_reduced_depends(
                [], vr, inc_id=False, inc_val=False)
            req_vars = [dict_from_list(all_vrs, i) for i in req_vars_defn]

            if vr_name in REQUIRES_CSI:
                func(out, req_vars, common_series_info=all_csi)
            else:
                func(out, req_vars)

    return out


def main(harvest_opt):

    sids = harvest_opt['sid']
    skip_idx = harvest_opt['skip_idx']
    overwrite = harvest_opt.get('overwrite', False)
    archive_path = harvest_opt['archive']['path']
    debug = harvest_opt.get('debug', False)

    if skip_idx is None or len(skip_idx) == 0:
        skip_idx = [[] for _ in range(len(sids))]
    elif len(skip_idx) > 0:
        no_skips = True
        for sk in skip_idx:
            if len(sk) > 0:
                no_skips = False
                break
        if no_skips:
            skip_idx = [[] for _ in range(len(sids))]

    for s_idx, s in enumerate(sids):
        read_results(s, archive_path, skip_idx=skip_idx[s_idx], overwrite=overwrite)

    # Compute additional properties
    out = collate_results(harvest_opt, skip_idx=skip_idx, debug=debug)

    # Save the JSON file in the results directory of the first listed SID
    res_dir = os.path.join(harvest_opt['output']['path'], out['rid'])

    os.makedirs(res_dir, exist_ok=True)
    json_fn = 'results.json'
    json_path = os.path.join(res_dir, json_fn)

    # Save a copy of the input results options
    src_path = os.path.join(SCRIPTS_PATH, 'set_up', 'harvest_opt.py')
    dst_path = os.path.join(res_dir, 'harvest_opt.py')
    shutil.copy(src_path, res_dir)

    with open(json_path, 'w', encoding='utf-8', newline='') as jf:
        print('Saving {} in {}'.format(json_fn, res_dir))
        json.dump(out, jf, sort_keys=True, indent=4)
