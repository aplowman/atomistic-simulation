import os
import numpy as np
import shutil
import json
import copy
from pathlib import Path
from atsim.readwrite import read_pickle, write_pickle, format_list, format_dict
from atsim import simsio, utils, plotting, vectors, SCRIPTS_PATH, REF_PATH
from atsim.analysis import compute_funcs
from atsim.analysis.compute_funcs import get_depends, SINGLE_COMPUTE_LOOKUP, MULTI_COMPUTE_LOOKUP
from atsim.analysis.postprocess import SINGLE_COMPUTES, MULTICOMPUTE_LOOKUP, compute_gb_energy, dict_from_list, get_required_defn
from atsim.set_up.harvest_opt import HARVEST_OPT


# List of multi computes which require the common series info list:
REQUIRES_CSI = []


def make_plots(out):

    def format_title(name, val):
        d = zip(name, val)
        t = ['{}: {},'.format(i, out) for i, out in d]
        return ' '.join(t)[:-1]

    series_names = out['series_name']
    sesh_ids = out['session_id_idx']
    num_sims = len(sesh_ids)

    vrs = out['variables']

    for pl_idx, pl in enumerate(out['plots']):

        fmt = pl['fmt']
        lib = pl['lib']
        fn = pl['filename']
        file_srs = pl['file_series']
        subplot_srs = pl['subplot_series']
        trace_srs = pl['trace_series']
        all_data_defn = pl['data']

        all_data = []
        for ii_idx, i in enumerate(all_data_defn):

            if i['type'] == 'poly':
                x_vals = np.linspace(i['xmin'], i['xmax'])
                coeff_vals = dict_from_list(vrs, {'id': i['coeff_id']})['vals']
                if i.get('coeff_idx') is not None:
                    for sub_idx in i['coeff_idx']:
                        coeff_vals = coeff_vals[sub_idx]

                poly_y = []
                poly_x = []
                for k in coeff_vals:
                    if k is None:
                        poly_y.append(None)
                        poly_x.append(None)
                    else:
                        p1d = np.poly1d(k)
                        poly_y.append(p1d(x_vals))
                        poly_x.append(x_vals)

                ex_lst = ['type', 'coeff_id', 'coeff_idx', 'xmin', 'xmax']
                d = {
                    'type': 'marker',
                    'x': {
                        'vals': poly_x,
                    },
                    'y': {
                        'vals': poly_y,
                    },
                    **{k: v for k, v in i.items() if k not in ex_lst}
                }

            else:
                x_defn = dict_from_list(vrs, {'id': i['x']['id']})
                y_defn = dict_from_list(vrs, {'id': i['y']['id']})

                x, y = i['x'], i['y']
                x['vals'], y['vals'] = x_defn['vals'], y_defn['vals']

                if i['x'].get('idx') is not None:
                    for sub_idx in i['x']['idx']:
                        x['vals'] = x['vals'][sub_idx]

                if i['y'].get('idx') is not None:
                    for sub_idx in i['y']['idx']:
                        y['vals'] = y['vals'][sub_idx]
                d = {
                    'x': x,
                    'y': y,
                    **{k: v for k, v in i.items() if k not in ['x', 'y', 'z']}
                }

            if i['type'] == 'contour':

                z_defn = dict_from_list(vrs, {'id': i['z']['id']})
                z = i['z']
                z['vals'] = z_defn['vals']
                d.update({'z': z, })

                if i['z'].get('idx') is not None:
                    for sub_idx in i['z']['idx']:
                        z['vals'] = z['vals'][sub_idx]

                if not i.get('grid', False):

                    row_idx = dict_from_list(vrs, {'id': i['row_idx_id']})
                    col_idx = dict_from_list(vrs, {'id': i['col_idx_id']})
                    shp = dict_from_list(vrs, {'id': i['shape_id']})

                    # Grid shape will be a list with a value for each sim.
                    # Later, we take the first valid shape for a given data
                    # set.

                    d.update({
                        'row_idx': row_idx['vals'],
                        'col_idx': col_idx['vals'],
                        'shape': shp['vals'],
                    })

            all_data.append(d)

        all_types_srs = [file_srs, subplot_srs, trace_srs]
        all_types_srs_vals = [[], [], []]
        for srs_type_idx, srs_type in enumerate(all_types_srs):

            for i in srs_type:

                if i in series_names:
                    i_idx = series_names.index(i)
                    i_vals = utils.get_col(out['series_id']['val'], i_idx)
                else:
                    i_vals = dict_from_list(
                        out['variables'], {'id': i})['vals']

                all_types_srs_vals[srs_type_idx].append(i_vals)

        all_types_srs_vals = [utils.transpose_list(
            i) for i in all_types_srs_vals]

        file_srs_vals = all_types_srs_vals[0]
        subplot_srs_vals = all_types_srs_vals[1]
        trace_srs_vals = all_types_srs_vals[2]

        if len(file_srs_vals) == 0:
            file_srs_vals = [[0] for _ in range(num_sims)]

        if len(subplot_srs_vals) == 0:
            subplot_srs_vals = [[0] for _ in range(num_sims)]

        if len(trace_srs_vals) == 0:
            trace_srs_vals = [[0] for _ in range(num_sims)]

        unique_fsv = []
        unique_fsv_idx = []
        for f_idx, f in enumerate(file_srs_vals):
            if f in unique_fsv:
                unique_fsv_idx[unique_fsv.index(f)].append(f_idx)
            elif None in f:
                continue
            else:
                unique_fsv.append(f)
                unique_fsv_idx.append([f_idx])

        unique_ssv = []
        unique_ssv_idx = []
        for s_idx, s in enumerate(subplot_srs_vals):
            if s in unique_ssv:
                unique_ssv_idx[unique_ssv.index(s)].append(s_idx)
            elif None in s:
                continue
            else:
                unique_ssv.append(s)
                unique_ssv_idx.append([s_idx])

        unique_tsv = []
        unique_tsv_idx = []
        for t_idx, t in enumerate(trace_srs_vals):
            if t in unique_tsv:
                unique_tsv_idx[unique_tsv.index(t)].append(t_idx)
            elif None in t:
                continue
            else:
                unique_tsv.append(t)
                unique_tsv_idx.append([t_idx])

        all_f = []
        for f in unique_fsv_idx:
            all_s = []
            for s in unique_ssv_idx:
                all_t = []
                for t in unique_tsv_idx:
                    all_i = []
                    for i in t:
                        if i in f and i in s:
                            all_i.append(i)
                    all_t.append(all_i)
                all_s.append(all_t)
            all_f.append(all_s)

        figs = []
        for f_idx, f in enumerate(all_f):

            subplots = []
            for s_idx, s in enumerate(f):
                traces = []
                for t_idx, t in enumerate(s):
                    sub_traces = []
                    for d_idx, d in enumerate(all_data):

                        x_d = utils.index_lst(d['x']['vals'], t)
                        y_d = utils.index_lst(d['y']['vals'], t)

                        trm_idx = utils.trim_common_nones(
                            x_d, y_d, ret_idx=True)

                        if trm_idx is None:
                            trm_idx = []

                        yn_idx = [i_idx for i_idx,
                                  i in enumerate(y_d) if i is None]

                        x_d = utils.index_lst(x_d, yn_idx, not_idx=True)
                        y_d = utils.index_lst(y_d, yn_idx, not_idx=True)

                        trm_idx += yn_idx

                        if d['type'] in ['line', 'marker']:

                            if isinstance(x_d[0], list):
                                x_d = x_d[0]
                                y_d = y_d[0]

                            if d.get('sort'):
                                srt_idx = np.argsort(x_d)
                                x_d = list(np.array(x_d)[srt_idx])
                                y_d = list(np.array(y_d)[srt_idx])

                        x = copy.deepcopy(d['x'])
                        x['vals'] = x_d

                        y = copy.deepcopy(d['y'])
                        y['vals'] = y_d

                        st_dict = {
                            'x': x,
                            'y': y,
                            'type': d['type'],
                            'title': format_title(trace_srs, unique_tsv[t_idx]),
                            **{k: v for k, v in d.items() if k not in ['x', 'y', 'z']},
                        }

                        if d['type'] == 'contour':

                            z_d = copy.copy(utils.index_lst(d['z']['vals'], t))
                            z = copy.deepcopy(d['z'])
                            z_d[:] = [j for j_idx, j in enumerate(
                                z_d) if j_idx not in trm_idx]
                            z['vals'] = z_d

                            st_dict.update({'z': z})

                            if not d.get('grid', False):

                                row_idx = utils.index_lst(d['row_idx'], t)
                                col_idx = utils.index_lst(d['col_idx'], t)
                                shp = utils.index_lst(d['shape'], t)

                                for i in [row_idx, col_idx, shp]:
                                    i[:] = [j for j_idx, j in enumerate(
                                        i) if j_idx not in trm_idx]

                                # Take the shape as that from the first
                                # datapoint:

                                shp = shp[0]

                                st_dict.update({
                                    'row_idx': row_idx,
                                    'col_idx': col_idx,
                                    'shape': shp,
                                })

                        sub_traces.append(st_dict)

                    traces.append(sub_traces)

                subplots.append({
                    'traces': traces,
                    'title': format_title(subplot_srs, unique_ssv[s_idx]),
                })

            figs.append({
                'subplots': subplots,
                'title': format_title(file_srs, unique_fsv[f_idx]),
                **{k: v for k, v in pl.items() if k not in ['subplots']},
            })

        if pl['lib'] == 'mpl':
            plotting.plot_many_mpl(figs, save_dir=out['output_path'])
        else:
            raise NotImplementedError(
                'Library "{}" not supported.'.format(pl['lib']))


def read_results(sid, skip_idx=None, overwrite=False, query_all=False):
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

    sid_path = os.path.join(HARVEST_OPT['archive_path'], sid)
    sims = read_pickle(os.path.join(sid_path, 'sims.pickle'))
    method = sims['base_options']['method']

    s_count = 0
    for s_idx, sim_i in enumerate(sims['all_sims']):

        if skip_idx is not None and s_idx in skip_idx:
            continue

        s_count += 1
        srs_paths = []
        srs_id = sim_i.options.get('series_id')
        if srs_id is not None:
            for srs_id_lst in srs_id:
                srs_paths.append('_'.join([i['path'] for i in srs_id_lst]))

        calc_path = os.path.join(sid_path, 'calcs', *srs_paths)

        if method == 'castep':
            out = simsio.castep.read_castep_output(calc_path)

        elif method == 'lammps':
            out = simsio.lammps.read_lammps_output(calc_path)

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
    plots = res_opt.get('plots', [])

    # Make a list of all variable ids
    var_ids = []

    # Allowed variable types and required keys for each type
    var_req_keys = ['type', 'name', 'id']
    var_allowed_required = {
        'result': var_req_keys,
        'parameter': var_req_keys,
        'compute': var_req_keys,
        'series_id': var_req_keys + ['col_id'],
    }

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

    # Keys which are not sent to compute functions. So can pass arbitrary
    # parameters to compute functions by specifying them in the variable dict.
    var_std_keys = var_req_keys + ['display_name', 'idx', 'vals', ]

    # Variables ordered such that dependenices are listed before
    ordered_vars = []

    # Loop though variables: do validation and find compute dependencies:
    for vr_idx, vr in enumerate(res_opt['variables']):

        vr_type = vr['type']
        vr_name = vr['name']
        vr_name_idx = vr.get('idx')
        vr_id = vr['id']

        # print('vr_id: {}'.format(vr_id))

        # Check type is allowed:
        if vr_type not in var_allowed_required:
            raise ValueError('"{}" is not an allowed variable type: {}'.format(
                vr_type, var_allowed_required.keys()))

        # Check all required keys are given:
        for rk in var_allowed_required[vr_type]:
            if rk not in vr:
                rk_error = 'Variable #{} must have key: {}'.format(vr_idx, rk)
                raise ValueError(rk_error)

        # Check `id` is not repeated
        if vr_id not in var_ids:
            var_ids.append(vr_id)
        else:
            raise ValueError('Variable #{} id is not unique.'.format(vr_idx))

        ov_idx, ov = var_in_list(ordered_vars, vr, ret_idx=True)

        if ov is not None:
            # If ID already exists in ordered_vals, modify in ordered_vals
            resolve_var_id_conflict(ordered_vars, vr_id)
            # If already exists, update variable ID to match user-specified ID:
            ordered_vars[ov_idx]['id'] = vr_id

        else:
            if vr_type != 'compute':
                # If ID already exists in ordered_vals, modify in ordered_vals
                resolve_var_id_conflict(ordered_vars, vr_id)
                vr_copy = copy.deepcopy(vr)
                vr_copy.update({'vals': []})
                ordered_vars.append(vr_copy)
                continue

            else:
                cmp_kw = {k: v for k, v in vr.items() if k not in var_std_keys}
                dpnds = get_depends(vr_name, inc_id=True,
                                    inc_val=True, **cmp_kw)

                for d_idx, d in enumerate(dpnds):

                    dov = var_in_list(ordered_vars, d)
                    if dov is None:

                        if d_idx == len(dpnds) - 1:
                            # Last dependency is the variable itself, so user-
                            # specified ID takes precedence.
                            resolve_var_id_conflict(ordered_vars, vr_id)
                            d['id'] = vr_id

                        else:
                            d['id'] = resolve_var_id_conflict(
                                ordered_vars, d['id'], modify_existing=False)

                        ordered_vars.append(copy.deepcopy(d))

    # Start building output dict, which will be saved as a JSON file:
    out = {
        'session_id': [],
        'session_id_idx': [],
        'idx': [],
        'series_name': [],
        'plots': plots,
        'variables': ordered_vars,
        'rid': rs_id,
        'output_path': res_opt['output_path'],
    }

    # Get a list of lists of sims:
    all_sims = []
    for sid in res_opt['sid']:
        path = os.path.join(res_opt['archive_path'], sid)
        pick_path = os.path.join(path, 'sims.pickle')
        pick = read_pickle(pick_path)
        all_sims.append(pick['all_sims'])

    # Get a flat list of series names for this sim series and get all sims:
    all_srs_name = []
    for series_sims in all_sims:
        sm_0 = series_sims[0]
        sm_0_opt = sm_0.options
        if sm_0_opt.get('series_id') is not None:
            for series_id_list in sm_0_opt['series_id']:
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
        path = os.path.join(res_opt['archive_path'], sid)

        # Open the pickle file associated with this simulation series:
        pick_path = os.path.join(path, 'sims.pickle')
        pick = read_pickle(pick_path)
        sims = pick['all_sims']
        base_opt = pick['base_options']
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
                args = {k: v for k, v in vr.items() if k not in var_std_keys}

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

    # Now calculate variables which are multi `compute`s and `series_id`s:
    for vr_idx, vr in enumerate(out['variables']):

        vr_type = vr['type']
        vr_name = vr['name']

        if vr_type == 'series_id':
            cid = all_srs_name.index(vr['col_id'])
            vals = utils.get_col(all_ids[vr_name], cid)
            if vr.get('col_idx') is not None:
                vals = utils.get_col_none(vals, vr['col_idx'])
            out['variables'][vr_idx]['vals'] = vals

        elif vr_type == 'compute' and SINGLE_COMPUTE_LOOKUP.get(vr_name) is None:
            func = MULTI_COMPUTE_LOOKUP[vr_name]
            args = {k: v for k, v in vr.items() if k not in var_std_keys}
            if vr_name in REQUIRES_CSI:
                args = {**args, 'common_series_info': all_csi}
            func(out, **args)

    return out


def main(harvest_opt):

    sids = harvest_opt['sid']
    skip_idx = harvest_opt['skip_idx']
    overwrite = harvest_opt.get('overwrite', False)
    debug = harvest_opt.get('debug', False)

    if skip_idx is None or len(skip_idx) == 0:
        skip_idx = [[] for _ in range(len(sids))]

    for s_idx, s in enumerate(sids):
        read_results(s, skip_idx=skip_idx[s_idx], overwrite=overwrite)

    # Compute additional properties
    out = collate_results(harvest_opt, skip_idx=skip_idx, debug=debug)

    # Save the JSON file in the results directory of the first listed SID
    res_dir = os.path.join(harvest_opt['output_path'], out['rid'])

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

    # Generate plots
    make_plots(out)


if __name__ == '__main__':
    main(HARVEST_OPT)
