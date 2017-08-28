from readwrite import read_pickle, write_pickle, format_list, format_dict
import simsio
import os
import numpy as np
import utils
import shutil
import json
import copy
from set_up.opt_test_res import RES_OPT
from postprocess import SINGLE_COMPUTES, MULTICOMPUTE_LOOKUP, compute_gb_energy, dict_from_list, get_required_defn
import plotting
import vectors

SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
REF_PATH = os.path.join(SCRIPTS_PATH, 'ref')
SU_PATH = os.path.join(SCRIPTS_PATH, 'set_up')
# HOME_PATH = r'C:\Users\{}\Dropbox (Research Group)\calcs'.format(
#     os.getlogin())
HOME_PATH = r'C:\calcs_archive'.format(os.getlogin())
RES_PATH = r'C:\Users\{}\Dropbox (Research Group)\calcs_results'.format(
    os.getlogin())


def do_plots(out):

    def format_title(name, val):
        d = zip(name, val)
        t = ['{}: {},'.format(i, out) for i, out in d]
        return ' '.join(t)[:-1]

    series_names = out['series_name']
    sesh_ids = out['session_id']
    num_sims = len(sesh_ids)

    vrs = out['variables']

    for pl in out['plots']:

        fmt = pl['fmt']
        lib = pl['lib']
        fn = pl['filename']
        file_srs = pl['file_series']
        subplot_srs = pl['subplot_series']
        trace_srs = pl['trace_series']
        all_data_defn = pl['data']

        all_data = []
        for i in all_data_defn:

            x = dict_from_list(vrs, {'id': i['x']['id']})['vals']
            y = dict_from_list(vrs, {'id': i['y']['id']})['vals']

            d = {
                'x': x,
                'y': y,
                **{k: v for k, v in i.items() if k not in ['x', 'y', 'z']}
            }
            if i['type'] == 'contour':
                row_idx = dict_from_list(vrs, {'id': i['row_idx_id']})['vals']
                col_idx = dict_from_list(vrs, {'id': i['col_idx_id']})['vals']
                shape = dict_from_list(vrs, {'id': i['shape_id']})['vals']
                z = dict_from_list(vrs, {'id': i['z']['id']})['vals']
                d.update({
                    'z': z,
                    'row_idx': row_idx,
                    'col_idx': col_idx,
                    'shape': shape,
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

        # print('unique_fsv_idx: \n{}\n'.format(unique_fsv_idx))
        # print('unique_ssv_idx: \n{}\n'.format(unique_ssv_idx))
        # print('unique_tsv_idx: \n{}\n'.format(unique_tsv_idx))

        # print('unique_fsv: \n{}\n'.format(unique_fsv))
        # print('unique_ssv: \n{}\n'.format(unique_ssv))
        # print('unique_tsv: \n{}\n'.format(unique_tsv))

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

        print('all_f: \n{}\n'.format(all_f))

        figs = []
        for f_idx, f in enumerate(all_f):
            subplots = []
            for s_idx, s in enumerate(f):
                traces = []
                for t_idx, t in enumerate(s):
                    sub_traces = []
                    for d in all_data:

                        x_d = utils.index_lst(d['x'], t)
                        y_d = utils.index_lst(d['y'], t)

                        if d['type'] in ['line', 'marker']:

                            utils.trim_common_nones(x_d, y_d)
                            if d.get('sort'):
                                srt_idx = np.argsort(x_d)
                                x_d = list(np.array(x_d)[srt_idx])
                                y_d = list(np.array(y_d)[srt_idx])

                        # if isinstance(x_d[0], list):
                        #     # In this case, we're getting a data set from a
                        #     # single simulation
                        #     x_d = x_d[0]
                        #     y_d = y_d[0]

                        st_dict = {
                            'x': x_d,
                            'y': y_d,
                            'type': d['type'],
                            'title': format_title(trace_srs, unique_tsv[t_idx]),
                            **{k: v for k, v in d.items() if k not in ['x', 'y', 'z']},
                        }

                        if d['type'] == 'contour':
                            z_d = utils.index_lst(d['z'], t)
                            st_dict.update({'z': z_d})

                        sub_traces.append(st_dict)

                    traces.append(sub_traces)

                subplots.append({
                    'traces': traces,
                    'title': format_title(subplot_srs, unique_ssv[s_idx]),
                })

            figs.append({
                'subplots': subplots,
                'title': format_title(file_srs, unique_fsv[f_idx]),
                'fmt': pl['fmt'],
                'filename': pl['filename'],
            })

        print('figs: \n{}\n'.format(format_list(figs)))

        if pl['lib'] == 'mpl':
            plotting.plot_many_mpl(figs, save_dir=out['dir'])
        else:
            raise NotImplementedError(
                'Library "{}" not supported.'.format(pl['lib']))


def read_results(sid, skip_idx=None, query_all=False):
    """
    """

    sid_path = os.path.join(HOME_PATH, sid)
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
            out = simsio.read_castep_output(calc_path)

        elif method == 'lammps':
            out = simsio.read_lammps_output(calc_path)

        query_i = False
        if query_all:
            query_i = True
        elif not query_all and s_count == 1:
            query_i = True

        save_res = True
        if query_i and hasattr(sim_i, 'results'):
            save_res = False
            msg = 'Results already collated for: {}'.format(sid)
            if query_all:
                msg += ' : {}'.format(s_idx)
            msg += '. Overwrite?'
            if utils.confirm(msg):
                save_res = True

        if save_res:
            sims['all_sims'][s_idx].results = out

    pick_path = os.path.join(sid_path, 'sims.pickle')
    write_pickle(sims, pick_path)


def collate_results(res_opt, debug=False):
    """
    Save a JSON file containing the results of one of more simulation series.

    Idea is to build a dict (saved as a JSON file) which has results from
    simulations in flat lists.

    """

    rs_date, rs_num = utils.get_date_time_stamp(split=True)
    rs_id = rs_date + '_' + rs_num

    # TEMP:
    if debug:
        rs_id = '0000-00-00-0000_00000'

    def get_series_items(series_id):
        out = {
            'path': []
        }
        idx = 0
        for i in series_id:
            path = []
            for j in i:
                for k, v in j.items():
                    if k == 'path':
                        path.append(v)
                        continue
                    if k not in out:
                        out.update({k: [None] * idx})
                    out[k].append(v)
                for k, v in out.items():
                    if k not in j and k != 'path':
                        out[k].append(None)
                idx += 1
            path_join = '_'.join([str(i) for i in path])
            out['path'].append(path_join)

        return out

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

    def get_series_vals(series_id):
        srs_vals = []
        for srs_id_lst in series_id:
            for i in srs_id_lst:
                v = i['val']
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                srs_vals.append(v)
        return srs_vals

    def get_series_paths(series_id):
        srs_paths = []
        for srs_id_lst in series_id:
            srs_paths.append('_'.join([i['path'] for i in srs_id_lst]))
        return srs_paths

    variables = []
    computes = []
    computes_idx = []
    additional_vars = []
    ids = []
    required_keys = ['name', 'type', 'id', ]
    std_keys = required_keys + ['display_name', 'idx', 'vals', ]
    allowed_types = ['result', 'parameter',
                     'compute', 'common_series_info', 'series_id']
    for vr_idx, vr in enumerate(res_opt['variables']):

        # Check type is allowed:
        if vr['type'] not in allowed_types:
            raise ValueError('"{}" is not an allowed `type`: {}'.format(
                vr['type'], allowed_types))

        # Check all required keys are given:
        for rk in required_keys:
            if rk not in vr:
                rk_error = 'Variable #{} must have key: {}'.format(vr_idx, rk)
                raise ValueError(rk_error)

        # Check `id` is not repeated
        if vr['id'] not in ids:
            ids.append(vr['id'])
        else:
            raise ValueError('Variable `id` must be unique.')

        vr_copy = copy.deepcopy(vr)
        vr_copy.update({'vals': []})

        if vr['type'] == 'compute':
            computes_idx.append(vr_idx)
            computes.append(vr_copy)
            compute_kwargs = {k: v for k, v in vr.items() if k not in std_keys}
            additional_vars += get_required_defn(vr['name'], **compute_kwargs)
        else:
            variables.append(vr_copy)

    # Merge additional variables due to computes to variables list
    # If the variable `name`, `type` and `idx` all match, combine, keeping
    # user-specified variable `id`, otherwise add as an additional variable.
    # If an `id` is already in use, append a number to the `id`.
    for advar in additional_vars:

        # Find if the variable already exists (i.e. if it was user-specified):
        conditions = {
            'name': advar['name'],
            'type': advar['type'],
        }
        advar_idx = advar.get('idx')
        if advar_idx is None:
            false_keys = ['idx']
        else:
            false_keys = None
            conditions.update({'idx': advar_idx})

        if advar['type'] == 'compute':
            append_target = computes
        else:
            append_target = variables

        m_idx, _ = dict_from_list(append_target, conditions,
                                  false_keys=false_keys,
                                  ret_index=True)

        if m_idx is not None:
            merged_var = {**advar, **append_target[m_idx]}
            append_target[m_idx] = merged_var

        else:
            # Check id does not exist, change if necessary
            trial_id = advar['id']
            id_match = dict_from_list(append_target, {'id': trial_id})
            count = 1
            while id_match is not None:
                trial_id += '_{:d}'.format(count)
                id_match = dict_from_list(append_target, {'id': trial_id})
                count += 1

            advar['id'] = trial_id
            append_target.append(advar)

    out = {
        'session_id': [],
        'idx': [],
        'series_name': [],
        'plots': res_opt['plots'],
        'variables': variables + computes,
    }

    all_srs_name = []
    sids = res_opt['sid']
    for sid in sids:

        path = os.path.join(HOME_PATH, sid)
        pick_path = os.path.join(path, 'sims.pickle')
        pick = read_pickle(pick_path)
        sims = pick['all_sims']
        base_opt = pick['base_options']

        # Get a flat list of series names for this sim series
        srs_name = []
        for series_list in base_opt['series']:
            for series_sublist in series_list:
                srs_name.append(series_sublist['name'])
        all_srs_name.extend(srs_name)

    if 'gamma_surface' in all_srs_name:
        all_srs_name[all_srs_name.index('gamma_surface')] = 'relative_shift'
    out['series_name'] = all_srs_name

    # Loop through series IDs and sims to append values to `result` and
    # `parameter` variable types:
    all_ids = {}
    all_sim_idx = 0
    for sid in sids:

        path = os.path.join(HOME_PATH, sid)
        pick_path = os.path.join(path, 'sims.pickle')
        pick = read_pickle(pick_path)
        sims = pick['all_sims']
        base_opt = pick['base_options']
        csi = pick.get('common_series_info')

        for vr_idx, vr in enumerate(out['variables']):

            vr_name = vr['name']
            vr_type = vr['type']

            if vr_type == 'common_series_info':

                if csi is None:
                    raise ValueError('No common series info was saved.')

                val = csi
                all_sub_idx = vr.get('idx')
                if all_sub_idx is not None:
                    try:
                        for sub_idx in all_sub_idx:
                            val = val[sub_idx]
                    except:
                        break

                if isinstance(val, np.ndarray):
                    val = val.tolist()

                out['variables'][vr_idx]['vals'] = val

        for sm_idx, sm in enumerate(sims):

            out['session_id'].append(sid)
            out['idx'].append(sm_idx)

            srs_id = sm.options.get('series_id')
            if srs_id is None:
                srs_id = [[]]

            all_ids = append_series_items(
                all_ids, srs_id, len(all_srs_name), all_sim_idx + sm_idx, all_srs_name)

            for vr_idx, vr in enumerate(out['variables']):

                vr_name = vr['name']
                vr_type = vr['type']
                val = None
                if vr_type == 'result':
                    val = sm.results[vr_name]
                elif vr_type == 'parameter':
                    val = sm.options[vr_name]
                elif vr_type == 'compute':
                    cmpt_props = SINGLE_COMPUTES.get(vr_name)
                    if cmpt_props is not None:
                        val = cmpt_props[0](sm, *cmpt_props[1:])
                    else:
                        continue
                else:
                    continue
                all_sub_idx = vr.get('idx')
                if all_sub_idx is not None:
                    for sub_idx in all_sub_idx:
                        # print('val: {}'.format(val))
                        val = val[sub_idx]
                if isinstance(val, np.ndarray):
                    val = val.tolist()

                out['variables'][vr_idx]['vals'].append(val)

        all_sim_idx += sm_idx

    all_ids = {k: v for k, v in all_ids.items() if k != 'name'}
    out['series_id'] = all_ids

    # Now calculate multi `compute`s:
    for vr_idx, vr in enumerate(out['variables']):

        if vr['type'] == 'series_id':
            cid = all_srs_name.index(vr['col_id'])
            vals = utils.get_col(all_ids[vr['name']], cid)

            if vr.get('col_idx') is not None:
                vals = utils.get_col_none(vals, vr['col_idx'])

            out['variables'][vr_idx]['vals'] = vals
            continue

        elif vr['type'] != 'compute':
            continue

        cmpt_name = vr['name']
        cmpt_props = SINGLE_COMPUTES.get(cmpt_name)
        if cmpt_props is None:
            func = MULTICOMPUTE_LOOKUP[cmpt_name]
            args = {k: v for k, v in vr.items() if k not in std_keys}
            func(out, **args)

    # Save the JSON file in the results directory of the first listed SID
    res_dir = os.path.join(RES_PATH, rs_id)
    out['dir'] = res_dir
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, 'results.json')

    # Save a copy of the input results options
    src_path = os.path.join(SCRIPTS_PATH, 'set_up', 'opt_test_res.py')
    dst_path = os.path.join(res_dir, 'opt_test_res.py')
    shutil.copy(src_path, res_dir)

    print('json_path: {}'.format(json_path))
    with open(json_path, 'w', encoding='utf-8', newline='') as jf:
        print('write json')
        json.dump(out, jf, sort_keys=True, indent=4)

    do_plots(out)


def main():

    sids = RES_OPT['sid']
    skip_idx = RES_OPT['skip_idx']

    if skip_idx is None or len(skip_idx) == 0:
        skip_idx = [[] for _ in range(len(sids))]

    # for s_idx, s in enumerate(sids):
    #     read_results(s, skip_idx=skip_idx[s_idx])

    # Compute additional properties
    collate_results(RES_OPT, debug=True)


if __name__ == '__main__':
    main()
