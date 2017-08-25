from readwrite import read_pickle, write_pickle, format_list, format_dict
import simsio
import dict_parser
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
# os.getlogin())
HOME_PATH = r'C:\calcs_archive'.format(os.getlogin())
RES_PATH = r'C:\Users\{}\Dropbox (Research Group)\calcs_results'.format(
    os.getlogin())


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
    allowed_types = ['result', 'parameter', 'compute', 'common_series_info', ]
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

        # print('considering advar: \n{}\n'.format(advar))

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

        # print('append_target is currently: \n\n{}'.format(append_target))

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

    print('variables 2: {}'.format(format_list(variables)))
    print('computes: {}'.format(format_list(computes)))

    common_series_info = []
    for csi_idx, csi in enumerate(res_opt['common_series_info']):
        common_series_info

    out = {
        'session_id': [],
        'idx': [],
        'series_name': [],
        'variables': variables + computes,
    }

    # print('variables: '.format(out['variables']))

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

    all_srs_name[all_srs_name.index('gamma_surface')] = 'relative_shift'
    out['series_name'] = all_srs_name

    # print('all_srs_names: {}'.format(all_srs_name))
    # print('variables: '.format(out['variables']))

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

        print('csi: {}'.format(csi))

        for vr_idx, vr in enumerate(out['variables']):

            vr_name = vr['name']
            vr_type = vr['type']

            print('vr_name: {}'.format(vr_name))

            if vr_type == 'common_series_info':

                if csi is None:
                    raise ValueError('No common series info was saved.')

                try:
                    val = csi[vr_name]
                    all_sub_idx = vr.get('idx')
                    if all_sub_idx is not None:
                        for sub_idx in all_sub_idx:
                            val = val[sub_idx]
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    out['variables'][vr_idx]['vals'] = val

                except:
                    val = None

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

                # print('vr_name: {}'.format(vr_name))

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

                # print('val ({}): {} ({})'.format(vr_name, val, type(val)))

                out['variables'][vr_idx]['vals'].append(val)

        all_sim_idx += sm_idx

    all_ids = {k: v for k, v in all_ids.items() if k != 'name'}
    out['series_id'] = all_ids
    print('out[vars]: \n{}\n'.format(format_list(out['variables'])))

    # exit()

    # Now calculate multi `compute`s:
    for vr_idx, vr in enumerate(out['variables']):

        if vr['type'] != 'compute':
            continue

        cmpt_name = vr['name']
        cmpt_props = SINGLE_COMPUTES.get(cmpt_name)
        if cmpt_props is None:
            func = MULTICOMPUTE_LOOKUP[cmpt_name]
            args = {k: v for k, v in vr.items() if k not in std_keys}
            func(out, **args)

    # Save the JSON file in the results directory of the first listed SID
    res_dir = os.path.join(RES_PATH, rs_id)
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, 'results.json')

    # Save a copy of the input results options
    src_path = os.path.join(SCRIPTS_PATH, 'set_up', 'opt_test_res.py')
    dst_path = os.path.join(res_dir, 'opt_test_res.py')
    shutil.copy(src_path, res_dir)

    with open(json_path, 'w', encoding='utf-8', newline='') as jf:
        json.dump(out, jf, sort_keys=True, indent=4)

    # Plots
    for pl in res_opt['plots']:

        x = pl['x']
        y = pl['y']

        x_id = x['id']
        y_id = y['id']
        x_label = x['label']
        y_label = y['label']

        z = pl.get('z')
        if z is not None:
            z_id = z['id']
            z_label = z['label']

        x_data = dict_from_list(out['variables'], {'id': x_id})['vals']
        all_x_subidx = pl['x'].get('idx')
        if all_x_subidx is not None:
            for x_subidx in all_x_subidx:
                x_data = x_data[x_subidx]
        # print('x_data: \n{}\n'.format(x_data))

        y_data = dict_from_list(out['variables'], {'id': y_id})['vals']
        all_y_subidx = pl['y'].get('idx')
        if all_y_subidx is not None:
            for y_subidx in all_y_subidx:
                y_data = y_data[y_subidx]
        # print('y_data: \n{}\n'.format(y_data))

        if z is not None:
            z_data = dict_from_list(out['variables'], {'id': z_id})['vals']
            all_z_subidx = z.get('idx')
            if all_z_subidx is not None:
                for z_subidx in all_z_subidx:
                    z_data = z_data[z_subidx]
            # print('z_data: \n{}\n'.format(z_data))

        pl_srs_id = pl.get('series_id')
        if pl_srs_id is not None:
            pl_srs = []
            for pid in pl_srs_id:
                pl_srs.append(dict_from_list(all_d, id=pid)['vals'])

        # plot_name = '{}_{}'.format(x_label, y_label)
        plot_name = pl['filename']
        # if z is not None:
        #     plot_name += '_{}'.format(z_label)
        plot_path = os.path.join(res_dir, plot_name)
        # print('plt_path: {}'.format(plot_path))

        if pl_srs_id is None:
            if z is None:
                traces = {
                    y_label: {
                        'x': x_data,
                        'y': y_data,
                        'xlabel': x_label,
                        'ylabel': y_label,
                    }
                }
            else:
                traces = {
                    z_label: {
                        'x': x_data,
                        'y': y_data,
                        'z': z_data,
                        'xlabel': x_label,
                        'ylabel': y_label,
                        'zlabel': z_label,
                    }
                }
        else:
            # Identify unique traces by series:
            traces = {}
            for x_idx, x in enumerate(x_data):

                if isinstance(pl_srs[0], list):
                    pl_srs = utils.transpose_list(pl_srs)
                trace_id = pl_srs[x_idx]
                trace_id_str = ' '.join([str(t) for t in trace_id])
                if trace_id_str in traces:
                    traces[trace_id_str]['x'].append(x)
                    traces[trace_id_str]['y'].append(y_data[x_idx])
                else:
                    traces.update({
                        trace_id_str: {
                            'x': [x],
                            'y': [y_data[x_idx]],
                        }
                    })
                traces[trace_id_str].update({
                    'xlabel': x_label,
                    'ylabel': y_label,
                })

        for k, v in traces.items():
            utils.trim_common_nones(v['x'], v['y'])

        all_traces = [traces]
        # print('all_traces: {}'.format(all_traces))

        if pl.get('plot_sequential_diff') is True:

            # Add a subplot showing difference between adjacent y values
            all_traces.append(copy.deepcopy(traces))
            for k, v in all_traces[1].items():
                all_traces[1][k]['x'] = v['x'][1:]
                all_traces[1][k]['y'] = np.ediff1d(v['y'])

        # print('traces: {}'.format(traces))

        if pl['lib'] == 'plotly':
            save_args = {
                'filename': plot_path + '.html'
            }
            # Make plot
            plotting.basic_plot_plotly(all_traces, save_args=save_args)

        elif pl['lib'] == 'mpl':
            plot_path += '.' + pl['fmt']
            if z is None:
                plotting.basic_plot_mpl(traces, plot_path)
            else:
                plotting.contour_plot_mpl(traces, plot_path)

        elif pl['lib'] == 'bokeh':
            plot_path += '.html'
            plotting.basic_plot_bokeh(traces, plot_path)


def main():

    sids = RES_OPT['sid']
    skip_idx = RES_OPT['skip_idx']

    if skip_idx is None or len(skip_idx) == 0:
        skip_idx = [[] for _ in range(len(sids))]

    for s_idx, s in enumerate(sids):
        read_results(s, skip_idx=skip_idx[s_idx])

    # Compute additional properties
    collate_results(RES_OPT, debug=True)


if __name__ == '__main__':
    main()
