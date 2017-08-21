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
from postprocess import SINGLE_COMPUTES, MULTI_COMPUTES, compute_gb_energy, dict_from_list
import plotting
import vectors

SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
REF_PATH = os.path.join(SCRIPTS_PATH, 'ref')
SU_PATH = os.path.join(SCRIPTS_PATH, 'set_up')
HOME_PATH = r'C:\Users\{}\Dropbox (Research Group)\calcs'.format(
    os.getlogin())
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


def collate_results(res_opt):
    """
    Save a JSON file containing the results of one of more simulation series.

    Idea is to build a dict (saved as a JSON file) which has results from
    simulations in flat lists.

    """

    rs_date, rs_num = utils.get_date_time_stamp(split=True)
    rs_id = rs_date + '_' + rs_num

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

    # Find required parameters/results neccessary for multi-computes
    additional_cmpt = []
    for cmpt in res_opt['computes']:
        cmpt_name = cmpt['name']
        cmpt_defn = MULTI_COMPUTES.get(cmpt_name)
        if cmpt_defn is not None:
            requires = cmpt_defn['requires']
            req_com = requires['computes']
            req_par = requires['parameters']
            req_res = requires['results']
            additional_cmpt.extend(req_com)

            # Very hacky, OK for now:
            for r_idx in range(len(req_res)):
                r = req_res[r_idx]
                if not isinstance(r, dict):
                    # Assume a function
                    req_res[r_idx] = r(cmpt['energy_src'])

            # TODO: Need to check that parameter/result is not already defined here
            res_opt['parameters'].extend(req_par)
            res_opt['results'].extend(req_res)

            print('req_res: {}'.format(req_res))

    res_opt['computes'].extend(additional_cmpt)

    computes = []
    for c_idx, c in enumerate(res_opt['computes']):
        c_copy = copy.deepcopy(c)
        c_copy.update({'vals': []})
        computes.append(c_copy)

    parameters = []
    for p_idx, p in enumerate(res_opt['parameters']):
        p_copy = copy.deepcopy(p)
        p_copy.update({'vals': []})
        parameters.append(p_copy)

    results = []
    for r_idx, r in enumerate(res_opt['results']):
        r_copy = copy.deepcopy(r)
        r_copy.update({'vals': []})
        results.append(r_copy)

    com_srs = []
    for cs_idx, cs in enumerate(res_opt['common_series_info']):
        cs_copy = copy.deepcopy(cs)
        cs_copy.update({'vals': []})
        com_srs.append(cs_copy)

    out = {
        'session_id': [],
        'idx': [],
        'series_name': [],
        'series_id_val': [],
        'series_id_path': [],
        'computes': computes,
        'parameters': parameters,
        'results': results,
        'common_series_info': com_srs,
    }

    # Loop through series IDs:
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
        out['series_name'] = srs_name

        for sm_idx, sm in enumerate(sims):

            out['session_id'].append(sid)
            out['idx'].append(sm_idx)

            srs_id = sm.options['series_id']
            out['series_id_path'].append(get_series_paths(srs_id))
            out['series_id_val'].append(get_series_vals(srs_id))

            sm_results = sm.results

            # Execute single-calc computes here:
            # Single computes are defined as functions which take the output from a single sim
            for c_idx, c in enumerate(out['computes']):
                c_name = c['name']
                c_props = SINGLE_COMPUTES.get(c_name)
                if c_props is not None:
                    v = c_props[0](sm, *c_props[1:])
                    all_sub_idx = c.get('idx')
                    if all_sub_idx is not None:
                        for sub_idx in all_sub_idx:
                            v = v[sub_idx]
                    out['computes'][c_idx]['vals'].append(v)

            for r_idx, r in enumerate(out['results']):

                v = sm.results[r['name']]
                all_sub_idx = r.get('idx')
                if all_sub_idx is not None:
                    for sub_idx in all_sub_idx:
                        v = v[sub_idx]
                        if isinstance(v, np.ndarray):
                            v = v.tolist()

                out['results'][r_idx]['vals'].append(v)

            for p_idx, p in enumerate(out['parameters']):

                v = sm.options[p['name']]
                all_sub_idx = p.get('idx')
                if all_sub_idx is not None:
                    for sub_idx in all_sub_idx:
                        v = v[sub_idx]
                        if isinstance(v, np.ndarray):
                            v = v.tolist()

                out['parameters'][p_idx]['vals'].append(v)

        for cs_idx, cs in enumerate(out['common_series_info']):

            v = pick['common_series_info'][cs['name']]
            print('v: \n{}\n'.format(v))
            all_sub_idx = cs.get('idx')
            if all_sub_idx is not None:
                for sub_idx in all_sub_idx:
                    v = v[sub_idx]
                    if isinstance(v, np.ndarray):
                        v = v.tolist()

            out['common_series_info'][cs_idx]['vals'].append(v)

    # Execute computes which require more than one simulation
    # GB compute:
    for c_idx, c in enumerate(out['computes']):
        c_props = MULTI_COMPUTES.get(c['name'])
        if c_props is not None:
            c_props['func'](out)

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

        x_name = pl['x']['name']
        y_name = pl['y']['name']

        # # Find this x data within parameters, results or computes:
        all_d = out['parameters'] + out['results'] + out['computes']
        x_data = dict_from_list(all_d, name=x_name)['vals']
        y_data = dict_from_list(all_d, name=y_name)['vals']

        pl_srs_id = pl.get('series_id')
        if pl_srs_id is not None:
            pl_srs = []
            for pid in pl_srs_id:
                pl_srs.append(dict_from_list(all_d, id=pid)['vals'])

        # print('pl_srs: {}'.format(pl_srs))

        plot_path = os.path.join(res_dir, '{}_{}'.format(x_name, y_name))

        # Identify unique traces by series:
        traces = {}
        for x_idx, x in enumerate(x_data):
            trace_id = utils.transpose_list(pl_srs)[x_idx]
            trace_id_str = ' '.join([str(t) for t in trace_id])
            if trace_id_str in traces:
                traces[trace_id_str]['x'].append(x)
                traces[trace_id_str]['y'].append(y_data[x_idx])
            else:
                traces.update({
                    trace_id_str: {
                        'x': [x],
                        'y': [y_data[x_idx]]
                    }
                })

        for k, v in traces.items():
            utils.trim_common_nones(v['x'], v['y'])

        all_traces = [traces]
        if pl.get('plot_sequential_diff') is True:

            # Add a subplot showing difference between adjacent y values
            all_traces.append(copy.deepcopy(traces))
            for k, v in all_traces[1].items():
                all_traces[1][k]['x'] = v['x'][1:]
                all_traces[1][k]['y'] = np.ediff1d(v['y'])

        # print('trace: {}'.format(traces))

        if pl['lib'] == 'plotly':
            save_args = {
                'filename': plot_path + '.html'
            }
            # Make plot
            plotting.basic_plot_plotly(all_traces, save_args=save_args)

        elif pl['lib'] == 'mpl':
            plot_path += '.' + pl['fmt']
            plotting.basic_plot_mpl(traces, plot_path)

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
    collate_results(RES_OPT)


if __name__ == '__main__':
    main()
