from atsim import plotting
from atsim import utils
from atsim import readwrite
from atsim import OPT_FILE_NAMES, SET_UP_PATH
import numpy as np
import json
import copy
import os
import shutil

RES_PATH = r'C:\Users\{}\Dropbox (Research Group)\calcs_results'.format(
    os.getlogin())


def split_by_series(vrs, pl, num_sims):

    fig_srs = pl.get('fig_series', [])
    subplot_srs = pl.get('subplot_series', [])
    trace_srs = pl.get('trace_series', [])

    # print('fig_srs: {}'.format(fig_srs))
    # print('subplot_srs: {}'.format(subplot_srs))
    # print('trace_srs: {}'.format(trace_srs))

    all_types_srs = [fig_srs, subplot_srs, trace_srs]
    all_types_srs_vals = [[], [], []]

    for srs_type_idx, srs_type in enumerate(all_types_srs):

        for i in srs_type:
            i_vals = utils.dict_from_list(vrs, {'id': i})['vals']
            all_types_srs_vals[srs_type_idx].append(i_vals)

    all_types_srs_vals = [utils.transpose_list(i) for i in all_types_srs_vals]

    fig_srs_vals = all_types_srs_vals[0]
    subplot_srs_vals = all_types_srs_vals[1]
    trace_srs_vals = all_types_srs_vals[2]

    if len(fig_srs_vals) == 0:
        fig_srs_vals = [[0] for _ in range(num_sims)]

    if len(subplot_srs_vals) == 0:
        subplot_srs_vals = [[0] for _ in range(num_sims)]

    if len(trace_srs_vals) == 0:
        trace_srs_vals = [[0] for _ in range(num_sims)]

    unique_fsv, unique_fsv_idx = utils.get_unique_idx(fig_srs_vals)
    unique_ssv, unique_ssv_idx = utils.get_unique_idx(subplot_srs_vals)
    unique_tsv, unique_tsv_idx = utils.get_unique_idx(trace_srs_vals)

    # print('unique_fsv: {}, unique_fsv_idx: {}'.format(unique_fsv, unique_fsv_idx))
    # print('unique_ssv: {}, unique_ssv_idx: {}'.format(unique_ssv, unique_ssv_idx))
    # print('unique_tsv: {}, unique_tsv_idx: {}'.format(unique_tsv, unique_tsv_idx))

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

    # print('all_f: {}'.format(all_f))

    all_traces = {}
    for f_idx, f in enumerate(all_f):
        for s_idx, s in enumerate(f):
            for t_idx, t in enumerate(s):

                if all_traces.get(f_idx) is None:
                    all_traces.update({f_idx: []})

                all_traces[f_idx].append({
                    'subplot_idx': s_idx,
                    'subplot_name': subplot_srs,
                    'subplot_val': unique_ssv[s_idx],
                    'fig_idx': f_idx,
                    'fig_name': fig_srs,
                    'fig_val': unique_fsv[f_idx],
                    'trace_name': trace_srs,
                    'trace_val': unique_tsv[t_idx],
                    'vals_idx': t
                })

    # Convert to a list
    all_traces_lst = [None] * len(all_traces)
    for k, v in all_traces.items():
        all_traces_lst[k] = v

    return all_traces_lst


def main(plots_defn):

    res_id = plots_defn['results_id']
    res_dir = os.path.join(RES_PATH, res_id)
    res_json_path = os.path.join(res_dir, 'results.json')
    with open(res_json_path, 'r') as f:
        results = json.load(f)

    # Save a copy of the input makeplots options
    src_path = os.path.join(SET_UP_PATH, OPT_FILE_NAMES['makeplots'])
    dst_path = os.path.join(res_dir, OPT_FILE_NAMES['makeplots'])
    shutil.copy(src_path, res_dir)

    series_names = results['series_name']
    sesh_ids = results['session_id_idx']
    num_sims = len(sesh_ids)

    vrs = results['variables']

    for pl_idx, pl in enumerate(plots_defn['plots']):

        figs = []
        lib = pl['lib']
        fn = pl['filename']
        all_data_defn = pl['data']
        axes = pl['axes']
        axes_props = pl['axes_props']
        subplot_rows = pl.get('subplot_rows')

        # Get data values from variable IDs
        all_data = []
        for ii_idx, i in enumerate(all_data_defn):

            if i['type'] == 'poly':

                coeffs = i['coeffs']
                coeffs_defn = utils.dict_from_list(vrs, {'id': coeffs['id']})
                coeffs['vals'] = coeffs_defn['vals']

                if coeffs.get('idx') is not None:
                    for subidx in coeffs['idx']:
                        coeffs['vals'] = coeffs['vals'][subidx]
                d = {
                    'coeffs': coeffs,
                    **{k: v for k, v in i.items() if k not in ['coeffs']}
                }

            elif i['type'] in ['line', 'marker', 'contour']:

                y_defn = utils.dict_from_list(vrs, {'id': i['y']['id']})
                # print("y_defn['vals']: {}".format(y_defn['vals']))

                # Allow setting x data to be an integer list if no x.id specified
                if i.get('x') is not None:
                    x_id = i['x']['id']
                    x_defn = utils.dict_from_list(vrs, {'id': x_id})
                else:
                    i['x'] = {}
                    if isinstance(y_defn['vals'][0], list):
                        x_defn_vals = [list(range(len(i)))
                                       for i in y_defn['vals']]
                    else:
                        x_defn_vals = list(range(len(y_defn['vals'])))
                    x_defn = {'vals': x_defn_vals}
                    print('Setting x data to be an integer lists of length of y data.')

                x, y = i['x'], i['y']
                x['vals'], y['vals'] = x_defn['vals'], y_defn['vals']

                if x.get('idx') is not None:
                    for subidx in x['idx']:
                        x['vals'] = x['vals'][subidx]

                if y.get('idx') is not None:
                    for subidx in y['idx']:
                        y['vals'] = y['vals'][subidx]

                d = {
                    'x': x,
                    'y': y,
                    **{k: v for k, v in i.items() if k not in ['x', 'y', 'z']}
                }

                if i['type'] == 'contour':
                    z_defn = utils.dict_from_list(vrs, {'id': i['z']['id']})
                    z = i['z']
                    z['vals'] = z_defn['vals']

                    if z.get('idx') is not None:
                        for subidx in z['idx']:
                            z['vals'] = z['vals'][subidx]

                    # print('z[vals]: {}'.format(z['vals']))

                    row_idx = utils.dict_from_list(
                        vrs, {'id': i['row_idx_id']})
                    col_idx = utils.dict_from_list(
                        vrs, {'id': i['col_idx_id']})
                    grid_shape = utils.dict_from_list(
                        vrs, {'id': i['shape_id']})

                    d.update({
                        'z': z,
                        'row_idx': row_idx,
                        'col_idx': col_idx,
                        'grid_shape': grid_shape,

                    })

            all_data.append(d)

        # Split up data according to figure, subplot and trace series
        all_traces = split_by_series(vrs, pl, num_sims)

        # Assign data to each trace for each in the figure
        all_traces_2 = []
        for f in all_traces:

            f_traces = []

            for tt in f:

                for d in all_data:

                    trc = copy.deepcopy(tt)

                    if d['type'] == 'poly':

                        coeffs = copy.deepcopy(d['coeffs'])
                        cv = utils.index_lst(coeffs['vals'], tt['vals_idx'])

                        non_none_idx = [idx for idx in range(
                            len(cv)) if cv[idx] is not None]

                        if len(non_none_idx) > 1:
                            raise ValueError(
                                'Multiple poly coeffs found for current trace.')
                        else:
                            cv = cv[non_none_idx[0]]

                        coeffs['vals'] = cv

                        # print("coeffs['vals']: {}".format(coeffs['vals']))

                        trc.update({
                            'coeffs': coeffs,
                            'xmin': d['xmin'],
                            'xmax': d['xmax'],
                            'sub_subplot_idx': d['axes_idx'][0],
                            'axes_idx': d['axes_idx'],
                            'type': d['type'],
                            'name': d['name'],
                            'legend': d.get('legend', True),
                        })

                    elif d['type'] in ['marker', 'line', 'contour']:

                        x = copy.deepcopy(d['x'])
                        y = copy.deepcopy(d['y'])

                        xv = utils.index_lst(x['vals'], tt['vals_idx'])
                        yv = utils.index_lst(y['vals'], tt['vals_idx'])

                        y_non_none_idx = [idx for idx in range(
                            len(yv)) if yv[idx] is not None]

                        if isinstance(xv[y_non_none_idx[0]], list):
                            if len(y_non_none_idx) == 1:
                                xv = xv[y_non_none_idx[0]]
                                yv = yv[y_non_none_idx[0]]
                            else:
                                raise ValueError('Multiple traces found.')
                        else:
                            xv = utils.index_lst(xv, y_non_none_idx)
                            yv = utils.index_lst(yv, y_non_none_idx)

                        xv = np.array(xv)
                        yv = np.array(yv)

                        if d.get('sort', False):
                            if d['type'] == 'contour':
                                raise ValueError(
                                    'Cannot sort `contour` `type` data.')
                            srt_idx = np.argsort(xv)
                            xv = xv[srt_idx]
                            yv = yv[srt_idx]

                        x['vals'] = xv
                        y['vals'] = yv

                        trc.update({
                            'x': x,
                            'y': y,
                            'sub_subplot_idx': d['axes_idx'][0],
                            'axes_idx': d['axes_idx'],
                            'type': d['type'],
                            'name': d['name'],
                            'legend': d.get('legend', True),
                        })

                        if d['type'] == 'marker':
                            trc.update({
                                'marker': d.get('marker', {})
                            })

                        elif d['type'] == 'line':
                            trc.update({
                                'line': d.get('line', {})
                            })

                        elif d['type'] == 'contour':

                            z = copy.deepcopy(d['z'])
                            row_idx = copy.deepcopy(d['row_idx'])
                            col_idx = copy.deepcopy(d['col_idx'])
                            grid_shape = copy.deepcopy(d['grid_shape'])

                            zv = np.array(z['vals'], dtype=float)
                            row_idx_vals = np.array(
                                row_idx['vals'], dtype=float)
                            col_idx_vals = np.array(
                                col_idx['vals'], dtype=float)
                            grid_shape_vals = np.array(
                                grid_shape['vals'], dtype=float)

                            zv = zv[tt['vals_idx']]
                            row_idx_vals = row_idx_vals[tt['vals_idx']]
                            col_idx_vals = col_idx_vals[tt['vals_idx']]
                            grid_shape_vals = grid_shape_vals[tt['vals_idx']]

                            z['vals'] = zv
                            row_idx['vals'] = row_idx_vals
                            col_idx['vals'] = col_idx_vals
                            grid_shape['vals'] = grid_shape_vals

                            trc.update({
                                'z': z,
                                'contour': {
                                    'row_idx': row_idx['vals'],
                                    'col_idx': col_idx['vals'],
                                    'grid_shape': grid_shape['vals']
                                }
                            })

                    f_traces.append(trc)

            all_traces_2.append(f_traces)

        # print('all_traces_2: {}'.format(readwrite.format_list(all_traces_2)))

        # Sort out subplots series
        all_traces_3 = copy.deepcopy(all_traces_2)
        for f_idx in range(len(all_traces_3)):

            f = all_traces_3[f_idx]

            resolved_idx = []  # trace indices for which final subplot index has been resolved
            for t_idx in range(len(f)):

                if t_idx not in resolved_idx:

                    si = f[t_idx]['subplot_idx']
                    ssi = f[t_idx]['sub_subplot_idx']
                    si_new = si + ssi

                    for t2_idx in range(t_idx, len(f)):

                        if f[t2_idx]['subplot_idx'] == si and f[t2_idx]['sub_subplot_idx'] == ssi:
                            f[t2_idx]['subplot_idx'] = si_new
                            resolved_idx.append(t2_idx)

                        elif f[t2_idx]['subplot_idx'] >= si_new:
                            f[t2_idx]['subplot_idx'] += ssi

        # Add subplots to figures list
        # print('all_traces_3: {}'.format(readwrite.format_list(all_traces_3)))
        for f in all_traces_3:

            # print('\nf: {}'.format(f))

            num_subplots = utils.get_key_max(f, 'subplot_idx') + 1
            subplots = []
            for sidx in range(num_subplots):

                new_sp = {
                    'axes_props': axes_props,
                }

                # Collect all traces at this subplot idx
                sidx_traces = []
                for t in f:
                    if t['subplot_idx'] == sidx:

                        axidx = t['axes_idx']
                        t.update({
                            'axes_idx': axidx[1],
                            'name': t['name'] + format_title(t['trace_name'], t['trace_val']),
                        })

                        sidx_traces.append(t)

                        if new_sp.get('axes') is None:
                            new_sp.update({'axes': axes[axidx[0]]})

                        if new_sp.get('title') is None:
                            new_sp.update({'title': format_title(
                                t['subplot_name'], t['subplot_val'])})

                new_sp.update({'traces': sidx_traces})
                subplots.append(new_sp)

            f_d = {
                'subplots': subplots,
                'title': fn + format_title(
                    subplots[0]['traces'][0]['fig_name'],
                    subplots[0]['traces'][0]['fig_val']),
                'subplot_rows': subplot_rows,
            }
            if f[0].get('save', True):
                f_d['save'] = True
                f_d['save_path'] = f[0].get('save_path', res_dir)

            figs.append(f_d)

        if lib == 'mpl':
            plotting.plot_many_mpl(figs)
        elif lib == 'plotly':
            plotting.plot_many_plotly(figs)


def format_title(names, vals):
    """Take two lists and combine like elements into strings."""

    if len(names) == 0:
        return ''

    if len(names) != len(vals):
        # print('names: {}, vals: {}'.format(names, vals))
        raise ValueError('Length of `names` ({}) and `vals` ({}) must match.'.format(
            len(names), len(vals)))

    out = []
    for n, v in zip(names, vals):
        out.append('{}: {}'.format(n, v))

    return '; ' + '; '.join(out)
