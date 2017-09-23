from atsim import plotting


def format_title(name, val):
    d = zip(name, val)
    t = ['{}: {},'.format(i, out) for i, out in d]
    return ' '.join(t)[:-1]


def main(out):

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

        unique_fsv, unique_fsv_idx = get_unique_idx(file_srs_vals)
        unique_ssv, unique_ssv_idx = get_unique_idx(subplot_srs_vals)
        unique_tsv, unique_tsv_idx = get_unique_idx(trace_srs_vals)

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
                    'title': format_title(subplot_srs, unique_ssv[s_idx]) or pl['filename'],
                })

            figs.append({
                'subplots': subplots,
                'title': format_title(file_srs, unique_fsv[f_idx]),
                **{k: v for k, v in pl.items() if k not in ['subplots']},
            })

        if pl['lib'] == 'mpl':
            save_dir = os.path.join(out['output_path'], out['rid'])
            plotting.plot_many_mpl(figs, save_dir=save_dir)
        else:
            raise NotImplementedError(
                'Library "{}" not supported.'.format(pl['lib']))


if __name__ == '__main__':
    # Generate plots
    main(out)
