import utils
import json
import yaml
import os
import numpy as np
import warnings
from atsim.simulation import makesims
from atsim.readwrite import format_dict, format_list
from atsim import geometry, SET_UP_PATH, OPT_FILE_NAMES


def main(opt):
    refine_gamma_surface(opt)


def refine_gamma_surface(opt):

    # Open results.json:
    res_dir = os.path.join(opt['output']['path'], opt['results_id'])
    res_pth = os.path.join(res_dir, 'results.json')
    with open(res_pth, 'r') as r:
        res = json.load(r)

    # Get minimised boundary expansion grid:
    mg = utils.dict_from_list(res['variables'], {'id': 'master_gamma'})
    x_frac = utils.dict_from_list(res['variables'], {'id': 'x_frac_vals'})
    y_frac = utils.dict_from_list(res['variables'], {'id': 'y_frac_vals'})
    em = mg['vals']['exp_master']

    exp_master_un = []
    xy_frac_un = []
    for i_idx, i in enumerate(em):
        if i is None:
            continue
        exp_master_un.append(i)
        xy_frac_un.append([x_frac['vals'][i_idx], y_frac['vals'][i_idx]])

    xy_frac_un = np.array(xy_frac_un)

    # Generate new grid series vals using prepare_series_update
    # Edge vectors don't matter.
    grd = geometry.Grid(np.eye(3)[:, 0:2], grid_spec=opt['grid_spec'])

    # For each new grid point, check exists on parent series grid and if so,
    # add an entry to the series_lookup
    parent_val = []
    child_series = []
    for gp in grd.get_grid_points()['points_frac'].T:

        w = np.where(np.all(np.isclose(xy_frac_un, gp), axis=1))
        if len(w[0]) > 0:

            vm = exp_master_un[w[0][0]]
            new_vms = [vm - 0.15, vm, vm + 0.15]

            parent_val.append([gp.tolist()])
            child_series.append({
                'name': 'boundary_vac',
                'vals': new_vms
            })

    if len(parent_val) == 0:
        warnings.warn('No matching grid positions found.')
    else:
        print('{} matching grid positions found.'.format(len(parent_val)))

    # Add GRID_SPEC and lookup to series for new OPT dict.
    series_lookup_src = {
        'parent_series': ['relative_shift'],
        'parent_val': parent_val,
        'child_series': child_series
    }
    series = [
        [
            {
                'name': 'gamma_surface',
                'grid_spec': opt['grid_spec'],
                'preview': True
            },
        ],
        [
            {
                'name': 'lookup',
                'src': series_lookup_src,
            },
        ],
    ]

    print('series: \n{}\n'.format(series))

    ms_opt_path = os.path.join(SET_UP_PATH, OPT_FILE_NAMES['makesims'])
    with open(ms_opt_path, 'a') as f:
        f.write(yaml.dump({'series': series}))

    # should add this programmatically to makesims.yml so the options are still
    # validated.
    # OPT['series'] = series
    # makesims.main(OPT)
