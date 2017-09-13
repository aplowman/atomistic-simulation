import utils
import json
import os
import numpy as np
import warnings
from atsim.simulation import makesims
from atsim.readwrite import format_dict, format_list
from atsim import geometry
from atsim.set_up.opt import OPT


# Temp hard code:
# SID = '2017-08-28-2218_84329'
rSID = '2017-09-04-0108_08243'

# TODO: add rSID as option in opt.py

HOME_PATH = r'C:\Users\adamj\Dropbox (Research Group)\calcs_results'

GRID_SPEC = {
    'size': (6, 10),
}


def refine_gamma_surface(sid):

    # Open results.json:
    res_pth = os.path.join(HOME_PATH, rSID, 'results.json')
    with open(res_pth, 'r') as r:
        res = json.load(r)

    # Get minimised boundary expansion grid:
    mg = utils.dict_from_list(res['variables'], {'name': 'master_gamma'})
    vac_min = np.array(mg['vals']['vac_min'])
    xy_frac = mg['vals']['XY_frac']

    # Generate new grid series vals using prepare_series_update
    # Edge vectors don't matter.
    grd = geometry.Grid(np.eye(3)[:, 0:2], grid_spec=GRID_SPEC)

    # For each new grid point, check exists on parent series grid and if so,
    # add an entry to the series_lookup
    parent_val = []
    child_series = []
    for gp in grd.get_grid_points()['points_frac'].T:

        w = np.where(np.all(np.isclose(xy_frac, gp), axis=2))
        if len(w[0]) > 0:

            vm = vac_min[w][0]
            new_vms = [vm - 0.1, vm, vm + 0.1]

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
                'grid_spec': GRID_SPEC,
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

    OPT['series'] = series
    makesims.main(OPT)


if __name__ == '__main__':
    refine_gamma_surface(rSID)
