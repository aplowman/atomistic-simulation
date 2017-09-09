import utils
from utils import dict_from_list
from readwrite import format_list
import numpy as np


def compute_time_fmt(sim):
    res = sim.results
    return utils.format_time(res['tot_time'])


def compute_per_atom_energies(sim, energy_idx):
    res = sim.results
    n = res['num_ions']
    final_energy_pa = res['final_energy'] / n
    final_fenergy_pa = res['final_fenergy'] / n
    final_zenergy_pa = res['final_zenergy'] / n

    return (final_energy_pa, final_fenergy_pa,
            final_zenergy_pa)[energy_idx]


def get_rms_force(forces):
    """
    Parameters
    ----------
    forces : ndarray of shape (M, 3, N)
        Array representing the force components on N atoms
        for M steps.
    """
    if len(forces) == 0:
        return None

    forces_rshp = forces.reshape(forces.shape[0], -1)
    forces_rms = np.sqrt(np.mean(forces_rshp ** 2, axis=1))

    return forces_rms


def compute_rms_force(sim, force_idx):
    res = sim.results
    if force_idx == 0:
        return get_rms_force(res['forces_constrained'])
    elif force_idx == 1:
        return get_rms_force(res['forces_unconstrained'])
    elif force_idx == 2:
        return get_rms_force(res['forces_constrained_sym'])


def compute_gb_area(sim):

    if sim.options['base_structure']['type'] == 'CSLBicrystal':
        return sim.structure.boundary_area
    else:
        return None


def compute_gb_thickness(sim):

    if sim.options['base_structure']['type'] == 'CSLBicrystal':
        return sim.structure.bicrystal_thickness
    else:
        return None


def atoms_gb_dist_initial(sim):
    try:
        return sim.structure.atoms_gb_dist_old
    except:
        return None


def atoms_gb_dist_final(sim):

    if sim.options['method'] == 'lammps':
        atom_sites_final = sim.results['atoms'][-1]
    elif sim.options['method'] == 'castep':
        atom_sites_final = sim.results['geom']['ions'][-1]

    try:
        n_unit = sim.structure.n_unit
        gb_dist = np.einsum('jk,jl->k', atom_sites_final, n_unit)
        return gb_dist

    except:
        return None


def atoms_gb_dist_δ(sim):
    try:
        return sim.structure.atoms_gb_dist_δ
    except:
        return None


def gamma_info(sim, info):

    sid_idx = None
    for s_idx, s in enumerate(sim.options['series']):
        cnd = {'name': 'gamma_surface'}
        g_idx, g = utils.dict_from_list(s, cnd, ret_index=True)[0]
        if g is not None:
            sid_idx = (sid_idx, g_idx)
            break
    try:
        d = sim.options['series_id'][sid_idx[0]][sid_idx[1]]
        if info in ['col_idx', 'row_idx', 'points_std']:
            out = d[info]
        return out

    except:
        return None


def atoms_gb_dist_δ_final(out):
    gb_dist_initial_cnd = {
        'type': 'compute',
        'name': 'atoms_gb_dist_initial',
    }
    gb_dist_final_cnd = {
        'type': 'compute',
        'name': 'atoms_gb_dist_final',
    }
    gb_dist_δ_cnd = {
        'type': 'compute',
        'name': 'atoms_gb_dist_δ',
    }
    this_cmpt_cnd = {
        'type': 'compute',
        'name': 'atoms_gb_dist_δ_final',
    }
    type_cnd = {
        'type': 'parameter',
        'name': 'base_structure',
        'idx': ('type', ),
    }

    variables = out['variables']
    this_cmpt = dict_from_list(variables, this_cmpt_cnd)

    # Required parameters/results:
    gb_dist_initial = dict_from_list(variables, gb_dist_initial_cnd)['vals']
    gb_dist_final = dict_from_list(variables, gb_dist_final_cnd)['vals']
    gb_dist_δ = dict_from_list(variables, gb_dist_δ_cnd)['vals']
    type_param = dict_from_list(variables, type_cnd)['vals']
    num_vals = len(type_param)

    print('len(gb_dist_final): {}'.format(len(gb_dist_final)))
    print('len(gb_dist_initial): {}'.format(len(gb_dist_initial)))
    print('len(gb_dist_δ): {}'.format(len(gb_dist_δ)))

    atoms_gb_dist_δ_final = [None, ] * num_vals
    for sim_idx, sup_type in enumerate(type_param):

        if sup_type == 'CSLBicrystal':
            atoms_in = np.array(gb_dist_initial[sim_idx])
            atoms_fi = np.array(gb_dist_final[sim_idx])
            δf = atoms_fi - atoms_in

            # Sort by dist from origin boundary plane
            srt_idx = np.argsort(atoms_in)
            atoms_gb_dist_δ_final[sim_idx] = list(δf[srt_idx])
            gb_dist_initial[sim_idx] = list(atoms_in[srt_idx])
            gb_dist_final[sim_idx] = list(atoms_fi[srt_idx])
            gb_dist_δ[sim_idx] = list(np.array(gb_dist_δ[sim_idx])[srt_idx])

    this_cmpt['vals'] = atoms_gb_dist_δ_final


def compute_gamma_info(out, info, common_series_info):

    sesh_id_idx = out['session_id_idx']

    csi_cnd = {'type': 'series_id',
               'name': 'csi_idx', 'col_id': 'relative_shift'}
    grd_cnd = {'type': 'series_id',
               'name': 'grid_idx', 'col_id': 'relative_shift'}
    pnt_cnd = {'type': 'series_id',
               'name': 'point_idx', 'col_id': 'relative_shift'}

    vrs = out['variables']

    all_csi_idx = dict_from_list(vrs, csi_cnd)['vals']
    all_grd_idx = dict_from_list(vrs, grd_cnd)['vals']
    all_pnt_idx = dict_from_list(vrs, pnt_cnd)['vals']

    row_idx = []
    col_idx = []
    x_std_vals = []
    y_std_vals = []
    x_frac_vals = []
    y_frac_vals = []
    x_num_den_vals = []
    y_num_den_vals = []
    gamma_shape = []

    # Loop through each sim, identified by session id to which it belongs
    for sii_idx, sii in enumerate(sesh_id_idx):

        try:

            csi = common_series_info[sii][all_csi_idx[sii_idx]]
            grid = csi['grids'][all_grd_idx[sii_idx]]
            pnt_idx = all_pnt_idx[sii_idx]

            all_ri = grid['row_idx']
            ri = all_ri[pnt_idx]

            all_ci = grid['col_idx']
            ci = all_ci[pnt_idx]

            all_x_std, all_y_std = grid['grid_points_std']
            all_x_frac, all_y_frac = grid['grid_points_frac']
            all_x_num_den, all_y_num_den = grid['grid_points_num_den']
            shp = grid['shape']

            x_std = all_x_std[pnt_idx]
            y_std = all_y_std[pnt_idx]

            x_frac = all_x_frac[pnt_idx]
            y_frac = all_y_frac[pnt_idx]

            x_num_den = all_x_num_den[pnt_idx]
            y_num_den = all_y_num_den[pnt_idx]

        except:

            ri = None
            ci = None
            x_std = None
            y_std = None
            x_frac = None
            y_frac = None
            x_num_den = None
            y_num_den = None
            shp = None

        row_idx.append(ri)
        col_idx.append(ci)
        x_std_vals.append(x_std)
        y_std_vals.append(y_std)
        x_frac_vals.append(x_frac)
        y_frac_vals.append(y_frac)
        x_num_den_vals.append(x_num_den)
        y_num_den_vals.append(y_num_den)
        gamma_shape.append(shp)

    this_cnd = {'name': 'gamma_info', 'type': 'compute', 'info': info}
    this_cmpt = dict_from_list(vrs, this_cnd)

    if info == 'row_idx':
        vals = row_idx
    elif info == 'col_idx':
        vals = col_idx
    elif info == 'x_std':
        vals = x_std_vals
    elif info == 'y_std':
        vals = y_std_vals
    elif info == 'x_frac':
        vals = x_frac_vals
    elif info == 'y_frac':
        vals = y_frac_vals
    elif info == 'x_num_den':
        vals = x_num_den_vals
    elif info == 'y_num_den':
        vals = y_num_den_vals
    elif info == 'shape':
        vals = gamma_shape

    this_cmpt['vals'] = vals


def compute_master_gamma(out, common_series_info):

    all_series_name = out['series_name']
    exp_idx = all_series_name.index('boundary_vac')
    exp_vals = utils.get_col(out['series_id']['val'], exp_idx)

    vrs = out['variables']

    gb_energy = utils.dict_from_list(vrs, {'id': 'gb_energy'})['vals']
    row_idx = utils.dict_from_list(vrs, {'id': 'gamma_row_idx'})['vals']
    col_idx = utils.dict_from_list(vrs, {'id': 'gamma_col_idx'})['vals']
    gamma_x = utils.dict_from_list(vrs, {'id': 'gamma_x_std'})['vals']
    gamma_y = utils.dict_from_list(vrs, {'id': 'gamma_y_std'})['vals']
    gamma_x_frac = utils.dict_from_list(vrs, {'id': 'gamma_x_frac'})['vals']
    gamma_y_frac = utils.dict_from_list(vrs, {'id': 'gamma_y_frac'})['vals']
    gamma_shape = utils.dict_from_list(vrs, {'id': 'gamma_shape'})['vals']

    this_cmpt = dict_from_list(vrs, {'id': 'master_gamma'})
    num_sims = len(gb_energy)

    # Get first valid gamma shape:
    for en_idx, en in enumerate(gb_energy):
        if en is None:
            continue
        shp = gamma_shape[en_idx]
        break

    all_E = {}
    X = np.ones(tuple(shp), dtype=float) * np.nan
    Y = np.ones(tuple(shp), dtype=float) * np.nan
    XY_frac = np.ones(tuple(shp) + (2,), dtype=float) * np.nan
    first_en_idx = np.ones(tuple(shp), dtype=float) * np.nan

    for en_idx, en in enumerate(gb_energy):

        if en is None:
            continue

        ri = row_idx[en_idx]
        ci = col_idx[en_idx]
        srs_v = exp_vals[en_idx]

        if all_E.get(srs_v) is None:
            blank = np.ones(tuple(shp), dtype=float) * np.nan
            all_E.update({srs_v: blank})

        all_E[srs_v][ri, ci] = en

        if np.isnan(X[ri][ci]):
            X[ri, ci] = gamma_x[en_idx]
            Y[ri, ci] = gamma_y[en_idx]
            XY_frac[ri, ci] = [gamma_x_frac[en_idx], gamma_y_frac[en_idx]]
            first_en_idx[ri, ci] = en_idx

    # Fitting
    nrows = shp[0]
    ncols = shp[1]
    fit_grid_E = [[[] for i in range(ncols)] for _ in range(nrows)]
    fit_grid_vac = [[[] for i in range(ncols)] for _ in range(nrows)]

    fitted_E = np.ones((nrows, ncols), dtype=float) * np.nan
    fitted_vac = np.ones((nrows, ncols), dtype=float) * np.nan
    fitted_p1d = np.ones((nrows, ncols, 3), dtype=float) * np.nan

    for ri in range(nrows):
        for ci in range(ncols):
            for k, v in sorted(all_E.items()):

                fit_grid_E[ri][ci].append(v[ri][ci])
                fit_grid_vac[ri][ci].append(k)

            x = np.array(fit_grid_vac[ri][ci])
            y = np.array(fit_grid_E[ri][ci])

            # Get nonNaN values from y
            y_fin_idx = np.isfinite(y)
            y = y[y_fin_idx]
            x = x[y_fin_idx]

            if len(x) > 2:
                z = np.polyfit(x, y, 2)
                p1d = np.poly1d(z)
                dpdx = np.polyder(p1d)
                min_x = -dpdx[0] / dpdx[1]
                min_y = p1d(min_x)

                fitted_vac[ri, ci] = min_x
                fitted_E[ri, ci] = min_y
                fitted_p1d[ri, ci] = p1d.coeffs

    first_en_idx = first_en_idx.reshape(-1,).astype(int)
    # print('first_en_idx: {}'.format(first_en_idx))

    fitted_p1d = fitted_p1d.reshape(-1, 3)
    all_fitted_p1d = np.ones((num_sims, 3), dtype=float) * np.nan
    all_fitted_p1d[first_en_idx] = fitted_p1d

    E_min_flat = fitted_E.reshape(-1)
    all_E_min_flat = np.ones((num_sims,), dtype=float) * np.nan
    all_E_min_flat[first_en_idx] = E_min_flat

    vac_min_flat = fitted_vac.reshape(-1)
    all_vac_min_flat = np.ones((num_sims,), dtype=float) * np.nan
    all_vac_min_flat[first_en_idx] = vac_min_flat

    # print('fits: {}'.format(all_fitted_p1d))

    this_cmpt['vals'] = {
        'X': X.tolist(),
        'Y': Y.tolist(),
        'XY_frac': XY_frac.tolist(),
        'E_min': fitted_E.tolist(),
        'vac_min': fitted_vac.tolist(),
        'fits': all_fitted_p1d.tolist(),
        'vac_min_flat': all_vac_min_flat.tolist(),
        'E_min_flat': all_E_min_flat.tolist(),
    }


def compute_gamma_energy(out, energy_src, opt_step, series=None):
    gamma_xy_cnd = {
        'type': 'common_series_info',
        'name': 'gamma_surface',
        'idx': ('grids', 0, 'grid_points_std'),
    }
    gamma_shape_cnd = {
        'type': 'common_series_info',
        'name': 'gamma_surface',
        'idx': ('grids', 0, 'shape'),
    }
    energy_cnd = {
        'type': 'result',
        'name': energy_src,
        'idx': (opt_step, ),
    }
    gb_energy_cnd = {
        'type': 'compute',
        'name': 'gb_energy',
    }
    this_cmpt_cnd = {
        'type': 'compute',
        'name': 'gamma_energy',
    }
    type_cnd = {
        'type': 'parameter',
        'name': 'base_structure',
        'idx': ('type', ),
    }

    variables = out['variables']
    this_cmpt = dict_from_list(variables, this_cmpt_cnd)

    # Required parameters/results:
    type_param = dict_from_list(variables, type_cnd)['vals']
    gamma_xy = dict_from_list(variables, gamma_xy_cnd)['vals']
    gamma_shape = dict_from_list(variables, gamma_shape_cnd)['vals']
    # energy = dict_from_list(variables, energy_cnd)['vals']
    gb_energy = dict_from_list(variables, gb_energy_cnd)['vals']

    all_series_name = out['series_name']
    rel_shift_idx = all_series_name.index('relative_shift')
    series_idx = all_series_name.index(series)

    gamma_row_idx = utils.get_col(out['series_id']['row_idx'], rel_shift_idx)
    gamma_col_idx = utils.get_col(out['series_id']['col_idx'], rel_shift_idx)
    series_vals = utils.get_col(out['series_id']['val'], series_idx)

    all_E = {}
    for en_idx, en in enumerate(gb_energy):

        if type_param[en_idx] != 'CSLBicrystal':
            continue

        ri = gamma_row_idx[en_idx]
        ci = gamma_col_idx[en_idx]
        srs_v = series_vals[en_idx]

        if all_E.get(srs_v) is None:
            all_E.update({srs_v: np.zeros(tuple(gamma_shape))})
        all_E[srs_v][ri, ci] = en

    for k, v in all_E.items():
        all_E[k] = v.tolist()

    # Fitting
    nrows = gamma_shape[0]
    ncols = gamma_shape[1]
    fit_grid_E = [[[] for i in range(ncols)] for _ in range(nrows)]
    fit_grid_vac = [[[] for i in range(ncols)] for _ in range(nrows)]

    fitted_E = np.ones((nrows, ncols), dtype=float) * np.nan
    fitted_vac = np.ones((nrows, ncols), dtype=float) * np.nan

    for ri in range(nrows):
        for ci in range(ncols):
            for k, v in sorted(all_E.items()):
                fit_grid_E[ri][ci].append(v[ri][ci])
                fit_grid_vac[ri][ci].append(k)

            x = fit_grid_vac[ri][ci]
            y = fit_grid_E[ri][ci]

            if len(x) > 2:
                z = np.polyfit(x, y, 2)
                p1d = np.poly1d(z)
                dpdx = np.polyder(p1d)
                min_x = -dpdx[0] / dpdx[1]
                min_y = p1d(min_x)

                fitted_vac[ri, ci] = min_x
                fitted_E[ri, ci] = min_y

    gamma_X = np.array(gamma_xy[0]).reshape(gamma_shape).tolist()
    gamma_Y = np.array(gamma_xy[1]).reshape(gamma_shape).tolist()

    this_cmpt['vals'] = {
        'X': gamma_X,
        'Y': gamma_Y,
        'E': all_E,
        'E_min': fitted_E.tolist(),
        'vac_min': fitted_vac.tolist(),
    }


def compute_gb_energy(out, series_id, bulk_src, energy_src, opt_step, unit,
                      common_series_info):

    # print('compute_gb_energy: csi: \n{}\n'.format(common_series_info))

    type_cnd = {
        'type': 'parameter',
        'name': 'base_structure',
        'idx': ('type', ),
    }
    # csatep:
    num_ions_cnd = {
        'type': 'result',
        'name': 'num_ions',
    }
    # num_ions_cnd = {
    #     'type': 'result',
    #     'name': 'dumps',
    #     'idx': (0, 'num_atoms',),
    # }
    energy_cnd = {
        'type': 'result',
        'name': energy_src,
        'idx': (opt_step, ),
    }
    areas_cnd = {
        'type': 'compute',
        'name': 'gb_area',
    }
    this_cmpt_cnd = {
        'type': 'compute',
        'name': 'gb_energy',
    }

    variables = out['variables']
    this_cmpt = dict_from_list(variables, this_cmpt_cnd)

    # Required parameters/results:
    type_param = dict_from_list(variables, type_cnd)['vals']
    num_ions = dict_from_list(variables, num_ions_cnd)['vals']
    energy = dict_from_list(variables, energy_cnd)['vals']

    # Required computes (if multi-computes, may not be computed yet)
    areas = dict_from_list(variables, areas_cnd)['vals']

    series_names = out['series_name']
    sesh_ids = np.array(out['session_id'])[out['session_id_idx']]
    num_sims = len(sesh_ids)

    srs_vals = []
    for i in series_id:

        if i in series_names:
            i_idx = series_names.index(i)
            i_vals = utils.get_col(out['series_id']['val'], i_idx)
        else:
            i_vals = dict_from_list(
                out['variables'], {'id': i})['vals']

        srs_vals.append(i_vals)

    srs_vals = utils.transpose_list(srs_vals)

    if len(srs_vals) == 0:
        srs_vals = [[0] for _ in range(num_sims)]

    # print('srs_vals: \n{}\n'.format(srs_vals))

    # print('energy: \n{}\n'.format(energy))
    # print('type_param: \n{}\n'.format(type_param))
    # print('num_ions: \n{}\n'.format(num_ions))
    # print('areas: \n{}\n'.format(areas))

    gb_idx = []
    bulk_idx = []
    for i_idx, i in enumerate(type_param):
        if i == 'CSLBicrystal':
            gb_idx.append(i_idx)
        elif i == 'CSLBulkCrystal':
            bulk_idx.append(i_idx)

    # print('gb_idx: {}'.format(gb_idx))
    # print('bulk_idx: {}'.format(bulk_idx))

    # For each GB supercell, find a bulk supercell whose series val matches
    # (Broadcasting logic would go here)
    all_E_gb = [None, ] * num_sims
    for gb_i in gb_idx:
        for bulk_i in bulk_idx:
            if srs_vals[gb_i] == srs_vals[bulk_i]:
                gb_num = num_ions[gb_i]
                bulk_num = num_ions[bulk_i]
                bulk_frac = gb_num / bulk_num
                E_gb = (energy[gb_i] - bulk_frac *
                        energy[bulk_i]) / (2 * areas[gb_i])

        if unit == 'J/m^2':
            E_gb *= 16.02176565
        all_E_gb[gb_i] = E_gb
    this_cmpt['vals'] = all_E_gb


MULTICOMPUTE_LOOKUP = {
    'gb_energy': compute_gb_energy,
    'master_gamma': compute_master_gamma,
    'gamma_energy': compute_gamma_energy,
    'atoms_gb_dist_δ_final': atoms_gb_dist_δ_final,
    'gamma_info': compute_gamma_info,
}

# Computed quantities which are dependent on exactly one simulation:
SINGLE_COMPUTES = {
    'time_fmt': (compute_time_fmt, ),
    'final_energy_pa': (compute_per_atom_energies, 0),
    'final_fenergy_pa': (compute_per_atom_energies, 1),
    'final_zenergy_pa': (compute_per_atom_energies, 2),
    'forces_cons_rms':  (compute_rms_force, 0),
    'forces_uncons_rms':  (compute_rms_force, 1),
    'forces_cons_sym_rms':  (compute_rms_force, 2),
    'gb_area': (compute_gb_area, ),
    'gb_thickness': (compute_gb_thickness, ),
    'atoms_gb_dist_initial': (atoms_gb_dist_initial, ),
    'atoms_gb_dist_final': (atoms_gb_dist_final, ),
    'atoms_gb_dist_δ': (atoms_gb_dist_δ, ),
}

# Variables which do not need to be parameterised:
PREDEFINED_VARS = {
    # castep:
    'num_ions': {
        'type': 'result',
        'name': 'num_ions',
        'id': 'num_ions',
        'vals': [],
    },
    # lammps:
    # 'num_ions': {
    #     'type': 'result',
    #     'name': 'dumps',
    #     'idx': (0, 'num_atoms'),
    #     'id': 'num_ions',
    #     'vals': [],
    # },
    'gb_area': {
        'type': 'compute',
        'name': 'gb_area',
        'id': 'gb_area',
        'vals': [],
    },
    'sup_type': {
        'type': 'parameter',
        'name': 'base_structure',
        'idx': ('type',),
        'id': 'supercell_type',
        'vals': [],
    },
    'gamma_row_idx': {
        'type': 'parameter',
        'name': 'series_id',
        'idx': (0, 0, 'row_idx', ),
        'id': 'gamma_row_idx',
        'vals': [],
    },
    'gamma_col_idx': {
        'type': 'parameter',
        'name': 'series_id',
        'idx': (0, 0, 'col_idx', ),
        'id': 'gamma_col_idx',
        'vals': [],
    },
    'gamma_surface_shape': {
        'type': 'common_series_info',
        'name': 'gamma_surface',
        'idx': ('grids', 0, 'shape'),
        'id': 'gamma_surface_shape',
        'vals': [],
    },
    'gamma_surface_xy': {
        'type': 'common_series_info',
        'name': 'gamma_surface',
        'idx': ('grids', 0, 'grid_points_std'),
        'id': 'gamma_surface_xy',
        'vals': [],
    },
    'gb_dist_initial': {
        'type': 'compute',
        'name': 'atoms_gb_dist_initial',
        'id': 'atoms_gb_dist_initial',
        'vals': [],
    },
    'gb_dist_final': {
        'type': 'compute',
        'name': 'atoms_gb_dist_final',
        'id': 'atoms_gb_dist_final',
        'vals': [],
    },
    'gb_dist_δ': {
        'type': 'compute',
        'name': 'atoms_gb_dist_δ',
        'id': 'atoms_gb_dist_δ',
        'vals': [],
    },
}


def get_required_defn(var_name, **kwargs):
    # print('get_required_defn: var_name: {}'.format(var_name))
    out = []
    if var_name == 'gb_energy':
        out += get_required_defn('energy',
                                 energy_src=kwargs['energy_src'],
                                 opt_step=kwargs['opt_step'])
        out += [
            PREDEFINED_VARS['num_ions'],
            PREDEFINED_VARS['gb_area'],
            PREDEFINED_VARS['sup_type'],
        ]

    elif var_name == 'gamma_energy':

        out += get_required_defn('energy',
                                 energy_src=kwargs['energy_src'],
                                 opt_step=kwargs['opt_step'])
        out += [
            PREDEFINED_VARS['gamma_surface_shape'],
            PREDEFINED_VARS['gamma_surface_xy'],
            PREDEFINED_VARS['sup_type'],
        ]

    elif var_name == 'energy':
        d = {
            'type': 'result',
            'name': kwargs['energy_src'],
            'id': kwargs['energy_src'],
            'vals': [],
        }
        opt_step = kwargs.get('opt_step')
        if opt_step is not None:
            d.update({
                'idx': (opt_step,),
            })
        out += [d]

    elif var_name == 'energy_pa':
        out += get_required_defn('energy',
                                 energy_src=kwargs['energy_src'],
                                 opt_step=kwargs['opt_step'])
        out += [
            PREDEFINED_VARS['num_ions']
        ]

    elif var_name == 'atoms_gb_dist_δ_final':
        out += [
            PREDEFINED_VARS['gb_dist_initial'],
            PREDEFINED_VARS['gb_dist_final'],
            PREDEFINED_VARS['gb_dist_δ'],
            PREDEFINED_VARS['sup_type'],
        ]

    # print('get_required_defn: out: {}'.format(format_list(out)))

    return out
