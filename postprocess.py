import utils


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


def compute_gamma_energy(out, energy_src, opt_step):
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
    gamma_row_idx_cnd = {
        'type': 'parameter',
        'name': 'series_id',
        'idx': (0, 0, 'row_idx', ),
    }
    gamma_col_idx_cnd = {
        'type': 'parameter',
        'name': 'series_id',
        'idx': (0, 0, 'col_idx', ),
    }
    this_cmpt_cnd = {
        'type': 'compute',
        'name': 'gamma_energy',
    }

    variables = out['variables']
    this_cmpt = dict_from_list(variables, this_cmpt_cnd)

    # Required parameters/results:
    gamma_xy = dict_from_list(variables, gamma_xy_cnd)['vals']
    gamma_shape = dict_from_list(variables, gamma_shape_cnd)['vals']
    energy = dict_from_list(variables, energy_cnd)['vals']
    gamma_row_idx = dict_from_list(variables, gamma_row_idx_cnd)['vals']
    gamma_col_idx = dict_from_list(variables, gamma_col_idx_cnd)['vals']

    gamma_X = np.array(gamma_xy[0]).reshape(gamma_shape).tolist()
    gamma_Y = np.array(gamma_xy[1]).reshape(gamma_shape).tolist()

    # Form a 2D energy array:
    print('gamma_shape: {}'.format(gamma_shape))
    E = np.zeros(tuple(gamma_shape))

    for en_idx, en in enumerate(energy):
        ri = gamma_row_idx[en_idx]
        ci = gamma_col_idx[en_idx]
        E[ri, ci] = en

    this_cmpt['vals'] = [gamma_X, gamma_Y, E.tolist()]


def compute_gb_energy(out, series_id, bulk_src, energy_src, opt_step, unit):

    type_cnd = {
        'type': 'parameter',
        'name': 'base_structure',
        'idx': ('type', ),
    }
    num_ions_cnd = {
        'type': 'result',
        'name': 'num_ions',
    }
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
    num_vals = len(energy)

    # Required computes (if multi-computes, may not be computed yet)
    areas = dict_from_list(variables, areas_cnd)['vals']

    if series_id is not None:
        gb_srs = []
        for s in series_id:
            srs_var = dict_from_list(variables, {'id': s})
            if srs_var is None:
                raise ValueError('Variable with `id` "{}" cannot be'
                                 ' found.'.format(s))
            gb_srs.append(srs_var['vals'])

    gb_idx = []
    bulk_idx = []
    for i_idx, i in enumerate(type_param):
        if i == 'CSLBicrystal':
            gb_idx.append(i_idx)
        elif i == 'CSLBulkCrystal':
            bulk_idx.append(i_idx)

    # Get series_id_val of each GB, bulk supercell
    srs_id_val = out['series_id_val']

    # For each GB supercell, find a bulk supercell whose series val matches
    # (Broadcasting logic would go here)
    all_E_gb = [None, ] * num_vals
    for gb_i in gb_idx:
        for bulk_i in bulk_idx:
            if srs_id_val[gb_i] == srs_id_val[bulk_i]:

                calc_gb = True
                if series_id is not None:
                    for s in gb_srs:
                        if s[gb_i] != s[bulk_i]:
                            calc_gb = False
                            break

                if calc_gb:
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
    'gamma_energy': compute_gamma_energy,
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
    'gb_thickness': (compute_gb_thickness, )
}

# Variables which do not need to be parameterised:
PREDEFINED_VARS = {
    'num_ions': {
        'type': 'result',
        'name': 'num_ions',
        'id': 'num_ions2',
        'vals': [],
    },
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
}


def get_required_defn(var_name, **kwargs):
    print('get_required_defn: var_name: {}'.format(var_name))
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
            PREDEFINED_VARS['gamma_row_idx'],
            PREDEFINED_VARS['gamma_col_idx'],
            PREDEFINED_VARS['gamma_surface_shape'],
            PREDEFINED_VARS['gamma_surface_xy'],
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

    print('get_required_defn: out: {}'.format(format_list(out)))

    return out
