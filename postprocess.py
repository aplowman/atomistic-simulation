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


def compute_gb_energy(results):

    # Get idx of results which are `CSLBicrystal` types
    params = results['parameters']
    res = results['results']
    comp = results['computes']

    ths_idx, ths_comp = dict_from_list(comp, name='gb_energy', ret_index=True)
    type_param = dict_from_list(
        params, name='base_structure', idx=('type',))['vals']
    num_ions = dict_from_list(res, name='num_ions')['vals']

    energy = dict_from_list(res, id='opt_energy')['vals']
    areas = dict_from_list(comp, name='gb_area')['vals']
    num_vals = len(type_param)

    srs_id_defn = ths_comp.get('series_id')
    if srs_id_defn is not None:
        all_d = params + res + comp
        gb_srs = []
        for s in srs_id_defn:
            gb_srs.append(dict_from_list(all_d, id=s)['vals'])
        print('gb_srs: {}'.format(gb_srs))

    gb_idx = []
    bulk_idx = []
    for i_idx, i in enumerate(type_param):
        if i == 'CSLBicrystal':
            gb_idx.append(i_idx)
        elif i == 'CSLBulkCrystal':
            bulk_idx.append(i_idx)

    # print('gb_idx: {}'.format(gb_idx))
    # print('bulk_idx: {}'.format(bulk_idx))

    # Get series_id_val of each GB, bulk supercell
    srs_id_val = results['series_id_val']

    # For each GB supercell, find a bulk supercell whose series val matches
    # (Broadcasting logic would go here)
    all_E_gb = [None, ] * num_vals
    for gb_i in gb_idx:
        for bulk_i in bulk_idx:
            if srs_id_val[gb_i] == srs_id_val[bulk_i]:

                calc_gb = True
                if srs_id_defn is not None:
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

                    # print('area: {:.20f}'.format(areas[gb_i]))

        if ths_comp['unit'] == 'J/m^2':
            E_gb *= 16.02176565
        all_E_gb[gb_i] = E_gb
    results['computes'][ths_idx]['vals'] = all_E_gb


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

# Computed quantities which are dependent on more than one simulation:


def get_energy(energy_type):
    energy = {
        'display_name': 'Final Energy',
        'id': 'opt_energy',
        'fmt': '{:15.10f}',
        'idx': (-1,),
        'name': 'final_energy',
        'unit': 'eV',
    }
    if energy_type == 'final_zenergy':
        energy.update({
            'name': 'final_zenergy',
            'display_name': 'Final 0K Energy',
        })
    elif energy_type == 'final_fenergy':
        energy.update({
            'name': 'final_fenergy',
            'display_name': 'Final Free Energy',
        })
    return energy


sup_type = {
    'display_name': 'Supercell type',
    'fmt': '{}',
    'idx': ('type',),
    'name': 'base_structure',
}
num_ions = {
    'display_name': '#Atoms',
    'fmt': '{:d}',
    'name': 'num_ions',
}
gb_area = {
    'name': 'gb_area',
    'display_name': 'GB Area',
    'unit': 'Ang^2'
}

MULTI_COMPUTES = {
    'gb_energy': {
        'requires': {
            'computes': [gb_area],
            'parameters': [sup_type],
            'results': [get_energy, num_ions],
        },
        'func': compute_gb_energy,
    },
    'surface_energy': {
        'requires': {
            'computes': [gb_area],
            'parameters': [sup_type],
            'results': [get_energy, num_ions]
        }
    },
    'cohesive_energy': {
        'requires': {
            'computes': [gb_area],
            'parameters': [sup_type],
            'results': [get_energy, num_ions]
        }
    },
}


def dict_from_list(lst, ret_index=False, **conditions):
    """
    Get the first dict from a list of dict given one or more matching
    key-values.

    Parameters
    ----------
    lst : list
    conditions : dict
    index : bool
        If True, return a tuple (element_index, element) else return element

    """

    for el_idx, el in enumerate(lst):

        condition_match = False
        for cnd_key, cnd_val in conditions.items():

            v = el.get(cnd_key)

            if v is not None and v == cnd_val:
                condition_match = True
            else:
                condition_match = False
                break

        if condition_match:
            if ret_index:
                return (el_idx, el)
            else:
                return el
