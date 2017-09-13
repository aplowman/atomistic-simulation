import numpy as np
import atsim.utils
from atsim.utils import dict_from_list
from atsim.readwrite import format_list, format_dict


# Variables which do not need to be parameterised:
PREDEFINED_VARS = {
    'gb_area': {
        'type': 'compute',
        'name': 'gb_area',
        'id': 'gb_area',
    },
    'sup_type': {
        'type': 'parameter',
        'name': 'base_structure',
        'idx': ('type',),
        'id': 'supercell_type',
    },
    'gb_dist_initial': {
        'type': 'compute',
        'name': 'atoms_gb_dist_initial',
        'id': 'atoms_gb_dist_initial',
    },
    'gb_dist_final': {
        'type': 'compute',
        'name': 'atoms_gb_dist_final',
        'id': 'atoms_gb_dist_final',
    },
    'gb_dist_change': {
        'type': 'compute',
        'name': 'atoms_gb_dist_change',
        'id': 'atoms_gb_dist_change',
    },
}


def get_depends(compute_name, inc_id=True, inc_val=True, **kwargs):
    """
    For a given compute, check if it has any dependencies. If it does,
    return a list of those as new definitions, in addition to the specified 
    compute, in the correct dependency order.

    Parameters
    ----------


    """
    # Validation:
    allowed_computes = (list(SINGLE_COMPUTE_LOOKUP.keys()) +
                        list(MULTI_COMPUTE_LOOKUP.keys()))

    if compute_name not in allowed_computes:
        raise ValueError('Compute "{}" is not allowed.'.format(compute_name))

    d = {
        'type': 'compute',
        'name': compute_name,
    }
    if inc_id:
        d.update({'id': compute_name})

    out = []
    if compute_name == 'gb_energy':

        d.update({
            'energy_src': kwargs['energy_src'],
            'opt_step': kwargs['opt_step'],
            'series_id': kwargs['series_id'],
        })
        out = (get_depends('energy', inc_id=inc_id, inc_val=inc_val,
                           energy_src=kwargs['energy_src'],
                           opt_step=kwargs['opt_step']) +
               get_depends('num_atoms', inc_id=inc_id, inc_val=inc_val) +
               [PREDEFINED_VARS['gb_area'],
                PREDEFINED_VARS['sup_type']]
               ) + out

    elif compute_name == 'num_atoms':
        pass

    elif compute_name == 'energy_per_atom':

        d.update({
            'energy_src': kwargs['energy_src'],
            'opt_step': kwargs['opt_step']
        })
        out = (get_depends('num_atoms', inc_id=inc_id, inc_val=inc_val) +
               get_depends('energy', inc_id=inc_id, inc_val=inc_val,
                           energy_src=kwargs['energy_src'],
                           opt_step=kwargs['opt_step']) + out)

    elif compute_name == 'gamma_energy':

        out = get_depends('energy', inc_id=inc_id, inc_val=inc_val,
                          energy_src=kwargs['energy_src'],
                          opt_step=kwargs['opt_step']) + out
        out = [
            PREDEFINED_VARS['gamma_surface_shape'],
            PREDEFINED_VARS['gamma_surface_xy'],
            PREDEFINED_VARS['sup_type'],
        ] + out

    elif compute_name == 'energy':

        d.update({
            'energy_src': kwargs['energy_src'],
        })
        opt_step = kwargs.get('opt_step')
        if opt_step is not None:
            d.update({
                'opt_step': opt_step,
            })

    elif compute_name == 'atoms_gb_dist_change':

        out = [
            PREDEFINED_VARS['gb_dist_initial'],
            PREDEFINED_VARS['gb_dist_final'],
            PREDEFINED_VARS['sup_type'],
        ] + out

    # If the d dict is not in out, add it:
    d_out = dict_from_list(out, d)
    if d_out is None:
        out += [d]

    # Add a vals key to each dict if inc_val is True:
    for v_idx, v in enumerate(out):
        if v.get('vals') is None and inc_val:
            out[v_idx].update({'vals': []})
        elif v.get('vals') is not None and not inc_val:
            del out[v_idx]['vals']

    return out


def num_atoms(out, sim, sim_idx):
    return sim.structure.num_atoms


def energy(out, sim, sim_idx, energy_src, opt_step=None):

    # Validation
    err_msg = 'Energy source: "{}" not available from {} output.'
    method = sim.options['method']
    allowed_en_srcs = {
        'castep': ['final_energy', 'final_fenergy', 'final_zenergy'],
        'lammps': ['final_energy'],
    }
    if energy_src not in allowed_en_srcs[method]:
        raise ValueError(err_msg.format(energy_src, method.upper()))

    energy = sim.results[energy_src]

    if opt_step is None:
        return energy
    else:
        if not isinstance(opt_step, int):
            raise ValueError('`opt_step` must be an integer.')
        return energy[opt_step]


def energy_per_atom(out, sim, sim_idx, energy_src, opt_step=None):

    rv_args = {
        'compute_name': 'energy_per_atom',
        'inc_id': False,
        'inc_val': False,
        'energy_src': energy_src,
        'opt_step': opt_step,
    }
    req_vars_defn = get_depends(**rv_args)
    vrs = out['variables']
    req_vars = [dict_from_list(vrs, i) for i in req_vars_defn]
    n = req_vars[0]['vals'][sim_idx]
    e = req_vars[1]['vals'][sim_idx]
    return e / n


def gb_area(out, sim, sim_idx):

    if sim.options['base_structure']['type'] == 'CSLBicrystal':
        return sim.structure.boundary_area
    else:
        return None


def gb_thickness(out, sim, sim_idx):

    if sim.options['base_structure']['type'] == 'CSLBicrystal':
        return sim.structure.bicrystal_thickness
    else:
        return None


def atoms_gb_dist_initial(out, sim, sim_idx):

    if sim.options['base_structure']['type'] == 'CSLBicrystal':
        return sim.structure.atoms_gb_dist
    else:
        return None


def atoms_gb_dist_final(out, sim, sim_idx):

    if sim.options['base_structure']['type'] == 'CSLBicrystal':

        if sim.options['method'] == 'lammps':
            atom_sites_final = sim.results['atoms'][-1]
        elif sim.options['method'] == 'castep':
            atom_sites_final = sim.results['geom']['ions'][-1]

        n_unit = sim.structure.n_unit
        gb_dist = np.einsum('jk,jl->k', atom_sites_final, n_unit)

        return gb_dist

    else:
        return None


def atoms_gb_dist_change(out, sim, sim_idx):

    rv_args = {
        'compute_name': 'atoms_gb_dist_change',
        'inc_id': False,
        'inc_val': False,
    }
    req_vars_defn = get_depends(**rv_args)
    vrs = out['variables']
    req_vars = [dict_from_list(vrs, i) for i in req_vars_defn]

    atoms_gb_dist_initial = req_vars[0]['vals'][sim_idx]
    atoms_gb_dist_final = req_vars[1]['vals'][sim_idx]
    sup_type = req_vars[2]['vals'][sim_idx]

    print('atoms_gb_dist_initial: {}'.format(atoms_gb_dist_initial))
    print('atoms_gb_dist_final: {}'.format(atoms_gb_dist_final))
    print('sup_type: {}'.format(sup_type))

    if sup_type == 'CSLBicrystal':
        print('blah!')
        return np.array(atoms_gb_dist_final) - np.array(atoms_gb_dist_initial)


def gb_energy(out, series_id, energy_src, opt_step, unit='J/m^2'):
    """
    Computes the grain boundary energy for multiple simulations.

    Parameters
    ----------
    out : dict
        Output dict in which to save the output GB energies.
    series_id : str
        Series name used to match bulk and grain boundary supercells for GB
        energy computation. Can either be a series name or a variable ID.
    energy_src : str
        Energy key from the simulation output dict to use in the GB energy
        computation.
    opt_step : int
        Optimisation step from which to take the energies in the GB energy
        computation.

    """

    rv_args = {
        'compute_name': 'gb_energy',
        'inc_id': False,
        'inc_val': False,
        'energy_src': energy_src,
        'opt_step': opt_step,
        'series_id': series_id,
    }
    req_vars_defn = get_depends(**rv_args)
    vrs = out['variables']
    req_vars = [dict_from_list(vrs, i) for i in req_vars_defn]
    energy, num_atoms, area, sup_type = [req_vars[i]['vals'] for i in range(4)]
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

    gb_idx = []
    bulk_idx = []
    for i_idx, i in enumerate(sup_type):
        if i == 'CSLBicrystal':
            gb_idx.append(i_idx)
        elif i == 'CSLBulkCrystal':
            bulk_idx.append(i_idx)

    all_E_gb = [None, ] * num_sims
    for gb_i in gb_idx:
        E_gb = None
        for bulk_i in bulk_idx:
            if srs_vals[gb_i] == srs_vals[bulk_i]:
                gb_num = num_atoms[gb_i]
                bulk_num = num_atoms[bulk_i]
                bulk_frac = gb_num / bulk_num
                E_gb = (energy[gb_i] - bulk_frac *
                        energy[bulk_i]) / (2 * area[gb_i])

        if E_gb is not None:
            if unit == 'J/m^2':
                E_gb *= 16.02176565
            all_E_gb[gb_i] = E_gb
    req_vars[-1]['vals'] = all_E_gb


# Single-compute functions are passed individual AtomisticSimulation objects:
SINGLE_COMPUTE_LOOKUP = {
    'num_atoms': num_atoms,
    'energy': energy,
    'energy_per_atom': energy_per_atom,
    'gb_area': gb_area,
    'gb_thickness': gb_thickness,
    'atoms_gb_dist_initial': atoms_gb_dist_initial,
    'atoms_gb_dist_final': atoms_gb_dist_final,
    'atoms_gb_dist_change': atoms_gb_dist_change,
    # 'atoms_gb_dist_δ': atoms_gb_dist_δ,
}

# Multi-compute functions are passed the whole output dict of harvest.py as it
# is being constructed:
MULTI_COMPUTE_LOOKUP = {
    'gb_energy': gb_energy,
}
