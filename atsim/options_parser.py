import yaml
import numpy as np
from atsim import utils
from atsim import SET_UP_PATH, SERIES_NAMES, ALLOWED_SERIES_KEYS
import copy
import os


def get_base_structure_defn(opt, opt_lookup):

    explicit_opt = {k: v for k, v in opt.items() if k != '<<lookup>>'}
    lookup_opt_spec = opt['<<lookup>>']

    lkup_name = lookup_opt_spec.split('<<')[1].split('>>')[0]
    lkup_params = lkup_name.split('__')

    allowed_sup_type = [
        'csl_bicrystal',
        #'csl_bicrystal_from_structure', add if we have a lookup for this one
        'csl_bulk_bicrystal_a'
        'csl_bulk_bicrsytal_b',
        'csl_surface_bicrystal_a',
        'csl_surface_bicrystal_b',
        'bulk',
    ]

    sup_type = lkup_params[0]
    if sup_type not in allowed_sup_type:
        raise ValueError('base_structure lookup key supercell type not '
                         'understood. Lookup key must start with one '
                         'of: {}'.format(allowed_sup_type))

    if sup_type == 'csl_bicrystal':
        lkup_params = lkup_params[1:]

    elif sup_type == 'csl_bulk_bicrystal':
        if lkup_params[1] == 'a':
            bulk_idx = 0
        elif lkup_params[1] == 'b':
            bulk_idx = 1
        else:
            raise ValueError('csl_bulk_bicrystal bulk idx must be a or b.')
        lkup_params = lkup_params[2:]

    elif sup_type == 'csl_surface_bicrystal':
        if lkup_params[1] == 'a':
            surface_idx = 0
        elif lkup_params[1] == 'b':
            surface_idx = 1
        else:
            raise ValueError(
                'csl_surface_bicrystal surface idx must be a or b.')
        lkup_params = lkup_params[2:]

    elif sup_type == 'bulk':
        lkup_params = lkup_params[1:]

    base_structure = {
        'type': sup_type,
        'cs_idx': 0,
    }

    if 'csl' in sup_type:

        size_str = lkup_params[-1]
        size = [int(i) for i in
                size_str.strip('[').strip(']').split(',')]
        if len(size) != 3:
            raise ValueError(
                'size must be of length 3: {}'.format(size_str))

        csl_str = lkup_params[0]
        if 'Σ' not in csl_str:
            raise ValueError(
                'Σ value not specified in CSL supercell type.')
        else:
            sigma = csl_str[-1]
            csl_vecs = opt_lookup['csl_vecs'][csl_str]

        gb_type = lkup_params[1]
        allowed_gb_types = ['tilt_A', 'twist']
        if gb_type not in allowed_gb_types:
            raise ValueError('gb_type not allowed: {}'.format(gb_type))

        base_structure.update({
            'sigma': sigma,
            'gb_type': gb_type,
            'gb_size': size,
            'csl_vecs': csl_vecs,
        })
    elif sup_type == 'bulk':

        size_str = lkup_params[-1]
        size = np.array([[int(j) for j in i.strip('[').strip(']').split(',')]
                         for i in size_str.split('_')])

        if size.shape != (3, 3):
            raise ValueError(
                'size must be of shape (3, 3): {}'.format(size_str))

        base_structure.update({
            'box_lat': size,
        })

    # Reassign opt:
    opt = {**base_structure, **explicit_opt}


def parse_string_as(val, data_type):
    """
    Parse a string as an `int`, `float`, or `bool`.

    Parameters
    ----------
    val : str
        String to be parsed. If a value is to be parsed as a float, strings
        like '34/2.3' allowed.
    data_type : type
        One of `int`, `float`, `bool` or `str`.

    Returns
    -------
    parsed val

    Examples
    --------
    >>> parse_string_as('2.5/0.5', float)
    5.0

    """

    bool_strs = {
        True: ['TRUE', 'YES', '1', '1.0'],
        False: ['FALSE', 'NO', '0', '0.0']
    }
    if isinstance(val, str):
        try:
            parsed_val = False
            if data_type is object or data_type is str:
                parsed_val = val
            elif data_type is int:
                parsed_val = int(val)
            elif data_type is float:
                if "/" in val:
                    num, den = val.split("/")
                    parsed_val = float(num) / float(den)
                else:
                    parsed_val = float(val)
            elif data_type is bool:
                v_up = val.upper()
                if v_up in bool_strs[True]:
                    parsed_val = True
                elif v_up in bool_strs[False]:
                    parsed_val = False
                else:
                    raise ValueError(
                        'Cannot parse string {} as type bool'.format(val))
            else:
                raise ValueError(
                    'Cannot parse string {} as type {}'.format(val, data_type))
        except ValueError:
            raise
        return parsed_val
    else:
        raise ValueError('Value passed to parse_string_as ({}) is not a'
                         'string.'.format(val))


def check_lookup(block_name, opt, opt_lookup):
    if '<<lookup>>' in opt:
        explicit_opt = {k: v for k, v in opt.items() if k != '<<lookup>>'}
        lookup_opt_spec = opt['<<lookup>>']
        lkup_name = lookup_opt_spec.split('<<')[1].split('>>')[0]
        opt = {**opt_lookup[block_name][lkup_name], **explicit_opt}
        return opt


def check_invalid_key(opt, allowed_keys):
    for k, v in opt.items():
        if k not in allowed_keys:
            raise ValueError('Key not allowed: {}'.format(k))


def validate_ms_opt(opt_fn, lookup_opt_fn):

    opt_path = os.path.join(SET_UP_PATH, opt_fn)
    opt_lookup_path = os.path.join(SET_UP_PATH, lookup_opt_fn)

    with open(opt_path, 'r', encoding='utf-8') as f:
        opt = yaml.load(f)

    with open(opt_lookup_path, 'r', encoding='utf-8') as f:
        opt_lookup = yaml.load(f)

    deep_keys = [
        'set_up',
        'base_structure',
        'constraints.cell',
        'constraints.atom',
        'castep',
        'castep.param',
        'lammps',
        'scratch',
        'stage',
        'archive',
        'database',
    ]
    for dk in deep_keys:
        if opt.get(dk) is not None:
            opt[dk + '.<<lookup>>'] = opt.pop(dk)

    opt_unflat = utils.unflatten_dict_keys(opt)

    allowed_keys = [
        'set_up',
        'base_structure',
        'constraints',
        'castep',
        'lammps',
        'crystal_structures',
        'series',
        'method',
        'make_plots',
        'sub_dirs',
        'append_db',
        'upload_plots',
        'scratch',
        'stage',
        'archive',
        'database',
        'offline_files',
    ]

    check_invalid_key(opt_unflat, allowed_keys)

    valid_opt = {}
    for k, v in opt_unflat.items():

        if k == 'archive':
            valid_opt.update({k: validate_ms_archive(v, opt_lookup)})

        elif k == 'stage':
            valid_opt.update({k: validate_ms_stage(v, opt_lookup)})

        elif k == 'database':
            valid_opt.update({k: validate_ms_database(v, opt_lookup)})

        elif k == 'scratch':
            valid_opt.update({k: validate_ms_scratch(v, opt_lookup)})

        elif k == 'base_structure':
            valid_opt.update({k: validate_ms_base_structure(v, opt_lookup)})

        elif k == 'crystal_structures':
            valid_opt.update(
                {k: validate_ms_crystal_structures(v, opt_lookup)})

        elif k == 'series':
            valid_opt.update({k: validate_ms_series(v)})

        elif k == 'constraints':
            valid_opt.update({k: validate_ms_constraints(v, opt_lookup)})

        elif k == 'castep':
            valid_opt.update({k: validate_ms_castep(v, opt_lookup)})

        elif k == 'lammps':
            valid_opt.update({k: validate_ms_lammps(v, opt_lookup)})

        else:
            valid_opt.update({k: v})

    return valid_opt


def validate_ms_archive(opt, opt_lookup):
    opt = check_lookup('archive', opt, opt_lookup) or opt
    allowed_keys = ['dropbox', 'path']
    check_invalid_key(opt, allowed_keys)
    return opt


def validate_ms_stage(opt, opt_lookup):
    opt = check_lookup('stage', opt, opt_lookup) or opt
    allowed_keys = ['path']
    check_invalid_key(opt, allowed_keys)
    return opt


def validate_ms_database(opt, opt_lookup):
    opt = check_lookup('database', opt, opt_lookup) or opt
    allowed_keys = ['dropbox', 'path']
    check_invalid_key(opt, allowed_keys)
    return opt


def validate_ms_scratch(opt, opt_lookup):

    def validate_offline_files(off_fls_opt):
        allowed_keys = [
            'path',
            'match',
        ]
        check_invalid_key(off_fls_opt, allowed_keys)
        valid_off_fls = copy.deepcopy(off_fls_opt)
        return valid_off_fls

    opt_flat = check_lookup('scratch', opt, opt_lookup) or opt
    opt = utils.unflatten_dict_keys(opt_flat)
    allowed_keys = [
        'num_cores',
        'path',
        'remote',
        'host',
        'os_name',
        'parallel_env',
        'sge',
        'module_load',
        'job_array',
        'offline_files',
    ]
    check_invalid_key(opt, allowed_keys)
    valid_scratch = {}
    for k, v in opt.items():
        if k == 'offline_files':
            valid_scratch.update({k: validate_offline_files(v)})
        else:
            valid_scratch.update({k: v})

    return opt


def validate_ms_base_structure(opt, opt_lookup):

    def validate_ms_csl_params(csl_params_opt, opt_lookup):
        csl_params_opt = check_lookup(
            'csl_params', csl_params_opt, opt_lookup) or csl_params_opt
        allowed_keys = [
            'cellfile',
            'repeats',
            'bound_vac',
            'transls',
            'term_plns',
        ]
        check_invalid_key(csl_params_opt, allowed_keys)
        return csl_params_opt

    if '<<lookup>>' in opt:
        opt = get_base_structure_defn(opt, opt_lookup)

    sup_type = opt['type']
    allowed_keys_all = [
        'type',
    ]
    allowed_keys_gb_from_structure = allowed_keys_all + [
        'csl',
        'csl_params',
    ]
    allowed_keys_all_with_cs = allowed_keys_all + [
        'cs_idx',
    ]
    allowed_keys_csl = allowed_keys_all_with_cs + [
        'sigma',
        'gb_type',
        'gb_size',
        'csl_vecs',
    ]
    allowed_keys_bulk = allowed_keys_all_with_cs + [
        'box_lat'
    ]
    allowed_keys = {
        'csl_bicrystal': allowed_keys_csl,
        'csl_bulk_bicrystal': allowed_keys_csl,
        'csl_surface_bicrystal': allowed_keys_csl,
        'csl_bicrystal_from_structure': allowed_keys_gb_from_structure,
        'bulk': allowed_keys_bulk,
    }
    allowed_types = list(allowed_keys.keys())
    if sup_type not in allowed_types:
        raise ValueError(
            'base_structure.type: {} is unknown.'.format(sup_type))
    check_invalid_key(opt, allowed_keys[sup_type])

    valid_bs = {}
    for k, v in opt.items():
        if k == 'csl_params':
            valid_bs.update({k: validate_ms_csl_params(v, opt_lookup)})

        elif k == 'csl_vecs':
            # Convert csl_vecs to numpy array:
            valid_csl = [np.array(copy.deepcopy(c)) for c in v]
            valid_bs.update({k: valid_csl})

        else:
            valid_bs.update({k: v})

    return valid_bs


def validate_ms_crystal_structures(opt, opt_lookup):

    def validate_lattice(lat_opt):
        allowed_keys = [
            'system',
            'centring',
            'a',
            'b',
            'c',
            'α',
            'β',
            'γ',
        ]
        check_invalid_key(lat_opt, allowed_keys)

        # Rename system to lattice system and centring to centring type
        # so it can be passed directly to bravais.BravaisLattice:
        lat_sys = lat_opt['system']
        cen_type = lat_opt['centring']
        valid_lat = copy.deepcopy(lat_opt)
        valid_lat.update({
            'lattice_system': lat_sys,
            'centring_type': cen_type,
        })
        del valid_lat['system']
        del valid_lat['centring']

        return valid_lat

    def validate_motif(motif_opt):
        allowed_keys = [
            'atom_sites',
            'species',
        ]
        check_invalid_key(motif_opt, allowed_keys)
        valid_motif = copy.deepcopy(motif_opt)

        # Convert atom_sites to a numpy array:
        as_floats = [[parse_string_as(j, float) for j in i]
                     for i in valid_motif['atom_sites']]
        valid_motif['atom_sites'] = np.array(as_floats)

        return valid_motif

    allowed_keys = [
        'lattice',
        'motif',
        'path',
    ]

    if not opt:
        raise ValueError('crystal_structures must be assigned.')
    valid_opt = []
    for cs_idx, cs in enumerate(opt):

        if isinstance(cs, str):
            if '<<' in cs and '>>' in cs:
                cs = {'<<lookup>>': cs}

        cs = check_lookup('crystal_structures', cs, opt_lookup) or cs

        cs_opt = utils.unflatten_dict_keys(cs)
        check_invalid_key(cs_opt, allowed_keys)
        valid_cs = {}
        for k, v in cs_opt.items():
            if k == 'lattice':
                valid_cs.update({k: validate_lattice(v)})
            elif k == 'motif':
                valid_cs.update({k: validate_motif(v)})
            else:
                valid_cs.update({k: v})

        valid_opt.append(valid_cs)

    return valid_opt


def validate_ms_series(opt):

    def validate_series_item(srs_itm):

        srs_nm = srs_itm['name']

        if srs_nm not in SERIES_NAMES:
            raise ValueError('Series name "{}" not allowed.'.format(srs_nm))
        check_invalid_key(srs_itm, ALLOWED_SERIES_KEYS[srs_nm])
        return srs_itm

    if opt is None:
        opt = [[]]
    valid_series = []
    for i in opt:
        valid_sub_series = []
        for j in i:
            valid_sub_series.append(
                validate_series_item(j))
        valid_series.append(valid_sub_series)
    return valid_series


def validate_ms_constraints(opt, opt_lookup):

    def validate_ms_cell_constraints(cell_opt, opt_lookup):
        cell_opt = check_lookup(
            'cell_constraints', cell_opt, opt_lookup) or cell_opt
        allowed_keys = [
            'fix_angles',
            'fix_lengths',
            'angles_equal',
            'lengths_equal',
        ]
        check_invalid_key(cell_opt, allowed_keys)
        return cell_opt

    def validate_ms_atom_constraints(atom_opt, opt_lookup):
        atom_opt = check_lookup(
            'atom_constraints', atom_opt, opt_lookup) or atom_opt
        allowed_keys = [
            'fix_xy_idx',
            'fix_xyz_idx',
        ]
        check_invalid_key(atom_opt, allowed_keys)
        return atom_opt

    allowed_keys = [
        'cell',
        'atom',
    ]
    check_invalid_key(opt, allowed_keys)

    valid_const = {}
    for k, v in opt.items():
        if k == 'cell':
            valid_const.update(
                {k: validate_ms_cell_constraints(v, opt_lookup)})
        elif k == 'atom':
            valid_const.update(
                {k: validate_ms_atom_constraints(v, opt_lookup)})
        else:
            valid_const.update({k: v})

    return valid_const


def validate_ms_castep(opt, opt_lookup):

    def validate_ms_castep_cell(cell_opt):
        cell_opt = check_lookup(
            'castep.cell', cell_opt, opt_lookup) or cell_opt
        return cell_opt

    def validate_ms_castep_param(param_opt):
        param_opt = check_lookup(
            'castep.param', param_opt, opt_lookup) or param_opt
        return param_opt

    allowed_keys = [
        'cell',
        'param',
        'find_inv_sym',
        'seedname',
        'checkpoint',
    ]
    check_invalid_key(opt, allowed_keys)
    valid_castep = {}
    for k, v in opt.items():
        if k == 'cell':
            valid_castep.update({k: validate_ms_castep_cell(v)})
        if k == 'param':
            valid_castep.update({k: validate_ms_castep_param(v)})
        else:
            valid_castep.update({k: v})

    return valid_castep


def validate_ms_lammps(opt, opt_lookup):
    opt = check_lookup('lammps', opt, opt_lookup) or opt
    allowed_keys = [
        'parameters',
        'atom_style',
        'atoms_file',
        'interactions',
        'potential_files',
        'dump_dt',
    ]
    check_invalid_key(opt, allowed_keys)
    return opt
