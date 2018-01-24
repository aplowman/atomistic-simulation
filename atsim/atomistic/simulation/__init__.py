"""matsim.atomistic.simulation.__init__.py"""
from atsim.atomistic.structure import atomistic, bicrystal, bravais, crystal

STRUCT_LOOKUP = {
    'BulkCrystal': atomistic.BulkCrystal,
    'csl_bicrystal_from_parameters': bicrystal.csl_bicrystal_from_parameters,
    'csl_bulk_bicrystal_from_parameters': bicrystal.csl_bulk_bicrystal_from_parameters,
    'csl_surface_bicrystal_from_parameters': bicrystal.csl_surface_bicrystal_from_parameters,
    'mon_bicrystal_180_u0w': bicrystal.mon_bicrystal_180_u0w,
}

SUPERCELL_TYPE_LOOKUP = {
    'default': atomistic.AtomisticStructure,
    'bulk': atomistic.BulkCrystal,
    'bicrystal': bicrystal.Bicrystal,
    'bulk_bicrystal': bicrystal.Bicrystal,
    'surface_bicrsytal': bicrystal.Bicrystal,
}


def generate_crystal_structure(cs_defn):
    """Generate a CrystalStructure object."""

    # prt(cs_defn, 'cs_defn')
    cs_params = {}

    if 'path' in cs_defn:
        # Generate CrystalStructure from file
        cs_params.update({
            'path': cs_defn['path'],
            **cs_defn['lattice'],
        })
        if cs_defn.get('motif') is not None:
            cs_params.update({
                'motif': cs_defn['motif'],
            })
        crys_struct = crystal.CrystalStructure.from_file(**cs_params)

    else:
        # Generate CrystalStructure from parameters
        cs_params.update({
            'bravais_lattice': bravais.BravaisLattice(**cs_defn['lattice']),
            'motif': cs_defn['motif'],
        })
        crys_struct = crystal.CrystalStructure(**cs_params)

    return crys_struct


def generate_structure(struct_opts):
    """Generate a new AtomisticStructure object.

    TODO: crystal_structure_modify stuff

    """

    remove_keys = [
        'source',
        'check',
        'func',
        'constraints',
        'crystal_structures',
    ]

    new_opt = {}
    for key, val in struct_opts.items():

        if key in remove_keys:
            continue

        elif key == 'cs_idx':
            crys_struct = generate_crystal_structure(
                struct_opts['crystal_structures'][val]
            )
            new_opt.update({
                'crystal_structure': crys_struct
            })

        else:
            new_opt.update({key: val})

    struct = STRUCT_LOOKUP[struct_opts['func']](**new_opt)
    return struct


def import_structure(import_opts):
    """Import an AtomisticStructure object."""

    raise NotImplementedError('Cannot yet import a structure.')

    # Retrieve the initial base AtomisticStructure object from a previous
    # simulation

    # 1. Connect to archive Resource and get sim_group.json

    # 2. Instantiate SimGroup from the state recorded in sim_group.json

    # 3. Retrieve the correct simulation object from the SimGroup

    # 4. Invoke Simulation.generate_structure to get an AtomisticStructure ojbect
    #    from the correct run and optimisation step.


def get_structure(opts):
    """Return an AtomisticStructure object; imported or generated."""

    import_opts = opts.get('import')
    struct_opts = opts.get('structure')

    if import_opts is not None and import_opts['is_import']:
        struct = import_structure(import_opts)
    else:
        struct = generate_structure(struct_opts)

    return struct
