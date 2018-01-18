import os
import copy
import shutil
import numpy as np
from plotly import graph_objs
from plotly.offline import plot, iplot, init_notebook_mode
from atsim import plotting, geometry, vectors, mathsutils, readwrite, utils
from atsim.structure.atomistic import AtomisticStructure
from atsim.structure.bicrystal import Bicrystal
from atsim.structure.crystal import CrystalBox, CrystalStructure
from atsim.simsio import castep, lammps

from atsim.utils import prt, mut_exc_args
from atsim.structure import atomistic, bicrystal
from atsim.structure.bravais import BravaisLattice, get_hex_a, get_hex_vol
from atsim import REF_PATH

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


def modify_crystal_structure(cs, vol_change, ca_change):
    """
    Regenerate a CrystalStructure with a modified bravais lattice.

    Parameters
    ----------
    cs : CrystalStructure object
    vol_change : float
        Percentage change in volume
    ca_change : float
        Percentage change in c/a ratio

    Returns
    -------
    CrystalStructure

    """
    bl = cs.bravais_lattice

    # Modify hexagonal CrystalStructure
    if bl.lattice_system != 'hexagonal':
        raise NotImplementedError('Cannot modify non-hexagonal crystal '
                                  'structure.')

    # Generate new a and c lattice parameters based on originals and volume and
    # c/a ratio changes:
    v = get_hex_vol(bl.a, bl.c)
    v_new = v * (1 + vol_change / 100)

    ca_new = (bl.c / bl.a) * (1 + ca_change / 100)
    a_new = get_hex_a(ca_new, v_new)
    c_new = ca_new * a_new

    bl_new = BravaisLattice('hexagonal', a=a_new, c=c_new)
    cs_new = CrystalStructure(bl_new, copy.deepcopy(cs.motif))
    return cs_new


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
        crys_struct = CrystalStructure.from_file(**cs_params)

    else:
        # Generate CrystalStructure from parameters
        cs_params.update({
            'bravais_lattice': BravaisLattice(**cs_defn['lattice']),
            'motif': cs_defn['motif'],
        })
        crys_struct = CrystalStructure(**cs_params)

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


class AtomisticSimulation(object):

    def __init__(self, options=None, state=None):

        # prt(readwrite.format_dict(options), 'options')
        # prt(state, 'state')

        mut_exc_args({'options': options}, {'state': state})

        if state:
            self.options = None
            self.structure = state['structure']

        else:
            self.options = options
            self.structure = get_structure(options)
            self.runs = []

    @classmethod
    def copy_reference_data(cls, sim_params, stage_path, scratch_path):
        """Copy any reference files necessary for these simulations to a
        suitable location."""
        pass

    def _process_options(self):
        """Additional processing on options to prepare for writing input files.

        Process constraint options. For atom constraints, convert `none` to
        None, and convert `all` to an index array of all atoms. Indexing starts
        from 1!

        TODO: check "indexing starts from 1" is correct.

        """

        constraints = self.options['structure']['constraints']
        cell_const = constraints['cell']
        atom_const = constraints['atoms']

        cell_const = {
            'fix_angles': 'none',
            'fix_lengths': 'none',
            'angles_equal': 'none',
            'lengths_equal': 'none',
            **cell_const,
        }
        atom_const = {
            'fix_xy_idx': 'none',
            'fix_xz_idx': 'none',
            'fix_yz_idx': 'none',
            'fix_xyz_idx': 'none',
            **atom_const,
        }

        valid_atom_cnst = {}
        for key, val in atom_const.items():

            if isinstance(val, list):
                valid_atom_cnst.update({key: np.array(val)})

            elif isinstance(val, str):

                if val.upper() == 'NONE':
                    valid_atom_cnst.update({key: None})

                elif val.upper() == 'ALL':
                    all_atm = np.arange(self.structure.atom_sites.shape[1]) + 1
                    valid_atom_cnst.update({key: all_atm})

        cell_fx = ['fix_angles', 'fix_lengths',
                   'angles_equal', 'lengths_equal']
        for fix in cell_fx:

            if cell_const[fix] == 'none':
                cell_const[fix] = None

            if cell_const[fix] == 'all':
                cell_const[fix] = 'abc'

        self.options['structure']['constraints']['atoms'] = valid_atom_cnst
        self.options['structure']['constraints']['cell'] = cell_const

        prt(self.options, 'self.options (after AtSim processing)')

    @classmethod
    def split_simulation_options(cls, options):
        """Split the options dict into base options valid for all Simulation
        types and those applicable to just AtomisticSimulation types."""

        struct_opt = options.pop('structure')
        ret = {
            'options': options,
            'structure': struct_opt,
        }

        return ret

    def to_jsonable(self):
        """Generate a dict representation that can be JSON serialised."""

        # `runs` have simulation-specific results and therefore should be
        # JSONified in their respective sub-classes.

        # Add all runs apart from `result` key (Simulation class-dependent)
        runs_js = []
        for run_idx, _ in enumerate(self.runs):
            run_info = copy.deepcopy(self.runs[run_idx])
            run_info['result'] = None
            runs_js.append(run_info)

        ret = {
            'structure': self.structure.to_jsonable(),
            'runs': runs_js,
        }
        return ret

    def generate_structure(self, opt_idx=-1, tile=None):
        """
        Generate an AtomisticStructure object for a given optimisation step
        after the simulation has run.

        Parameters
        ----------
        opt_idx : int, optional
            The optimisation step from which to generate the structure.

        tile : tuple of int, optional
            The integer number of times to tile the supercell in each
            supercell direction. Default is None, in which cas the supercell
            is not tiled.

        """

        if self.results is None:
            raise ValueError('Cannot generate an optimised AtomisticStructure '
                             'object for a simulation whose results have '
                             'not been harvested.')

        # Get supercell and atom sites from given optimisation step
        method = self.options['method']

        species_orig = self.structure.species
        species_idx_orig = self.structure.species_idx
        supercell_orig = self.structure.supercell
        supercell_orig_inv = np.linalg.inv(supercell_orig)

        if method == 'castep':

            supercell = self.results['geom']['cells'][opt_idx]
            atom_sites = self.results['geom']['ions'][opt_idx]

            reorder_map = castep.map_species_to_castep(
                species_orig, species_idx_orig)
            reorder_map_inv = np.argsort(reorder_map)

            # Atoms reordered based on original species_idx
            atom_sites = atom_sites[:, reorder_map_inv]

        elif method == 'lammps':

            supercell = self.results['supercell'][opt_idx]
            atom_sites = self.results['atoms'][opt_idx]

        # Assume the bounds of each crystal to remain at fixed fractional
        # coordinates within the supercell during relaxation.

        crystals_new = []
        for c_idx, c in enumerate(self.structure.crystals):

            c_vecs = c['crystal']
            c_origin = c['origin']

            # Transform to fractional coordinates in the original supercell
            c_vecs_sup = np.dot(supercell_orig_inv, c_vecs)
            c_origin_sup = np.dot(supercell_orig_inv, c_origin)

            # Transform to Cartesian coordinates in the new supercell
            c_vecs_new = np.dot(supercell, c_vecs_sup)
            c_origin_new = np.dot(supercell, c_origin_sup)

            crystals_new.append({
                'crystal': c_vecs_new,
                'origin': c_origin_new,
            })

            additional_info = ['cs_idx', 'cs_orientation', 'cs_origin']
            for i in additional_info:
                if c.get(i) is not None:
                    crystals_new[-1].update({i: c[i]})

        as_params = {
            'atom_sites': atom_sites,
            'atom_labels': copy.deepcopy(self.structure.atom_labels),
            'supercell': supercell,
            'crystals': self.structure.crystals,
            'crystal_structures': self.structure.crystal_structures,
        }

        if 'bicrystal' in self.structure.meta:

            # Regenerate a bicrystal
            bc_params = {
                'as_params': as_params,
                'maintain_inv_sym': self.structure.maintain_inv_sym,
                'non_gb_idx': self.structure.nbi,
                'rot_mat': self.structure.rot_mat
            }
            opt_structure = bicrystal.Bicrystal(**bc_params)

        else:
            opt_structure = AtomisticStructure(**as_params)

        if tile is not None:
            opt_structure.tile_supercell(tile)

        return opt_structure


class CastepSimulation(AtomisticSimulation):
    """Class to represent a CASTEP simulation."""

    def __init__(self, options=None, state=None):
        """Initialise a CastepSimulation."""
        super().__init__(options=options, state=state)
        self._process_options()

    def _process_options(self):
        """Additional processing on LAMMPS options to prepare for writing input
        files.

        """
        super()._process_options()

        castep_opt = self.options['params']['castep']

        # Sort out checkpointing:
        if castep_opt.get('checkpoint') is True:
            if castep_opt.get('backup_interval') is not None:
                castep_opt['param'].update(
                    {'backup_interval': castep_opt['backup_interval']})

        else:
            castep_opt['param'].update({'write_checkpoint': 'none'})

        castep_opt.pop('backup_interval', None)
        castep_opt.pop('checkpoint', None)

        # Remove geometry optimisation parameters if not necessary:
        castep_task = castep_opt['param']['task'].upper()
        geom_opt_str = ['GEOMETRYOPTIMISATION', 'GEOMETRYOPTIMIZATION']
        if castep_task not in geom_opt_str:

            geom_keys = []
            for param_key in castep_opt['param']:
                if param_key.upper().startswith('GEOM_'):
                    geom_keys.append(param_key)

            for geom_k in geom_keys:
                castep_opt['param'].pop(geom_k, None)

        # Remove constraints if task is single point:
        if castep_task == 'SINGLEPOINT':

            constraints = self.options['structure']['constraints']
            cell_const = constraints['cell']
            atom_const = constraints['atoms']

            cell_const.pop('cell_angles_equal', None)
            cell_const.pop('cell_lengths_equal', None)
            cell_const.pop('fix_cell_angles', None)
            cell_const.pop('fix_cell_lengths', None)
            atom_const.pop('fix_xy_idx', None)
            atom_const.pop('fix_xz_idx', None)
            atom_const.pop('fix_yz_idx', None)
            atom_const.pop('fix_xyz_idx', None)

        # Add symmetry operations:
        if castep_opt['find_inv_sym']:

            if castep_opt['cell'].get('symmetry_generate') is True:
                msg = ('Cannot add inversion symmetry operation to CASTEP '
                       'CELL file if `symmetry_generate` is `True`.')
                raise ValueError(msg)

            sym_ops = self.structure.get_sym_ops()

            sym_rots = sym_ops['rotations']
            sym_trans = sym_ops['translations']
            inv_sym_rot = -np.eye(3, dtype=int)
            inv_sym_idx = np.where(
                np.all(sym_rots == inv_sym_rot, axis=(1, 2)))[0]

            if not inv_sym_idx:
                msg = 'The supercell does not have inversion symmetry.'
                raise ValueError(msg)

            inv_sym_trans = sym_trans[inv_sym_idx[0]]

            castep_opt['sym_ops'] = [
                np.vstack([np.eye(3), np.zeros((3,))]),
                np.vstack([inv_sym_rot, inv_sym_trans])
            ]

    def write_input_files(self, path):

        common_params = {
            'supercell': self.structure.supercell,
            'atom_sites': self.structure.atom_sites,
            'species': self.structure.species,
            'species_idx': self.structure.species_idx,
            'path': path,
        }

        cst_opt = self.options['params']['castep']
        cst_in_params = {
            'seedname': cst_opt['seedname'],
            'cell': cst_opt['cell'],
            'param': cst_opt['param'],
            # 'sym_ops': cst_opt['sym_ops'],
            **common_params
        }

        castep.write_castep_inputs(**cst_in_params)

    def to_jsonable(self):
        """Generate a dict representation that can be JSON serialised."""

        # `structure` and `options` are dealt with in the super-class:
        ret = super().to_jsonable()

        # Add `result` key from each run in `runs`:
        for i, j in zip(ret['runs'], self.runs):
            if j['result'] is not None:
                i['result'] = {
                    # TODO, check which CASTEP results vals need `tolist()`ing.
                    **j['result'],
                }

        return ret

    @classmethod
    def from_jsonable(cls, state):
        """Instantiate from a JSONable dict."""

        runs_native = state['runs']
        for idx, _ in enumerate(runs_native):
            if runs_native[idx]['result'] is not None:
                runs_native[idx]['result'] = {
                    # TODO, check which CASTEP results vals need `array`ing.
                    **runs_native[idx]['result'],
                }

        sup_types_str = state['structure']['meta'].get('supercell_type')
        sup_type_class = 'default'
        for sup_type in sup_types_str:
            if sup_type in SUPERCELL_TYPE_LOOKUP:
                sup_type_class = sup_type
                break
        struct_class = SUPERCELL_TYPE_LOOKUP[sup_type_class]

        state.update({
            'structure': struct_class.from_jsonable(state['structure']),
            'runs': runs_native,
        })
        return cls(state=state)


class LammpsSimulation(AtomisticSimulation):
    """Class to represent a LAMMPS simulation."""

    def __init__(self, options=None, state=None):
        """Initialise a LammpsSimulation."""
        super().__init__(options=options, state=state)
        self._process_options()

    @classmethod
    def copy_reference_data(cls, sim_params, stage_path, scratch_path):
        """Copy potential files."""

        for key, pot_fn in sim_params['potential_files'].items():

            # Copy potential files to the sim group directory:
            pot_path = os.path.join(REF_PATH, 'potentials', pot_fn)
            path_stage = stage_path.joinpath(pot_fn)
            path_scratch = scratch_path.joinpath(pot_fn)

            try:
                shutil.copy(pot_path, path_stage)
            except:
                msg = ('Check potential file: "{}" exists.')
                raise ValueError(msg.format(pot_path))

            # Map correct file names for the potentials:
            for int_idx, inter in enumerate(sim_params['interactions']):

                if key in inter:
                    path_scratch = '"' + str(path_scratch) + '"'
                    new_inter = inter.replace(key, path_scratch)
                    sim_params['interactions'][int_idx] = new_inter

        # Remove potential files list from options:
        del sim_params['potential_files']

    def _process_options(self):
        """Additional processing on LAMMPS options to prepare for writing input
        files.

        """
        super()._process_options()

        lammps_opts = self.options['params']['lammps']

        # Set charges list
        charges_dict = lammps_opts.get('charges')
        if charges_dict is not None:

            charges = []
            for species in self.structure.species:
                try:
                    charges.append(charges_dict[species])
                except:
                    msg = ('Cannot find charge specification for species: {}')
                    raise ValueError(msg.format(species))

            lammps_opts['charges'] = charges

        prt(lammps_opts, 'lammps_opts (after all processing)')

    def write_input_files(self, path):

        lmp_in_params = {
            **self.options['params']['lammps'],
            'supercell': self.structure.supercell,
            'atom_sites': self.structure.atom_sites,
            'species': self.structure.species,
            'species_idx': self.structure.species_idx,
            'path': path,
            'atom_constraints': self.options['structure']['constraints']['atoms'],
            'cell_constraints': self.options['structure']['constraints']['cell'],
        }
        lammps.write_lammps_inputs(**lmp_in_params)

    def to_jsonable(self):
        """Generate a dict representation that can be JSON serialised."""

        # `structure` and `options` are dealt with in the super-class:
        ret = super().to_jsonable()

        # Add `result` key from each run in `runs`:
        for i, j in zip(ret['runs'], self.runs):
            if j['result'] is not None:
                i['result'] = {
                    # TODO, check which LAMMPS results vals need `tolist()`ing.
                    **j['result'],
                }

        return ret

    @classmethod
    def from_jsonable(cls, state):
        """Instantiate from a JSONable dict."""

        runs_native = state['runs']
        for idx, _ in enumerate(runs_native):
            if runs_native[idx]['result'] is not None:
                runs_native[idx]['result'] = {
                    # TODO, check which LAMMPS results vals need `array`ing.
                    **runs_native[idx]['result'],
                }

        state.update({
            'structure': AtomisticStructure.from_jsonable(state['structure']),
            'runs': runs_native,
        })
        return cls(state=state)
