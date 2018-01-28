"""matsim.atomistic.simulation.sim.py"""
import copy

import numpy as np

from atsim.utils import mut_exc_args, prt
from atsim.simulation.sim import Simulation
from atsim.atomistic.simulation import get_structure
from atsim.atomistic.structure.atomistic import AtomisticStructure
from atsim.atomistic.structure.bicrystal import Bicrystal


class AtomisticSimulation(Simulation):
    """Class to represent an atomistic simulation."""

    def __init__(self, options=None, state=None):

        mut_exc_args(
            {'options': options},
            {'state': state})

        if state:
            self.options = None
            self.structure = state['structure']
            self.runs = state['runs']

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
            opt_structure = Bicrystal(**bc_params)

        else:
            opt_structure = AtomisticStructure(**as_params)

        if tile is not None:
            opt_structure.tile_supercell(tile)

        return opt_structure
