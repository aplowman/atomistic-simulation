import os
import numpy as np
from plotly import graph_objs
from plotly.offline import plot, iplot, init_notebook_mode
from atsim import plotting, geometry, vectors, mathsutils, readwrite, utils
from atsim.structure.atomistic import AtomisticStructure
from atsim.structure.bicrystal import Bicrystal
from atsim.structure.crystal import CrystalBox, CrystalStructure
from atsim.simsio import castep, lammps

from atsim.utils import prt


class AtomisticSimulation(object):

    def __init__(self, atomistic_structure, options):

        self.structure = atomistic_structure
        self.options = options
        self.results = None

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

        species_orig = self.structure.all_species
        species_idx_orig = self.structure.all_species_idx
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
            'supercell': supercell,
            'all_species': species_orig,
            'all_species_idx': species_idx_orig,
            'crystals': self.structure.crystals,
            'crystal_idx': self.structure.crystal_idx,
            'crystal_structures': self.structure.crystal_structures,
        }

        if 'bicrystal' in self.structure.meta:

            # Regenerate a bicrystal
            bc_params = {
                'as_params': as_params,
                'maintain_inv_sym': self.structure.maintain_inv_sym,
                'nbi': self.structure.nbi,
                'rot_mat': self.structure.rot_mat
            }
            opt_structure = Bicrystal(**bc_params)

        else:
            opt_structure = AtomisticStructure(**as_params)

        if tile is not None:
            opt_structure.tile_supercell(tile)

        return opt_structure

    def write_input_files(self):

        common_params = {
            'supercell': self.structure.supercell,
            'atom_sites': self.structure.atom_sites,
            'species': self.structure.all_species,
            'species_idx': self.structure.all_species_idx,
            'path': self.options['stage_series_path'],
            'atom_constraints': self.options['constraints']['atom'],
            'cell_constraints': self.options['constraints']['cell'],
        }

        if self.options['method'] == 'castep':

            cst_opt = self.options['castep']
            cst_in_params = {
                'seedname': cst_opt['seedname'],
                'cell': cst_opt['cell'],
                'param': cst_opt['param'],
                'sym_ops': cst_opt['sym_ops'],
                **common_params
            }

            castep.write_castep_inputs(**cst_in_params)

        elif self.options['method'] == 'lammps':

            lmp_opt = self.options['lammps']
            lmp_in_params = {
                **lmp_opt,
                **common_params
            }
            lammps.write_lammps_inputs(**lmp_in_params)
