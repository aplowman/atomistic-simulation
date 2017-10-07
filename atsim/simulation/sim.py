import os
import numpy as np
from plotly import graph_objs
from plotly.offline import plot, iplot, init_notebook_mode
from atsim import plotting, geometry, vectors, mathsutils, readwrite, utils
from atsim.structure.atomistic import AtomisticStructure
from atsim.structure.crystal import CrystalBox, CrystalStructure
from atsim.simsio import castep, lammps


class AtomisticSimulation(object):

    def __init__(self, atomistic_structure, options):

        self.structure = atomistic_structure
        self.options = options
        self.results = None

    def generate_structure(self, opt_idx=-1):
        """
        Generates an AtomisticStructure object for a given optimisation step 
        after the simulation has run.

        Parameters
        ----------
        opt_idx : int, optional
            The optimisation step from which to generate the structure.

        """

        # TODO: consider reordering of atoms (in e.g. CASTEP) when generating
        # new structure. I.e. generating a new AS object from opt_step 0 should
        # write identical input files to the originals.

        if self.results is None:
            raise ValueError('Cannot generate an optimised AtomisticStructure '
                             'object for a simulation whose results have '
                             'not been harvested.')

        # Get supercell and atom sites from given optimisation step
        method = self.options['method']

        if method == 'castep':
            supercell = self.results['geom']['cells'][opt_idx]
            atom_sites = self.results['geom']['ions'][opt_idx]
            all_species = self.results['geom']['species']
            all_species_idx = self.results['geom']['species_idx']

        elif method == 'lammps':
            supercell = self.results['supercell'][opt_idx]
            atom_sites = self.results['atoms'][opt_idx]

            # Assume no reordering of species
            all_species = self.structure.all_species
            all_species_idx = self.structure.all_species_idx

        opt_structure = AtomisticStructure(
            atom_sites, supercell, all_species=all_species,
            all_species_idx=all_species_idx)

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
