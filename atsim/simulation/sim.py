import os
import numpy as np
from plotly import graph_objs
from plotly.offline import plot, iplot, init_notebook_mode
from atsim import plotting, simsio, geometry, vectors, mathsutils, readwrite, utils
from atsim.structure.crystal import CrystalBox, CrystalStructure


class AtomisticSimulation(object):

    def __init__(self, atomistic_structure, options):

        self.structure = atomistic_structure
        self.options = options
        self.results = None

    def write_input_files(self):

        set_opt = self.options['set_up']
        common_params = {
            'supercell': self.structure.supercell,
            'atom_sites': self.structure.atom_sites,
            'species': self.structure.all_species,
            'species_idx': self.structure.all_species_idx,
            'path': set_opt['stage_series_path'],
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

            simsio.castep.write_castep_inputs(**cst_in_params)

        elif self.options['method'] == 'lammps':

            lmp_opt = self.options['lammps']
            lmp_in_params = {
                **lmp_opt,
                **common_params
            }
            simsio.lammps.write_lammps_inputs(**lmp_in_params)
