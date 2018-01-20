"""matsim.atomistic.simulation.lammps.py"""

import os
import shutil

from atsim import REF_PATH
from atsim.utils import prt
from atsim.atomistic.software.lammps import write_lammps_inputs
from atsim.atomistic.simulation import SUPERCELL_TYPE_LOOKUP
from atsim.atomistic.simulation.sim import AtomisticSimulation


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
        write_lammps_inputs(**lmp_in_params)

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
