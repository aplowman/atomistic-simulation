"""matsim.atomistic.simulation.lammps.py"""

import os
import shutil
import copy

import numpy as np

from atsim import REF_PATH
from atsim.utils import prt
from atsim.atomistic.software import lammps as lammpsio
from atsim.atomistic.simulation import SUPERCELL_TYPE_LOOKUP
from atsim.atomistic.simulation.sim import AtomisticSimulation


class LammpsSimulation(AtomisticSimulation):
    """Class to represent a LAMMPS simulation."""

    def __init__(self, options=None, state=None):
        """Initialise a LammpsSimulation."""
        super().__init__(options=options, state=state)
        if options:
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
        lammpsio.write_lammps_inputs(**lmp_in_params)

    def to_jsonable(self):
        """Generate a dict representation that can be JSON serialised."""

        # `structure` and `options` are dealt with in the super-class:
        ret = super().to_jsonable()

        runs_js = copy.deepcopy(self.runs)

        dumps_arr_keys = ['box', 'supercell', 'atom_sites', 'atom_types',
                          'atom_pot_energy', 'atom_disp', 'vor_vols', 'vor_faces']

        for run_idx, run in enumerate(self.runs):

            run_res = run['result']
            run_res_js = runs_js[run_idx]['result']

            if not run_res:
                continue

            dumps = run_res['dumps']
            dumps_js = {}

            for dump_step, dump in dumps.items():

                dump_js = {}
                for key, val in dump.items():

                    if key in dumps_arr_keys:
                        dump_js.update({key: val.tolist()})
                    else:
                        dump_js.update({key: val})

                dumps_js.update({dump_step: dump_js})

            thermo = run_res['thermo']
            thermo_js = {}
            for key, val in thermo.items():
                thermo_js.update({
                    key: val.tolist()
                })

            # list-ify parsed LAMMPS output
            runs_js[run_idx]['result'].update({
                'time_steps': run_res_js['time_steps'].tolist(),
                'atoms': run_res_js['atoms'].tolist(),
                'atom_disp': run_res_js['atom_disp'].tolist(),
                'atom_pot_energy': run_res_js['atom_pot_energy'].tolist(),
                'vor_vols': run_res_js['vor_vols'].tolist(),
                'vor_faces': run_res_js['vor_faces'].tolist(),
                'supercell': run_res_js['supercell'].tolist(),
                'box': run_res_js['box'].tolist(),
                'thermo': thermo_js,
                'final_energy': run_res_js['final_energy'].tolist(),
                'dumps': dumps_js,
            })

        ret['runs'] = runs_js

        # print('saving state of LammpsSimulation...')
        # prt(ret, 'ret')

        # prt(self.runs[0]['result'], 'self.runs[0][result]')
        # exit()

        return ret

    def check_success(self, path):
        """Check a given run of this simulation has succeeded."""

        # Can add logic here to decide which check_success function to invoke.
        return lammpsio.check_success(path)

    def parse_result(self, path, run_idx):
        """Parse results from path and add to runs[run_idx]['result']"""

        run_res = self.runs[run_idx]['result']
        if run_res:
            msg = 'Result has already been parsed for run_idx {}'
            raise ValueError(msg.format(run_idx))

        out = lammpsio.read_lammps_output(path)
        self.runs[run_idx]['result'] = out

    @classmethod
    def _run_from_jsonable(cls, runs):
        """Parse a run from a JSONable dict."""

        runs_nt = copy.deepcopy(runs)

        dumps_arr_keys = ['box', 'supercell', 'atom_sites', 'atom_types',
                          'atom_pot_energy', 'atom_disp', 'vor_vols', 'vor_faces']

        for run_idx, run in enumerate(runs):

            run_res = run['result']

            if not run_res:
                continue

            run_res_nt = runs_nt[run_idx]['result']

            dumps = run_res['dumps']
            dumps_nt = {}

            for dump_step, dump in dumps.items():

                dump_nt = {}
                for key, val in dump.items():

                    if key in dumps_arr_keys:
                        dump_nt.update({key: np.array(val)})
                    else:
                        dump_nt.update({key: val})

                dumps_nt.update({dump_step: dump_nt})

            thermo = run_res['thermo']
            thermo_nt = {}
            for key, val in thermo.items():
                thermo_nt.update({
                    key: np.array(val)
                })

            # list-ify parsed LAMMPS output
            runs_nt[run_idx]['result'].update({
                'time_steps': np.array(run_res_nt['time_steps']),
                'atoms': np.array(run_res_nt['atoms']),
                'atom_disp': np.array(run_res_nt['atom_disp']),
                'atom_pot_energy': np.array(run_res_nt['atom_pot_energy']),
                'vor_vols': np.array(run_res_nt['vor_vols']),
                'vor_faces': np.array(run_res_nt['vor_faces']),
                'supercell': np.array(run_res_nt['supercell']),
                'box': np.array(run_res_nt['box']),
                'thermo': thermo_nt,
                'final_energy': np.array(run_res_nt['final_energy']),
                'dumps': dumps_nt,
            })

        return runs_nt

    @classmethod
    def from_jsonable(cls, state):
        """Instantiate from a JSONable dict."""

        runs_nt = LammpsSimulation._run_from_jsonable(state['runs'])

        sup_types_str = state['structure']['meta'].get('supercell_type')
        sup_type_class = 'default'
        for sup_type in sup_types_str:
            if sup_type in SUPERCELL_TYPE_LOOKUP:
                sup_type_class = sup_type
                break
        struct_class = SUPERCELL_TYPE_LOOKUP[sup_type_class]

        state.update({
            'structure': struct_class.from_jsonable(state['structure']),
            'runs': runs_nt,
        })
        return cls(state=state)
