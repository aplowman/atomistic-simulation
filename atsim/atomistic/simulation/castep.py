"""matsim.atomistic.simulation.castep.py"""

import numpy as np

from atsim.atomistic.software.castep import write_castep_inputs
from atsim.atomistic.simulation import SUPERCELL_TYPE_LOOKUP
from atsim.atomistic.simulation.sim import AtomisticSimulation


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

        write_castep_inputs(**cst_in_params)

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
