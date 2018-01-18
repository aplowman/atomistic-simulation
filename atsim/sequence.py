"""Module containing a class to represent a sequence of simulations."""

import warnings
import copy
import numpy as np
import yaml
from atsim.utils import (set_nested_dict, get_recursive,
                         update_dict, prt, mut_exc_args)
from atsim import readwrite, BaseUpdate


class SimSequence(object):
    """Options which parameterise a sequence of simulations."""

    seq_defn = None

    def __init__(self, spec):

        params, spec = self._validate_spec(spec)

        # Store this for easily saving/loading as JSON:
        self.spec = copy.deepcopy(spec)

        self.base_dict = params['base_dict']
        self.func = params['func']
        self.range_allowed = params['range_allowed']
        self.update_mode = params['update_mode']
        self.val_seq_type = params['val_seq_type']
        self.map_to_dict = params['map_to_dict']
        self.val_name = params['val_name']

        self.name = spec.pop('name')
        self.nest_idx = spec.pop('nest_idx')
        self.val_fmt = spec.pop('val_fmt')
        self.path_fmt = spec.pop('path_fmt')

        # Remove vals from spec to parse and to get remaining additional spec:
        vals_spec_keys = ['vals', 'start', 'step', 'stop']
        vals_spec = {i: spec.pop(i, None) for i in vals_spec_keys}

        self.vals = self._parse_vals(**vals_spec)
        self.additional_spec = spec

        self.updates = self._get_updates()

    def to_jsonable(self):
        """Generate a dict representation that can be JSON serialised."""
        return {'spec': self.spec}

    @classmethod
    def from_jsonable(cls, state):
        """Generate new instance from JSONable dict"""
        return cls(state['spec'])

    @classmethod
    def load_sequence_definitions(cls, path):
        """Load sequence definitions from a YAML file."""
        with open(path, 'r') as seq_file:
            seq_defn = yaml.safe_load(seq_file)

        cls.seq_defn = seq_defn

    def _validate_spec(self, spec):
        """
        TODO: if `map_to_dict`, check `val_name` is not None.

        """
        msg = 'SimSequence failed validation: '

        # Keys allowed in the sequence spec (from makesims options):
        req_specs = [
            'name',
            'nest_idx',
        ]
        ok_specs = req_specs + [
            'val_fmt',
            'path_fmt',
            'vals',
            'start',
            'step',
            'stop',
        ]

        # Keys allowed in the sequence definition (from sequences.yml):
        req_params = [
            'base_dict',
            'func',
            'range_allowed',
            'val_seq_type',
            'update_mode',
            'defaults',
            'additional_spec',
            'val_name',
            'map_to_dict',
        ]
        ok_params = req_params

        for i in req_specs:
            if i not in spec:
                req_spec_msg = msg + 'Sequence spec '
                if spec.get('name'):
                    req_spec_msg += '"{}" '.format(spec.get('name'))
                req_spec_msg += 'must have `{}` key.'.format(i)
                raise ValueError(req_spec_msg)

        params = SimSequence.seq_defn[spec['name']]

        for i in req_params:
            if i not in params:
                raise ValueError(msg + 'Sequence definition must have '
                                 '`{}` key.'.format(i))

        for param_key in params:
            if param_key not in ok_params:
                raise ValueError(msg + 'Sequence definition key "{}" is not '
                                 'allowed in sequence "{}".'.format(param_key, spec['name']))

        spec = {**params['defaults'], **spec}

        for spec_key in spec.keys():
            if spec_key not in ok_specs and spec_key not in params['additional_spec']:
                ok_spec_msg = (msg + 'Sequence spec key "{}" is not allowed in'
                               ' sequence "{}".'.format(spec_key, spec['name']))
                raise ValueError(ok_spec_msg)

        if spec['name'] not in SimSequence.seq_defn:
            raise ValueError(
                msg + 'Sequence name "{}" not known. Sequence name should be '
                'one of: {}'.format(spec['name'], list(
                    SimSequence.seq_defn.keys()))
            )

        if not params['range_allowed']:
            if any([i in spec for i in ['start', 'step', 'stop']]):
                raise ValueError(msg + 'Range is not allowed for '
                                 'sequence name: {}'.format(spec['name']))
        return params, spec

    def _parse_vals(self, vals=None, start=None, step=None, stop=None):
        """Parse sequence spec vals and """

        mut_exc_args({'vals': vals},
                     {'start': start, 'step': step, 'stop': stop})

        if vals is None:

            if not self.range_allowed:
                raise ValueError('Specifying a range for sequence "{}" is not '
                                 'allowed.'.format(self.name))

            step_wrn_msg = ('Spacing between series values will not be exactly'
                            ' as specified.')

            if not np.isclose((start - stop) % step, 0):
                warnings.warn(step_wrn_msg)

            diff = start - stop if start > stop else stop - start
            num = int(np.round((diff + step) / step))
            vals = np.linspace(start, stop, num=num)

        # Parse vals Numpy array or tuple if necessary (lists are represented
        # natively by YAML):
        if self.val_seq_type:

            vals_prsd = []
            for val in vals:

                if self.val_seq_type == 'array':
                    vals_prsd.append(np.array(val))

                elif self.val_seq_type == 'tuple':
                    vals_prsd.append(tuple(val))

            vals = vals_prsd

        return vals

    @property
    def num_vals(self):
        """Get the number of values (simulations) in the sequence."""
        return len(self.vals)

    def _get_updates(self):
        """
        Build a list of update dicts to be applied to the base options for each
        element in the group.

        """

        # Run any additional processing on the `vals`
        if self.func:
            self.func(self)

        fmt_arr_opt = {
            'col_delim': '_',
            'row_delim': '__',
            'format_spec': self.path_fmt,
        }

        name_add = ['sequence_id', 'names']
        paths_add = ['sequence_id', 'paths']
        vals_add = ['sequence_id', 'vals']
        nest_add = ['sequence_id', 'nest_idx']

        prt(self.vals, 'self.vals')
        prt(self.additional_spec, 'self.additional_spec')

        updates = []
        for val in self.vals:

            if self.val_seq_type == 'array':
                path_str = readwrite.format_arr(val, **fmt_arr_opt)[:-2]

            else:
                path_str = self.path_fmt.format(val)

            # If `map_to_dict` replace the val which updates the options with a
            # dict mapping `val_name`: val and all other `additional_spec`:
            upd_val = val
            if self.map_to_dict:
                # TODO move this check to _validate_spec
                if not self.val_name:
                    msg = ('`val_name` must be set if `map_to_dict` True.')
                    raise ValueError(msg)

                upd_val = {
                    self.val_name: val,
                    **self.additional_spec,
                }

            # Update that affects the generation of the Simulation:
            elem_upd = BaseUpdate(self.base_dict, upd_val,
                                  self.val_seq_type, self.update_mode)

            # Updates that parameterise this effect:
            seqid_upd = [
                BaseUpdate(['sequence_id'], {}, None, 'replace'),
                BaseUpdate(name_add, self.name, None, 'append'),
                BaseUpdate(paths_add, path_str, None, 'append'),
                BaseUpdate(vals_add, val, self.val_seq_type, 'append'),
                BaseUpdate(nest_add, self.nest_idx, None, 'append'),
            ]

            updates.append([elem_upd] + seqid_upd)

        return updates
