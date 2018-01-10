"""Module containing a class to represent a sequence of simulations."""

import warnings
import numpy as np
import yaml
from atsim.utils import set_nested_dict, get_recursive, update_dict
from atsim import readwrite


class BaseUpdate(object):
    """
    TODO: consider replacing this class with a dict.

    """

    def __init__(self, address, val, mode):
        """
        Parameters
        ----------
        """
        self.address = address
        self.val = val
        self.mode = mode

    def __str__(self):
        return 'BaseUpdate(val={!r})'.format(self.val)

    def __repr__(self):
        return 'BaseUpdate(val={!r})'.format(self.val)

    def apply_to(self, base_dict):
        """Apply the update to a dict."""

        if self.mode == 'replace':
            upd_dict = set_nested_dict(self.address, self.val)

        elif self.mode == 'append':
            upd_val = get_recursive(base_dict, self.address, [])
            upd_val.append(self.val)
            upd_dict = set_nested_dict(self.address, upd_val)

        base_dict = update_dict(base_dict, upd_dict)

        return base_dict


class SimSequence(object):
    """Options which parameterise a sequence of simulations."""

    seq_defn = None

    def __init__(self, spec):

        params, spec = self._validate_spec(spec)

        self.base_dict = params['base_dict']
        self.func = params['func']
        self.range_allowed = params['range_allowed']
        self.update_mode = params['update_mode']

        self.name = spec.pop('name')
        self.nest_idx = spec.pop('nest_idx')
        self.val_fmt = spec.pop('val_fmt')
        self.path_fmt = spec.pop('path_fmt')

        self.vals = self._parse_vals(spec)
        self.additional_spec = spec

        self._generate_updates()

    @classmethod
    def load_sequence_definitions(cls, path):
        """Load sequence definitions from a YAML file."""
        with open(path, 'r') as seq_file:
            seq_defn = yaml.safe_load(seq_file)

        cls.seq_defn = seq_defn

    def _validate_spec(self, spec):

        msg = 'SimSequence failed validation: '

        req_specs = [
            'name',
            'nest_idx',
        ]
        allowed_specs = req_specs + [
            'val_fmt',
            'path_fmt',
            'vals',
            'start',
            'step',
            'stop',
        ]

        for i in req_specs:
            if i not in spec:
                raise ValueError(msg + ' Sequence must have '
                                 '`{}` key.'.format(i))

        params = SimSequence.seq_defn[spec['name']]
        spec = {**params['defaults'], **spec}

        for spec_name in spec.keys():
            if spec_name not in allowed_specs:
                raise ValueError(
                    msg + ' Key "{}" is not allowed.'.format(spec_name)
                )

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

    def _parse_vals(self, kwargs):

        if kwargs.get('vals') is not None:
            vals = kwargs.pop('vals')

        else:
            start = kwargs.pop('start')
            step = kwargs.pop('step')
            stop = kwargs.pop('stop')

            step_wrn_msg = ('Spacing between series values '
                            'will not be exactly as specified.')

            if not np.isclose((start - stop) % step, 0):
                warnings.warn(step_wrn_msg)

            diff = start - stop if start > stop else stop - start
            num = int(np.round((diff + step) / step))
            vals = np.linspace(start, stop, num=num)

        return vals

    @property
    def num_vals(self):
        """Get the number of values (simulations) in the sequence."""
        return len(self.vals)

    def _generate_updates(self):
        """
        Build a list of update dicts to be applied to the base options for each
        element in the group.

        """

        # Run any additional processing on the `vals`
        if self.func:
            self.func(self)

        updates = []
        for val in self.vals:

            if isinstance(val, np.ndarray):
                fmt_arr_opt = {
                    'col_delim': '_',
                    'row_delim': '__',
                    'format_spec': self.path_fmt,
                }
                path_str = readwrite.format_arr(val, **fmt_arr_opt)[:-2]

            else:
                path_str = self.path_fmt.format(val)

            seqid_val = {
                'name': self.name,
                'val': val,
                'path': path_str,
                'nest_idx': self.nest_idx,
            }

            elem_upd = {
                'address': self.base_dict,
                'val': val,
                'mode': self.update_mode,
            }
            seqid_upd = {
                'address': ['sequence_id'],
                'val': seqid_val,
                'mode': 'append',
            }

            updates.append([BaseUpdate(**elem_upd), BaseUpdate(**seqid_upd)])

        self.updates = updates
