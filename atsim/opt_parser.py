"""Module for parsing options."""

import copy
import numpy as np
import yaml
from atsim.utils import unflatten_dict_keys, dict_from_list, prt, parse_float


def validate_opt_spec(opt_specs):
    """Validate an options specification dict.

    TODO:
        - check allowed values of each of the required/allowed keys (e.g. subdict_type
            is either "list" or "dict", seq_type (if present) if array/float/...)
        - check default keys are included in ok_keys
        - check "seq_type" is only present for "is_subdict" False and
            "val_type" includes [list, array, tuple] (i.e. only for sequences)

    """

    def check_ok_keys(to_check, ok_keys):
        """Check a dict has particular keys of particular types.

        Params
        ------
        to_check : dict
        ok_keys : list of tuple of (keyname, type)

        """
        for key in ok_keys:

            if key[0] not in to_check:
                msg = ('does not have key "{}" of type {}.')
                raise ValueError(msg.format(*key))

            check_key = to_check[key[0]]
            if not isinstance(check_key, key[1]):
                msg = ('key "{}" must be of type: {}, but is of type: {}.')
                raise ValueError(msg.format(
                    key[0], key[1], type(check_key)))

    gen_msg = 'Validation error in option specification #{} (address: {}): '
    all_address = []
    for spec_idx, spec in enumerate(opt_specs):

        gen_msg_fmt = (spec_idx, spec.get('address'))

        # Check required keys in each specification, and their types:
        exact_keys = [
            ('address', list),
            ('is_subdict', bool),
        ]
        if spec['is_subdict']:
            exact_keys += [
                ('subdict_type', str),
                ('ok_keys', list),
                ('req_keys', list),
                ('defaults', dict),
                ('conditional_keys', list),
            ]
        else:
            exact_keys += [('val_type', list)]

        try:
            check_ok_keys(spec, exact_keys)

        except ValueError as val_err:
            msg = gen_msg + '{}'
            raise ValueError(msg.format(*gen_msg_fmt, str(val_err)))

        # Don't allowed specifications with the same address:
        add = spec['address']
        if add in all_address:
            msg = gen_msg + ('specification with address: {} is defined '
                             'multiple times.')
            raise ValueError(msg.format(*gen_msg_fmt, add))
        else:
            all_address.append(add)

        # Don't allow "val_type" containing multiple sequence types:
        num_seqs = None
        seq_types = None
        if not spec['is_subdict']:
            seq_types = ('list', 'tuple', 'array')
            seq_in_val_type = [i in spec['val_type'] for i in seq_types]

            num_seqs = sum(seq_in_val_type)
            if num_seqs > 1:
                msg = gen_msg + ('`val_type` can only contain one sequence '
                                 'type of {}')
                raise ValueError(msg.format(*gen_msg_fmt, seq_types))

        # Check "seq_type" is only present if "is_subdict" False and "val_type"
        # contains a sequence type (list, tuple, array):
        if spec.get('seq_type') is not None:

            if spec['is_subdict']:
                msg = gen_msg + ('`seq_type` key is only allowed when '
                                 '`is_subdict` is `False`.')
                raise ValueError(msg.format(*gen_msg_fmt))

            if num_seqs == 0:
                msg = gen_msg + ('`seq_type` key is only allowed when '
                                 '`val_type` contains a sequence type: {}.')
                raise ValueError(msg.format(*gen_msg_fmt, seq_types))


def parse_opt(opts, opts_spec, address=None, key_spec=None, nest_syntax='.'):
    """Parse options according to an options specification.

    TODO:
    - improve error messages (include "address" of problem)
    - more flexible conditional requirements (e.g. condtions based on other
      addresses).
    - don't allow repeated keys (some effort, would need to define own loader:
      see: https://stackoverflow.com/a/34363871/5042280)

    """

    val_type_lookup = {
        'list': list,
        'dict': dict,
        'str': str,
        'bool': bool,
        'int': int,
        'float': float,
        'tuple': tuple,
    }

    if address is None:
        address = []

    if nest_syntax is not None:
        opts = unflatten_dict_keys(opts, delim=nest_syntax)

    key_spec = key_spec or dict_from_list(opts_spec, {'address': address})
    key_spec = copy.deepcopy(key_spec)

    # Loop though conditional keys and if condition is met, add to required keys,
    # defaults and OK keys accordingly:
    for cond_k in key_spec['conditional_keys']:

        if ((cond_k['key'] in key_spec['ok_keys'])
                and (opts[cond_k['key']] == cond_k['val'])):
            key_spec['req_keys'] += cond_k['req_keys']
            key_spec['ok_keys'] += cond_k['ok_keys']
            key_spec['defaults'].update({
                **cond_k['defaults']
            })

    for req_k in key_spec['req_keys']:
        if req_k not in opts:
            raise ValueError('Key "{}" is missing but required.'.format(req_k))

    for def_k, def_v in key_spec['defaults'].items():
        if def_k not in opts:
            opts.update({def_k: def_v})

    opts_parsed = {}
    for key, val in opts.items():

        if key not in key_spec['ok_keys']:
            raise ValueError('Key "{}" not allowed.'.format(key))

        new_address = address + [key]
        sub_key_spec = dict_from_list(opts_spec, {'address': new_address})

        if sub_key_spec is None:
            raise ValueError('No option specification for key: {}'.format(key))

        # If `is_subdict`, does val match `subdict_type`?
        if sub_key_spec['is_subdict']:

            msg = ('Value at address {} must be a {}, but is of type: {}')

            if sub_key_spec['subdict_type'] == 'dict':
                if not isinstance(val, dict):
                    raise ValueError(msg.format(address, 'dict', type(val)))

                opts_parsed.update({
                    key: parse_opt(
                        val, opts_spec, new_address, sub_key_spec, nest_syntax)
                })

            elif sub_key_spec['subdict_type'] == 'list':
                if not isinstance(val, list):
                    raise ValueError(msg.format(address, 'list', type(val)))

                opts_parsed.update({key: []})
                for list_elem in val:
                    opts_parsed[key].append(
                        parse_opt(list_elem, opts_spec, new_address,
                                  sub_key_spec, nest_syntax)
                    )

        else:

            # Check val type matches a type in key spec val_type:

            ks_val_type = copy.copy(sub_key_spec['val_type'])
            array_ok = False
            tup_ok = False

            if 'array' in ks_val_type:
                array_ok = True
                ks_val_type[ks_val_type.index('array')] = 'list'

            elif 'tuple' in ks_val_type:
                tup_ok = True
                ks_val_type[ks_val_type.index('tuple')] = 'list'

            ok_val_types = tuple([val_type_lookup[i] for i in ks_val_type])
            if not isinstance(val, ok_val_types):
                msg = ('Key "{}" must be one of these types: {}, but is'
                       ' of type: {}')
                raise ValueError(msg.format(
                    key, sub_key_spec['val_type'], type(val)))

            seq_type = sub_key_spec.get('seq_type')

            if seq_type == 'array':
                val = [np.array(i) for i in val]

            # Parse as array/tuple if necessary:
            if array_ok and isinstance(val, list):

                val = np.array(val)
                # Parse array elements as floats if necessary:
                if seq_type == 'float':
                    val_new = np.ones_like(val, dtype=float) * np.nan
                    for i_idx, i in np.ndenumerate(val):
                        val_new[i_idx] = parse_float(i)
                    val = val_new

            elif tup_ok and isinstance(val, list):
                val = tuple(val)

            opts_parsed.update({key: val})

    return opts_parsed


def parse_opt_file(opts_path, opts_spec_path, opt_name, nest_syntax='.'):
    """Parse an options file according to a options specification file."""

    with open(opts_spec_path, 'r') as opts_spec_fs:
        opts_spec = yaml.load(opts_spec_fs)[opt_name]

    validate_opt_spec(opts_spec)

    with open(opts_path, 'r') as opts_path_fs:
        opts = yaml.load(opts_path_fs)

    opts_parsed = parse_opt(opts, opts_spec, address=None,
                            key_spec=None, nest_syntax=nest_syntax)

    return opts_parsed, opts_spec
