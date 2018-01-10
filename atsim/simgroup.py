"""Module containing a class to represent a group of simulations defined by one
or more simulation sequences.

"""
import copy
import numpy as np
from atsim.sequence import SimSequence
from atsim.structure.crystal import CrystalStructure
from atsim.structure.bravais import BravaisLattice
from atsim.utils import get_date_time_stamp, nest, merge


class SimGroup(object):
    """Class to represent a group of related simulations."""

    # Default path formatting options:
    path_options_def = {
        'parallel_sims_join': '_+_',
        'sim_numbers': True,
        'sim_numbers_join': '__',
        'sequence_names': False,
        'subdirs': [],
        'run_fmt': 'run_{}',
    }

    def __init__(self, base_options, sequences, sim_updates=None):

        SimSequence.load_sequence_definitions('sequences.yml')

        self.sequences = [SimSequence(i) for i in sequences]
        self.sim_updates = sim_updates
        if sim_updates is None:
            self._set_sim_updates()

        self.stage_id = base_options.pop('stage_id')
        self.scratch_id = base_options.pop('scratch_id')
        self.archive_id = base_options.pop('archive_id', None)
        path_options = base_options.pop('path_options', {})
        self.path_options = {**SimGroup.path_options_def, **path_options}
        self.base_options = base_options

#         self._generate_crystal_structures()

        # Generate a human-useful id:
        hid_date, hid_num = get_date_time_stamp(split=True)
        hid = hid_date + '_' + hid_num
        self.hid = hid

        self.sims = None  # maybe turn this into a property or function?

    def _get_merged_updates(self):
        """Merge updates from 'parallel' sequences (those with the same `nest_idx`)"""

        # Get the updates for each sequence:
        seq_upds = []
        for i in self.sequences:
            i._generate_updates()
            seq_upds.append(i.updates)

        # Merge parallel sequences (those with same `nest_idx`):
        seq_upds_mergd = {}
        for idx, seq_i in enumerate(self.sequences):
            nest_idx = seq_i.nest_idx

            if nest_idx in seq_upds_mergd:
                mergd = merge(seq_upds_mergd[nest_idx], seq_upds[idx])
                seq_upds_mergd[nest_idx] = mergd

            else:
                seq_upds_mergd.update({
                    nest_idx: seq_upds[idx]
                })

        # Sort by `nest_idx`
        merged_updates = [val for _, val in sorted(seq_upds_mergd.items())]
        return merged_updates

    def _set_sim_updates(self):

        grp_upd = nest(*self._get_merged_updates())

        grp_upd_flat = []
        for upd_lst in grp_upd:
            upd_lst_flat = [j for i in upd_lst for j in i]
            grp_upd_flat.append(upd_lst_flat)

        self.sim_updates = grp_upd_flat

    @property
    def num_sims(self):
        return len(self.sim_updates)

    def _generate_crystal_structures(self):

        crys_structs = self.base_options['structure']['crystal_structures']

        for cs_idx in range(len(crys_structs)):

            cs_defn = crys_structs[cs_idx]
            cs_params = {}

            if 'path' in cs_defn:

                # Generate CrystalStructure from file
                cs_params.update({
                    'path': cs_defn['path'],
                    **cs_defn['lattice'],
                })
                if cs_defn.get('motif') is not None:
                    cs_params.update({
                        'motif': cs_defn['motif'],
                    })
                crys_structs[cs_idx] = CrystalStructure.from_file(**cs_params)

            else:

                # Generate CrystalStructure from parameters
                cs_params.update({
                    'bravais_lattice': BravaisLattice(**cs_defn['lattice']),
                    'motif': cs_defn['motif'],
                })
                crys_structs[cs_idx] = CrystalStructure(**cs_params)

    def get_sim_options(self, sim_idx):

        sim_opt = copy.deepcopy(self.base_options)
        for upd in self.sim_updates[sim_idx]:
            sim_opt = upd.apply_to(sim_opt)

        return sim_opt

    @property
    def sequence_lengths(self):

        all_nest = [i.nest_idx for i in self.sequences]
        _, uniq_idx = np.unique(all_nest, return_index=True)
        ret = tuple([self.sequences[i].num_vals
                     for i in uniq_idx])
        return ret

    def get_path_labels(self, sim_idx):

        num_elems = self.sequence_lengths
        ind = [sim_idx]
        for j in range(len(num_elems) - 1):
            prod_range = num_elems[::-1][:j + 1]
            prod = np.product(prod_range)
            ind.append(prod * int(ind[-1] / prod))

        return ind[::-1]

    def to_jsonable(self):
        pass

    @classmethod
    def from_json(cls, path):
        pass

    def write_run_inputs(self):
        """
        Write input files for (a subset of) this simulation group.
        Files may be copied/modified from pre-existing runs (on scratch) of this simulation group.

        """

    def generate_structure(self, sim_idx, opt_idx):
        pass

    def get_run_path(self, sim_idx, run_idx):

        sim_path = self.get_sim_path(sim_idx)
        run_path = sim_path + [self.path_options['run_fmt'].format(run_idx)]
        return run_path

    def get_sim_path(self, sim_idx):

        sim_opt = self.get_sim_options(sim_idx)
        seq_id = sim_opt['sequence_id']

        path = []
        nest_idx = seq_id[0]['nest_idx'] - 1
        for sid in seq_id:

            add_path = sid['path']

            if self.path_options['sequence_names']:
                add_path = sid['name'] + '_' + add_path

            if sid['nest_idx'] == nest_idx:
                path[-1] += self.path_options['parallel_sims_join'] + add_path
            else:
                path.append(add_path)

            nest_idx = sid['nest_idx']

        if self.path_options['sim_numbers']:
            path_labs = self.get_path_labels(sim_idx)
            path = [str(i) + self.path_options['sim_numbers_join'] + j
                    for i, j in zip(path_labs, path)]

        path = [self.hid, 'calcs'] + path
        return path
