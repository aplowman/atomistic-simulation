"""matsim.analysis.process"""

import sys
import os
from copy import deepcopy
from distutils.dir_util import copy_tree
import shutil
import subprocess

from atsim import database
from atsim import update
from atsim.resources import ResourceConnection
from atsim.utils import prt
from atsim.simulation.simgroup import SimGroup


def search_database_by_session_id(database, s_id):

    base_opt = None
    for k, v in database.items():

        set_up_opts = v.get('set_up')
        if set_up_opts is not None:
            db_sid = set_up_opts['session_id']
        else:
            db_sid = v['session_id']

        if db_sid == s_id:
            base_opt = v
            break

    return base_opt


def check_errors(sms_path, src_path, skip_idx=None):
    """Some basic checks for errors in sim series output files."""

    # Open the sims pickle, get list of AtomisticSimulation objects:
    sms = read_pickle(sms_path)
    all_sms = sms['all_sims']
    # Get options from first sim if they don't exist (legacy compatiblity)
    base_opt = sms.get('base_options', all_sms[0].options)
    method = base_opt['method']

    error_idx = []
    s_count = 0
    for s_idx, sim_i in enumerate(all_sms):

        if skip_idx is not None and s_idx in skip_idx:
            continue

        s_count += 1
        srs_paths = []
        srs_id = sim_i.options.get('series_id')

        if srs_id is not None:

            # (legacy compatibility)
            if isinstance(srs_id, dict) and len(srs_id) == 1:

                new_srs_id = []
                for k, v in srs_id.items():
                    new_srs_id.append([{'name': k, **v}])
                srs_id = new_srs_id

            if not isinstance(srs_id, dict):
                for srs_id_lst in srs_id:
                    srs_paths.append('_'.join([i['path'] for i in srs_id_lst]))

            else:
                raise ValueError('Cannot parse `series_id` option from '
                                 's_idx: {}'.format(s_idx))

        calc_path = os.path.join(src_path, 'calcs', *srs_paths)

        if method == 'castep':
            out = castep.read_castep_output(calc_path)

        elif method == 'lammps':
            out = lammps.read_lammps_output(calc_path)

        if len(out['errors']) > 0:
            error_idx.append(s_idx)

    return error_idx


def move_offline_files(s_id, src_path, offline_files):

    arch_dir = offline_files['path']
    fl_types = offline_files.get('file_types') or offline_files.get('match')

    fls_paths = []
    for t in fl_types:
        fls_paths.extend(find_files_in_dir_glob(src_path, t, recursive=True))

    print('Offline files dir: {}'.format(arch_dir))
    print('Offline files types: {}'.format(fl_types))

    fls_paths = []
    for t in fl_types:
        fls_paths.extend(find_files_in_dir_glob(src_path, t, recursive=True))

    print('Offline files paths: {}'.format(fls_paths))

    if len(fls_paths) > 0:

        # Generate the offline files directories
        s_path = os.path.join(arch_dir, s_id)
        os.makedirs(s_path, exist_ok=True)

        for fp in fls_paths:
            dr, fl = os.path.split(fp)
            os.makedirs(os.path.join(s_path, dr), exist_ok=True)

            # Move the offline files to the offline dirs
            fl_src = os.path.join(src_path, fp)
            fl_dst = os.path.join(s_path, fp)
            print('Moving file from {} to {}'.format(fl_src, fl_dst))
            shutil.move(fl_src, fl_dst)

        print('{} offline files moved.'.format(len(fls_paths)))

    else:
        print('No offline files found.')


def main(opts, seq_defn, up_opts):
    """
    Process a given SimGroup:
    -   Run update to update run states in the database
    -   Find all runs in state 6 ("pending_process")
    -   Invoke check success method on each run
    -   If check success True, parse results and add to result attribute in
        SimGroup -> sim -> runs.
    -   Overwrite JSON file to Scratch (make backup of previous on Scratch)
    -   Copy new JSON file Archive (make backup of previous on Archive)

    """

    # Update (SGE) run states in database:
    # update.main(up_opts)

    # Instantiate SimGroup object:
    sim_group = SimGroup.load_state(opts['human_id'], 'scratch', seq_defn)
    sim_group.check_is_scratch_machine()
    sg_id = sim_group.db_id

    # prt(sg_id, 'sg_id')

    # Find all runs belonging to this run group in state 6
    pending_runs = database.get_sim_group_runs(sg_id, 6)
    # prt(pending_runs, 'pending_runs')

    # Set state to 7 ("processing") for these runs
    run_ids = [i['id'] for i in pending_runs]
    database.set_many_run_states(run_ids, 7)

    no_errs_pen_idx = []
    errs_pen_idx = []
    sim_run_idx = []

    for pen_run_idx, pen_run in enumerate(pending_runs):

        # Get path on scratch of run:
        sim_idx = pen_run['sim_order_id'] - 1
        run_idx = pen_run['run_group_order_id'] - 1
        sim_run_idx.append([sim_idx, run_idx])

        run_success = sim_group.check_run_success(sim_idx, run_idx)

        if run_success:
            no_errs_pen_idx.append(pen_run_idx)
        else:
            errs_pen_idx.append(pen_run_idx)

    # Update states to 9 ("process_errors")
    err_ids = [pending_runs[i]['id'] for i in errs_pen_idx]
    database.set_many_run_states(err_ids, 9)

    # Parse full output and add to sim.results[run_idx]
    for pen_run_idx in no_errs_pen_idx:
        sim_group.parse_result(*sim_run_idx[pen_run_idx])

    # Update states to 8 ("process_no_errors")
    no_err_ids = [pending_runs[i]['id'] for i in no_errs_pen_idx]
    database.set_many_run_states(no_err_ids, 8)

    if no_errs_pen_idx:

        # Copy new sim_group.json to Archive location
        arch_conn = ResourceConnection(sim_group.scratch, sim_group.archive)

        # Overwrite sim_group.json with new results:
        sim_group.save_state('scratch')

        if not database.check_archive_started(sg_id):
            # Copy everything to archive apart from calcs directory:
            arch_conn.copy_to_dest(ignore=['calcs'])
            database.set_archive_started(sg_id)

        else:
            # Copy updated sim_group.json
            subpath = ['sim_group.json']
            arch_conn.copy_to_dest(subpath=subpath, file_backup=True)

        archived_ids = []
        for pen_run_idx in no_errs_pen_idx:
            # Copy relevent sim/run/ directories to Archive location
            subpath = sim_group.get_run_path(*sim_run_idx[pen_run_idx])
            arch_conn.copy_to_dest(subpath=subpath)

            archived_ids.append(pending_runs[pen_run_idx]['id'])

        # Update states to 10 ("archived")
        database.set_many_run_states(archived_ids, 10)
