"""matsim.update"""

import datetime
import copy

from atsim import CONFIG
from atsim import database
from atsim import resources
from atsim.utils import prt, SpinnerThread


def parse_task_id(task_id_str):
    """Parse a task id str from `qstat` output."""

    split_low = task_id_str.split('-')
    start = int(split_low[0])
    split_high = split_low[1].split(':')
    stop = int(split_high[0]) + 1
    step = int(split_high[1])
    task_range = list(range(start, stop, step))

    return task_range


def expand_job_arrays(qstat_job):
    """Expand a single dict job array into a list of dict jobs."""

    ret = []

    # Parse the task IDs
    for task_idx in parse_task_id(qstat_job['task_id']):

        single_job = copy.deepcopy(qstat_job)
        single_job.update({
            'task_id': task_idx
        })
        ret.append(single_job)

    return ret


def parse_raw_qstat(qstat_lines):
    """Parse output of SGE qstat command into a dict."""

    qstat_jobs = []
    for job in qstat_lines[2:-1]:

        jb_items = job.split()
        state = jb_items[4]

        datetime_raw = ' '.join([jb_items[5], jb_items[6]])
        datetime_parsed = datetime.datetime.strptime(
            datetime_raw, '%m/%d/%Y %H:%M:%S')

        # Convert into datetime compatible with MySQL:
        dt_fmt = '%Y-%m-%d %H:%M:%S'
        datetime_str = datetime.datetime.strftime(datetime_parsed, dt_fmt)

        if state == 'qw':
            num_slots_idx = 7
        else:
            num_slots_idx = 8

        num_slots = int(jb_items[num_slots_idx])

        if (len(jb_items) - 1) == num_slots_idx:
            task_id = None
        else:
            task_id = jb_items[-1]

        job_dict = {
            'job_id': int(jb_items[0]),
            'name': jb_items[2],
            'datetime': datetime_str,
            'state': state,
            'slots': num_slots,
            'task_id': task_id
        }

        if task_id and '-' in task_id:
            # Expand job array into multiple jobs:
            job_dicts = expand_job_arrays(job_dict)

        else:
            if task_id:
                job_dict.update({'task_id': int(job_dict['task_id'])})
            job_dicts = [job_dict]

        qstat_jobs.extend(job_dicts)

    # Turn into one dict, keyed by tuple (job-ID, task-ID):
    ret = {}
    for job in qstat_jobs:
        job_id = job.pop('job_id')
        task_id = job.pop('task_id')
        ret.update({
            (job_id, task_id): job,
        })

    return ret


def main(opts):
    """Main function to update run states of SGE run groups.

    Task is to check if any run state need updating. Two possible updates:

    1.  If a run is currently recorded in any of the states:
            3 ("in_queue"), 4 ("queue_error")
        and it is now found according to `qstat` in:
            -   a running state, the state should be updated to 5 ("running").
            -   no state, the stat should be updated to 6 ("pending_process").

    2.  If a run is currently recorded in state 5 ("running") and it is now
        not found in `qstat` output, the state should be updated to
        6 ("pending_process").

    """

    # First check stage specified in options exists on this machine:
    stage = resources.Stage(opts['stage'])
    # print('stage: {}'.format(stage.__dict__))

    mach_name = CONFIG['machine_name']
    machine_defn = database.get_machine_by_name(mach_name)

    # TODO: move this check to Stage.__init__:
    if stage.machine_id != machine_defn['id']:
        msg = 'The Stage specified "{}" does not reside on this machine.'
        raise ValueError(msg.format(mach_name))

    # Get resource connections from this Stage to all available Scratches:
    res_conns_defn = database.get_resource_connections_by_source(
        stage.resource_id)
    dest_res_ids = [i['destination_id'] for i in res_conns_defn]

    spinner_thread = SpinnerThread('Updating database')
    spinner_thread.start()

    for res_id in dest_res_ids:

        scratch = resources.Scratch(
            database.get_scratch_by_resource_id(res_id)['name'])

        # Ignore non-SGE Scratch:
        if not scratch.sge:
            continue

        res_conn = resources.ResourceConnection(stage, scratch)

        qstat_raw = res_conn.run_command(['qstat'], block=True)
        qstat = parse_raw_qstat(qstat_raw.split('\n'))

        # Find runs on this scratch in states 3, 4, 5
        run_states = [3, 4, 5]
        all_runs = database.get_runs_by_scratch(scratch.scratch_id, run_states)

        state_3_run_ids = []
        state_4_run_ids = []
        state_5_run_ids = []
        state_6_run_ids = []

        # Now loop over runs_a:
        for run in all_runs:

            # Current state in database:
            db_state_id = run['run_state_id']
            r_id = run['run_id']

            # Does this run exist on qstat output:
            run_a_qstat = qstat.get((run['job_id'], run['run_order_id']))
            if run_a_qstat is not None:

                if run_a_qstat['state'] == 'qw' and db_state_id != 3:
                    # Update database run to be in state 3 ("in_queue")
                    state_3_run_ids.append(r_id)

                elif run_a_qstat['state'] == 'eq' and db_state_id != 4:
                    # Update database run to be in state 4 ("queue_error")
                    state_4_run_ids.append(r_id)

                if run_a_qstat['state'] == 'r' and db_state_id != 5:
                    # Update database run to be in state 5 ("running")
                    state_5_run_ids.append(r_id)

            elif db_state_id != 6:

                # Update database run to be in state 6 ("pending_process")
                state_6_run_ids.append(r_id)

        # Update run states in database:
        database.set_many_run_states(state_3_run_ids, 3)
        database.set_many_run_states(state_4_run_ids, 4)
        database.set_many_run_states(state_5_run_ids, 5)
        database.set_many_run_states(state_6_run_ids, 6)

    spinner_thread.stop()
    print("Database updated!")
