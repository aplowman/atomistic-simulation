"""matsim.simulation.makesims.py"""

from atsim.utils import prt
from atsim.simulation.simgroup import SimGroup
from atsim.resources import Stage, Scratch, Archive, ResourceConnection


def main(makesim_opts, makesim_opts_raw, seq_defn):
    """Main function to generate a simulation group."""

    # # Make resource objects:
    # stage = Stage(makesim_opts['run_opt'].pop('stage'))
    # scratch = Scratch(makesim_opts['run_opt'].pop('scratch'))
    # archive = Archive(makesim_opts['run_opt'].pop('archive'))

    # Generate a new SimGroup object
    sim_group = SimGroup(makesim_opts, makesim_opts_raw, seq_defn)

    # # Assign resource objects:
    # sim_group.set_stage(stage)
    # sim_group.set_scratch(scratch)
    # sim_group.set_archive(archive)

    prt(sim_group, 'sim_group')

    sim_group.add_to_db()

    # Generate input files on stage (current machine)
    sim_group.write_initial_runs()

    # Save current state of sim group as JSON file
    sim_group.save_state()

    print('exiting: {}'.format(sim_group.human_id))
    exit()

    # Copy to scratch
    sim_group.copy_to_scratch()

    # Submit initial run groups on scratch
    sim_group.submit_run_groups([0])

    print('Finished setting up simulation group: {}'.format(sim_group.human_id))
