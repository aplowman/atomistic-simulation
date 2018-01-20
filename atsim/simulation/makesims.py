"""matsim.simulation.makesims.py"""

from atsim.utils import prt
from atsim.simulation.simgroup import SimGroup


def main(makesim_opts, makesim_opts_raw, seq_defn):
    """Main function to generate a simulation group."""

    # Generate a new SimGroup object
    sim_group = SimGroup(makesim_opts, makesim_opts_raw, seq_defn)

    prt(sim_group, 'sim_group')

    # Generate input files on stage (current machine)
    sim_group.write_initial_runs()

    # Save current state of sim group as JSON file
    sim_group.save_state()

    # Copy to scratch
    sim_group.copy_to_scratch()

    # Submit initial run groups on scratch
    sim_group.submit_run_groups([0])

    print('Finished setting up simulation group: {}'.format(sim_group.hid))
