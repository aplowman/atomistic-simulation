import sys

args = sys.argv

if len(args) == 1 or args[1] == 'make':
    from atsim.simulation import makesims
    from atsim.set_up.opt import OPT
    makesims.main(OPT)

elif args[1] == 'harvest':
    from atsim.analysis import harvest
    from atsim.set_up.harvest_opt import HARVEST_OPT
    harvest.main(HARVEST_OPT)

elif args[1] == 'process':
    if len(args) != 3:
        print('Specify SID to process.')
    from atsim.analysis import process
    process.main(args[2])

elif args[1] == 'submit_process':
    if len(args) != 3:
        print('Specify SID to process.')
    from atsim.analysis import submit_process
    submit_process.main(args[2])

elif args[1] == 'series_helper':
    if len(args) != 3:
        print('Specify SID to series_helper.')
    from atsim.simulation import series_helpers
    series_helpers.refine_gamma_surface(args[2])

else:
    print('Invalid option.')
