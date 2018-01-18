import sys
import os
import yaml
from atsim import readwrite
from atsim import options_parser  # old one
from atsim import opt_parser  # new one
from atsim import OPT_FILE_NAMES
from atsim import SET_UP_PATH
from atsim.utils import prt

args = sys.argv
OPT_SPEC_FN = os.path.join(SET_UP_PATH, OPT_FILE_NAMES['opt_spec'])

opt_lkup_fn = OPT_FILE_NAMES['lookup']
opt_def_fn = OPT_FILE_NAMES['defaults']

OPT_CONFIG_FN = os.path.join(SET_UP_PATH, OPT_FILE_NAMES['config'])
OPT_RESOURCES_FN = os.path.join(SET_UP_PATH, OPT_FILE_NAMES['resources'])
OPT_SOFTWARE_FN = os.path.join(SET_UP_PATH, OPT_FILE_NAMES['software'])

if len(args) == 1 or args[1] == 'make':
    from atsim.simulation import makesims

    MS_OPT_FN = os.path.join(SET_UP_PATH, OPT_FILE_NAMES['makesims'])

    with open(OPT_CONFIG_FN, 'r') as config_fs:
        OPT_CONFIG = yaml.safe_load(config_fs)

    with open(OPT_RESOURCES_FN, 'r') as resources_fs:
        OPT_RESOURCES = yaml.safe_load(resources_fs)

    with open(OPT_SOFTWARE_FN, 'r') as software_fs:
        OPT_SOFTWARE = yaml.safe_load(software_fs)

    makesims.main2(MS_OPT_FN, OPT_SPEC_FN, OPT_CONFIG,
                   OPT_RESOURCES, OPT_SOFTWARE)


elif args[1] == 'load':

    from atsim.simgroup import SimGroup
    sim_group = SimGroup.load_state(args[2], OPT_SPEC_FN)
    prt(sim_group, 'sim_group')

    with open(OPT_CONFIG_FN, 'r') as config_fs:
        OPT_CONFIG = yaml.safe_load(config_fs)

    # a = sim_group.sims[0].structure
    # prt(a, 'a')

    # cs = sim_group.sims[0].structure.crystal_structures[0]
    # prt(cs, 'cs')
    sim_group.save_state()
    SimGroup.set_machine_id(OPT_CONFIG['machine_id'])
    # sim_group.write_initial_runs()

# elif args[1] == 'process':
#     if len(args) != 3:
#         raise ValueError('Specify SID to process.')
#     from atsim.analysis import process
#     opt_fn = OPT_FILE_NAMES['process']
#     ps_opt = options_parser.validate_ps_opt(opt_fn, opt_lkup_fn)
#     process.main(ps_opt, args[2])

# elif args[1] == 'submit_process':
#     if len(args) != 3:
#         raise ValueError('Specify SID to process.')
#     from atsim.analysis import submit_process
#     submit_process.main(args[2])

# elif args[1] == 'harvest':
#     from atsim.analysis import harvest
#     opt_fn = OPT_FILE_NAMES['harvest']
#     hv_opt = options_parser.validate_hv_opt(opt_fn, opt_lkup_fn, opt_def_fn)
#     harvest.main(hv_opt)

# elif args[1] == 'plot':
#     from atsim.analysis import makeplots
#     opt_fn = OPT_FILE_NAMES['makeplots']
#     mp_opt = options_parser.validate_mp_opt(opt_fn, opt_lkup_fn, opt_def_fn)
#     makeplots.main(mp_opt)

# elif args[1] == 'series_helper':
#     from atsim.simulation import serieshelper
#     opt_fn = OPT_FILE_NAMES['serieshelper']
#     sh_opt = options_parser.validate_sh_opt(opt_fn, opt_lkup_fn, opt_def_fn)
#     serieshelper.main(sh_opt)
#     # series_helpers.refine_gamma_surface(args[2])

else:
    print('Invalid option.')
