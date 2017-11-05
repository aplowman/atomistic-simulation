import sys
import os
import yaml
from atsim import readwrite
from atsim import options_parser
from atsim import OPT_FILE_NAMES

args = sys.argv
opt_lkup_fn = OPT_FILE_NAMES['lookup']
opt_def_fn = OPT_FILE_NAMES['defaults']

if len(args) == 1 or args[1] == 'make':
    from atsim.simulation import makesims
    opt_fn = OPT_FILE_NAMES['makesims']
    ms_opt = options_parser.validate_ms_opt(opt_fn, opt_lkup_fn)
    makesims.main(ms_opt)

elif args[1] == 'process':
    if len(args) != 3:
        raise ValueError('Specify SID to process.')
    from atsim.analysis import process
    opt_fn = OPT_FILE_NAMES['process']
    ps_opt = options_parser.validate_ps_opt(opt_fn, opt_lkup_fn)
    process.main(ps_opt, args[2])

elif args[1] == 'submit_process':
    if len(args) != 3:
        raise ValueError('Specify SID to process.')
    from atsim.analysis import submit_process
    submit_process.main(args[2])

elif args[1] == 'harvest':
    from atsim.analysis import harvest
    opt_fn = OPT_FILE_NAMES['harvest']
    hv_opt = options_parser.validate_hv_opt(opt_fn, opt_lkup_fn, opt_def_fn)
    harvest.main(hv_opt)

elif args[1] == 'plot':
    from atsim.analysis import makeplots
    opt_fn = OPT_FILE_NAMES['makeplots']
    mp_opt = options_parser.validate_mp_opt(opt_fn, opt_lkup_fn, opt_def_fn)
    makeplots.main(mp_opt)

elif args[1] == 'series_helper':
    from atsim.simulation import serieshelper
    opt_fn = OPT_FILE_NAMES['serieshelper']
    sh_opt = options_parser.validate_sh_opt(opt_fn, opt_lkup_fn, opt_def_fn)
    serieshelper.main(sh_opt)
    # series_helpers.refine_gamma_surface(args[2])

else:
    print('Invalid option.')
