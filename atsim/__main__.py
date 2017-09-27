import sys
import os
import yaml
from atsim import readwrite
from atsim import options_parser

args = sys.argv

if len(args) == 1 or args[1] == 'make':
    from atsim.simulation import makesims
    opt_fn = 'makesims_opt.yml'
    opt_lkup_fn = 'makesims_opt_lookup.yml'
    ms_opt = options_parser.validate_ms_opt(opt_fn, opt_lkup_fn)
    makesims.main(ms_opt)

elif args[1] == 'process':
    if len(args) != 3:
        raise ValueError('Specify SID to process.')
    from atsim.analysis import process
    opt_fn = 'process_opt.yml'
    opt_lkup_fn = 'makesims_opt_lookup.yml'
    ps_opt = options_parser.validate_ps_opt(opt_fn, opt_lkup_fn)
    process.main(ps_opt, args[2])

elif args[1] == 'submit_process':
    if len(args) != 3:
        raise ValueError('Specify SID to process.')
    from atsim.analysis import submit_process
    submit_process.main(args[2])

elif args[1] == 'harvest':
    from atsim.analysis import harvest
    opt_fn = 'harvest.yml'
    opt_lkup_fn = 'makesims_opt_lookup.yml'
    opt_def_fn = 'defaults.yml'
    hv_opt = options_parser.validate_hv_opt(opt_fn, opt_lkup_fn, opt_def_fn)

    print('hv_opt: {}'.format(hv_opt))

    harvest.main(hv_opt)

elif args[1] == 'plot':
    if len(args) != 3:
        raise ValueError('Specify RID for plotting data source.')    
    from atsim.analysis import makeplots    
    opt_fn = 'makeplots.yml'
    opt_lkup_fn = 'makesims_opt_lookup.yml'
    opt_def_fn = 'defaults.yml'
    mp_opt = options_parser.validate_mp_opt(opt_fn, opt_lkup_fn, opt_def_fn)
    makeplots.main(args[2], mp_opt)
    
elif args[1] == 'series_helper':
    if len(args) != 3:
        raise ValueError('Specify SID to series_helper.')
    from atsim.simulation import series_helpers
    series_helpers.refine_gamma_surface(args[2])

else:
    print('Invalid option.')
