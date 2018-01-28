"""matsim.__init__.py"""

from pathlib import Path
from collections import namedtuple
import os
import yaml
from atsim.set_up.dbconfig import DB_CONFIG
from atsim.opt_parser import parse_opt, validate_all_opt_specs

_SCRIPTS_PATH = os.path.join(str(Path(__file__).parents[0]))
REF_PATH = os.path.join(_SCRIPTS_PATH, 'ref')
SET_UP_PATH = os.path.join(_SCRIPTS_PATH, 'set_up')

# Get file names of all options files
_CONFIG_FN = os.path.join(SET_UP_PATH, 'config.yml')
_OPT_SPEC_FN = os.path.join(SET_UP_PATH, 'opt_spec.yml')

# Load configuration data:
with open(_CONFIG_FN, 'r') as _config_fp:
    CONFIG = yaml.safe_load(_config_fp)

# Load options specification data:
with open(_OPT_SPEC_FN, 'r') as _spec_fp:
    OPTSPEC = yaml.safe_load(_spec_fp)

# Validation option specification file
validate_all_opt_specs(OPTSPEC)

# Parse config file:
CONFIG = parse_opt(CONFIG, OPTSPEC['config'])

# Get additional option file n
SEQ_FN = os.path.join(SET_UP_PATH, 'sequences.yml')
MAKESIMS_FN = os.path.join(SET_UP_PATH, CONFIG['option_paths']['makesims'])
UPDATE_FN = os.path.join(SET_UP_PATH, CONFIG['option_paths']['update'])
PROCESS_FN = os.path.join(SET_UP_PATH, CONFIG['option_paths']['process'])

# Jobscript template directory:
JS_TEMPLATE_DIR = os.path.join(_SCRIPTS_PATH, 'set_up', 'jobscript_templates')
