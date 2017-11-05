from pathlib import Path
import os

# Useful paths
SCRIPTS_PATH = os.path.join(str(Path(__file__).parents[0]))
REF_PATH = os.path.join(SCRIPTS_PATH, 'ref')
SET_UP_PATH = os.path.join(SCRIPTS_PATH, 'set_up')

# Allowed series names and their allowed keys:
VLD_SRS_NM = ['name']
VLD_SRS_NM_VLS = VLD_SRS_NM + ['vals']
VLD_SRS_NM_VLS_SSS = VLD_SRS_NM_VLS + ['start', 'step', 'stop']
ALLOWED_SERIES_KEYS = {
    'gb_size': VLD_SRS_NM_VLS,
    'box_lat': VLD_SRS_NM_VLS,
    'relative_shift': VLD_SRS_NM_VLS,
    'kpoint': VLD_SRS_NM_VLS_SSS,
    'cut_off_energy': VLD_SRS_NM_VLS_SSS,
    'smearing_width': VLD_SRS_NM_VLS_SSS,
    'nextra_bands': VLD_SRS_NM_VLS_SSS,
    'geom_energy_tol': VLD_SRS_NM_VLS_SSS,
    'geom_stress_tol': VLD_SRS_NM_VLS_SSS,
    'boundary_vac': VLD_SRS_NM_VLS_SSS,
    'boundary_vac_flat': VLD_SRS_NM_VLS_SSS,
    'point_defect_charge': VLD_SRS_NM_VLS_SSS,
    'point_defect_idx': VLD_SRS_NM_VLS_SSS,
    'gamma_surface': VLD_SRS_NM_VLS + ['grid_spec', 'preview'],
    'cs_vol_range': VLD_SRS_NM_VLS_SSS,
    'cs_ca_range': VLD_SRS_NM_VLS_SSS,
    'lookup': VLD_SRS_NM + ['src', 'parent_series', 'parent_val']
}
SERIES_NAMES = list(ALLOWED_SERIES_KEYS.keys())

# Options file names
OPT_FILE_NAMES = {
    'makesims': 'makesims.yml',
    'process': 'process.yml',
    'harvest': 'harvest.yml',
    'makeplots': 'makeplots.yml',
    'serieshelper': 'serieshelper.yml',
    'lookup': 'lookup.yml',
    'defaults': 'defaults.yml',
}
