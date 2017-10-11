import os
import itertools
import numpy as np
from atsim import vectors, utils
from atsim import readwrite
from atsim.readwrite import format_arr


def get_castep_cell_constraints(lengths_equal, angles_equal, fix_lengths,
                                fix_angles):
    """
    Get CASTEP cell constraints encoded from a set of more user-friendly
    parameters.

    Parameters
    ----------
    lengths_equal : str
        Some combination of 'a', 'b' and 'c', or 'none'. Represents which
        supercell vectors are to remain equal to one another.
    angles_equal : str
        Some combination of 'a', 'b' and 'c', or 'none'. Represents which
        supercell angles are to remain equal to one another.
    fix_lengths : str
        Some combination of 'a', 'b' and 'c', or 'none'. Represents which
        supercell vectors are to remain fixed.
    fix_angles : str
        Some combination of 'a', 'b' and 'c', or 'none'. Represents which
        supercell angles are to remain fixed.

    Returns
    -------
    list of list of int

    TODO:
    -   Improve robustness; should return exception for some cases where it
        currently is not: E.g. angles_equal = bc; fix_angles = ab should not
        be allowed. See OneNote notebook on rules.

    """

    if lengths_equal is None:
        lengths_equal = ''

    if angles_equal is None:
        angles_equal = ''

    if fix_lengths is None:
        fix_lengths = ''

    if fix_angles is None:
        fix_angles = ''

    eqs_params = [lengths_equal, angles_equal]
    fxd_params = [fix_lengths, fix_angles]
    encoded = [[1, 2, 3], [1, 2, 3]]

    ambig_cell_msg = 'Ambiguous cell constraints specified.'

    # Loop through lengths, then angles:
    for idx, (fp, ep) in enumerate(zip(fxd_params, eqs_params)):

        if len(fp) == 3 and all([x in fp for x in ['a', 'b', 'c']]):
            encoded[idx] = [0, 0, 0]

        elif len(fp) == 2 and all([x in fp for x in ['a', 'b']]):
            encoded[idx] = [0, 0, 3]

        elif len(fp) == 2 and all([x in fp for x in ['a', 'c']]):
            encoded[idx] = [0, 2, 0]

        elif len(fp) == 2 and all([x in fp for x in ['b', 'c']]):
            encoded[idx] = [1, 0, 0]

        elif fp == 'a':

            if ep == '':
                encoded[idx] = [0, 2, 3]

            elif ep == 'bc':
                encoded[idx] = [0, 2, 2]

            else:
                raise ValueError(ambig_cell_msg)

        elif fp == 'b':

            if ep == '':
                encoded[idx] = [1, 0, 3]

            elif ep == 'ac':
                encoded[idx] = [1, 0, 1]

            else:
                raise ValueError(ambig_cell_msg)

        elif fp == 'c':

            if ep == '':
                encoded[idx] = [1, 2, 0]

            elif ep == 'ab':
                encoded[idx] = [1, 1, 0]

            else:
                raise ValueError(ambig_cell_msg)

        else:

            if len(ep) == 3 and all([x in ep for x in ['a', 'b', 'c']]):
                encoded[idx] = [1, 1, 1]

            elif any([x in ep for x in ['ab', 'ba']]):
                encoded[idx] = [1, 1, 3]

            elif any([x in ep for x in ['ac', 'ca']]):
                encoded[idx] = [1, 2, 1]

            elif any([x in ep for x in ['bc', 'cb']]):
                encoded[idx] = [1, 2, 2]

    for idx in range(len(encoded[1])):
        if encoded[1][idx] > 0:
            encoded[1][idx] += 3

    return encoded


def write_castep_inputs(supercell, atom_sites, species, species_idx, path,
                        seedname='sim', cell=None, param=None, sym_ops=None,
                        cell_constraints=None, atom_constraints=None):
    """
    Generate CASTEP input files.

    Parameters
    ----------
    supercell : ndarray of shape (3, 3)
        Array of column vectors representing the edge vectors of the supercell.
    atom_sites : ndarray of shape (3, N)
        Array of column vectors representing the positions of each atom.
    species : ndarray of str of shape (M, )
        The distinct species of the atoms.
    species_idx : list or ndarray of shape (N, )
        Maps each atom site to a given species in `species`.
    path : str
        Directory in which to generate input files.
    seedname : str, optional
        The seedname of the CASTEP calculation. Default is `sim`.
    cell : dict, optional
        Key value pairs to add to the cell file.
    param : dict, optional
        Key value pairs to add to the param file.
    sym_ops: list of ndarray of shape (4, 3)
        Each array represents a symmetry operation, where the first three rows
        are the rotation matrix and the final row is the translation.
    cell_constraints : dict, optional
        A dict with the following keys:
            lengths_equal : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell vectors are to remain equal to one another.
            angles_equal : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell angles are to remain equal to one another.
            fix_lengths : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell vectors are to remain fixed.
            fix_angles : str
                Some combination of 'a', 'b' and 'c'. Represents which
                supercell angles are to remain fixed.
    atom_constraints : dict, optional
        A dict with the following keys:
            fix_`mn`_idx : ndarray of dimension 1
                The atom indices whose `m` and `n` coordinates are to
                be fixed, where valid pairs of `mn` are (`xy`, `xz`, `yz`).
                By default, set to None.
            fix_xyz_idx : ndarray of dimension 1
                The atom indices whose `x`, `y` and `z` coordinates
                are to be fixed. By default, set to None.

    TODO:
    -   Generalise atom constraints.

    """

    # Validation:

    species_idx = utils.parse_as_int_arr(species_idx)
    if species_idx.min() < 0 or species_idx.max() > (atom_sites.shape[1] - 1):
        raise IndexError('`species_idx` must index `atom_sites`'.format(k))

    for k, v in atom_constraints.items():

        if isinstance(v, (np.ndarray, list)):

            atom_constraints[k] = utils.parse_as_int_arr(v)
            v = atom_constraints[k]

            if v.ndim != 1:
                raise ValueError('`atom_constraints[{}]` must be a 1D list, '
                                 '1D array or str.'.format(k))

            if v.min() < 1 or v.max() > atom_sites.shape[1]:
                raise IndexError('`atom_constraints[{}]` must index '
                                 '`atom_sites`'.format(k))

        elif v is not None:
            raise ValueError('`atom_constraints[{}]` must be a 1D list or 1D '
                             'array.'.format(k))

    f_xy = atom_constraints.get('fix_xy_idx')
    f_xz = atom_constraints.get('fix_xz_idx')
    f_yz = atom_constraints.get('fix_yz_idx')
    f_xyz = atom_constraints.get('fix_xyz_idx')

    if f_xy is None:
        f_xy = np.array([])
    if f_xz is None:
        f_xz = np.array([])
    if f_yz is None:
        f_yz = np.array([])
    if f_xyz is None:
        f_xyz = np.array([])

    atom_constr_opt = [f_xy, f_xz, f_yz, f_xyz]
    atom_constr_pairs = list(itertools.combinations(atom_constr_opt, 2))

    for pair in atom_constr_pairs:
        if len(pair[0]) > 0 and len(pair[1]) > 0:
            if len(np.intersect1d(pair[0], pair[1])) > 0:
                raise ValueError('`{}_idx` and `{}_idx`  cannot '
                                 'contain the same indices.'.format(pair[0], pair[1]))

    os.makedirs(path, exist_ok=True)

    task = param['task'].upper()
    geom_opt_str = ['GEOMETRYOPTIMISATION', 'GEOMETRYOPTIMIZATION']

    # Write CELL file:
    cell_path = os.path.join(path, seedname + '.cell')
    with open(cell_path, 'w') as cf:   # to do: test if i need universal newline mode here

        # Supercell (need to tranpose to array of row vectors):
        cf.write('%block lattice_cart\n')
        cf.write(format_arr(
            supercell.T, format_spec='{:24.15f}', col_delim=' '))
        cf.write('%endblock lattice_cart\n\n')

        # Atoms (need to tranpose to array of row vectors):
        atom_species = species[species_idx][:, np.newaxis]

        cf.write('%block positions_abs\n')
        cf.write(format_arr([atom_species, atom_sites.T],
                            format_spec=['{:5}', '{:24.15f}'],
                            col_delim=' '))
        cf.write('%endblock positions_abs\n')

        # Cell constraints:
        encoded_params = get_castep_cell_constraints(**cell_constraints)

        if (not (encoded_params[0] == [1, 2, 3] and
                 encoded_params[1] == [4, 5, 6])) and (
                task in geom_opt_str):

            if (encoded_params[0] == [0, 0, 0] and
                    encoded_params[1] == [0, 0, 0]):
                cf.write('\nfix_all_cell = True\n')

            else:
                cf.write('\n%block cell_constraints\n')
                cf.write('{}\t{}\t{}\n'.format(*encoded_params[0]))
                cf.write('{}\t{}\t{}\n'.format(*encoded_params[1]))
                cf.write('%endblock cell_constraints\n')

        # Atom constraints:
        if any([len(x) for x in atom_constr_opt]) > 0:

            # For each atom, get the index within like-species atoms:
            # 1-based indexing instead of 0-based!
            sub_idx = np.zeros((atom_sites.shape[1]), dtype=int) - 1
            for sp_idx in range(len(species)):
                w = np.where(species_idx == sp_idx)[0]
                sub_idx[w] = np.arange(w.shape[0]) + 1

            cnst_fs = ['{:<5d}', '{:<5}', '{:<5d}', '{:24.15f}']
            cf.write('\n%block ionic_constraints\n')

            nc_xyz = f_xyz.shape[0]
            nc_xy = f_xy.shape[0]
            nc_xz = f_xz.shape[0]
            nc_yz = f_yz.shape[0]

            if nc_xyz > 0:
                f_xyz -= 1
                f_xyz_sp = np.tile(
                    atom_species[f_xyz], (1, 3)).reshape(nc_xyz * 3, 1)
                f_xyz_sub_idx = np.repeat(sub_idx[f_xyz], 3)[:, np.newaxis]
                f_xyz_cnst_idx = (np.arange(nc_xyz * 3) + 1)[:, np.newaxis]
                f_xyz_cnst_coef = np.tile(np.eye(3), (nc_xyz, 1))

                cnst_arrs_xyz = [f_xyz_cnst_idx, f_xyz_sp, f_xyz_sub_idx,
                                 f_xyz_cnst_coef]

                cf.write(format_arr(cnst_arrs_xyz,
                                    format_spec=cnst_fs,
                                    col_delim=' '))

            if nc_xy > 0:
                f_xy -= 1
                f_xy_sp = np.tile(
                    atom_species[f_xy], (1, 2)).reshape(nc_xy * 2, 1)
                f_xy_sub_idx = np.repeat(sub_idx[f_xy], 2)[:, np.newaxis]
                f_xy_cnst_idx = (np.arange(nc_xy * 2) + 1 +
                                 (nc_xyz * 3))[:, np.newaxis]
                f_xy_cnst_coef = np.tile(np.eye(3)[[0, 1]], (nc_xy, 1))

                cnst_arrs_xy = [f_xy_cnst_idx, f_xy_sp, f_xy_sub_idx,
                                f_xy_cnst_coef]

                cf.write(format_arr(cnst_arrs_xy,
                                    format_spec=cnst_fs,
                                    col_delim=' '))

            if nc_xz > 0:
                f_xz -= 1
                f_xz_sp = np.tile(
                    atom_species[f_xz], (1, 2)).reshape(nc_xz * 2, 1)
                f_xz_sub_idx = np.repeat(sub_idx[f_xz], 2)[:, np.newaxis]
                f_xz_cnst_idx = (np.arange(nc_xz * 2) + 1 +
                                 (nc_xy * 2) + (nc_xyz * 3))[:, np.newaxis]
                f_xz_cnst_coef = np.tile(np.eye(3)[[0, 2]], (nc_xz, 1))

                cnst_arrs_xz = [f_xz_cnst_idx, f_xz_sp, f_xz_sub_idx,
                                f_xz_cnst_coef]

                cf.write(format_arr(cnst_arrs_xz,
                                    format_spec=cnst_fs,
                                    col_delim=' '))

            if nc_yz > 0:
                f_yz -= 1
                f_yz_sp = np.tile(
                    atom_species[f_yz], (1, 2)).reshape(nc_yz * 2, 1)
                f_yz_sub_idx = np.repeat(sub_idx[f_yz], 2)[:, np.newaxis]
                f_yz_cnst_idx = (np.arange(nc_yz * 2) + 1 + (nc_xz * 2) +
                                 (nc_xy * 2) + (nc_xyz * 3))[:, np.newaxis]

                f_yz_cnst_coef = np.tile(np.eye(3)[[1, 2]], (nc_yz, 1))

                cnst_arrs_yz = [f_yz_cnst_idx, f_yz_sp, f_yz_sub_idx,
                                f_yz_cnst_coef]

                cf.write(format_arr(cnst_arrs_yz,
                                    format_spec=cnst_fs,
                                    col_delim=' '))

            cf.write('%endblock ionic_constraints\n')

        # Symmetry ops
        if sym_ops is not None:

            sym_ops = np.vstack(sym_ops)
            cf.write('\n%block symmetry_ops\n')
            cf.write(format_arr(sym_ops,
                                format_spec='{:24.15f}',
                                col_delim=' '))
            cf.write('%endblock symmetry_ops\n')

        # Other cell file items:
        if cell is not None:
            cf.write('\n')
            for k, v in sorted(cell.items()):
                cf.write('{:25s}= {}\n'.format(k, v))

    # Write PARAM file:
    if param is not None:
        param_path = os.path.join(path, seedname + '.param')
        with open(param_path, 'w') as pf:
            for k, v in sorted(param.items()):
                pf.write('{:25s}= {}\n'.format(k, v))


def read_castep_output(dir_path, seedname=None, ignore_missing_output=False):
    """
        Function to parse the output files from CASTEP. The main output file is
        `seedname`.castep.

        If `seedname` is specified, raise IOError if there is no `seedname`.castep
        file in `dir_path`.

        If `seedname` is not specified, raise IOError if there is not exactly one
        .castep file in `dir_path.

        Depending on the calculation task, additional output files are generated
        (e.g. `seedname`.geom for geometry optimisations). If these files are not
        found in `dir_path`, raise an IOError, unless `ignore_missing_output` is True.

        CASTEP versions tested:
            17.2

        CASTEP tasks tested:
            SinglePoint, GeometryOptimisation

    """

    # Find the files ending in .castep in `dir_path`:
    all_cst_files = readwrite.find_files_in_dir(
        dir_path, r'.castep$', recursive=False)

    # If no .castep files in `dir_path, raise IOError.
    if len(all_cst_files) == 0:
        raise IOError(
            'No .castep files found in directory {}'.format(dir_path))

    if seedname is None:

        if len(all_cst_files) > 1:
            raise IOError(
                'Seedname not specified, but multiple .castep files found.')

        else:
            seedname = all_cst_files[0].split('.castep')[0]

    cst_fn = '{}.castep'.format(seedname)
    cst_path = os.path.join(dir_path, cst_fn)

    if not os.path.isfile(cst_path):
        raise IOError(
            'File not found: {} in directory {}'.format(cst_fn, dir_path))

    # Parse the .castep file:
    out = read_castep_file(cst_path)

    # Parse additional ouput files:
    vrs = out['version']
    is_geom = out['params']['calc_type'] == 'geometry optimization'
    if (vrs in ['17.2'] and out['write_geom']) or is_geom:

        # Parse the .geom file:
        geom_fn = '{}.geom'.format(seedname)
        geom_path = os.path.join(dir_path, geom_fn)

        if not ignore_missing_output and not os.path.isfile(geom_path):
            raise IOError(
                'File not found: {} in directory {}'.format(geom_fn, dir_path))

        out['geom'] = read_castep_geom_file(geom_path)

    return out


def read_castep_file(cst_path):
    """
    Function to parse a .castep file. Returns a dict.

    CASTEP versions tested:
        17.2

    CASTEP tasks tested:
        SinglePoint, GeometryOptimisation

    Notes:

    -   Unconstrained forces reported in the BFGS: Final Configuration section
    -   Constrained forces reported after each SCF cycle and in BFGS: Final
        Configuration section

    -   BFGS Final Configuration:
        - Cell contents (if ions relaxed)
        - Unit cell (if cell relaxed)
        - BFGS Final Enthalpy
        - BFGS Final <frequency> (if cell fixed?)
        - BFGS Final bulk modulus (if cell relaxed)
        - Unconstrained Forces
        - Constrained (symmetrised) Forces
        - (Symmetrised) Stress Tensor
        - Atomic populations

    -   cell contents will be listed more than scf_num_cycles if BFGS reverts
        to an earlier configuration.

    TODO:
    -   Test version 16.1.1
    -   Add error lines to `errors` list

    """

    TESTED_VERS = ['17.2', '16.11']

    HEADER = '+-------------------------------------------------+'
    VERS = 'CASTEP version'
    CALC_TYPE = 'type of calculation                            :'
    PARAM_ECUT = 'plane wave basis set cut-off                   :'
    PARAM_FINE_GRID = 'size of   fine   gmax                          :'
    PARAM_NUM_ELEC = 'number of  electrons                           :'
    PARAM_NET_CHARGE = 'net charge of system                           :'
    PARAM_NUM_BANDS = 'number of bands                                :'
    PARAM_METALLIC = 'Method: Treating system as metallic'
    PARAM_ELEC_EN_TOL = 'total energy / atom convergence tol.           :'
    BASIS_SET_PARAM_FBC = 'finite basis set correction                    :'
    BASIS_SET_PARAM_NUM_EN = 'number of sample energies                      :'
    FBC = 'Calculating finite basis set correction'
    FBC_RESULT = 'For future reference: finite basis dEtot/dlog(Ecut) ='
    SCF_START_END = '------------------------------------------------------------------------ <-- SCF'
    SCF_FIRST = 'Initial'
    BFGS_START = 'BFGS: starting iteration'
    BFGS_IMPROVE = 'BFGS: improving iteration'
    BFGS_END = 'BFGS: finished iteration'
    BFGS_IMPROVE_LN = BFGS_IMPROVE + '{:10d} with line'
    BFGS_IMPROVE_QD = BFGS_IMPROVE + '{:10d} with quad'
    SCF_FINAL_EN = 'Final energy ='
    SCF_FINAL_FEN = 'Final free energy (E-TS)    ='
    SCF_FINAL_ZEN = 'NB est. 0K energy (E-0.5TS)      ='
    BFGS_DE = '|  dE/ion   |'
    BFGS_S_MAX = '|   Smax    |'
    BFGS_MAG_F_MAX = '|  |F|max   |'
    BFGS_MAG_DR_MAX = '|  |dR|max  |'
    NUM_IONS = 'Total number of ions in cell ='
    TOT_TIME = 'Total time          ='
    GO_PARAM_WRITE_GEOM = 'write geom trajectory file                     :'
    UNCON_FORCES_START = '******************* Unconstrained Forces *******************'
    UNCON_FORCES_END = '*                                                          *'
    CON_FORCES_SP_START = '******************** Constrained Forces ********************'
    CON_FORCES_SP_END = UNCON_FORCES_END
    CON_FORCES_GO_START = '******************************** Constrained Forces ********************************'
    CON_FORCES_GO_END = '*                                                                                  *'
    CON_SYM_FORCES_GO_START = '************************** Constrained Symmetrised Forces **************************'
    CON_SYM_FORCES_GO_END = CON_FORCES_GO_END
    CON_SYM_FORCES_SP_START = '************** Constrained Symmetrised Forces **************'
    CON_SYM_FORCES_SP_END = UNCON_FORCES_END
    BFGS_CYCLE_TYPE_STR = np.array(
        ['Initial', 'Trial guess', 'Line minimization', 'Quad minimization'])
    CELL_CONTS_START_END = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    CELL_CONTS_INI_START_END = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
    CELL_START = 'Unit Cell'
    CELL_END = 'Current cell volume ='
    KP_MP = 'MP grid size for SCF calculation is'
    KP_OFF = 'with an offset of'
    KP_NUM = 'Number of kpoints used ='
    SYM_MAX_DEV = 'Maximum deviation from symmetry ='
    SYM_NUM = 'Number of symmetry operations   ='
    ION_CON_NUM = 'Number of ionic constraints     ='
    POINT_GRP = 'Point group of crystal ='
    SPACE_GRP = 'Space group of crystal ='
    CELL_CON_NUM = 'Number of cell constraints='
    CELL_CON = 'Cell constraints are:'
    WARNING = 'Warning:'

    header_lns = 0  # Header line is repeated three times for each header
    version = None
    ecut = None
    fine_grid = None
    num_elec = None
    num_bands = None
    metallic = False
    scf_num_cols = 3
    elec_energy_tol = None
    net_charge = None
    sym_max_deviation = None
    sym_num_ops = None
    ion_constraints_num = None
    point_group = None
    space_group = None
    cell_constraints = None
    cell_constraints_num = None
    finite_basis_correction = None
    calc_type_str = None
    errors = []
    warnings = []

    bfgs_iter_idx = 0

    # Each BFGS iteration is associated with one or more SCF cycles:
    # Here, scf_cycle refers to a whole SCF convergence process to find the groundstate wavefunction
    scf_cycle_idx = 0
    # Index list where indices correspond to 0: 'initial, 1: 'trial guess', 2: 'line', 3: 'quad'
    scf_cycle_type_idx = []
    bfgs_lambda = []
    cell_contents = []
    current_cell_conts = []

    real_lattice = []
    recip_lattice = []
    lattice_params = []
    cell_angles = []
    cell_volume = []

    current_real_lat = []
    current_recip_lat = []
    current_lat_params = []
    current_cell_angles = []

    # Each SCF cycle finishes with a final (free) energy:
    final_energy = []
    final_fenergy = []
    final_zenergy = []

    # Each BFGS iteration finishes with a final enthalpy:
    bfgs_enthalpy = []
    dE_per_ion = []
    mag_F_max = []
    mag_dR_max = []
    s_max = []

    finite_basis_parsed = False
    finite_basis_corr = False
    finite_basis_num_en = -1

    tot_time = np.nan
    num_ions = np.nan
    write_geom = None
    scf_iter_data = np.ones(4) * np.nan
    scf_cycle_data = []
    all_scf_data = []

    SCF_HEADER_LNS = 3  # 3 lines between scf header and start of block
    FORCES_HEADER_LNS = 5  # 5 lines between forces header and start of block
    # 3 lines between cell contents header and start of block
    CELL_CONTENTS_HEADER_LNS = 3
    CELL_LAT_IDX_START = 2
    CELL_LAT_IDX_END = 4
    CELL_PARAMS_IDX_START = 7
    CELL_PARAMS_IDX_END = 9

    # Set line indices for parsing blocks of data; must reset after each block is parsed
    force_ion_idx = -FORCES_HEADER_LNS
    cell_conts_idx = -CELL_CONTENTS_HEADER_LNS
    scf_iter_idx = -SCF_HEADER_LNS
    cell_idx = 0

    species_set = False
    species = []
    species_idx = []

    current_bfgs_forces = []
    all_constrained_forces = []
    all_unconstrained_forces = []
    all_constrained_symmetrised_forces = []

    kpoint_mp_grid = None
    kpoint_mp_offset = None
    kpoint_num = None

    # For data blocks parsed over multiple lines, mode is changed from scan to parse_<something>
    mode = 'scan'
    prev_mode = mode

    with open(cst_path, 'r') as cst:

        for ln_idx, ln in enumerate(cst):

            ln_s = ln.strip().split()

            if len(ln_s) > 0 and ln_s[0] == WARNING:
                prev_mode = mode
                mode = 'parse_warning'
                warnings.append([])

            if ln.strip() == HEADER:
                header_lns += 1
                if header_lns % 3 != 0:
                    mode = 'parse_header'
                else:
                    force_ion_idx = -FORCES_HEADER_LNS
                    cell_conts_idx = -CELL_CONTENTS_HEADER_LNS
                    scf_iter_idx = -SCF_HEADER_LNS
                    cell_idx = 0
                    mode = 'scan'

            if not finite_basis_parsed:

                if BASIS_SET_PARAM_FBC in ln:

                    if ln_s[-1] == 'automatic':
                        finite_basis_corr = True

                    elif ln_s[-1] == 'none':
                        finite_basis_num_en = 0
                        finite_basis_parsed = True

                elif finite_basis_corr and BASIS_SET_PARAM_NUM_EN in ln:

                    finite_basis_num_en = int(ln_s[-1])
                    scf_cycle_type_idx += [0, ] * finite_basis_num_en
                    bfgs_lambda += [np.nan, ] * finite_basis_num_en
                    finite_basis_parsed = True

            if mode == 'parse_warning':

                if ln.strip() != '':
                    warnings[-1].append(ln.strip())
                else:
                    mode = prev_mode

            elif mode == 'parse_header':
                if VERS in ln:
                    version = ln_s[7].split('|')[0]
                    if version not in TESTED_VERS:
                        raise NotImplementedError(
                            'Parser not tested on this version of CASTEP: '
                            '{}'.format(version))

            # Parse a unit cell block
            elif mode == 'parse_cell':

                if (cell_idx >= CELL_LAT_IDX_START and
                        cell_idx <= CELL_LAT_IDX_END):

                    # Parse real and reciprocal lattice blocks. Theses are row
                    # vectors in the file, but we will return as column vectors.

                    rl_lat_ijk = [float(ln_s[i]) for i in [0, 1, 2]]
                    rc_lat_ijk = [float(ln_s[i]) for i in [3, 4, 5]]
                    current_real_lat.append(
                        np.array(rl_lat_ijk)[:, np.newaxis])
                    current_recip_lat.append(
                        np.array(rc_lat_ijk)[:, np.newaxis])

                    if cell_idx == CELL_LAT_IDX_END:

                        real_lattice.append(np.hstack(current_real_lat))
                        recip_lattice.append(np.hstack(current_recip_lat))

                        current_real_lat = []
                        current_recip_lat = []

                    cell_idx += 1

                elif (cell_idx >= CELL_PARAMS_IDX_START and
                      cell_idx <= CELL_PARAMS_IDX_END):

                    current_lat_params.append(float(ln_s[2]))
                    current_cell_angles.append(float(ln_s[5]))

                    if cell_idx == CELL_PARAMS_IDX_END:

                        lattice_params.append(np.array(current_lat_params))
                        cell_angles.append(np.array(current_cell_angles))

                        current_lat_params = []
                        current_cell_angles = []

                    cell_idx += 1

                elif CELL_END in ln:

                    # Parse cell volume and finish parsing cell block
                    cell_volume.append(float(ln_s[-2]))
                    mode = 'scan'
                    cell_idx = 0

                else:
                    cell_idx += 1

            # Parse a cell contents block
            elif mode == 'parse_cell_contents':

                if cell_conts_idx < 0:
                    cell_conts_idx += 1

                elif cell_conts_idx > 0 and (CELL_CONTS_START_END in ln or
                                             CELL_CONTS_INI_START_END in ln):

                    # Finish parsing cell contents block
                    mode = 'scan'

                    cell_conts_idx = -CELL_CONTENTS_HEADER_LNS
                    cell_contents.append(np.hstack(current_cell_conts))
                    current_cell_conts = []

                    if not species_set:
                        species_set = True

                elif cell_conts_idx >= 0:

                    sp = ln_s[1]
                    ion_idx = ln_s[2]

                    if not species_set:

                        if sp not in species:
                            species.append(sp)

                        species_idx.append(species.index(sp))

                    ion_uvw = [float(ln_s[i]) for i in [3, 4, 5]]
                    current_cell_conts.append(
                        np.array(ion_uvw)[:, np.newaxis])

                    cell_conts_idx += 1

            # Parse an SCF cycle block:
            elif mode == 'parse_scf':

                if scf_iter_idx < 0:
                    scf_iter_idx += 1

                elif scf_iter_idx > 0 and SCF_START_END in ln:

                    # Finish parsing SCF block
                    mode = 'scan'
                    scf_iter_idx = -SCF_HEADER_LNS
                    scf_cycle_idx += 1
                    all_scf_data.append(np.array(scf_cycle_data))

                    scf_cycle_data = []

                elif scf_iter_idx >= 0:

                    scf_iter_data = np.ones(scf_num_cols) * np.nan

                    if scf_iter_idx == 0:

                        if metallic:

                            scf_iter_data[0:2] = [
                                float(ln_s[i]) for i in [1, 2]]
                            scf_iter_data[3] = float(ln_s[3])

                        else:

                            scf_iter_data[0] = float(ln_s[1])
                            scf_iter_data[2] = float(ln_s[2])

                    else:

                        scf_iter_data = [float(ln_s[i])
                                         for i in range(1, scf_num_cols + 1)]

                    scf_cycle_data.append(scf_iter_data)
                    scf_iter_idx += 1

            # Parse a forces block:
            elif mode in ['parse_con_forces',
                          'parse_uncon_forces',
                          'parse_con_sym_forces']:

                if force_ion_idx < 0:
                    force_ion_idx += 1

                elif force_ion_idx > 0 and (UNCON_FORCES_END in ln or
                                            CON_FORCES_GO_END in ln):

                    # Finish parsing forces block
                    force_ion_idx = -FORCES_HEADER_LNS
                    current_bfgs_forces = np.hstack(current_bfgs_forces)

                    if mode == 'parse_con_forces':
                        all_constrained_forces.append(current_bfgs_forces)

                    elif mode == 'parse_uncon_forces':
                        all_unconstrained_forces.append(current_bfgs_forces)

                    elif mode == 'parse_con_sym_forces':
                        all_constrained_symmetrised_forces.append(
                            current_bfgs_forces)

                    current_bfgs_forces = []
                    mode = 'scan'

                elif force_ion_idx >= 0:

                    ln_s = [i.split("(cons'd)")[0] for i in ln_s]
                    force_xyz = [float(ln_s[i]) for i in [3, 4, 5]]
                    current_bfgs_forces.append(
                        np.array(force_xyz)[:, np.newaxis])

                    force_ion_idx += 1

            elif mode == 'scan':

                if CALC_TYPE in ln:
                    cur_calc_type_str = ln.split(':')[1].strip()
                    if calc_type_str is not None:
                        if cur_calc_type_str != calc_type_str:
                            raise ValueError('Caclulation type changed: was '
                                             '"{}", changed to: "{}"'.format(
                                                 calc_type_str, cur_calc_type_str))
                    calc_type_str = cur_calc_type_str

                elif PARAM_ECUT in ln:
                    ecut = float(ln_s[-2])

                elif PARAM_FINE_GRID in ln:
                    fine_grid = float(ln_s[-2])

                elif PARAM_NUM_ELEC in ln:
                    num_elec = float(ln_s[-1])

                elif PARAM_NUM_BANDS in ln:
                    num_bands = int(ln_s[-1])

                elif PARAM_NET_CHARGE in ln:
                    net_charge = float(ln_s[-1])

                elif PARAM_METALLIC in ln:
                    metallic = True
                    SCF_FINAL_EN = 'Final energy, E             ='
                    scf_num_cols = 4

                elif PARAM_ELEC_EN_TOL in ln:
                    elec_energy_tol = float(ln_s[-2])

                elif GO_PARAM_WRITE_GEOM in ln:
                    write_geom = True if ln_s[-1] == 'on' else False

                elif NUM_IONS in ln:
                    num_ions = int(ln_s[-1])

                elif CELL_CONTS_START_END in ln or CELL_CONTS_INI_START_END in ln:
                    mode = 'parse_cell_contents'

                elif CELL_START in ln:
                    mode = 'parse_cell'

                elif SCF_START_END in ln:
                    mode = 'parse_scf'

                elif KP_MP in ln:
                    kpoint_mp_grid = [int(ln_s[i]) for i in [-3, -2, -1]]

                elif KP_OFF in ln:
                    kpoint_mp_offset = [float(ln_s[i]) for i in [-3, -2, -1]]

                elif KP_NUM in ln:
                    kpoint_num = int(ln_s[-1])

                elif SYM_MAX_DEV in ln:
                    sym_max_deviation = float(ln_s[-2])

                elif SYM_NUM in ln:
                    sym_num_ops = int(ln_s[-1])

                elif ION_CON_NUM in ln:
                    ion_constraints_num = int(ln_s[-1])

                elif POINT_GRP in ln:
                    point_group = ln.split('=')[1].strip()

                elif CELL_CON_NUM in ln:
                    cell_constraints_num = int(ln.split('=')[1].strip())

                elif CELL_CON in ln:
                    cell_constraints = [int(ln_s[i]) for i in range(3, 9)]

                elif SPACE_GRP in ln:
                    space_group = ln.split('=')[1].strip()

                elif FBC_RESULT in ln:
                    finite_basis_correction = float(ln_s[-1].split('eV')[0])

                elif CON_FORCES_GO_START in ln or CON_FORCES_SP_START in ln:
                    mode = 'parse_con_forces'

                elif UNCON_FORCES_START in ln:
                    mode = 'parse_uncon_forces'

                elif CON_SYM_FORCES_GO_START in ln or CON_SYM_FORCES_SP_START in ln:
                    mode = 'parse_con_sym_forces'

                elif BFGS_END in ln:
                    bfgs_enthalpy.append(float(ln_s[-2]))

                elif BFGS_START in ln or BFGS_IMPROVE in ln:

                    if BFGS_START in ln and BFGS_IMPROVE not in ln:

                        bfgs_iter_idx += 1
                        scf_cycle_type_idx.append(1)

                    try:
                        λ = float(ln_s[-1].split('=')[-1].split(')')[0])

                    except ValueError:

                        # Sometimes get: "(lambda=**********)"; perhaps if larger than 999.000000?
                        λ = np.nan

                    bfgs_lambda.append(λ)

                    if BFGS_IMPROVE_LN.format(bfgs_iter_idx) in ln:

                        scf_cycle_type_idx.append(2)

                    elif BFGS_IMPROVE_QD.format(bfgs_iter_idx) in ln:

                        scf_cycle_type_idx.append(3)

                elif SCF_FINAL_EN in ln:
                    final_energy.append(float(ln_s[-2]))

                elif SCF_FINAL_FEN in ln:
                    final_fenergy.append(float(ln_s[-2]))

                elif SCF_FINAL_ZEN in ln:
                    final_zenergy.append(float(ln_s[-2]))

                elif BFGS_DE in ln:
                    dE_per_ion.append(float(ln_s[3]))

                elif BFGS_MAG_F_MAX in ln:
                    mag_F_max.append(float(ln_s[3]))

                elif BFGS_MAG_DR_MAX in ln:
                    mag_dR_max.append(float(ln_s[3]))

                elif BFGS_S_MAX in ln:
                    s_max.append(float(ln_s[3]))

                elif TOT_TIME in ln:
                    tot_time = float(ln_s[-2])

        # Change to numpy arrays where sensible:
        scf_cycle_type = BFGS_CYCLE_TYPE_STR[scf_cycle_type_idx]
        bfgs_lambda = np.array(bfgs_lambda)
        final_energy = np.array(final_energy)
        final_fenergy = np.array(final_fenergy)
        final_zenergy = np.array(final_zenergy)
        dE_per_ion = np.array(dE_per_ion)
        mag_F_max = np.array(mag_F_max)
        mag_dR_max = np.array(mag_dR_max)
        s_max = np.array(s_max)
        all_constrained_forces = np.array(all_constrained_forces)
        all_unconstrained_forces = np.array(all_unconstrained_forces)
        all_constrained_symmetrised_forces = np.array(
            all_constrained_symmetrised_forces)
        real_lattice = np.array(real_lattice)
        recip_lattice = np.array(recip_lattice)
        lattice_params = np.array(lattice_params)
        cell_angles = np.array(cell_angles)
        cell_volume = np.array(cell_volume)
        cell_contents = np.array(cell_contents)
        species = np.array(species)
        species_idx = np.array(species_idx)

        # Constrained forces are repeated at the end of BFGS output in Final
        # config, so remove the last entry if geometry optimisation:
        if calc_type_str == 'geometry optimization':
            all_constrained_forces = all_constrained_forces[:-1]

        tot_time_hrs = tot_time / 3600

        params = {
            'calc_type':                calc_type_str,
            'cut_off_energy':           ecut,
            'fine_grid_size':           fine_grid,
            'num_electrons':            num_elec,
            'num_bands':                num_bands,
            'metallic':                 metallic,
            'net_charge':               net_charge,
            'elec_energy_tol':          elec_energy_tol,
            'kpoint_mp_grid':           kpoint_mp_grid,
            'kpoint_mp_offset':         kpoint_mp_offset,
            'kpoint_num':               kpoint_num,
            'sym_max_deviation':        sym_max_deviation,
            'sym_num_ops':              sym_num_ops,
            'ion_constraints_num':      ion_constraints_num,
            'point_group':              point_group,
            'space_group':              space_group,
            'cell_constraints':         cell_constraints,
            'cell_constraints_num':     cell_constraints_num,
            'finite_basis_correction':  finite_basis_correction,

        }

        out = {
            'params':                   params,
            'version':                  version,
            'scf':                      all_scf_data,
            'scf_num_cycles':           len(all_scf_data),
            'scf_cycle_type_idx':       scf_cycle_type_idx,
            'scf_cycle_type':           scf_cycle_type,
            'cell_contents':            cell_contents,
            'real_lattice':             real_lattice,
            'recip_lattice':            recip_lattice,
            'lattice_params':           lattice_params,
            'cell_angles':              cell_angles,
            'cell_volume':              cell_volume,
            'bfgs_lambda':              bfgs_lambda,
            'bfgs_enthalpy':            bfgs_enthalpy,
            'bfgs_num_iters':           bfgs_iter_idx + 1,
            'bfgs_dE_per_ion':          dE_per_ion,
            'bfgs_mag_F_max':           mag_F_max,
            'bfgs_mag_dR_max':          mag_dR_max,
            'bfgs_s_max':               s_max,
            'final_energy':             final_energy,
            'final_fenergy':            final_fenergy,
            'final_zenergy':            final_zenergy,
            'tot_time':                 tot_time,
            'tot_time_hrs':             tot_time_hrs,
            'num_ions':                 num_ions,
            'write_geom':               write_geom,
            'forces_constrained':       all_constrained_forces,
            'forces_unconstrained':     all_unconstrained_forces,
            'forces_constrained_sym':   all_constrained_symmetrised_forces,
            'species':                  species,
            'species_idx':              species_idx,
            'errors':                   errors,
            'warnings':                 warnings,
        }

        return out


def read_castep_geom_file(geom_path):
    """
        Function to parse a .geom geomtery trajectory file from a CASTEP geometry
        optimisation run. The .geom file includes the main results at the
        end of each BFGS step. All quantities in the .geom file are expressed in
        atomic units. Returned data is in eV (energies), Angs (ionic positions)
        and eV/Angs (ionic forces).

        Important note regarding returned dict:
            - Returned arrays of vectors are formed of **column** vectors.

        Tested with CASTEP versions:
            17.2,
            16.11

        Structure of .geom file:

        - HEADER, ends with "END header" line.

        - Blocks representing the end of each BFGS step, separated by a blank line

        - Block line #1 (ends in "<-- c")::
            BFGS iteration number (and an indication of which quantities are convergerd
            in later CASTEP versions).

        - Block line #2 (ends in "<-- E"):
            Final energy followed by final free energy.

        - Block lines #3,4,5
            Unit cell row vectors, Cartesian coordinates, units of Bohr radius.

        - Next three lines (if the cell is being relaxed; ends in "<-- S"):
            Stress vectors

        - Next n lines for n ions (starts with ion symbol and number; ends in "<-- R"):
            Ion positions, Cartesian coordinates, units of Bohr radius

        - Next n lines for n ions (starts with ion symbol and number; ends in "<-- F"):
            Ion forces, Cartesian coordinates, atomic units of force.

            Note: we assume that the ions are listed in the same order for ion positions
            and forces.

        - Note:
            I don't Understand how the energy and free energy values are calculated as
            reported in the .geom file. For metallic systems, this energy seems to be
            closest to the "Final free energy (E-TS)" reported at the end of the last
            SCF cycle. (but not exactly the same (given rounding)).

            In the .castep file, the "BFGS: Final Enthalpy     =" seems to match the
            "Final free energy (E-TS)" reported at the end of the last SCF cycle
            (given rounding).

    """

    # As used in CASTEP v16.11 and v17.2, physical constants from:
    #   [1] CODATA RECOMMENDED VALUES OF THE FUNDAMENTAL PHYSICAL CONSTANTS: 2010

    A_0_SI = 0.52917721092e-10      # m  [1]
    E_h_SI = 4.35974434e-18         # J  [1]
    E_h = 27.21138505               # eV [1]
    e_SI = 1.602176565e-19          # C  [1]
    F_SI = E_h_SI / A_0_SI          # N

    A_0 = A_0_SI * 1e10             # Angstrom
    F = F_SI * (1e-10 / e_SI)       # eV / Angstrom

    if not os.path.isfile(geom_path):
        raise IOError('File not found: {}'.format(geom_path))

    ITER_NUM = '<-- c'
    ENERGY = '<-- E'
    CELL = '<-- h'
    STRESS = '<-- S'
    POS = '<-- R'
    FORCE = '<-- F'

    bfgs_iter_idx = 0

    energies = []
    free_energies = []

    cell_ln_idx = 0
    all_cells = []
    current_cell = np.zeros((3, 3))

    stress_ln_idx = 0
    all_stresses = []
    current_stress = np.zeros((3, 3))

    all_ions = []
    all_forces = []
    current_bfgs_ions = []
    current_bfgs_forces = []

    species_set = False
    iter_num_line = False
    species = []
    species_idx = []

    force_ln_idx = 0

    with open(geom_path, 'r') as f:

        for ln in f:

            ln_s = ln.strip().split()

            # For CASTEP version 16.11, iteration number is a lone integer line
            #   - we assume there are no other lines which are a single integer
            # For CASTEP version 17.2, iteration number line has ITER_NUM in line
            if len(ln_s) == 1:

                try:
                    int(ln_s[0])
                    iter_num_line = True

                except ValueError as e:
                    pass

            if (ITER_NUM in ln) or iter_num_line:

                iter_num_line = False

                if bfgs_iter_idx > 0:

                    # Save the ion positions for the previous BFGS step:
                    all_ions.append(np.hstack(current_bfgs_ions))
                    current_bfgs_ions = []

                    all_forces.append(np.hstack(current_bfgs_forces))
                    current_bfgs_forces = []

                # No need to record the species and ion indices after the first BFGS block
                if bfgs_iter_idx == 1:
                    species_set = True

                bfgs_iter_idx += 1

            elif ENERGY in ln:
                energies.append(float(ln_s[0]))
                free_energies.append(float(ln_s[1]))

            elif CELL in ln:

                current_cell[cell_ln_idx] = [float(ln_s[i]) for i in [0, 1, 2]]

                if cell_ln_idx == 2:
                    cell_ln_idx = 0
                    all_cells.append(current_cell.T)
                    current_cell = np.zeros((3, 3))

                else:
                    cell_ln_idx += 1

            elif STRESS in ln:

                current_stress[stress_ln_idx] = [
                    float(ln_s[i]) for i in [0, 1, 2]]

                if stress_ln_idx == 2:
                    stress_ln_idx = 0
                    all_stresses.append(current_stress)
                    current_stress = np.zeros((3, 3))

                else:
                    stress_ln_idx += 1

            elif POS in ln:

                sp = ln_s[0]
                ion_idx = ln_s[1]

                if not species_set:

                    if sp not in species:
                        species.append(sp)

                    species_idx.append(species.index(sp))

                ion_xyz = [float(ln_s[i]) for i in [2, 3, 4]]
                current_bfgs_ions.append(np.array(ion_xyz)[:, np.newaxis])

            elif FORCE in ln:

                force_xyz = [float(ln_s[i]) for i in [2, 3, 4]]
                current_bfgs_forces.append(np.array(force_xyz)[:, np.newaxis])

        # Save the ion positions and forces for the final BFGS step:
        all_ions.append(np.hstack(current_bfgs_ions))
        all_forces.append(np.hstack(current_bfgs_forces))

    # Convert to numpy arrays where sensible
    energies = np.array(energies)
    free_energies = np.array(free_energies)
    all_cells = np.array(all_cells)
    all_stresses = np.array(all_stresses)
    species = np.array(species)
    species_idx = np.array(species_idx)
    ions = np.array(all_ions)
    forces = np.array(all_forces)

    # Convert units to eV and Angstroms:
    energies *= E_h
    free_energies *= E_h
    ions *= A_0
    all_cells *= A_0
    forces *= F

    out = {
        'energies':         energies,
        'free_energies':    free_energies,
        'cells':            all_cells,
        'cell_stresses':    all_stresses,
        'species_idx':      species_idx,
        'species':          species,
        'ions':             ions,
        'forces':           forces,
        'bfgs_num_iter':    len(energies)
    }

    return out


def read_cell_file(cellfile):
    """
    Read data from a castep .cell file.

    Parameters
    ----------
    cellfile : string
        The path and name of the .cell file to be read

    Returns
    -------
    lattice_data : dict of (str : ndarray or list or dict)
        `cell_vecs` : ndarray
            Array of shape (3, 3), where the row vectors are the three lattice
            vectors.
        `latt_params` : list
            List containing the lattice parameters of the unit cell,
            [a, b, c, α, β, γ], where the units of angles are radians.
        `motif` : dict
            `atom_sites` : ndarray
                Array of shape (3, n), where the column vectors are the fractional
                coordinates of the atoms and n is the number of atoms.
            `species` : list
                List of length n associated with each atom in `atom_sites`.


    Notes
    -----
    Currently only reads lattice data - lattice parameters and motif.


    """

    st_i = 0
    fin_i = 0
    with open(cellfile) as f:
        lines = f.readlines()
        cell_vecs = np.zeros((3, 3))
        pos_f_str = []
        species_str = []

        for ln_i, ln in enumerate(lines):
            if '%BLOCK lattice_cart' in ln:
                cell_vecs[0, :] = [float(x) for x in lines[ln_i + 2].split()]
                cell_vecs[1, :] = [float(x) for x in lines[ln_i + 3].split()]
                cell_vecs[2, :] = [float(x) for x in lines[ln_i + 4].split()]

            if '%BLOCK positions_frac' in ln:
                st_i = ln_i

            if '%ENDBLOCK positions_frac' in ln:
                end_i = ln_i
                break

        for ln in lines[st_i + 1:end_i]:
            species_str.append(ln.split()[0])
            pos_f_str.append(ln.split()[1:])

        # Fractional coordinates
        pos_f = np.asarray(pos_f_str, dtype=float)

        # Find lattice parameters
        a = np.linalg.norm(cell_vecs[0, :])
        b = np.linalg.norm(cell_vecs[1, :])
        c = np.linalg.norm(cell_vecs[2, :])
        α = np.arccos(np.dot(cell_vecs[1, :], cell_vecs[2, :]) / (
            np.linalg.norm(cell_vecs[1, :]) * np.linalg.norm(cell_vecs[2, :])))
        β = np.arccos(np.dot(cell_vecs[0, :], cell_vecs[2, :]) / (
            np.linalg.norm(cell_vecs[0, :]) * np.linalg.norm(cell_vecs[2, :])))
        γ = np.arccos(np.dot(cell_vecs[0, :], cell_vecs[1, :]) / (
            np.linalg.norm(cell_vecs[0, :]) * np.linalg.norm(cell_vecs[1, :])))

        latt_params = [a, b, c, α, β, γ]

        motif = {}
        motif['atom_sites'] = pos_f.T
        motif['species'] = species_str

        lattice_data = {
            'cell_vecs': cell_vecs,
            'latt_params': latt_params,
            'motif': motif
        }

        return lattice_data
