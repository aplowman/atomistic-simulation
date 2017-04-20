import os
import numpy as np

import readwrite

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
    all_cst_files = readwrite.find_files_in_dir(dir_path, r'.castep$', recursive=False)

    # If no .castep files in `dir_path, raise IOError.
    if len(all_cst_files) == 0:
        raise IOError('No .castep files found in directory {}'.format(dir_path))

    if seedname is None:

        if len(all_cst_files) > 1:
            raise IOError('Seedname not specified, but multiple .castep files found.')

        else:
            seedname = all_cst_files[0].split('.castep')[0]

    cst_fn = '{}.castep'.format(seedname)
    cst_path = os.path.join(dir_path, cst_fn)

    if not os.path.isfile(cst_path):
        raise IOError('File not found: {} in directory {}'.format(cst_fn, dir_path))

    # Parse the .castep file:
    out = read_castep_file(cst_path)

    # Parse additional ouput files:
    if out['write_geom']:

        # Parse the .geom file:
        geom_fn = '{}.geom'.format(seedname)
        geom_path = os.path.join(dir_path, geom_fn)

        if not ignore_missing_output and not os.path.isfile(geom_path):
            raise IOError('File not found: {} in directory {}'.format(geom_fn, dir_path))

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

    """

    TESTED_VERS = ['17.2']

    VERS = 'CASTEP version'
    PARAM_ECUT = 'plane wave basis set cut-off                   :'
    PARAM_NUM_ELEC = 'number of  electrons                           :'
    PARAM_NET_CHARGE = 'net charge of system                           :'
    PARAM_NUM_BANDS = 'number of bands                                :'
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
    SCF_FINAL_EN = 'Final energy, E             ='
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
    CON_FORCES_GO_START = '******************************** Constrained Forces ********************************'
    CON_FORCES_GO_END = '*                                                                                  *'
    CON_SYM_FORCES_GO_START = '************************** Constrained Symmetrised Forces **************************'
    CON_SYM_FORCES_GO_END = CON_FORCES_GO_END
    CON_SYM_FORCES_SP_START = '************** Constrained Symmetrised Forces **************'    
    CON_SYM_FORCES_SP_END = UNCON_FORCES_END
    BFGS_CYCLE_TYPE_STR = np.array(['Initial', 'Trial guess', 'Line minimization', 'Quad minimization'])
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

    version = None
    ecut = None
    num_elec = None
    num_bands = None
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

    bfgs_iter_idx = 0

    # Each BFGS iteration is associated with one or more SCF cycles:
    # Here, scf_cycle refers to a whole SCF convergence process to find the groundstate wavefunction
    scf_cycle_idx = 0
    scf_cycle_type_idx = [] # Index list where indices correspond to 0: 'initial, 1: 'trial guess', 2: 'line', 3: 'quad'
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

    SCF_HEADER_LNS = 3              # 3 lines between scf header and start of block
    FORCES_HEADER_LNS = 5           # 5 lines between forces header and start of block
    CELL_CONTENTS_HEADER_LNS = 3    # 3 lines between cell contents header and start of block
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

    with open(cst_path, 'r') as cst:

        for ln_idx, ln in enumerate(cst):

            ln_s = ln.strip().split()

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

            # Parse a unit cell block
            if mode == 'parse_cell':

                if cell_idx >= CELL_LAT_IDX_START and cell_idx <= CELL_LAT_IDX_END:

                    # Parse real and reciprocal lattice blocks. Theses are row
                    # vectors in the file, but we will return as column vectors.

                    rl_lat_ijk = [float(ln_s[i]) for i in [0,1,2]]
                    rc_lat_ijk = [float(ln_s[i]) for i in [3,4,5]]
                    current_real_lat.append(np.array(rl_lat_ijk)[:, np.newaxis])
                    current_recip_lat.append(np.array(rc_lat_ijk)[:, np.newaxis])

                    if cell_idx == CELL_LAT_IDX_END:

                        real_lattice.append(np.hstack(current_real_lat))
                        recip_lattice.append(np.hstack(current_recip_lat))

                        current_real_lat = []
                        current_recip_lat = []

                    cell_idx += 1

                elif cell_idx >= CELL_PARAMS_IDX_START and cell_idx <= CELL_PARAMS_IDX_END:

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

                elif cell_conts_idx > 0 and (CELL_CONTS_START_END in ln or CELL_CONTS_INI_START_END in ln):

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

                    ion_uvw = [float(ln_s[i]) for i in [3,4,5]]
                    current_cell_conts.append(np.array(ion_uvw)[:, np.newaxis])

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

                    scf_iter_data = np.ones(4) * np.nan

                    if scf_iter_idx == 0:
                        scf_iter_data[0:2] = [float(ln_s[i]) for i in [1,2]]
                        scf_iter_data[3] = float(ln_s[3])

                    else:
                        scf_iter_data[0:4] = [float(ln_s[i]) for i in [1,2,3,4]]

                    scf_cycle_data.append(scf_iter_data)
                    scf_iter_idx += 1

            # Parse a forces block:
            elif mode in ['parse_con_forces', 'parse_uncon_forces', 'parse_con_sym_forces']:

                if force_ion_idx < 0:
                    force_ion_idx += 1

                elif force_ion_idx > 0 and (UNCON_FORCES_END in ln or CON_FORCES_GO_END in ln):

                    # Finish parsing forces block
                    force_ion_idx = -FORCES_HEADER_LNS
                    current_bfgs_forces = np.hstack(current_bfgs_forces)

                    if mode == 'parse_con_forces':
                        all_constrained_forces.append(current_bfgs_forces)

                    elif mode == 'parse_uncon_forces':
                        all_unconstrained_forces.append(current_bfgs_forces)

                    elif mode == 'parse_con_sym_forces':
                        all_constrained_symmetrised_forces.append(current_bfgs_forces)

                    current_bfgs_forces = []
                    mode = 'scan'

                elif force_ion_idx >= 0:

                    force_xyz = [float(ln_s[i]) for i in [3, 4, 5]]
                    current_bfgs_forces.append(np.array(force_xyz)[:, np.newaxis])

                    force_ion_idx += 1

            elif mode == 'scan':

                if VERS in ln:
                    version = ln_s[7].split('|')[0]
                    if version not in TESTED_VERS:
                        raise NotImplementedError('Parser not tested on this version of CASTEP: {}'.format(version))

                elif PARAM_ECUT in ln:
                    ecut = float(ln_s[-2])

                elif PARAM_NUM_ELEC in ln:
                    num_elec = float(ln_s[-1])

                elif PARAM_NUM_BANDS in ln:
                    num_bands = int(ln_s[-1])

                elif PARAM_NET_CHARGE in ln:
                    net_charge = float(ln_s[-1])

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
                    cell_constraints = [int(ln_s[i]) for i in range(3,9)]

                elif SPACE_GRP in ln:
                    space_group = ln.split('=')[1].strip()

                elif FBC_RESULT in ln:
                    finite_basis_correction = float(ln_s[-1].split('eV')[0])

                elif CON_FORCES_GO_START in ln:
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

                elif SCF_FINAL_EN in ln:
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
        all_constrained_symmetrised_forces = np.array(all_constrained_symmetrised_forces)
        real_lattice = np.array(real_lattice)
        recip_lattice = np.array(recip_lattice)
        lattice_params = np.array(lattice_params)
        cell_angles = np.array(cell_angles)
        cell_volume = np.array(cell_volume)
        cell_contents = np.array(cell_contents)
        species = np.array(species)
        species_idx = np.array(species_idx)

        # Constrained forces are repeated at the end of BFGS output in Final config,
        # so remove the last entry:
        all_constrained_forces = all_constrained_forces[:-1]

        tot_time_hrs = tot_time / 3600

        params = {
            'cut_off_energy':           ecut,
            'num_electrons':            num_elec,
            'num_bands':                num_bands,
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
            'finite_basis_correction':  finite_basis_correction

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
            'species_idx':              species_idx

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

        Tested with CASTEP versions: 17.2

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

    """

    # Seems to be small discrepencies between ionic positions, forces in
    # GEOM file and those reported in .castep file. Need to figure this out.

    # Physical constants from:
    #   CRC Handbook of Chemistry and Physics, 95th ed. (2014-2015)

    # Atomic unit of length is the Bohr radius:
    A_0_SI = 0.52917721092e-10      # m
    E_h_SI = 4.35974434e-18         # J
    e_SI = 1.602176565e-19          # C
    F_SI = E_h_SI / A_0_SI          # N
    F_SI_w = 8.2387225e-8           # N
    F_SI_n = 8.23872336e-8

    A_0 = A_0_SI * 1e10             # Angstrom
    E_h = E_h_SI * e_SI             # eV
    F = F_SI * (1e-10 / e_SI)       # eV / Angstrom
    F_w = F_SI_w * (1e-10 / e_SI)   # eV / Angstrom
    F_n = F_SI_n * (1e-10 / e_SI)   # eV / Angstrom

    # Atomic unit of energy is the Hartree:
    E_h = 27.21138505   # eV

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
    current_cell = np.zeros((3,3))

    stress_ln_idx = 0
    all_stresses = []
    current_stress = np.zeros((3,3))

    all_ions = []
    all_forces = []
    current_bfgs_ions = []
    current_bfgs_forces = []

    species_set = False
    species = []
    species_idx = []

    force_ln_idx = 0

    with open(geom_path, 'r') as f:

        for ln in f:

            ln_s = ln.strip().split()

            if ITER_NUM in ln:

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

                current_cell[cell_ln_idx] = [float(ln_s[i]) for i in [0,1,2]]

                if cell_ln_idx == 2:
                    cell_ln_idx = 0
                    all_cells.append(current_cell.T)
                    current_cell = np.zeros((3,3))

                else:
                    cell_ln_idx += 1

            elif STRESS in ln:

                current_stress[stress_ln_idx] = [float(ln_s[i]) for i in [0,1,2]]

                if stress_ln_idx == 2:
                    stress_ln_idx = 0
                    all_stresses.append(current_stress)
                    current_stress = np.zeros((3,3))

                else:
                    stress_ln_idx += 1

            elif POS in ln:

                sp = ln_s[0]
                ion_idx = ln_s[1]

                if not species_set:

                    if sp not in species: species.append(sp)

                    species_idx.append(species.index(sp))

                ion_xyz = [float(ln_s[i]) for i in [2,3,4]]
                current_bfgs_ions.append(np.array(ion_xyz)[:, np.newaxis])

            elif FORCE in ln:

                force_xyz = [float(ln_s[i]) for i in [2,3,4]]
                current_bfgs_forces.append(np.array(force_xyz)[:, np.newaxis])


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
    forces *= F_n

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