import os
import numpy as np
import readwrite
from readwrite import format_arr as fmt_arr
from readwrite import replace_in_file, delete_line, add_line
import vectors
import utils
import shutil

SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
JS_TEMPLATE_DIR = os.path.join(SCRIPTS_PATH, 'set_up', 'jobscript_templates')


def write_jobscript(path, calc_paths, method, num_cores, sge, job_array,
                    scratch_os=None, scratch_path=None, parallel_env=None,
                    selective_submission=False, module_load=None,
                    job_name=None, seedname=None):
    """
    Write a jobscript file whose execution runs calculation input files.

    Parameters
    ----------
    path : str
        Directory in which to save the generated jobscript file.
    calc_paths : list of str
        Directories in which calculations are to be run.
    method : str
        Either 'castep' or 'lammps'
    num_cores : int
        Number of processor cores to use for the calculations.
    sge : bool
        If True, jobscript is generated for the SGE batch scheduler. If False,
        jobscript is generated to immediately run the calculations.
    job_array : bool
        Only applicable if `sge` is True. If True, calculations are submitted
        as an SGE job array. If False, calculations are submitted in one go. If
        the number of calculations is one (i.e. len(`calc_paths`) == 1), this
        will be set to False. Setting this to False can be handy for many small
        calculations which won't take a long time to complete. If submitted as
        a job array, a significant fraction of the total completion time may be
        queuing.
    scratch_os : str, optional
        Either 'nt' (Windows) or 'posix' (Unix-like, MacOS). The operating
        system on which the jobscript file will be executed. Default is to
        query to system on which this script is invoked i.e. `os.name`.
    scratch_path : str, optional
        Directory in which the jobscript is to be executed. Specify this path
        if the jobscript will be executed in a location different to the
        directory `path`, in which it is generated. By default, this is set to
        the same string as `path`.
    parallel_env : str, optional
        The SGE parallel environment on which to submit the calculations. Only
        applicable in `num_cores` > 1. Default is None.
    selective_submission : bool, optional
        Only applicable if `sge` is True. If True, the SGE task id flag `-t`
        [1] will be excluded from the jobscript file and instead this flag will
        be expected as a command line argument when executing the jobscript:
        e.g. "qsub jobscript.sh -t 1-10:2". Default is False.
    module_load : str, optional
        A string representing the path to a module to load within the
        jobscript. If specified, the statement "module load `module_load`" will
        be added to the top of the jobscript.
    job_name : str, optional
        Only applicable if `sge` is True. Default is None.
    seedname : str, optional
        Must be set if `method` is 'castep'.

    Returns
    -------
    None

    References
    ----------
    [1] http://gridscheduler.sourceforge.net/htmlman/htmlman1/qsub.html

    TODO:
    -   Add option for specifying parallel environment type for sge jobscripts

    """

    # General validation:

    if method == 'castep' and seedname is None:
        raise ValueError('`seedname` must be specified for CASTEP jobscripts.')

    if num_cores > 1 and parallel_env is None:
        raise ValueError('`parallel_env` must be set if `num_cores` > 1.')

    num_calcs = len(calc_paths)

    if num_cores <= 0:
        raise ValueError('Num cores not valid.')
    elif num_cores == 1:
        multi_type = 'serial'
    else:
        multi_type = 'parallel'

    if sge:
        sge_str = 'sge'
    else:
        sge_str = 'no_sge'

    if num_calcs == 1:
        job_array = False

    if job_array:
        job_arr_str = 'job_array'
    else:
        job_arr_str = 'single_job'

    if scratch_path is None:
        scratch_path = path

    if scratch_os is None:
        scratch_os = os.name

    if scratch_os == 'nt':
        scratch_path_sep = '\\'
    elif scratch_os == 'posix':
        scratch_path_sep = '/'

    # Get the template file name:
    tmp_fn = method + '_' + sge_str + '_' + \
        multi_type + '_' + scratch_os + '_' + job_arr_str + '.txt'

    # Get the template file path:
    tmp_path = os.path.join(JS_TEMPLATE_DIR, tmp_fn)

    # Write text file with all calc paths
    dirlist_fn = 'dir_list.txt'
    dir_list_path_stage = os.path.join(path, dirlist_fn)
    dir_list_path_scratch = scratch_path_sep.join([scratch_path, dirlist_fn])
    readwrite.write_list_file(dir_list_path_stage, calc_paths)

    if scratch_os == 'posix':
        js_ext = 'sh'
    elif scratch_os == 'nt':
        js_ext = 'bat'

    js_name = 'jobscript.' + js_ext

    # Copy template file to path
    js_path = os.path.join(path, js_name)
    shutil.copy(tmp_path, js_path)

    # Add module load to jobscript:
    if module_load is not None:
        add_line(js_path, 1, '')
        add_line(js_path, 2, 'module load {}'.format(module_load))

    # Make replacements in template file:
    replace_in_file(js_path, '<replace_with_dir_list>', dir_list_path_scratch)

    if multi_type == 'parallel':
        replace_in_file(js_path, '<replace_with_num_cores>', str(num_cores))
        replace_in_file(js_path, '<replace_with_pe>', parallel_env)

    if sge:
        if job_name is not None:
            replace_in_file(js_path, '<replace_with_job_name>', job_name)
        else:
            delete_line(js_path, '<replace_with_job_name>')

        if selective_submission:
            delete_line(js_path, '#$ -t')
        else:
            replace_in_file(js_path, '<replace_with_job_index_range>',
                            '1-' + str(num_calcs))

    if method == 'castep':
        replace_in_file(js_path, '<replace_with_seed_name>', seedname)

    # For `method` == 'lammps', `sge` == True, `job_array` == False, we need
    # a helper jobscript, called by the SGE jobscript:
    if method == 'lammps' and sge and not job_array:

        if multi_type != 'serial' or scratch_os != 'posix':
            raise NotImplementedError('Jobscript parameters not supported.')

        help_tmp_path = os.path.join(
            JS_TEMPLATE_DIR, 'lammps_no_sge_serial_posix_single_job.txt')

        help_js_path = os.path.join(path, 'lammps_single_job.sh')
        shutil.copy(help_tmp_path, help_js_path)
        replace_in_file(help_js_path, '<replace_with_dir_list>', dir_list_path)

    # Make a directory for job-related output. E.g. .o and .e files from CSF.
    os.makedirs(os.path.join(path, 'output'))


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

    if lengths_equal == 'none':
        lengths_equal == ''

    if angles_equal == 'none':
        angles_equal == ''

    if fix_lengths == 'none':
        fix_lengths == ''

    if fix_angles == 'none':
        fix_angles == ''

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


def write_lammps_atoms(supercell, atom_sites, species, species_idx, path):
    """
    Write file defining atoms for a LAMMPS simulation, using the
    `atomic` atom style.

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
        Directory in which to generate the atoms file.

    Returns
    -------
    None

    Notes
    -----
    Generates atom file in the `atomic` atom style so columns of the body of
    the generated file are: `atom-ID`, `atom-type`, `x`, `y`, `z`.

    """

    num_atoms = atom_sites.shape[1]
    num_atom_types = len(species)

    xhi = supercell[0, 0]
    yhi = supercell[1, 1]
    zhi = supercell[2, 2]
    xy = supercell[0, 1]
    xz = supercell[0, 2]
    yz = supercell[1, 2]

    atom_id = np.arange(1, num_atoms + 1)[:, np.newaxis]
    atom_type = (species_idx + 1)[:, np.newaxis]

    os.makedirs(path, exist_ok=True)
    atom_path = os.path.join(path, 'atoms.lammps')

    with open(atom_path, 'w', newline='') as at_f:

        at_f.write('Header\n\n')

        at_f.write('{:d} atoms\n'.format(num_atoms))
        at_f.write('{:d} atom types\n\n'.format(num_atom_types))

        at_f.write('0.0 {:14.9f} xlo xhi\n'.format(xhi))
        at_f.write('0.0 {:14.9f} ylo yhi\n'.format(yhi))
        at_f.write('0.0 {:14.9f} zlo zhi\n'.format(zhi))
        at_f.write('\n')

        tilt_str = '{:<24.15f} {:<24.15f} {:<24.15f} xy xz yz'
        at_f.write(tilt_str.format(xy, xz, yz))
        at_f.write('\n\n')

        at_f.write('Atoms\n\n')

        fmt_sp = ['{:>8d}', '{:>5d}', '{:>24.15f}']
        arr_fm = readwrite.format_arr([atom_id, atom_type, atom_sites.T],
                                      format_spec=fmt_sp)
        at_f.write(arr_fm)


def write_lammps_inputs(supercell, atom_sites, species, species_idx, path,
                        potential_path, potential_type, potential_species,
                        atom_constraints=None, computes=None, thermos_dt=1,
                        dump_dt=1):
    """
    Generate LAMMPS input files for energy minimisation of a 3D periodic
    supercell.

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
    potential_path : str
        Absolute path to the potential file.
    potential_type : str
        Type of potential which `potential_path` points to.
    potential_species : str
        Species which is modelled by the potential which `potential_path`
        points to.
    atom_constraints : dict, optional
        A dict with the following keys:
            fix_xy_idx : list or ndarray of dimension 1
                The atom indices whose `x` and `y` coordinates are to
                be fixed. By default, set to None.
            fix_xyz_idx : list or ndarray of dimension 1
                The atom indices whose `x`, `y` and `z` coordinates
                are to be fixed. By default, set to None.
    computes : list of str, optional
        A list of quantities to compute during the simulation. If not
        specified, set to a list with elements: `pe/atom`, `displace/atom`,
        `voronoi/atom`.
    thermos_dt : int, optional
        After this number of timesteps, ouput thermodynamics to the log file.
    dump_dt : int, optional
        After this number of timesteps, output dump file containing atom
        positions.

    Notes
    -----
    See [1] for an explanation of the LAMMPS input script syntax.

    References
    ----------
    [1] http://lammps.sandia.gov/doc/Section_commands.html

    TODO:
    -   Implement cell constraints, see:
        http://lammps.sandia.gov/doc/fix_box_relax.html

    """

    computes_info = {
        'pe/atom': {
            'name': 'peatom',
            'dims': 1,
            'fmt': ['%20.10f'],
        },
        'displace/atom': {
            'name': 'datom',
            'dims': 4,
            'fmt': ['%20.10f'] * 4,
        },
        'voronoi/atom': {
            'name': 'voratom',
            'dims': 2,
            'fmt': ['%20.10f', '%5.f'],
        }
    }

    if computes is None:
        computes = ['pe/atom', 'displace/atom', 'voronoi/atom']

    # Validation
    for c in computes:
        if computes_info.get(c) is None:
            raise NotImplementedError(
                'Compute "{}" is not understood.'.format(c))

    # Write file defining atom positions:
    write_lammps_atoms(supercell, atom_sites, species, species_idx, path)

    command_lns = [
        'units        metal',
        'dimension    3',
        'boundary     p p p',
        'atom_style   atomic',
        'box          tilt large',
        'read_data    atoms.lammps',
        'pair_style   {}'.format(potential_type),
        'pair_coeff   * * "{}" {}'.format(potential_path, potential_species)
    ]

    fix_lns = []
    # Atom constraints
    fix_xy_idx = atom_constraints.get('fix_xy_idx')
    fix_xyz_idx = atom_constraints.get('fix_xyz_idx')

    if fix_xy_idx is not None:

        nfxy = len(fix_xy_idx)
        if nfxy == atom_sites.shape[1]:
            fxy_grp = 'all'
        else:
            fxy_grp = 'fix_xy'
            fxy_grp_ln = 'group {} id '.format(fxy_grp)
            fxy_grp_ln += ('{:d} ' * nfxy).format(*fix_xy_idx)
            fix_lns.append(fxy_grp_ln)

        fix_lns.append('fix 1 {} setforce 0.0 0.0 NULL'.format(fxy_grp))

    if fix_xyz_idx is not None:

        nfxyz = len(fix_xyz_idx)
        if nfxyz == atom_sites.shape[1]:
            fxyz_grp = 'all'
        else:
            fxyz_grp = 'fix_xyz'
            fxyz_grp_ln = 'group {} id '.format(fxyz_grp)
            fxyz_grp_ln += ('{:d} ' * nfxyz).format(*fix_xyz_idx)
            fix_lns.append(fxyz_grp_ln)

        fix_lns.append('fix 2 {} setforce 0.0 0.0 0.0'.format(fxy_grp))

    # computes are used in the dump files
    dmp_computes = ''
    dmp_fmt = ''
    compute_lns = []
    for c in computes:

        cinf = computes_info[c]
        c_nm = cinf['name']
        c_dm = cinf['dims']
        c_fm = cinf['fmt']
        compute_lns.append('compute {} all {}'.format(c_nm, c))

        if c_dm == 1:
            dmp_computes += ' c_{}'.format(c_nm)
            dmp_fmt += ' {}'.format(c_fm[0])

        else:
            for i in range(c_dm):
                dmp_computes += ' c_{}['.format(c_nm) + '{:d}]'.format(i + 1)
                dmp_fmt += ' {}'.format(c_fm[i])

    # thermo prints info to the log file
    thermo_args = ['step', 'atoms', 'pe', 'ke', 'etotal']
    thermo_args_all = ' '.join(thermo_args)
    thermo_lns = [
        'thermo_style custom {}'.format(thermo_args_all),
        'thermo_modify format float %20.10f',
        'thermo {:d}'.format(thermos_dt)
    ]

    dump_str = 'dump 1 all custom {} dump.*.txt id type x y z'.format(dump_dt)
    dump_str += dmp_computes
    dump_mod = 'dump_modify 1 format "%5d %5d %20.10f %20.10f %20.10f'
    dump_mod += dmp_fmt + '"'
    dump_lns = [dump_str, dump_mod]

    # Run simulation

    # minimize args:
    #   energy_tol: energy change / energy magnitude
    #   force tol: ev / ang for units = metal
    #   max iterations:
    #   max force/energy evaluations:
    max_iters = int(1e4)
    max_force_per_energy_evals = int(1e5)
    sim_lns = []
    sim_lns.append('min_style cg')
    sim_lns.append('minimize 0.0 1.0e-6 {} {}'.format(
        str(max_iters),
        str(max_force_per_energy_evals)
    ))

    all_lns = [command_lns, fix_lns, compute_lns, thermo_lns,
               dump_lns, sim_lns]

    # Write all lines to input script file
    os.makedirs(path, exist_ok=True)
    in_path = os.path.join(path, 'in.lammps')
    with open(in_path, 'w', newline='') as in_f:
        for i_idx, i in enumerate(all_lns):
            if i_idx > 0:
                in_f.write('\n')
            in_f.writelines([j + '\n' for j in i])


def read_lammps_log(path):
    """"""

    # Map `thermo_style` args to how they appears in the thermo output,
    # and their data type.
    thermo_map = {
        'step': {
            'name': 'Step',
            'dtype': int
        },
        'atoms': {
            'name': 'Atoms',
            'dtype': int
        },
        'pe': {
            'name': 'PotEng',
            'dtype': float
        },
        'ke': {
            'name': 'KinEng',
            'dtype': float
        },
        'etotal': {
            'name': 'TotEng',
            'dtype': float
        }
    }

    DUMP = 'dump'
    THERMO_STYLE = 'thermo_style'
    THERMO_OUT_END = 'Loop'
    WARN = 'WARNING:'
    ERR = 'ERROR:'
    VERS = 'LAMMPS'

    dump_name = None
    thermo_style_args = None
    thermo_style_out = None
    vers = None

    warns = []
    errs = []

    with open(path, 'r', encoding='utf-8', newline='') as lf:

        mode = 'scan'
        for ln in lf:

            ln = ln.strip()
            ln_s = ln.split()

            if ln == '':
                continue

            if mode == 'scan':

                if VERS in ln:
                    vers = ln.split('(')[1].split(')')[0]

                if ln_s[0] == DUMP:
                    dump_name = ln_s[5]

                elif ln_s[0] == THERMO_STYLE:
                    thermo_style_args = ln_s[2:]
                    tm = thermo_map
                    thermo_style_out = [tm[i]['name']
                                        for i in thermo_style_args]
                    thermo_out = {i: [] for i in thermo_style_out}

                if thermo_style_out is not None:
                    if ln_s[0] == thermo_style_out[0]:
                        mode = 'thermo'
                        continue

                if WARN in ln:
                    warns.append(ln)

                if ERR in ln:
                    errs.append(ln)

            if mode == 'thermo':
                if ln_s[0] == THERMO_OUT_END:
                    mode = 'scan'
                else:
                    for i_idx, i in enumerate(thermo_style_out):
                        thermo_out[i].append(ln_s[i_idx])

    # Parse thermo as correct dtypes
    for k, v in thermo_out.items():
        dt = thermo_map[thermo_style_args[thermo_style_out.index(k)]]['dtype']
        thermo_out[k] = np.array(v, dtype=dt)

    out = {
        'version': vers,
        'warnings': warns,
        'errors': errs,
        'thermo': thermo_out,
        'dump_name': dump_name
    }

    return out


def read_lammps_dump(path):
    """
    Read a LAMMPS dump file.

    Notes
    -----
    This is not generalised, in terms of the fields present in the `ATOMS`
    block, but could be developed to be more-generalised in the future.

    """

    # Search strings
    TS = 'ITEM: TIMESTEP'
    NUMATOMS = 'ITEM: NUMBER OF ATOMS'
    BOX = 'ITEM: BOX BOUNDS'
    ATOMS = 'ITEM: ATOMS'
    TILT_FACTORS = 'xy xz yz'

    ts = None
    num_atoms = None
    box_tilt = False
    box_periodicity = None
    box = []
    atom_sites = None
    atom_types = None
    atom_disp = None
    atom_pot = None
    vor_vols = None
    vor_faces = None

    out = {}
    with open(path, 'r', encoding='utf-8', newline='') as df:
        mode = 'scan'
        for ln in df:

            ln = ln.strip()
            ln_s = ln.split()

            if TS in ln:
                mode = 'ts'
                continue

            elif NUMATOMS in ln:
                mode = 'num_atoms'
                continue

            elif BOX in ln:
                mode = 'box'
                box_ln_idx = 0
                box_periodicity = [ln_s[-i] for i in [3, 2, 1]]
                if TILT_FACTORS in ln:
                    box_tilt = True
                continue

            elif ATOMS in ln:
                mode = 'atoms'
                headers = ln_s[2:]

                x_col = headers.index('x')
                y_col = headers.index('y')
                z_col = headers.index('z')

                atom_type_col = headers.index('type')
                vor_vol_col = headers.index('c_voratom[1]')
                vor_face_col = headers.index('c_voratom[2]')
                d1c = headers.index('c_datom[1]')
                d2c = headers.index('c_datom[2]')
                d3c = headers.index('c_datom[3]')
                d4c = headers.index('c_datom[4]')
                pot_col = headers.index('c_peatom')

                atom_ln_idx = 0
                atom_sites = np.zeros((3, num_atoms))
                atom_types = np.zeros((num_atoms,), dtype=int)
                atom_pot = np.zeros((num_atoms,))
                atom_disp = np.zeros((4, num_atoms))
                vor_vols = np.zeros((num_atoms,))
                vor_faces = np.zeros((num_atoms,), dtype=int)

                continue

            if mode == 'ts':
                ts = int(ln)
                mode = 'scan'

            elif mode == 'num_atoms':
                num_atoms = int(ln)
                mode = 'scan'

            elif mode == 'box':
                box.append([float(i) for i in ln_s])
                box_ln_idx += 1
                if box_ln_idx == 3:
                    mode = 'scan'

            elif mode == 'atoms':

                atom_sites[:, atom_ln_idx] = [
                    float(i) for i in (ln_s[x_col], ln_s[y_col], ln_s[z_col])]

                atom_disp[:, atom_ln_idx] = [
                    float(i) for i in [ln_s[j] for j in (d1c, d2c, d3c, d4c)]]

                atom_pot[atom_ln_idx] = float(ln_s[pot_col])
                atom_types[atom_ln_idx] = int(ln_s[atom_type_col])
                vor_vols[atom_ln_idx] = float(ln_s[vor_vol_col])
                vor_faces[atom_ln_idx] = int(ln_s[vor_face_col])

                atom_ln_idx += 1
                if atom_ln_idx == num_atoms:
                    mode = 'scan'

    # Form supercell edge vectors as column vectors:
    supercell = np.array([
        [box[0][1] - box[0][0], box[0][2], box[1][2]],
        [0, box[1][1] - box[1][0], box[2][2]],
        [0, 0, box[2][1] - box[2][0]]
    ])

    out = {
        'time_step': ts,
        'num_atoms': num_atoms,
        'box_tilt': box_tilt,
        'box_periodicity': box_periodicity,
        'box': box,
        'supercell': supercell,
        'atom_sites': atom_sites,
        'atom_types': atom_types,
        'atom_pot_energy': atom_pot,
        'atom_disp': atom_disp,
        'vor_vols': vor_vols,
        'vor_faces': vor_faces
    }

    # print('ts: \n{}\n'.format(ts))
    # print('num_atoms: \n{}\n'.format(num_atoms))
    # print('box_tilt: \n{}\n'.format(box_tilt))
    # print('box_periodicity: \n{}\n'.format(box_periodicity))
    # print('box: \n{}\n'.format(box))
    # print('supercell: \n{}\n'.format(supercell))
    # print('atom_sites: \n{}\n'.format(atom_sites))
    # print('atom_pot: \n{}\n'.format(atom_pot))
    # print('atom_types: \n{}\n'.format(atom_types))
    # print('atom_disp: \n{}\n'.format(atom_disp))
    # print('vor_vols: \n{}\n'.format(vor_vols))
    # print('vor_faces: \n{}\n'.format(vor_faces))

    return out


def read_lammps_output(dir_path, log_name='log.lammps'):
    """

    References
    ----------
    [1] https://technet.microsoft.com/en-us/library/gg440701.aspx

    """

    # Get the format of dummp files from the log file
    log_path = os.path.join(dir_path, log_name)
    log_out = read_lammps_log(log_path)
    dump_name = log_out['dump_name']

    # Convert * wildcard in `dump_name` to regex (see ref. [1]):
    dump_name = dump_name.replace('.', '\.').replace('*', '.*')
    dmp_fns = readwrite.find_files_in_dir(dir_path, dump_name)

    """

            'time_step': ts,
            'num_atoms': num_atoms,
            'box_tilt': box_tilt,
            'box_periodicity': box_periodicity,
            'box': box,
            'supercell': supercell,
            'atom_sites': atom_sites,
            'atom_types': atom_types,
            'atom_pot_energy': atom_pot,
            'atom_disp': atom_disp,
            'vor_vols': vor_vols,
            'vor_faces': vor_faces
    """

    all_dmps = {}
    atoms = []
    atom_disp = []
    atom_pot_energy = []
    vor_vols = []
    vor_faces = []
    supercell = []
    time_steps = []

    for dfn in dmp_fns:
        dmp_path = os.path.join(dir_path, dfn)
        dmp_i = read_lammps_dump(dmp_path)
        dmp_ts = dmp_i['time_step']
        all_dmps.update({dmp_ts: dmp_i})
        atoms.append(dmp_i['atom_sites'])
        atom_disp.append(dmp_i['atom_disp'])
        atom_pot_energy.append(dmp_i['atom_pot_energy'])
        vor_vols.append(dmp_i['vor_vols'])
        vor_faces.append(dmp_i['vor_faces'])
        supercell.append(dmp_i['supercell'])
        time_steps.append(dmp_i['time_step'])

    atoms = np.array(atoms)
    atom_disp = np.array(atom_disp)
    atom_pot_energy = np.array(atom_pot_energy)
    vor_vols = np.array(vor_vols)
    vor_faces = np.array(vor_faces)
    supercell = np.array(supercell)
    time_steps = np.array(time_steps)

    final_energy = log_out['thermo']['TotEng']

    out = {
        **log_out,
        'dumps': all_dmps,
        'atoms': atoms,
        'atom_disp': atom_disp,
        'atom_pot_energy': atom_pot_energy,
        'vor_vols': vor_vols,
        'vor_faces': vor_faces,
        'supercell': supercell,
        'time_steps': time_steps,
        'final_energy': final_energy
    }

    return out


def write_castep_inputs(supercell, atom_sites, species, species_idx, path,
                        seedname='sim', cell=None, param=None,
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
            fix_xy_idx : list or ndarray of dimension 1
                The atom indices whose `x` and `y` coordinates are to
                be fixed. By default, set to None.
            fix_xyz_idx : list or ndarray of dimension 1
                The atom indices whose `x`, `y` and `z` coordinates
                are to be fixed. By default, set to None.

    TODO:
    -   Generalise atom constraints.

    """

    # Validation:

    species_idx = utils.parse_as_int_arr(species_idx)
    if species_idx.min() < 0 or species_idx.max() > (atom_sites.shape[1] - 1):
        raise IndexError('`species_idx` must index `atom_sites`'.format(k))

    cell_cnst_def = {
        'lengths_equal': '',
        'angles_equal': '',
        'fix_lengths': '',
        'fix_angles': ''
    }

    atom_cnst_def = {
        'fix_xy_idx': None,
        'fix_xyz_idx': None
    }

    if cell_constraints is None:
        cell_constraints = cell_cnst_def
    else:
        cell_constraints = {**cell_cnst_def, **cell_constraints}

    if atom_constraints is None:
        atom_constraints = atom_cnst_def
    else:
        atom_constraints = {**atom_cnst_def, **atom_constraints}

    for k, v in atom_constraints.items():

        if isinstance(v, (np.ndarray, list)):

            atom_constraints[k] = utils.parse_as_int_arr(v)
            v = atom_constraints[k]

            if v.ndim != 1:
                raise ValueError('`atom_constraints[{}]` must be a 1D list, '
                                 '1D array or str.'.format(k))

            if v.min() < 0 or v.max() > (atom_sites.shape[1] - 1):
                raise IndexError('`atom_constraints[{}]` must index '
                                 '`atom_sites`'.format(k))

        elif v is not None:
            raise ValueError('`atom_constraints[{}]` must be a 1D list or 1D '
                             'array.'.format(k))

    f_xy = atom_constraints.get('fix_xy_idx')
    f_xyz = atom_constraints.get('fix_xyz_idx')

    if f_xy is None:
        f_xy = np.array([])
    if f_xyz is None:
        f_xyz = np.array([])

    if len(f_xyz) > 0 and len(f_xy) > 0:
        if len(np.intersect1d(f_xyz, f_xy)) > 0:
            raise ValueError('`fix_xyz_idx` and `fix_xy_idx` cannot '
                             'contain the same indices.')

    os.makedirs(path, exist_ok=True)

    # Write CELL file:
    cell_path = os.path.join(path, seedname + '.cell')
    with open(cell_path, 'w') as cf:   # to do: test if i need universal newline mode here

        # Supercell (need to tranpose to array of row vectors):
        cf.write('%block lattice_cart\n')
        cf.write(fmt_arr(supercell.T, format_spec='{:24.15f}', col_delim=' '))
        cf.write('%endblock lattice_cart\n\n')

        # Atoms (need to tranpose to array of row vectors):
        atom_species = species[species_idx][:, np.newaxis]

        cf.write('%block positions_abs\n')
        cf.write(fmt_arr([atom_species, atom_sites.T],
                         format_spec=['{:5}', '{:24.15f}'],
                         col_delim=' '))
        cf.write('%endblock positions_abs\n')

        # Cell constraints:
        encoded_params = get_castep_cell_constraints(**cell_constraints)

        if not (encoded_params[0] == [1, 2, 3] and
                encoded_params[1] == [4, 5, 6]):

            if (encoded_params[0] == [0, 0, 0] and
                    encoded_params[1] == [0, 0, 0]):
                cf.write('\nfix_all_cell = True\n')

            else:
                cf.write('\n%block cell_constraints\n')
                cf.write('{}\t{}\t{}\n'.format(*encoded_params[0]))
                cf.write('{}\t{}\t{}\n'.format(*encoded_params[1]))
                cf.write('%endblock cell_constraints\n')

        # Atom constraints:
        if len(f_xyz) > 0 or len(f_xy) > 0:

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

            if nc_xyz > 0:

                f_xyz_sp = np.tile(atom_species[f_xyz], (3, 1))
                f_xyz_sub_idx = np.repeat(sub_idx[f_xyz], 3)[:, np.newaxis]
                f_xyz_cnst_idx = (np.arange(nc_xyz * 3) + 1)[:, np.newaxis]
                f_xyz_cnst_coef = np.tile(np.eye(3), (nc_xyz, 1))

                cnst_arrs_xyz = [f_xyz_cnst_idx, f_xyz_sp, f_xyz_sub_idx,
                                 f_xyz_cnst_coef]

                cf.write(fmt_arr(cnst_arrs_xyz,
                                 format_spec=cnst_fs,
                                 col_delim=' '))

            if nc_xy > 0:

                f_xy_sp = np.tile(atom_species[f_xy], (2, 1))
                f_xy_sub_idx = np.repeat(sub_idx[f_xy], 2)[:, np.newaxis]
                f_xy_cnst_idx = (np.arange(nc_xy * 2) + 1 +
                                 (nc_xyz * 3))[:, np.newaxis]
                f_xy_cnst_coef = np.tile(np.eye(3)[:2], (nc_xy, 1))

                cnst_arrs_xy = [f_xy_cnst_idx, f_xy_sp, f_xy_sub_idx,
                                f_xy_cnst_coef]

                cf.write(fmt_arr(cnst_arrs_xy,
                                 format_spec=cnst_fs,
                                 col_delim=' '))

            cf.write('%endblock ionic_constraints\n')

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
    if out['write_geom']:

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
    -   Parse SCF warning lines like in: "\2017-04-01-2206_64626\calcs\0.360\"

    """

    TESTED_VERS = ['17.2']

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

    SCF_HEADER_LNS = 3              # 3 lines between scf header and start of block
    FORCES_HEADER_LNS = 5           # 5 lines between forces header and start of block
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

                    ion_uvw = [float(ln_s[i]) for i in [3, 4, 5]]
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

                elif force_ion_idx > 0 and (UNCON_FORCES_END in ln or CON_FORCES_GO_END in ln):

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

                if VERS in ln:
                    version = ln_s[7].split('|')[0]
                    if version not in TESTED_VERS:
                        raise NotImplementedError(
                            'Parser not tested on this version of CASTEP: {}'.format(version))

                elif CALC_TYPE in ln:
                    calc_type_str = ln.split(':')[1].strip()

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
        }

        return out


def process_castep_out(out):
    """Some simple calculations for convenience."""

    def get_rms_force(forces):
        """
        Parameters
        ----------
        forces : ndarray of shape (M, 3, N)
            Array representing the force components on N atoms 
            for M steps.
        """
        if len(forces) == 0:
            return None

        forces_rshp = forces.reshape(forces.shape[0], -1)
        forces_rms = np.sqrt(np.mean(forces_rshp ** 2, axis=1))

        return forces_rms

    n = out['num_ions']

    # Energies per atom:
    final_energy_pa = out['final_energy'] / n
    final_fenergy_pa = out['final_fenergy'] / n
    final_zenergy_pa = out['final_zenergy'] / n

    # RMS forces
    forces_cons_rms = get_rms_force(out['forces_constrained'])
    forces_uncons_rms = get_rms_force(out['forces_unconstrained'])
    forces_cons_sym_rms = get_rms_force(out['forces_constrained_sym'])

    # Nicely formatted time
    d = out['tot_time'] / (24 * 60 * 60)
    h = (d - np.floor(d)) * 24
    m = (h - np.floor(h)) * 60
    s = np.round((m - np.floor(m)) * 60, decimals=0)
    t = [int(np.floor(i)) for i in [d, h, m]]
    time_strs = ['days', 'hrs', 'mins']
    time_fmt = ''.join(['{} {} '.format(i, j)
                        for i, j in zip(t, time_strs) if i > 0])
    time_fmt += '{:.0f} sec'.format(s)

    out.update({
        'final_energy_pa': final_energy_pa,
        'final_fenergy_pa': final_fenergy_pa,
        'final_zenergy_pa': final_zenergy_pa,
        'forces_cons_rms': forces_cons_rms,
        'forces_uncons_rms': forces_uncons_rms,
        'forces_cons_sym_rms': forces_cons_sym_rms,
        'time_fmt': time_fmt,
    })


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


def get_LAMMPS_compatible_box(box_cart):
    """
        `box_cart` is an array of column vectors
        Returns array of column vectors.

        LAMMPS requirements:
            -   simulation cell vectors must form a right-handed basis
            -   1st vector lies along x-axis
            -   1st and 2nd vectors lie in xy-plane
            -   simulation cell is specified with numbers: xhi,xlo,yhi,hlo,zhi,zlo,xy,xz,yz

            a = (xhi-xlo, 0, 0)
            b = (xy, yhi-hlo, 0)
            c = (xz, yz, zhi-zlo)

            -   xy,xz,yz are "tilt factors", which by default may not be larger than 0.5 * x in xy
                0.5 * y in yz and 0.5 * x in xz. Default can be turned off by "box tilt large"

            -   Let the original supercell vectors be column vectors A, B and C in `box_cart`.
            -   Let γ be the angle between A and B, and β be the angle between C and A.
            -   New vectors are a,b,c.
    """

    A = box_cart[:, 0:1]
    B = box_cart[:, 1:2]
    C = box_cart[:, 2:3]

    A_mag, B_mag, C_mag = np.linalg.norm(
        A), np.linalg.norm(B), np.linalg.norm(C)

    cos_γ = vectors.col_wise_cos(A, B)[0]
    sin_γ = vectors.col_wise_sin(A, B)[0]
    cos_β = vectors.col_wise_cos(C, A)[0]

    a_x = A_mag
    a_y = 0
    a_z = 0

    b_x = B_mag * cos_γ
    b_y = B_mag * sin_γ
    b_z = 0

    c_x = C_mag * cos_β
    c_y = (vectors.col_wise_dot(B, C)[0] - (b_x * c_x)) / b_y
    c_z = np.sqrt(C_mag**2 - c_x**2 - c_y**2)

    return np.array([
        [a_x, a_y, a_z],
        [b_x, b_y, b_z],
        [c_x, c_y, c_z]]).T
