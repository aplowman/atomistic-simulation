import os
import numpy as np
from atsim import vectors
from atsim import readwrite
from atsim.readwrite import format_arr


def write_lammps_atoms(supercell, atom_sites, species, species_idx, path, atom_style, charges=None):
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
    atom_style : str ("atomic" or "full")
        Corresponds to the LAMMPs command `atom_style`. Determines which
        columns are neccessary in writing the atom data.
    charges : list or ndarray of float of shape (M, ), optional
        The charge associated with each species type. Only used if `atom_style`
        is "full".

    Returns
    -------
    None

    Notes
    -----
    For `atom_style` "atomic", output columns of the body of the generated file
    are: `atom-ID`, `atom-type`, `x`, `y`, `z`. For `atom_style` "full",
    output columns are: `atom-ID`, `molecule-ID`, `atom-type`, `q` (charge),
    `x`, `y`, `z`.


    """

    if isinstance(charges, list):
        charges = np.array(charges)

    # Validation
    all_atom_styles = ['full', 'atomic']
    if atom_style not in all_atom_styles:
        raise ValueError('Atom style: "{}" not understood. Must be one of '
                         ': {}'.format(all_atom_styles))

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

    if atom_style == 'full':
        mol_id = np.zeros((num_atoms, 1), dtype=int)
        ch = charges[species_idx][:, np.newaxis]

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

        if atom_style == 'atomic':
            fmt_sp = ['{:>8d}', '{:>5d}', '{:>24.15f}']
            arrs = [atom_id, atom_type, atom_sites.T]

        elif atom_style == 'full':
            fmt_sp = ['{:>8d}', '{:>5d}', '{:>5d}', '{:>15.10f}', '{:>24.15f}']
            arrs = [atom_id, mol_id, atom_type, ch, atom_sites.T]

        arr_fm = format_arr(arrs, format_spec=fmt_sp, col_delim=' ')
        at_f.write(arr_fm)


def write_lammps_inputs(supercell, atom_sites, species, species_idx, path,
                        parameters, interactions, atoms_file, atom_style,
                        atom_constraints=None, cell_constraints=None,
                        computes=None, thermos_dt=1, dump_dt=1, charges=None):
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
    atom_style : str ("atomic" or "full")
        Corresponds to the LAMMPs command `atom_style`. Determines which
        columns are neccessary in writing the atom data.
    atom_constraints : dict, optional
        A dict with the following keys:
            fix_xy_idx : list or ndarray of dimension 1
                The atom indices whose `x` and `y` coordinates are to
                be fixed. By default, set to None.
            fix_xyz_idx : list or ndarray of dimension 1
                The atom indices whose `x`, `y` and `z` coordinates
                are to be fixed. By default, set to None.
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
    computes : list of str, optional
        A list of quantities to compute during the simulation. If not
        specified, set to a list with elements: `pe/atom`, `displace/atom`,
        `voronoi/atom`.
    thermos_dt : int, optional
        After this number of timesteps, ouput thermodynamics to the log file.
    dump_dt : int, optional
        After this number of timesteps, output dump file containing atom
        positions.
    charges : list or ndarray of float of shape (M, ), optional
        The charge associated with each species type. Only used if `atom_style`
        is "full".

    Notes
    -----
    See [1] for an explanation of the LAMMPS input script syntax.

    References
    ----------
    [1] http://lammps.sandia.gov/doc/Section_commands.html

    TODO:
    -   Add stress compute
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
        },
    }

    if computes is None:
        computes = ['pe/atom', 'displace/atom', 'voronoi/atom']

    # Validation
    for c in computes:
        if computes_info.get(c) is None:
            raise NotImplementedError(
                'Compute "{}" is not understood.'.format(c))

    # Write file defining atom positions:
    write_lammps_atoms(supercell, atom_sites, species, species_idx, path,
                       atom_style, charges=charges)

    command_lns = list(parameters)
    command_lns.append('atom_style   {}'.format(atom_style))
    command_lns.append('')
    command_lns.append('read_data    {}'.format(atoms_file))
    command_lns.append('')
    command_lns += interactions

    # Cell constraints (cell is fixed by default)
    fix_lengths = cell_constraints.get('fix_lengths')
    fix_angles = cell_constraints.get('fix_angles')
    angles_eq = cell_constraints.get('angles_equal')
    lengths_eq = cell_constraints.get('lengths_equal')

    fix_count = 1

    # Define arguments for the LAMMPS `fix box/relax` command:
    relax_fp = ['fixedpoint 0.0 0.0 0.0']
    relax_A = ['x 0.0']
    relax_B = ['y 0.0', 'scalexy yes']
    relax_C = ['z 0.0', 'scaleyz yes', 'scalexz yes']
    relax_all = ['tri 0.0']
    relax_couple_xy = ['couple xy']
    relax_couple_xz = ['couple xz']
    relax_couple_yz = ['couple yz']
    relax_couple_xyz = ['couple xyz']

    cell_cnst = []

    if not (fix_angles == 'abc' and fix_lengths == 'abc'):

        cell_cnst.append('fix {:d} all box/relax'.format(fix_count))
        fix_count += 1

        if fix_angles == 'abc':
            if fix_lengths is None:
                cell_cnst.extend(relax_A + relax_B + relax_C)
            elif fix_lengths == 'bc':
                cell_cnst.extend(relax_A)
            elif fix_lengths == 'ac':
                cell_cnst.extend(relax_B)
            elif fix_lengths == 'ab':
                cell_cnst.extend(relax_C)
            elif fix_lengths == 'a':
                cell_cnst.extend(relax_B + relax_C)
            elif fix_lengths == 'b':
                cell_cnst.extend(relax_A + relax_C)
            elif fix_lengths == 'c':
                cell_cnst.extend(relax_A + relax_B)

        elif fix_angles is None:

            if fix_lengths is None:
                cell_cnst.extend(relax_all)

            else:
                raise NotImplementedError('Relaxing supercell angles and '
                                          'fixing some or all supercell '
                                          'lengths is not implemented in the '
                                          'LAMMPS input file writer.')
        else:
            raise NotImplementedError('Fixing only some supercell angles is '
                                      'not implemented in the LAMMPS input '
                                      'file writer.')

        cell_cnst += relax_fp

    cell_cnst_str = ' '.join(cell_cnst)

    fix_lns = []
    if cell_cnst_str is not '':
        fix_lns = [cell_cnst_str]

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

        fix_lns.append('fix {:d} {} setforce 0.0 0.0 NULL'.format(fix_count,
                                                                  fxy_grp))
        fix_count += 1

    if fix_xyz_idx is not None:

        nfxyz = len(fix_xyz_idx)
        if nfxyz == atom_sites.shape[1]:
            fxyz_grp = 'all'
        else:
            fxyz_grp = 'fix_xyz'
            fxyz_grp_ln = 'group {} id '.format(fxyz_grp)
            fxyz_grp_ln += ('{:d} ' * nfxyz).format(*fix_xyz_idx)
            fix_lns.append(fxyz_grp_ln)

        fix_lns.append('fix {:d} {} setforce 0.0 0.0 0.0'.format(fix_count,
                                                                 fxy_grp))
        fix_count += 1

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
    thermo_args = ['step', 'atoms', 'pe', 'ke', 'etotal', 'fmax']
    thermo_args_all = ' '.join(thermo_args)
    thermo_lns = [
        'thermo_style custom {}'.format(thermo_args_all),
        'thermo_modify format float %20.10f',
        'thermo {:d}'.format(thermos_dt)
    ]

    if atom_style == 'atomic':
        dump_str = 'dump 1 all custom {} dump.*.txt id type x y z'.format(
            dump_dt)
        dump_mod = 'dump_modify 1 format line "%5d %5d %20.10f %20.10f %20.10f'

    elif atom_style == 'full':
        dump_str = 'dump 1 all custom {} dump.*.txt id type x y z q'.format(
            dump_dt)
        dump_mod = 'dump_modify 1 format line "%5d %5d %20.10f %20.10f %20.10f %20.10f'

    dump_str += dmp_computes
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
        },
        'fmax': {
            'name': 'Fmax',
            'dtype': float,
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
    box = np.array(box)
    xlo_bnd = box[0, 0]
    xhi_bnd = box[0, 1]
    ylo_bnd = box[1, 0]
    yhi_bnd = box[1, 1]
    zlo_bnd = box[2, 0]
    zhi_bnd = box[2, 1]
    xy = box[0, 2]
    xz = box[1, 2]
    yz = box[2, 2]

    xlo = xlo_bnd - np.min([0, xy, xz, xy + xz])
    xhi = xhi_bnd - np.max([0, xy, xz, xy + xz])
    ylo = ylo_bnd - np.min([0, yz])
    yhi = yhi_bnd - np.max([0, yz])
    zlo = zlo_bnd
    zhi = zhi_bnd

    supercell = np.array([
        [xhi - xlo, xy, xz],
        [0, yhi - ylo, yz],
        [0, 0, zhi - zlo],
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

    all_dmps = {}
    atoms = []
    atom_disp = []
    atom_pot_energy = []
    vor_vols = []
    vor_faces = []
    supercell = []
    box = []
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
        box.append(dmp_i['box'])
        time_steps.append(dmp_i['time_step'])

    time_steps = np.array(time_steps)
    srt_idx = np.argsort(time_steps)
    atoms = np.array(atoms)[srt_idx]
    atom_disp = np.array(atom_disp)[srt_idx]
    atom_pot_energy = np.array(atom_pot_energy)[srt_idx]
    vor_vols = np.array(vor_vols)[srt_idx]
    vor_faces = np.array(vor_faces)[srt_idx]
    supercell = np.array(supercell)[srt_idx]
    box = np.array(box)[srt_idx]

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
        'box': box,
        'time_steps': time_steps,
        'final_energy': final_energy
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
