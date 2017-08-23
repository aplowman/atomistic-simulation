import numpy as np
from vectors import rotation_matrix
import simsio
from functools import reduce

# Construct grain boundaries functions
def construct_180_001_mZrO2(cellfile, repeats=[3, 1, 1], bound_vac=0.2,
                            transls=[0.0, 0.0], term_plns=['100', '100']):
    """
    Construct a 180° [001] twin monoclinic ZrO2 boundary.

    Parameters
    ----------
    cell_input : string
        Input crystal structure as a .cell file.
    repeats : list
        Number of unit cell repeats [Nx, Ny, Nz] for both crystals, where 
        x-direction is normal to GB plane. Default value is [3, 1, 1].
    bound_vac : int
        Vacuum thickness to add at boundary (Angstrom)
    transls : list
        Translation steps in the two directions of boundary plane 
        (fractions of unit cell vectors)
    term_plns : list
        Termination planes (h00) for grains `a` and `b`. 
        Allowed values: '100' or '200' (equivalent to '-200').

    Returns
    -------
    dict of (str : ndarray/list)
        'supercell' : ndarray
            Array of shape (3, 3), where the column vectors are the three 
            lattice vectors.
        'pos_gb_a(b)' : ndarrays
            Arrays of shape (n, 3), where the row vectors are the fractional 
            coordinates of the atoms and n is the number of atoms in a given
            grain.
        'atom_sites' : ndarray of shape (3, 2*n)
            Array of column vectors representing the atom positions. 
        'crystals' : list of dict of (str : ndarray or int)
            `crystal` : ndarray of shape (3, 3)
                Array of column vectors representing the crystal edge vectors.
            `origin` : ndarray of shape (3, 1)
                Column vector specifying the origin of this crystal.
        'crystals_idx' : ndarray of shape (N,)
            Atom indices according to which crystal they belong to.
        'species_gb_x' : list
            List of length n, containing indices according to `species_key` for
            each atom as ordered in `pos_gb_a(b)`.
        'all_species' : ndarray
            Array of species elements.
        'all_species_idx' : ndarray of shape (N, )
            Two copies of `species_gb_x` concatenated (indices for the whole structure).
            Needed in order to construct an AtomisticStructure.

    Notes
    -----
            'supercell': supercell.T,
        'pos_gb_a': pos_gb_a,
        'pos_gb_b': pos_gb_b,
        'atom_sites' : np.concatenate((pos_gb_a, pos_gb_b)).T,
        'crystals' : crystals,
        'crystals_idx' : crystals_idx,
        'species_gb_x': species_gb_x,
        'all_species' : np.array(species_key),
        'all_species_idx' : np.concatenate((species_gb_x, species_gb_x))
    """

    # Boundary parameters
    # Numbers of unit cell repeats for each half of bicrystal
    Nx = repeats[0]
    Ny = repeats[1]
    Nz = repeats[2]

    # Boundary expansion in angstrom
    Bx = bound_vac

    # Translation step in boundary plane (fractions of unit cell vectors)
    Dy = transls[0]
    Dz = transls[1]

    # Termination planes for grains `a` and `b`
    term_plane_a = term_plns[0]
    term_plane_b = term_plns[1]

    # Read in data for input structure and set up cell, pos_f, species, natoms
    lat_data = simsio.read_cell_file(cellfile)
    
    cell_vecs = lat_data['cell_vecs']

    cell = align_mon_cart(cell_vecs, axis='c')  # Align c with z
    pos_f = lat_data['motif']['atom_sites'].T
    species_key = get_species_key_idx(lat_data['motif']['species'])[0]
    species = get_species_key_idx(lat_data['motif']['species'])[1]
    natoms = len(species)
    print(pos_f)
    
    # Translate atoms based on termination planes for grains `a` and `b`.
    pos_f_a = np.zeros((natoms, 3))
    pos_f_b = np.zeros((natoms, 3))

    pos_f_a = term_h00(pos_f, plane=term_plane_a)
    pos_a_a = frac2abs(cell, pos_f_a)

    pos_f_b = term_h00(pos_f, plane=term_plane_b)
    pos_a_b = frac2abs(cell, pos_f_b)

    # Rotation matrix
    R = rotation_matrix(np.array([0, 0, 1]), np.array([180]), degrees=True)
    R = zero_prec(R).squeeze()

    # Calculate positions of atoms in rotated cell
    pos_r_b = np.transpose(np.dot(R, np.transpose(pos_a_b)))

    # Calculate cell vectors for rotated cell. All vectors positive.
    cell_r = np.zeros((3, 3), dtype=float)
    cell_r = np.transpose(np.dot(R, np.transpose(cell)))

    # Vectors of rotated and translated cell to octant 1.
    cell_r_t = np.zeros((3, 3), dtype=float)
    cell_r_t[:, :] = cell[:, :]
    cell_r_t[0, 2] = - cell[0, 2]

    Nr = np.array([Nx, Ny, Nz])
    # Total number of repeats
    Nt = Nx * Ny * Nz

    # Vector of half the boundary expansion
    Dv = np.array([0.5 * Bx, 0.0, 0.0])

    # Boundary translation vector
    Dr = np.array([0.0, Dy * cell[1, 1], Dz * cell[2, 2]])

    # Define supercell vectors
    supercell = np.zeros((3, 3), dtype=float)
    # Set the right perpendicular length, including vacuum
    supercell[0, 0] = 2 * Nx * cell[0, 0] + 2 * Bx
    # Add required skew to maintain boundary equivalence
    # (twice the translation vector)
    supercell[0, :] = supercell[0, :] + 2 * Dr
    supercell[1, :] = Ny * cell[1, :]
    supercell[2, :] = Nz * cell[2, :]
    
    
    # Crystals 'a' and 'b' vectors and origin
    crystal = np.zeros((3,3),dtype=float)
    crystal[:,0] = 0.5 * supercell[0] 
    crystal[:,1:3] = supercell[1:3].T  

    crystals = [{},{}]

    for item in crystals:
        item.update( {'crystal': crystal})
        item.update( {'origin': np.zeros((3,1))}) 
        #  Don't really understand how these work
        #         item.update( {'cs_idx': 0})
        #         item.update( {'cs_orientation': R})
        #         item.update( {'cs_origin': [0.0,0.0]})
        # #     crystals[1]['origin'] = 0.5 * supercell[0]



    # Populate arrays of atom positions and types for a grain boundary supercell
    natoms_t = Nt * natoms
    species_gb_x = np.zeros(natoms_t, dtype=int)
    pos_gb_a = np.zeros((natoms_t, 3), dtype=float)
    pos_gb_b = np.zeros((natoms_t, 3), dtype=float)
    crystals_idx = np.concatenate((np.zeros(natoms_t), np.ones(natoms_t))) # atom belonging to crystal index

    # Populate arrays with atom positions
    a = 0
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                pos_gb_b[natoms * a:natoms * (1 + a), :] = \
                    pos_r_b - i * cell_r_t[0, :] + j * cell[1, :] + \
                    k * cell[2, :] + supercell[0, :] - Dr - Dv + cell[1, :]

                pos_gb_a[natoms * a:natoms * (1 + a), :] = pos_a_a + \
                    i * cell[0, :] + j * cell[1, :] + k * cell[2, :] + Dv
                species_gb_x[natoms * a:natoms * (1 + a)] = species
                a = a + 1

    out = {
        'supercell': supercell.T,
        'pos_gb_a': pos_gb_a,
        'pos_gb_b': pos_gb_b,
        'atom_sites' : np.concatenate((pos_gb_a, pos_gb_b)).T,
        'crystals' : crystals,
        'crystals_idx' : crystals_idx,
        'species_gb_x': species_gb_x,
        'all_species' : np.array(species_key),
        'all_species_idx' : np.concatenate((species_gb_x, species_gb_x))
    }

    return out


# Helper functions
def  term_h00(pos_f, plane='100'):
    """
    Translates and wraps fractional coordinates of m-ZrO2 primitive unit cell 
    for (h00) termination planes.
    
    Notes
    ----
    Only works for fractional coordinates and primitive unit cell for now.
    
    """
    if plane == '200' or plane == '-200':
        tvecs = np.array([1 / int(plane[0]), 0, 0]) # translation vector
        pos_f_t = pos_f + tvecs
        pos_f_t = wrap(pos_f_t)
        
        return pos_f_t

    elif plane == '100':
        pos_f_t = pos_f

        return pos_f_t
        
    else:
        raise ValueError ('Specify a valid (h00) termination plane: `100` or `200`')


def get_species_key_idx(species_str):
    """
    Create a species key and list of species indices based on that key.
    
    Parameters
    ----------
    species_str : list of str
        A list of strings for the atomic species symbols.
        
    Returns
    -------
    species_key : list of str
        List of unique species present.
    species_idx : list of str
        List of length n, containing indices according to `species_key` for
        each atom as ordered in `species_str `.
    
    """

    # Species convert string to number according to dictionary
    species_key = reduce(lambda l, x: l if x in l else l+[x], species_str, [])
    species_idx = np.zeros([len(species_str)])
    for i, atom in enumerate(species_str):
        species_idx[i] = species_key.index(atom)
        
    return species_key, species_idx


def align_mon_cart(cell_vecs, axis='c', pos_a=None):
    """
    Align monoclinic ZrO2 with Cartesian reference frame.
    
    Parameters
    ----------
    cell_vecs : ndarray
        Array of shape (3, 3), where the row vectors are the three lattice 
        vectors.
    axis : string
        Axis to be aligned: if 'c', then the c-axis is aligned parallel to 
        z-axis. If 'a', then the a-axis is aligned parallel to x-axis.
    pos_a : ndarray, optional 
        Array of shape (n, 3), where the row vectors are the absolute
        coordinates of the atoms and n is the number of atoms.

    Returns
    -------
    cell_r : ndarray
        Rotated lattice vectors.
    pos_a_r : ndarray, optional
        Rotated absolute atom coordinates.

    """

    if axis == 'c':
        d = np.arctan2(cell_vecs[2,2], cell_vecs[2,0]) - np.pi/2
        R_c = rotation_matrix(np.array([0,1,0]),d).squeeze()
        cell_r = np.dot(R_c, cell_vecs.T).T

    if pos_a is not None:
        pos_a_r = np.transpose(np.dot(R_c,np.transpose(pos_a)))
        return cell_r, pos_a_r
    else:
        return cell_r
    

def frac2abs(cell_vecs, pos_f):
    """
    Convert fractional to absolute coordinates.
    Parameters
    ----------
    cell_vecs : ndarray
        Array of shape (3, 3), where the row vectors are the three lattice 
        vectors.
    pos_f : ndarray, optional 
        Array of shape (n, 3), where the row vectors are the fractional
        coordinates of the atoms and n is the number of atoms.
    Returns
    -------
    pos_a : ndarray, optional 
        Array of shape (n, 3), where the row vectors are the absolute
        coordinates of the atoms and n is the number of atoms.
    """
    
    
    pos_a = np.zeros((pos_f.shape[0],3),dtype=float)   # Number of atoms = pos_f_rows
    
    for i in range(pos_f.shape[0]):
        for j in range(3):
            pos_a[i,:] = pos_a[i,:] + pos_f[i,j] * cell_vecs[j,:]

    return pos_a


def zero_prec(A, tol = 1e-12):
    """
    Sets elements of an array within some tolerance `tol` to 0.0.

    """
    A[abs(A) < tol] = 0.0
    return A

