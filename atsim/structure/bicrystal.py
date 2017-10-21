import numpy as np
import spglib
from atsim.structure.atomistic import AtomisticStructure
from atsim import vectors, mathsutils
from atsim.structure.crystal import CrystalBox
from atsim.structure import gbhelper


CSL_FROM_PARAMS_GB_TYPES = {
    'tilt_A': np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]),
    'tilt_B': np.array([
        [1, 0, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]),
    'twist': np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]),
    'mixed_A': np.array([
        [1, 0, 0],
        [0, 1, 0],
        [1, 0, 1]
    ])
}


class Bicrystal(AtomisticStructure):
    """
    Class to represent a bicrystal supercell.

    Attributes
    ----------
    atoms_gb_dist : ndarray of shape (N,)
        Perpendicular distances of each atom from the origin boundary plane

    TODO: Finish docstrings.

    """

    def __init__(self, as_params, maintain_inv_sym=False, reorient=False,
                 boundary_vac_args=None, relative_shift_args=None, wrap=True,
                 nbi=None, rot_mat=None):

        # Call parent constructor
        super().__init__(**as_params)

        # Set meta info
        self.meta.update({'supercell_type': ['bicrystal']})

        # Non-boundary (column) index of `box_csl` and grain arrays
        NBI = nbi
        BI = [0, 1, 2]
        BI.remove(nbi)

        # Boundary normal vector:
        n = np.cross(self.supercell[:, BI[0]],
                     self.supercell[:, BI[1]])[:, np.newaxis]
        n_unit = n / np.linalg.norm(n)

        # Non-boundary supercell unit vector
        u = self.supercell[:, NBI:NBI + 1]
        u_unit = u / np.linalg.norm(u)

        # Set instance CSLBicrystal-specific attributes:
        self.maintain_inv_sym = maintain_inv_sym
        self.n_unit = n_unit
        self.u_unit = u_unit
        self.non_boundary_idx = NBI
        self.boundary_idx = BI
        self.boundary_vac = 0
        self.relative_shift = [0, 0]
        self.rot_mat = rot_mat

        self.check_inv_symmetry()

        # Invoke additional methods:
        if reorient:
            self.reorient_to_lammps()

        if boundary_vac_args is not None:
            self.apply_boundary_vac(**boundary_vac_args)

        if relative_shift_args is not None:
            self.apply_relative_shift(**relative_shift_args)

        if wrap:
            self.wrap_atoms_to_supercell()

    @property
    def bicrystal_thickness(self):
        """Computes bicrystal thickness in grain boundary normal direction."""

        nbi = self.non_boundary_idx
        sup_nb = self.supercell[:, nbi:nbi + 1]
        return np.einsum('ij,ij', sup_nb, self.n_unit)

    @property
    def boundary_area(self):
        """Computes the grain boundary area, in square Angstroms."""
        bv = self.boundary_vecs
        return np.linalg.norm(np.cross(bv[:, 0], bv[:, 1]))

    @property
    def boundary_vecs(self):
        bi = self.boundary_idx
        s = self.supercell
        return np.vstack([s[:, bi[0]], s[:, bi[1]]]).T

    @property
    def atoms_gb_dist(self):
        """
        Computes the distance from each atom to the origin grain boundary
        plane.

        """
        return np.einsum('jk,jl->k', self.atom_sites, self.n_unit)

    def reorient_to_lammps(self):

        R = super().reorient_to_lammps()

        # Reorient objects which are CSLBicrystal specific
        self.n_unit = np.dot(R, self.n_unit[:, 0])[:, np.newaxis]
        self.u_unit = np.dot(R, self.u_unit[:, 0])[:, np.newaxis]

        return R

    def apply_boundary_vac(self, vac_thickness, sharpness=1):
        """
        Apply boundary vacuum to the supercell, atoms and grains according
        to a sigmoid function.

        TODO:
        -   Also apply to lattice sites

        """

        sup_type = self.meta['supercell_type']

        if 'bulk_bicrystal' in sup_type:
            raise NotImplementedError(
                'Cannot apply boundary vacuum to a bulk_bicrystal.')

        elif all([i not in sup_type for i in ['bicrystal', 'surface_bicrystal']]):
            raise NotImplementedError(
                'Cannot apply boundary vacuum to this supercell type.')

        vt = vac_thickness

        # For convenience:
        grn_a = self.crystals[0]
        grn_b = self.crystals[1]
        bt = self.bicrystal_thickness
        grn_a_org = grn_a['origin']
        grn_b_org = grn_b['origin']
        grn_a_full = grn_a['crystal'] + grn_a_org
        grn_b_full = grn_b['crystal'] + grn_b_org
        nbi = self.non_boundary_idx

        # Get perpendicular distances from the origin boundary plane:
        grn_a_gb_dist = np.einsum('ij,ik->j', grn_a_full, self.n_unit)
        grn_b_gb_dist = np.einsum('ij,ik->j', grn_b_full, self.n_unit)
        grn_a_org_gb_dist = np.einsum('ij,ik->j', grn_a_org, self.n_unit)
        grn_b_org_gb_dist = np.einsum('ij,ik->j', grn_b_org, self.n_unit)

        # Set which Sigmoid function to use:
        if self.maintain_inv_sym:
            sig_fnc = mathsutils.double_sigmoid
        else:
            sig_fnc = mathsutils.single_sigmoid

        # Get displacements in the boundary normal directions according
        # to a Sigmoid function:
        as_dx = sig_fnc(self.atoms_gb_dist, vt, sharpness, bt)
        grn_a_dx = sig_fnc(grn_a_gb_dist, vt, sharpness, bt)
        grn_b_dx = sig_fnc(grn_b_gb_dist, vt, sharpness, bt)
        grn_a_org_dx = sig_fnc(grn_a_org_gb_dist, vt, sharpness, bt)
        grn_b_org_dx = sig_fnc(grn_b_org_gb_dist, vt, sharpness, bt)

        # Snap atoms close to zero
        as_dx = vectors.snap_arr_to_val(as_dx, 0, 1e-14)

        # Find new positions with vacuum applied:
        as_vac = self.atom_sites + (as_dx * self.n_unit)
        grn_a_org_vac = grn_a_org + (grn_a_org_dx * self.n_unit)
        grn_b_org_vac = grn_b_org + (grn_b_org_dx * self.n_unit)
        grn_a_vac = grn_a_full + (grn_a_dx * self.n_unit) - grn_a_org_vac
        grn_b_vac = grn_b_full + (grn_b_dx * self.n_unit) - grn_b_org_vac

        # Apply vacuum to the supercell
        if self.maintain_inv_sym:
            sup_gb_dist = np.einsum('ij,ik->j', self.supercell, self.n_unit)
            sup_dx = sig_fnc(sup_gb_dist, vt, sharpness, bt)
            sup_vac = self.supercell + (sup_dx * self.n_unit)

        else:
            vac_add = (self.u_unit * vt) / \
                np.einsum('ij,ij->', self.n_unit, self.u_unit)
            sup_vac = np.copy(self.supercell)
            sup_vac[:, nbi:nbi + 1] += vac_add

        # Add new attributes:
        self.atoms_gb_dist_old = self.atoms_gb_dist
        self.atoms_gb_dist_δ = as_dx

        if self.boundary_vac != 0:
            warnings.warn('`boundary_vac` is already non-zero ({}). Resetting to '
                          'new value.'.format(self.boundary_vac))

        self.boundary_vac = vac_thickness

        # Update attributes:
        self.atom_sites = as_vac
        self.supercell = sup_vac
        self.crystals[0].update({
            'crystal': grn_a_vac,
            'origin': grn_a_org_vac
        })
        self.crystals[1].update({
            'crystal': grn_b_vac,
            'origin': grn_b_org_vac
        })

        self.check_overlapping_atoms(self._overlap_tol)
        self.check_inv_symmetry()

    def apply_boundary_vac_flat(self, vac_thickness):
        """
        Apply boundary vacuum to the supercell without affecting the atom
        blocks.

        """
        self.apply_boundary_vac(vac_thickness, sharpness=1000)

    def apply_relative_shift(self, shift):
        """
        Apply in-boundary-plane shifts to the grain further away from the origin 
        (right hand side of boundary) to explore the microscopic degrees of freedom.

        `shift` is a 2 element array whose elements are the
        relative shift in fractional coords of the boundary area.

        TODO:
        -   Also apply to lattice sites

        """

        sup_type = self.meta['supercell_type']

        if 'bulk_bicrystal' in sup_type:
            raise NotImplementedError(
                'Cannot apply relative shift to a bulk_bicrystal.')

        elif all([i not in sup_type for i in ['bicrystal', 'surface_bicrystal']]):
            raise NotImplementedError(
                'Cannot apply relative shift to this supercell type.')

        if isinstance(shift, np.ndarray):
            if shift.shape[0] == 1:
                shift = shift[0]
        else:
            shift = np.array(shift)
        if np.any(shift < -1) or np.any(shift > 1):
            raise ValueError('Elements of `shift` should be between -1 and 1.')

        # Convenience:
        nbi = self.non_boundary_idx
        bi = self.boundary_idx
        sup_nb = self.supercell[:, nbi:nbi + 1]

        # Find the right grain to shift
        # Crystals origins
        crys_orgns = np.array([self.crystals[i]['origin'] for i in range(2)])
        # Find crystals with origin at supercell origin if any
        crys_sup_orgn = np.all((crys_orgns == 0.0), axis=1)

        if len(np.where(crys_sup_orgn == False)[0]) == 1:
            sh_idx = np.where(crys_sup_orgn == False)[0][0]
        else:
            for i in range(2):
                if self.crystals[i]['crystal'][:, nbi][nbi] > 0:
                    sh_idx = i
        grn_sh = self.crystals[sh_idx]

        shift_gb = np.zeros((3, 1))
        shift_gb[bi] = shift[:, np.newaxis]
        shift_std = np.dot(grn_sh['crystal'], shift_gb)

        # Translate shifted grain atoms:
        as_shift = np.copy(self.atom_sites)
        as_shift[:, np.where(self.crystal_idx == sh_idx)[0]] += shift_std

        # Translate shifted grain origin:
        grn_sh_org_shift = grn_sh['origin'] + shift_std

        # Update attributes:
        self.atom_sites = as_shift
        self.crystals[sh_idx].update({
            'origin': grn_sh_org_shift
        })

        if self.relative_shift != [0, 0]:
            warnings.warn('`relative_shift` is already non-zero. Resetting to '
                          'new value.')
        self.relative_shift = [i + j for i,
                               j in zip(shift.tolist(), self.relative_shift)]

        if self.maintain_inv_sym:
            # Modify out-of-boundary supercell vector
            sup_shift = np.copy(self.supercell)
            sup_shift[:, nbi:nbi + 1] += (2 * shift_std)

            # Update attribute:
            self.supercell = sup_shift

        self.check_overlapping_atoms(self._overlap_tol)
        self.check_inv_symmetry()

    def wrap_atoms_to_supercell(self):
        """
        Wrap atoms to within the boundary plane as defined by the supercell.

        """
        sup_type = self.meta['supercell_type']

        if 'bulk_bicrystal' in sup_type:
            raise NotImplementedError(
                'Cannot wrap atoms within a bulk_bicrystal.')

        elif all([i not in sup_type for i in ['bicrystal', 'surface_bicrystal']]):
            raise NotImplementedError(
                'Cannot wrap atoms within this supercell type.')

        super().wrap_atoms_to_supercell(dirs=self.boundary_idx)
        self.check_inv_symmetry()

    def check_inv_symmetry(self):
        """
        Check atoms exhibit inversion symmetry through the two crystal centres,
        if `self.maintain_inv_sym` is True.

        """

        if self.maintain_inv_sym:

            sym_ops = spglib.get_symmetry(self.spglib_cell)
            sym_rots = sym_ops['rotations']
            sym_trans = sym_ops['translations']
            inv_sym_rot = -np.eye(3, dtype=int)
            inv_sym = np.where(np.all(sym_rots == inv_sym_rot, axis=(1, 2)))[0]
            if len(inv_sym) == 0:
                raise ValueError('The bicrystal does not have inversion '
                                 'symmetry.')


def csl_bicrystal_from_parameters(crystal_structure, csl_vecs, box_csl=None,
                                  gb_type=None, gb_size=None,
                                  edge_conditions=None,
                                  overlap_tol=1, reorient=True, wrap=True,
                                  maintain_inv_sym=False,
                                  boundary_vac_args=None,
                                  relative_shift_args=None):
    """
    Parameters
    ----------
    crystal_structure : CrystalStructure
    csl_vecs : list of length 2 of ndarray of shape (3, 3)
        List of two arrays of three column vectors representing CSL vectors
        in the lattice basis. The two CSL unit cells defined here rotate onto
        each other by the CSL rotation angle. The rotation axis is taken as the
        third vector, which must therefore be the same for both CSL unit cells.
    box_csl : ndarray of shape (3, 3), optional
        The linear combination of CSL unit vectors defined in `csl_vecs` used
        to construct each half of the bicrystal. The first two columns
        represent vectors defining the boundary plane. The third column
        represents a vector in the out-of-boundary direction. Only one of
        `box_csl` and `gb_type` may be specified.
    gb_type : str, optional
        Default is None. Must be one of 'tilt_A', 'tilt_B', 'twist' or
        'mixed_A'. Only one of `box_csl` and `gb_type` may be specified.
    gb_size : ndarray of shape (3,) of int, optional
        If `gb_type` is specified, the unit grain vectors associated with that
        `gb_type` are scaled by these integers. Default is None, in which case
        it is set to np.array([1, 1, 1]).
    edge_conditions : list of list of str
        Edge conditions for each grain in the bicrystal. See `CrystalBox` for
        details.
    maintain_inv_sym : bool, optional
        If True, the supercell atoms will be checked for inversion symmetry
        through the centres of both crystals. This check will be repeated
        following methods which manipulate atom positions. In this way, the two
        grain boundaries in the bicrystal are ensured to be identical.
    reorient : bool, optional
        If True, after construction of the boundary, reorient_to_lammps() is
        invoked. Default is True.
    boundary_vac_args : dict, optional
        If not None, after construction of the boundary, apply_boundary_vac()
        is invoked with this dict as keyword arguments. Default is None.
    apply_boundary_vac_flat_args: dict, optional
        If not None, after construction of the boundary,
        apply_boundary_vac_flat() is invoked with this dict as keyword
        arguments. Default is None.
    relative_shift_args : dict, optional
        If not None, after construction of the boundary, apply_relative_shift()
        is invoked with this dict as keyword arguments. Default is None.
    wrap : bool, optional
        If True, after construction of the boundary, wrap_atoms_to_supercell()
        is invoked. Default is True.

    Notes
    -----
    Algorithm proceeds as follows:
    1.  Apply given linear combinations of given CSL unit vectors to form grain
        vectors of the bicrystal.
    2.  Multiply the out-of-boundary vector of the second grain by -1, such
        that rotation of the second grain by the CSL rotation angle will form a
        bicrystal of two grains.
    3.  Check grain A is formed from a right-hand basis - since we want the
        supercell vectors to be formed from a right-hand basis. If not, for
        both grain A and B, swap the first and second vectors to do this.
    4.  Fill the two grains with atoms

    TODO:
    -   Sort out lattice sites in apply_boundary_vac() & apply_relative_shift()
    -   Rename wrap_atoms_to_supercell to wrap_to_supercell and apply wrapping
        to lattice sites, and crystal boxes as well.

    """

    if np.all(csl_vecs[0][:, 2] != csl_vecs[1][:, 2]):
        raise ValueError('Third vectors in `csl_vecs[0]` and csl_vecs[1] '
                         'represent the CSL rotation axis and must '
                         'therefore be equal.')

    if box_csl is not None and gb_type is not None:
        raise ValueError('Only one of `box_csl` and `gb_type` may be '
                         'specified.')

    if box_csl is None and gb_type is None:
        raise ValueError('Exactly one of `box_csl` and `gb_type` must be '
                         'specified.')

    if gb_type is not None:

        if gb_size is None:
            gb_size = np.array([1, 1, 1])

        if gb_type not in CSL_FROM_PARAMS_GB_TYPES:
            raise ValueError(
                'Invalid `gb_type`: {}. Must be one of {}'.format(
                    gb_type, list(CSL_FROM_PARAMS_GB_TYPES.keys())))

        box_csl = CSL_FROM_PARAMS_GB_TYPES.get(gb_type) * gb_size

    lat_vecs = crystal_structure.bravais_lattice.vecs
    rot_ax_std = np.dot(lat_vecs, csl_vecs[0][:, 2:3])
    csl_vecs_std = [np.dot(lat_vecs, c) for c in csl_vecs]

    # Non-boundary (column) index of `box_csl` and grain arrays:
    NBI = 2
    BI = [0, 1]

    # Enforce a rule that out of boundary grain vector has to be
    # (a multiple of) a single CSL unit vector. This reduces the
    # potential "skewness" of the supercell.
    if np.count_nonzero(box_csl[:, NBI]) > 1:
        raise ValueError('The out of boundary vector, `box_csl[:, {}]`'
                         ' must have exactly one non-zero '
                         'element.'.format(NBI))

    # Scale grains in lattice basis
    grn_a_lat = np.dot(csl_vecs[0], box_csl)
    grn_b_lat = np.dot(csl_vecs[1], box_csl)
    grn_b_lat[:, NBI] *= -1

    # Get grain vectors in standard Cartesian basis
    grn_a_std = np.dot(lat_vecs, grn_a_lat)
    grn_b_std = np.dot(lat_vecs, grn_b_lat)

    # Get rotation matrix for rotating grain B onto grain A
    if np.all(csl_vecs[0] == csl_vecs[1]):
        rot_angles = [0, 0, 0]
        rot_mat = np.eye(3)

    else:
        rot_angles = vectors.col_wise_angles(
            csl_vecs_std[0], csl_vecs_std[1])

        if not np.isclose(*rot_angles[0:2]):
            raise ValueError('Non-equivalent rotation angles found '
                             'between CSL vectors.')

        rot_mat = vectors.rotation_matrix(
            rot_ax_std[:, 0], rot_angles[0])[0]

    rot_angle_deg = np.rad2deg(rot_angles[0])
    grn_vols = [np.dot(np.cross(g[:, 0], g[:, 1]), g[:, 2])
                for g in (grn_a_std, grn_b_std)]

    # Check grain volumes are the same:
    if not np.isclose(*np.abs(grn_vols)):
        raise ValueError('Grain A and B have different volumes.')

    # Check if grain A forms a right-handed coordinate system:
    if grn_vols[0] < 0:
        # Swap boundary vectors to make a right-handed coordinate system:
        grn_a_lat[:, [0, 1]] = grn_a_lat[:, [1, 0]]
        grn_b_lat[:, [0, 1]] = grn_b_lat[:, [1, 0]]
        grn_a_std[:, [0, 1]] = grn_a_std[:, [1, 0]]
        grn_b_std[:, [0, 1]] = grn_b_std[:, [1, 0]]
        box_csl[0, 1] = box_csl[1, 0]

    # Specify bounding box edge conditions for including atoms:
    if edge_conditions is None:
        edge_conditions = [
            ['10', '10', '10'],
            ['10', '10', '10']
        ]
        edge_conditions[1][NBI] = '01'

    # Make two crystal boxes:
    crys_a = CrystalBox(crystal_structure, grn_a_std,
                        edge_conditions=edge_conditions[0])
    crys_b = CrystalBox(crystal_structure, grn_b_std,
                        edge_conditions=edge_conditions[1])

    # Get atom and lattice sites from crystal boxes:
    as_a = crys_a.atom_sites_std
    as_b = crys_b.atom_sites_std
    ls_a = crys_a.lat_sites_std
    ls_b = crys_b.lat_sites_std

    # Rotate crystal B onto A:
    as_b_rot = np.dot(rot_mat, as_b)
    ls_b_rot = np.dot(rot_mat, ls_b)
    grn_b_rot_std = np.dot(rot_mat, grn_b_std)

    # Shift crystals to form a supercell at the origin
    zs_std = - grn_b_rot_std[:, NBI:NBI + 1]
    as_a_zs = as_a + zs_std
    as_b_zs = as_b_rot + zs_std
    ls_a_zs = ls_a + zs_std
    ls_b_zs = ls_b_rot + zs_std

    crystal_idx = np.array([0] * as_a_zs.shape[1] + [1] * as_b_zs.shape[1])
    lat_crystal_idx = np.array(
        [0] * ls_a_zs.shape[1] + [1] * ls_b_zs.shape[1])
    atom_sites = np.hstack([as_a_zs, as_b_zs])
    lattice_sites = np.hstack([ls_a_zs, ls_b_zs])

    # Define the supercell:
    sup_std = np.copy(grn_a_std)
    sup_std[:, NBI] = grn_a_std[:, NBI] - grn_b_rot_std[:, NBI]

    crystals = [
        {
            'crystal': grn_a_std,
            'origin': zs_std,
            'cs_idx': 0,
            'cs_orientation': np.eye(3),
            'cs_origin': [0, 0, 0]
        },
        {
            'crystal': grn_b_rot_std,
            'origin': zs_std,
            'cs_idx': 0,
            'cs_orientation': rot_mat,
            'cs_origin': [0, -1, 0]
        }
    ]

    species_idx = np.hstack([crys_a.species_idx, crys_b.species_idx])
    motif_idx = np.hstack([crys_a.motif_idx, crys_b.motif_idx])

    # AtomisticStructure parameters
    as_params = {
        'atom_sites': atom_sites,
        'supercell': sup_std,
        'lattice_sites': lattice_sites,
        'crystals': crystals,
        'crystal_structures': [crystal_structure],
        'crystal_idx': crystal_idx,
        'lat_crystal_idx': lat_crystal_idx,
        'species_idx': species_idx,
        'motif_idx': motif_idx,
        'overlap_tol': overlap_tol,
    }

    # Bicrystal parameters
    bc_params = {
        'as_params': as_params,
        'maintain_inv_sym': maintain_inv_sym,
        'reorient': reorient,
        'boundary_vac_args': boundary_vac_args,
        'relative_shift_args': relative_shift_args,
        'wrap': wrap,
        'nbi': 2,
        'rot_mat': rot_mat,
    }

    return Bicrystal(**bc_params)


def csl_bulk_bicrystal_from_parameters(crystal_structure, csl_vecs,
                                       box_csl=None, gb_type=None,
                                       gb_size=None, edge_conditions=None,
                                       overlap_tol=1, reorient=True):
    """
    Parameters
    ----------
    csl_vecs: ndarray of int of shape (3, 3)

    """

    bc_params = {
        'crystal_structure': crystal_structure,
        'csl_vecs': [csl_vecs, csl_vecs],
        'box_csl': box_csl,
        'gb_type': gb_type,
        'gb_size': gb_size,
        'edge_conditions': edge_conditions,
        'overlap_tol': overlap_tol,
        'reorient': reorient,
        'wrap': False,
    }

    print('bc_params: \n{}\n'.format(bc_params))

    bc = csl_bicrystal_from_parameters(**bc_params)
    bc.meta.update({'supercell_type': ['bulk', 'bulk_bicrystal']})
    return bc


def csl_surface_bicrystal_from_parameters(crystal_structure, csl_vecs,
                                          box_csl=None, gb_type=None,
                                          gb_size=None, edge_conditions=None,
                                          overlap_tol=1,
                                          reorient=True, wrap=True,
                                          maintain_inv_sym=False,
                                          boundary_vac_args=None,
                                          relative_shift_args=None,
                                          surface_idx=0):
    """
    Parameters
    ----------
    csl_vecs: list of length 2 of ndarray of int of shape (3, 3)

    """

    bc_params = {
        'crystal_structure': crystal_structure,
        'csl_vecs': csl_vecs,
        'box_csl': box_csl,
        'gb_type': gb_type,
        'gb_size': gb_size,
        'edge_conditions': edge_conditions,
        'overlap_tol': overlap_tol,
        'reorient': reorient,
        'wrap': wrap,
        'maintain_inv_sym': maintain_inv_sym,
        'boundary_vac_args': boundary_vac_args,
        'relative_shift_args': relative_shift_args,
    }

    bc = csl_bicrystal_from_parameters(**bc_params)

    # Remove atoms from removed crystal
    atoms_keep = np.where(bc.crystal_idx == surface_idx)[0]
    bc.atom_sites = bc.atom_sites[:, atoms_keep]
    bc.species_idx = bc.species_idx[atoms_keep]
    bc.motif_idx = bc.motif_idx[atoms_keep]
    bc.crystal_idx = bc.crystal_idx[atoms_keep]

    # Remove lattice sites from removed crystal
    lat_keep = np.where(bc.lat_crystal_idx == surface_idx)[0]
    bc.lattice_sites = bc.lattice_sites[:, lat_keep]
    bc.lat_crystal_idx = bc.lat_crystal_idx[lat_keep]

    bc.meta.update({'supercell_type': ['surface', 'surface_bicrystal']})
    return bc


def csl_bicrystal_from_structure(csl, csl_params,
                                 overlap_tol=0.1, reorient=True,
                                 maintain_inv_sym=False,
                                 boundary_vac_args=None,
                                 relative_shift_args=None,
                                 wrap=False):
    """
    Create a CSL Bicrystal from a structure.

    Parameters
    ----------
    csl : string
        '[angle_axis_structure]' to be constructed using a method 
        defined as 'construct_'in gbhelper.py.
        (for example '180_001_mZrO2').
    csl_params : dict of (str : string or ndarray or int)
        Parameters needed to construct `csl`. Vary depending on method.
    maintain_inv_sym : bool, optional
        If True, the supercell atoms will be checked for inversion symmetry
        through the centres of both crystals. This check will be repeated
        following methods which manipulate atom positions. In this way, the two
        grain boundaries in the bicrystal are ensured to be identical.
    reorient : bool, optional
        If True, after construction of the boundary, reorient_to_lammps() is
        invoked. Default is True.
    boundary_vac_args : dict, optional
        If not None, after construction of the boundary, apply_boundary_vac()
        is invoked with this dict as keyword arguments. Default is None.
    apply_boundary_vac_flat_args: dict, optional
        If not None, after construction of the boundary,
        apply_boundary_vac_flat() is invoked with this dict as keyword
        arguments. Default is None.
    relative_shift_args : dict, optional
        If not None, after construction of the boundary, apply_relative_shift()
        is invoked with this dict as keyword arguments. Default is None.
    wrap : bool, optional
        If True, after construction of the boundary, wrap_atoms_to_supercell()
        is invoked. Default is True.
    Notes
    -----
    `reorient`=True doesn't work since no lattice_sites are passed.
    `crystals` not yet available.

    """

    create_bound = getattr(gbhelper, 'construct_' + csl)
    bound_struct = create_bound(**csl_params)

    # AtomisticStructure parameters
    as_params = {
        'atom_sites': bound_struct['atom_sites'],
        'supercell': bound_struct['supercell'],
        'all_species': bound_struct['all_species'],
        'all_species_idx': bound_struct['all_species_idx'],
        'overlap_tol': overlap_tol,
        'crystals': bound_struct['crystals'],
        #     'crystal_structures' : crystal_structures,
        'crystal_idx': bound_struct['crystals_idx'],
    }

    # Bicrystal parameters
    bc_params = {
        'as_params': as_params,
        'maintain_inv_sym': maintain_inv_sym,
        'reorient': reorient,
        'boundary_vac_args': boundary_vac_args,
        'relative_shift_args': relative_shift_args,
        'wrap': wrap,
        'nbi': 0,
    }

    return Bicrystal(**bc_params)
