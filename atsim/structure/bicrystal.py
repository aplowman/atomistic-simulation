import warnings
import numpy as np
import spglib
from functools import partial
from atsim.structure.atomistic import AtomisticStructure
from atsim import vectors, mathsutils
from atsim.structure.crystal import CrystalBox
from atsim.structure import gbhelper
from atsim.utils import prt
from atsim.vectors import rotation_matrix


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
                 boundary_vac=None, relative_shift=None,
                 wrap=True, nbi=None, rot_mat=None):

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
        self.boundary_vac_type = None
        self.relative_shift = [0, 0]
        self.rot_mat = rot_mat

        # Invoke additional methods:
        if reorient:
            self.reorient_to_xy()

        if boundary_vac is not None:
            for bv in boundary_vac:
                self.apply_boundary_vac(**bv)

        if relative_shift is not None:
            self.apply_relative_shift(**relative_shift)

        if wrap:
            self.wrap_sites_to_supercell()

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

    def distance_from_gb(self, points):
        """
        Computes the distance from each in an array of column vector to the
        origin grain boundary plane.

        """
        return np.einsum('jk,jl->k', points, self.n_unit)

    def reorient_to_xy(self):
        """
        Reorient the supercell to a LAMMPS-compatible orientation in such a 
        way that the boundary plane is in the xy plane.

        """

        # Reorient so the boundary plane is in xy
        if self.non_boundary_idx != 2:

            # Ensure non-boundary supercell vector is last vector, whilst
            # maintaining handedness of the supercell coordinate system.
            self.supercell = np.roll(self.supercell,
                                     (2 - self.non_boundary_idx), axis=1)

            self.non_boundary_idx = 2
            self.boundary_idx = [0, 1]

        R = super().reorient_to_lammps()

        # Reorient objects which are CSLBicrystal specific
        self.n_unit = np.dot(R, self.n_unit[:, 0])[:, np.newaxis]
        self.u_unit = np.dot(R, self.u_unit[:, 0])[:, np.newaxis]

        return R

    def apply_boundary_vac(self, thickness, func, wrap=False, **kwargs):
        """
        Apply vacuum to the Bicrystal in the direction normal to the grain
        boundary, distributed according to a function.

        Parameters
        ----------
        thickness : float
            The length by which the supercell will be extended in the GB
            normal direction. This is not necessarily a supercell vector
            direction.
        func : str
            One of "sigmoid", "flat" or "linear". Describes how the vacuum 
            should be distributed across the supercell in the GB normal
            direction.
        kwargs : dict
            Additional arguments to pass to the function.

        """

        # Validation
        allowed_func = [
            'sigmoid',
            'flat',
            'linear',
        ]

        if func not in allowed_func:
            raise ValueError('"{}" is not an allowed function name to use for'
                             ' applying grain boundary vacuum.'.format(func))

        sup_type = self.meta['supercell_type']

        if 'bulk_bicrystal' in sup_type:
            raise NotImplementedError(
                'Cannot apply boundary vacuum to a bulk_bicrystal.')

        elif all([i not in sup_type for i in ['bicrystal',
                                              'surface_bicrystal']]):
            raise NotImplementedError(
                'Cannot apply boundary vacuum to this supercell type.')

        # For convenience:
        vt = thickness
        bt = self.bicrystal_thickness
        nu = self.n_unit
        uu = self.u_unit
        grn_a = self.crystals[0]
        grn_b = self.crystals[1]
        grn_a_cry = grn_a['crystal']
        grn_b_cry = grn_b['crystal']
        grn_a_org = grn_a['origin']
        grn_b_org = grn_b['origin']

        if func in ['sigmoid', 'flat']:

            if func == 'sigmoid':
                b = kwargs.get('sharpness', 1)
            elif func == 'flat':
                b = 1000

            func_args = {
                'a': thickness,
                'b': b,
                'width': bt,
            }

            if self.maintain_inv_sym:
                func_cll = mathsutils.double_sigmoid
            else:
                func_cll = mathsutils.single_sigmoid

        elif func == 'linear':

            func_args = {
                'm': vt / bt,
            }
            func_cll = mathsutils.linear

        # Callable which will return dx for a given x
        f = partial(func_cll, **func_args)

        def expand_box(edge_vecs, box_origin):

            vecs_full = edge_vecs + box_origin
            vecs_dist = self.distance_from_gb(vecs_full - self.origin)
            org_dist = self.distance_from_gb(box_origin - self.origin)

            vecs_dx = f(vecs_dist)
            org_dx = f(org_dist)

            org_vac = box_origin + (org_dx * nu)
            vecs_vac = vecs_full + (vecs_dx * nu) - org_vac

            return (vecs_vac, org_vac)

        def expand_sites(sites):

            dist = self.distance_from_gb(sites - self.origin)
            dx = f(dist)
            sites_vac = sites + (dx * nu)

            return (sites_vac, dist, dx)

        atm_sts, atm_dst, atm_dx = expand_sites(self.atom_sites)
        self.atom_sites = atm_sts

        # Store info in the meta dict regarding this change in site positions:
        bv_dict = {
            'thickness': vt,
            'func': func,
            'kwargs': kwargs,
            'atom_sites': {
                'x': atm_dst,
                'dx': atm_dx,
            }
        }

        if self.lattice_sites is not None:
            lat_sts, lat_dst, lat_dx = expand_sites(self.lattice_sites)
            self.lattice_sites = lat_sts
            bv_dict.update({
                'lattice_sites': {
                    'x': lat_dst,
                    'dx': lat_dx,
                }
            })

        if self.interstice_sites is not None:
            int_sts, int_dst, int_dx = expand_sites(self.interstice_sites)
            self.interstice_sites = int_sts
            bv_dict.update({
                'interstice_sites': {
                    'x': int_dst,
                    'dx': int_dx,
                }
            })

        cry_keys = ('crystal', 'origin')
        crys_a = dict(zip(cry_keys, expand_box(grn_a_cry, grn_a_org)))
        crys_b = dict(zip(cry_keys, expand_box(grn_b_cry, grn_b_org)))

        self.crystals[0].update(crys_a)
        self.crystals[1].update(crys_b)

        # Apply vacuum to the supercell
        if self.maintain_inv_sym:
            self.supercell, _ = expand_box(self.supercell, self.origin)

        else:
            vac_add = (uu * vt) / np.einsum('ij,ij->', nu, uu)
            sup_vac = np.copy(self.supercell)
            nbi = self.non_boundary_idx
            sup_vac[:, nbi:nbi + 1] += vac_add
            self.supercell = sup_vac

        # Add new meta information:
        bv_meta = self.meta.get('boundary_vac')
        if bv_meta is None:
            self.meta.update({
                'boundary_vac': [bv_dict]
            })
        else:
            bv_meta.append(bv_dict)

        new_vac_thick = self.boundary_vac + vt

        if self.boundary_vac != 0:
            warnings.warn('`boundary_vac` is already non-zero ({}). '
                          'Summing to new value: {}'.format(
                              self.boundary_vac, new_vac_thick))

        new_vac_type = func
        if self.boundary_vac_type is not None:
            new_vac_type = self.boundary_vac_type + '+' + new_vac_type

        # Update attributes:
        self.boundary_vac = new_vac_thick
        self.boundary_vac_type = new_vac_type

        if wrap:
            self.wrap_sites_to_supercell()

    def apply_relative_shift(self, shift, crystal_idx, wrap=False):
        """
        Apply in-boundary-plane shifts to the specified grain to enable an 
        exploration of the grain boundary's microscopic degrees of freedom.

        Parameters
        ----------
        shift : ndarray of size two
            A two-element array whose elements are the relative shift in of one
            crystal relative to the other in fractional coordinates s of the
            boundary area.
        crystal_idx : int
            Which crystal to shift. Zero-indexed.
        wrap : bool, optional
            If True, sites (atoms, lattice sites, etc) within the supercell are
            wrapped to the supercell volume after shifting. By default, False. 

        """

        sup_type = self.meta['supercell_type']

        if 'bulk_bicrystal' in sup_type:
            raise NotImplementedError(
                'Cannot apply relative shift to a bulk_bicrystal.')

        elif all([i not in sup_type for i in ['bicrystal', 'surface_bicrystal']]):
            raise NotImplementedError(
                'Cannot apply relative shift to this supercell type.')

        shift_frac_gb = np.array(shift).squeeze()

        if np.any(shift_frac_gb < -1) or np.any(shift_frac_gb > 1):
            raise ValueError('Elements of `shift` should be between -1 and 1.')
     
        shift_crystal = self.crystals[crystal_idx]
        shift_frac = np.zeros((3, 1))
        shift_frac[self.boundary_idx, [0, 0]] = shift_frac_gb
        shift_std = np.dot(shift_crystal['crystal'], shift_frac)

        # Translate shifted grain sites:
        atm_shiftd = np.copy(self.atom_sites)
        atm_crys_lab = self.atom_labels['crystal_idx']
        atm_crys_idx = atm_crys_lab[0][atm_crys_lab[1]]
        atm_shiftd[:, np.where(atm_crys_idx == crystal_idx)[0]] += shift_std
        self.atom_sites = atm_shiftd

        if self.lattice_sites is not None:
            lat_shiftd = np.copy(self.lattice_sites)
            lat_crys_lab = self.lattice_labels['crystal_idx']
            lat_crys_idx = lat_crys_lab[0][lat_crys_lab[1]]
            lat_shiftd[:, np.where(lat_crys_idx == crystal_idx)[0]] += shift_std
            self.lattice_sites = lat_shiftd

        if self.interstice_sites is not None:
            int_shiftd = np.copy(self.interstice_sites)
            int_crys_lab = self.interstice_labels['crystal_idx']
            int_crys_idx = int_crys_lab[0][int_crys_lab[1]]
            int_shiftd[:, np.where(int_crys_idx == crystal_idx)[0]] += shift_std
            self.interstice_sites = int_shiftd

        # Update attributes:
        self.crystals[crystal_idx].update({
            'origin': shift_crystal['origin'] + shift_std
        })

        if self.relative_shift != [0, 0]:
            warnings.warn('`relative_shift` is already non-zero. Resetting to '
                          'new value.')
        self.relative_shift = [i + j for i,
                               j in zip(shift_frac_gb.tolist(), self.relative_shift)]

        if self.maintain_inv_sym:
            # Modify out-of-boundary supercell vector
            sup_shift = np.copy(self.supercell)
            sup_shift[:, self.non_boundary_idx][:, None] += (2 * shift_std)

            # Update attribute:
            self.supercell = sup_shift

        if wrap:
            self.wrap_sites_to_supercell()

    def wrap_sites_to_supercell(self, sites='all'):
        """
        Wrap atoms to within the boundary plane as defined by the supercell.

        """
        sup_type = self.meta['supercell_type']

        if 'bulk_bicrystal' in sup_type:
            raise NotImplementedError(
                'Cannot wrap atoms within a bulk_bicrystal.')

        elif all([i not in sup_type for i in ['bicrystal',
                                              'surface_bicrystal']]):
            raise NotImplementedError(
                'Cannot wrap atoms within this supercell type.')

        super().wrap_sites_to_supercell(sites=sites, dirs=self.boundary_idx)

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

    def check_atomic_environment(self, checks_list):
        """Invoke checks of the atomic environment."""

        super().check_atomic_environment(checks_list)

        allowed_checks = {
            'bicrystal_inversion_symmetry': self.check_inv_symmetry,
        }

        for chk, func in allowed_checks.items():
            if chk in checks_list:
                func()


def csl_bicrystal_from_parameters(crystal_structure, csl_vecs, box_csl=None,
                                  gb_type=None, gb_size=None,
                                  edge_conditions=None,
                                  overlap_tol=1, reorient=True, wrap=True,
                                  maintain_inv_sym=False,
                                  boundary_vac=None,
                                  relative_shift=None):
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
    boundary_vac_flat_args : dict, optional
        If not None, after construction of the boundary,
        apply_boundary_vac_flat() is invoked with this dict as keyword
        arguments. Default is None.
    boundary_vac_linear_args : dict, optional
        If not None, after construction of the boundary,
        apply_boundary_vac_linear() is invoked with this dict as keyword
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

    # Rotate crystal B onto A:
    crys_b.rotate(rot_mat)

    # Shift crystals to form a supercell at the origin
    zs = - crys_b.box_vecs[:, NBI:NBI + 1]
    crys_a.translate(zs)
    crys_b.translate(zs)

    atom_sites = np.hstack((crys_a.atom_sites, crys_b.atom_sites))
    lattice_sites = np.hstack((crys_a.lattice_sites, crys_b.lattice_sites))

    atm_crystal_idx = np.array([0] * crys_a.atom_sites.shape[1] +
                               [1] * crys_b.atom_sites.shape[1])
    atm_cry_idx_dict = {'crystal_idx': (np.array([0, 1]), atm_crystal_idx)}

    atom_labels = {**atm_cry_idx_dict}
    for k, v in crys_a.atom_labels.items():
        atom_labels.update({
            k: (v[0], np.hstack((v[1], crys_b.atom_labels[k][1])))
        })

    lat_crystal_idx = np.array([0] * crys_a.lattice_sites.shape[1] +
                               [1] * crys_b.lattice_sites.shape[1])
    lat_cry_idx_dict = {'crystal_idx': (np.array([0, 1]), lat_crystal_idx)}

    lat_labels = {**lat_cry_idx_dict}
    for k, v in crys_a.lattice_labels.items():
        lat_labels.update({
            k: (v[0], np.hstack((v[1], crys_b.lattice_labels[k][1])))
        })

    int_sites, int_labels = None, None
    if crys_a.interstice_sites is not None:

        int_crystal_idx = np.array([0] * crys_a.interstice_sites.shape[1] +
                                   [1] * crys_b.interstice_sites.shape[1])
        int_cry_idx_dict = {'crystal_idx': (np.array([0, 1]), int_crystal_idx)}

        int_sites = np.hstack(
            (crys_a.interstice_sites, crys_b.interstice_sites))
        int_labels = {**int_cry_idx_dict}
        for k, v in crys_a.interstice_labels.items():
            int_labels.update({
                k: (v[0], np.hstack((v[1], crys_b.interstice_labels[k][1])))
            })

    # Define the supercell:
    sup_std = np.copy(crys_a.box_vecs)
    sup_std[:, NBI] = crys_a.box_vecs[:, NBI] - crys_b.box_vecs[:, NBI]

    crystals = [
        {
            'crystal': crys_a.box_vecs,
            'origin': zs,
            'cs_idx': 0,
            'cs_orientation': np.eye(3),
            'cs_origin': [0, 0, 0]
        },
        {
            'crystal': crys_b.box_vecs,
            'origin': zs,
            'cs_idx': 0,
            'cs_orientation': rot_mat,
            'cs_origin': [0, -1, 0]
        }
    ]

    # AtomisticStructure parameters
    as_params = {
        'supercell': sup_std,
        'atom_sites': atom_sites,
        'atom_labels': atom_labels,
        'lattice_sites': lattice_sites,
        'lattice_labels': lat_labels,
        'interstice_sites': int_sites,
        'interstice_labels': int_labels,
        'crystals': crystals,
        'crystal_structures': [crystal_structure],
        'overlap_tol': overlap_tol,
    }

    # Bicrystal parameters
    bc_params = {
        'as_params': as_params,
        'maintain_inv_sym': maintain_inv_sym,
        'reorient': reorient,
        'boundary_vac': boundary_vac,
        'relative_shift': relative_shift,
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

    bc = csl_bicrystal_from_parameters(**bc_params)
    bc.meta.update({'supercell_type': ['bulk', 'bulk_bicrystal']})
    return bc


def csl_surface_bicrystal_from_parameters(crystal_structure, csl_vecs,
                                          box_csl=None, gb_type=None,
                                          gb_size=None, edge_conditions=None,
                                          overlap_tol=1,
                                          reorient=True, wrap=True,
                                          maintain_inv_sym=False,
                                          boundary_vac=None,
                                          relative_shift=None,
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
        'boundary_vac': boundary_vac,
        'relative_shift': relative_shift,
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


def mon_bicrystal_180_u0w(crystal_structure, gb_params, overlap_tol=0.1,
                          reorient=True, maintain_inv_sym=False,
                          boundary_vac=None, relative_shift=None,
                          wrap=False):
    """
    Construct a 180° [u0w] twin monoclinic boundary.

    Parameters
    ----------
    crystal_structure : CrystalStructure
    gb_params: dict (str : string or ndarray or int)
        uvw_vecs : ndarray of size (3, 3)
            The vectors of one of the crystals expressed in Miller indices of 
            the primitive lattice. 
        repeats : list
            Number of unit cell repeats [Nx, Ny, Nz] for both crystals, where 
            x-direction is normal to GB plane. Default value is [3, 1, 1].
        bound_vac : int
            Vacuum thickness to add at boundary (Angstrom)
        transls : list (REMOVE)
            Translation steps in the two directions of boundary plane 
            (fractions of unit cell vectors)
        term_plns : list
            Termination planes for grains `a` and `b` if any exist as listed below. 
            Allowed values for [001] 180°: '100' or '200' (equivalent to '-200').
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
    Returns
    -------
    Bicrystal
    """

    gb_params_def = {
        'repeats': [3, 1, 1],
        'bound_vac': 0.0,
        'transls': [0.0, 0.0],
        'term_plns': None
    }

    gb_params = {**gb_params_def, **gb_params}
    gb_params_req = ['uvw_vecs']

    for p in gb_params_req:
        if p not in gb_params:
            raise ValueError('Missing key in `gb_params`: {}'.format(p))

    # Get parameters from dict
    uvw_vecs = gb_params['uvw_vecs']
    repeats = gb_params['repeats']
    bound_vac = gb_params['bound_vac']
    transls = gb_params['transls']
    term_plns = gb_params['term_plns']

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

    if term_plns:
        # Termination planes for grains `a` and `b`
        term_plane_a = term_plns[0]
        term_plane_b = term_plns[1]

    # Read in data for input structure and set up cell, pos_f, species, natoms
    # lat_data = castep.read_cell_file(cellfile)

    # Non-primitive lattice
    uvw_vecs = np.array(uvw_vecs)
    nonprim_latt = gbhelper.create_nonprimitive_unitcell(uvw_vecs,
                                                         crystal_structure,
                                                         abs_coord=False)
    cell = nonprim_latt[0]
    pos_f = nonprim_latt[1]
    species_key = nonprim_latt[2]
    species = nonprim_latt[3]
    natoms = len(species)

    # Translate atoms based on termination planes for grains `a` and `b`.
    pos_f_a = np.zeros((natoms, 3))
    pos_f_b = np.zeros((natoms, 3))

    if term_plns:
        pos_f_a = term_h00(pos_f, plane=term_plane_a)
        pos_f_b = term_h00(pos_f, plane=term_plane_b)
    else:
        pos_f_a = np.copy(pos_f)
        pos_f_b = np.copy(pos_f)

    pos_a_a = gbhelper.frac2abs(cell, pos_f_a)
    pos_a_b = gbhelper.frac2abs(cell, pos_f_b)

    # Rotation matrix
    R = rotation_matrix(np.array([0, 0, 1]), np.array([180]), degrees=True)
    R = gbhelper.zero_prec(R).squeeze()

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
    crystal = np.zeros((3, 3), dtype=float)
    crystal[:, 0] = 0.5 * supercell[0]
    crystal[:, 1:3] = supercell[1:3].T

    crystals = [{}, {}]

    for item in crystals:
        item.update({'crystal': crystal})
        item.update({'origin': np.zeros((3, 1))})
        #  Don't really understand how these work
        #         item.update( {'cs_idx': 0})
        #         item.update( {'cs_orientation': R})
        #         item.update( {'cs_origin': [0.0,0.0]})

    crystals[1]['origin'] = 0.5 * supercell[:, 0:1]

    # Populate arrays of atom positions and types for a grain boundary supercell
    natoms_t = Nt * natoms
    species_gb_x = np.zeros(natoms_t, dtype=int)
    pos_gb_a = np.zeros((natoms_t, 3), dtype=float)
    pos_gb_b = np.zeros((natoms_t, 3), dtype=float)
    # atom belonging to crystal index
    crystals_idx = np.concatenate((np.zeros(natoms_t, dtype=int),
                                   np.ones(natoms_t,  dtype=int)))

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

    atom_labels = {
        'species': (np.array(species_key), np.concatenate((species_gb_x, species_gb_x))),
        'crystal_idx': (np.array([0, 1]), crystals_idx)
    }

    # AtomisticStructure parameters
    as_params = {
        'supercell': supercell.T,
        'atom_sites': np.concatenate((pos_gb_a, pos_gb_b)).T,
        'atom_labels': atom_labels,
        #         'lattice_sites': lattice_sites,
        #         'lattice_labels': lat_labels,
        #         'interstice_sites': int_sites,
        #         'interstice_labels': int_labels,
        'crystals': crystals,
        'crystal_structures': [crystal_structure],
        'overlap_tol': overlap_tol,
    }

    # Bicrystal parameters
    bc_params = {
        'as_params': as_params,
        'maintain_inv_sym': maintain_inv_sym,
        'reorient': reorient,
        'boundary_vac': boundary_vac,
        'relative_shift': relative_shift,
        'wrap': wrap,
        'nbi': 0,
        #         'rot_mat': rot_mat, what is this?
    }

    return Bicrystal(**bc_params)
