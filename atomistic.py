"""
Classes for generating atomistic structures useful for atomistic simulations.

Attributes
----------
REF_PATH : str
    The path of the `ref` directory, which contains reference data.

TODO:
-   Investigate building a helper class for structure visualisations which can
    be shared between BravaisLattice, CrystalStructure, CrystalBox etc. Or
    maybe not a class, but some way of sharing some code at least...

"""

import numpy as np
import os
from plotly import graph_objs
from plotly.offline import init_notebook_mode, plot, iplot
import simsio
import geometry
import vectors
import mathsutils
import utils
import readwrite

REF_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref')


class AtomisticStructure(object):
    """
    Class to represent crystals of atoms

    Attributes
    ----------
    atom_sites : ndarray of shape (3, N)
        Array of column vectors representing the atom positions.
    supercell : ndarray of shape (3, 3)
        Array of column vectors representing supercell edge vectors.
    lattice_sites : ndarray of shape (3, M), optional
        Array of column vectors representing lattice site positions.
    crystals : list of dict of (str : ndarray or int), optional
        Each dict contains at least these keys:
            `crystal` : ndarray of shape (3, 3)
                Array of column vectors representing the crystal edge vectors.
            `origin` : ndarray of shape (3, 1)
                Column vector specifying the origin of this crystal.
        Additional keys are:
            'cs_idx': int
                Index of `crystal_structures`, defining to which
                CrystalStructure this crystal belongs.
            `cs_orientation`: ndarray of shape (3, 3)
                Rotation matrix which rotates the CrystalStructure lattice
                unit cell from the initialised BravaisLattice object to some
                other desired orientation.
            'cs_origin': list of float or int
                Origin of the CrystalStructure unit cell in multiples of the
                CrystalStructure unit cell vectors. For integer values, this
                will not affect the atomic structure of the crystal. To
                produce a rigid translation of the atoms within the crystal,
                non-integer values can be used.

    crystal_structures : list of CrystalStructure, optional
    crystal_idx : ndarray of shape (N,), optional
        Defines to which crystal each atom belongs.
    lat_crystal_idx : ndarray of shape (M,), optional
        Defines to which crystal each lattice site belongs
    species_idx : ndarray of shape (N,), optional
        Defines to which species each atom belongs, indexed within the atom's
        crystal_structure. For atom index `i`, this indexes
        `crystal_structures[
            crystals[crystal_idx[i]]['cs_idx']]['species_set']`
    motif_idx : ndarray of shape (N,), optional
        Defines to which motif atom each atom belongs, indexed within the
        atom's crystal_structure. For atom index `i`, this indexes
        `crystal_structures[
            crystals[crystal_idx[i]]['cs_idx']]['species_motif']`

    Methods
    -------
    get_atom_species():
        Return a str ndarray representing the species of each atom.
    visualise():
        Use a plotting library to visualise the atomistic structure.

    """

    def __init__(self, atom_sites, supercell, lattice_sites=None,
                 crystals=None, crystal_structures=None, crystal_idx=None,
                 lat_crystal_idx=None, species_idx=None, motif_idx=None):
        """Constructor method for AtomisticStructure object."""

        # Input validation
        # ----------------
        # 1.    Check length of `crystal_idx`, `species_idx`, and `motif_idx`
        #       match number of atoms in `atom_sites`.
        # 2.    Check length of 'lat_crystal_idx' matches number of lattice
        #       sites in `lattice_sites'.
        # 3.    Check set of indices in `crystal_idx` resolve in `crystals`.

        if crystal_idx is not None:
            if len(crystal_idx) != atom_sites.shape[1]:
                raise ValueError('Length of `crystal_idx` must match number '
                                 'of atoms specified as column vectors in '
                                 '`atom_sites`.')

            c_idx_set = sorted(list(set(crystal_idx)))
            if c_idx_set[0] < 0 or c_idx_set[-1] >= len(crystals):
                raise ValueError('Indices in `crystal_idx` must index elements'
                                 ' in `crystals`.')

        if lat_crystal_idx is not None:
            if len(lat_crystal_idx) != lattice_sites.shape[1]:
                raise ValueError('Length of `lat_crystal_idx` must match '
                                 'number of lattice sites specified as column '
                                 'vectors in `lattice_sites`.')

        if species_idx is not None:
            if len(species_idx) != atom_sites.shape[1]:
                raise ValueError('Length of `species_idx` must match number '
                                 'of atoms specified as column vectors in '
                                 '`atom_sites`.')

        if motif_idx is not None:
            if len(motif_idx) != atom_sites.shape[1]:
                raise ValueError('Length of `motif_idx` must match number '
                                 'of atoms specified as column vectors in '
                                 '`atom_sites`.all')

        # Set attributes
        # --------------

        self.atom_sites = atom_sites
        self.supercell = supercell

        self.lattice_sites = lattice_sites
        self.crystals = crystals
        self.crystal_structures = crystal_structures
        self.crystal_idx = crystal_idx
        self.lat_crystal_idx = lat_crystal_idx
        self.species_idx = species_idx
        self.motif_idx = motif_idx

    def visualise(self, show_iplot=True, save=False, save_args=None):
        """
        TODO:
        -   Add 3D arrows/cones to supercell vectors using mesh3D.
        -   Add lattice vectors/arrows to lattice unit cells

        """

        # Validation:
        if not show_iplot and not save:
            raise ValueError('Visualisation will not be displayed or saved!')

        crystal_cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Get colours for atom species:
        atom_cols = readwrite.read_pickle(
            os.path.join(REF_PATH, 'jmol_colours.pickle'))

        data = []

        # Supercell box
        sup_xyz = geometry.get_box_xyz(self.supercell)[0]
        sup_props = {
            'mode': 'lines',
            'line': {
                'color': '#98df8a'
            }
        }
        data.append(
            graph_objs.Scatter3d(
                x=sup_xyz[0],
                y=sup_xyz[1],
                z=sup_xyz[2],
                name='Supercell',
                **sup_props
            )
        )

        # Supercell edge vectors
        sup_edge_props = {
            'mode': 'lines',
            'line': {
                'color': '#000000',
                'width': 3
            }
        }
        sup_vec_labs = ['a', 'b', 'c']
        for i in range(3):
            data.append(
                graph_objs.Scatter3d(
                    x=[0, self.supercell[0, i]],
                    y=[0, self.supercell[1, i]],
                    z=[0, self.supercell[2, i]],
                    name='Supercell vectors',
                    legendgroup='Supercell vectors',
                    showlegend=True if i == 0 else False,
                    **sup_edge_props
                )
            )
            data.append(
                graph_objs.Scatter3d(
                    mode='text',
                    x=[self.supercell[0, i]],
                    y=[self.supercell[1, i]],
                    z=[self.supercell[2, i]],
                    text=[sup_vec_labs[i]],
                    legendgroup='Supercell vectors',
                    showlegend=False,
                    textfont={
                        'size': 20
                    }
                )
            )

        # Crystal boxes and atoms
        for c_idx, c in enumerate(self.crystals):

            # Crystal box
            c_xyz = geometry.get_box_xyz(c['crystal'], origin=c['origin'])[0]
            c_props = {
                'mode': 'lines',
                'line': {
                    'color': crystal_cols[c_idx]
                }
            }
            data.append(
                graph_objs.Scatter3d(
                    x=c_xyz[0],
                    y=c_xyz[1],
                    z=c_xyz[2],
                    name='Crystal #{}'.format(c_idx + 1),
                    **c_props
                )
            )

            # Get CrystalStructure associated with this crystal:
            cs = self.crystal_structures[c['cs_idx']]

            # Lattice unit cell, need to rotate by given orientation
            unit_cell = np.dot(c['cs_orientation'], cs.bravais_lattice.vecs)

            cs_origin = np.dot(unit_cell, c['cs_origin'])
            uc_origin = c['origin'] + cs_origin[:, np.newaxis]
            uc_xyz = geometry.get_box_xyz(unit_cell, origin=uc_origin)[0]

            uc_trace_name = 'Unit cell (crystal #{})'.format(c_idx + 1)
            uc_props = {
                'mode': 'lines',
                'line': {
                    'color': 'gray'
                },
                'name': uc_trace_name,
                'legendgroup': uc_trace_name,
                'visible': 'legendonly'
            }
            data.append(
                graph_objs.Scatter3d(
                    x=uc_xyz[0],
                    y=uc_xyz[1],
                    z=uc_xyz[2],
                    **uc_props
                )
            )

            # Lattice sites
            ls_idx = np.where(self.lat_crystal_idx == c_idx)[0]
            ls = self.lattice_sites[:, ls_idx]
            ls_trace_name = 'Lattice sites (crystal #{})'.format(c_idx + 1)
            lat_site_props = {
                'mode': 'markers',
                'marker': {
                    'symbol': 'x',
                    'size': 5,
                    'color': crystal_cols[c_idx]
                },
                'name': ls_trace_name,
                'legendgroup': ls_trace_name,
                'visible': 'legendonly',
            }

            data.append(
                graph_objs.Scatter3d(
                    x=ls[0],
                    y=ls[1],
                    z=ls[2],
                    **lat_site_props
                )
            )

            # Get motif associated with this crystal:
            sp_motif = cs.species_motif

            # Get indices of atoms in this crystal
            crys_atm_idx = np.where(self.crystal_idx == c_idx)[0]

            # Atoms by species
            for sp_idx, sp_name in enumerate(sp_motif):

                atom_idx = np.where(self.motif_idx[crys_atm_idx] == sp_idx)[0]
                atom_sites_sp = self.atom_sites[:, crys_atm_idx[atom_idx]]
                sp_col = str(atom_cols[cs.motif['species'][sp_idx]])

                trace_name = sp_name + ' (crystal #{})'.format(c_idx + 1)

                atom_site_props = {
                    'mode': 'markers',
                    'marker': {
                        'symbol': 'o',
                        'size': 7,
                        'color': 'rgb' + sp_col
                    },
                    'name': trace_name,
                    'legendgroup': trace_name,
                }

                data.append(
                    graph_objs.Scatter3d(
                        x=atom_sites_sp[0],
                        y=atom_sites_sp[1],
                        z=atom_sites_sp[2],
                        **atom_site_props
                    )
                )

            # TODO: Add traces for atom numbers

        layout = graph_objs.Layout(
            width=1000,
            height=800,
            scene={
                'aspectmode': 'data'
            }
        )

        fig = graph_objs.Figure(data=data, layout=layout)

        if show_iplot:
            iplot(fig)

        if save:
            if save_args is None:
                save_args = {}
            plot(fig, **save_args)

    def reorient_to_lammps(self):
        """
        Reorient the supercell and its contents to a LAMMPS-compatible
        orientation.

        Returns
        -------
        ndarray of shape (3, 3)
            Rotation matrix used to reorient the supercell and its contents

        """

        # Reorient supercell
        sup_inv = np.linalg.inv(self.supercell)
        sup_lmps = simsio.get_LAMMPS_compatible_box(self.supercell)
        R = np.dot(sup_lmps, sup_inv)
        self.supercell = sup_lmps

        # Reorient atom sites
        self.atom_sites = np.dot(R, self.atom_sites)

        # Reorient lattice sites
        self.lattice_sites = np.dot(R, self.lattice_sites)

        # Reorient CrystalStructure lattice objects
        for c_idx in range(len(self.crystals)):

            c = self.crystals[c_idx]
            c['crystal'] = np.dot(R, c['crystal'])
            c['origin'] = np.dot(R, c['origin'])
            c['cs_orientation'] = np.dot(R, c['cs_orientation'])

        return R

    def wrap_atoms_to_supercell(self, dirs=None, wrap_lattice_sites=False):
        """
        Wrap atoms to within the supercell.

        Parameters
        ----------
        dirs : list of int, optional
            Supercell direction indices to apply wrapping. Default is None, in
            which case atoms are wrapped in all directions.
        wrap_lattice_sites : bool, optional
            If True, lattice sites are also wrapped. Default is False.

        TODO:
        -   Implement `wrap_lattice_sites`.

        """

        # Validation
        if dirs is not None:
            if len(set(dirs)) != len(dirs):
                raise ValueError('Indices in `dirs` must not be repeated.')

            if len(dirs) not in [1, 2, 3]:
                raise ValueError('`dirs` must be a list of length 1, 2 or 3.')

            for d in dirs:
                if d not in [0, 1, 2]:
                    raise ValueError('`dirs` must be a list whose elements are'
                                     '0, 1 or 2.')

        # Get atom sites in supercell basis:
        sup_inv = np.linalg.inv(self.supercell)
        as_sup = np.dot(sup_inv, self.atom_sites)

        # Wrap atoms:
        as_sup_wrp = np.copy(as_sup)
        as_sup_wrp[dirs] -= np.floor(as_sup_wrp[dirs])

        # Snap to 0 and 1:
        as_sup_wrp = vectors.snap_arr_to_val(as_sup_wrp, 0, 1e-12)
        as_sup_wrp = vectors.snap_arr_to_val(as_sup_wrp, 1, 1e-12)
        as_sup_wrp[as_sup_wrp == 1] = 0

        # Convert back to Cartesian basis
        as_std_wrp = np.dot(self.supercell, as_sup_wrp)

        # Update attributes:
        self.atom_sites = as_std_wrp

    @property
    def num_atoms_per_crystal(self):
        """Computes number of atoms in each crystal, returns a list."""
        na = []
        for c_idx in range(len(self.crystals)):
            na.append(np.where(self.crystal_idx == c_idx)[0].shape[0])

        return na

    @property
    def num_atoms(self):
        """Computes total number of atoms."""
        return self.atom_sites.shape[1]

    @property
    def num_crystals(self):
        """Returns number of crystals."""
        return len(self.crystals)

    @property
    def reciprocal_supercell(self):
        """Returns the reciprocal supercell as array of column vectors."""

        v = self.supercell
        cross_1 = np.cross(v[:, 1], v[:, 2])
        cross_2 = np.cross(v[:, 0], v[:, 2])
        cross_3 = np.cross(v[:, 0], v[:, 1])

        B = np.zeros((3, 3))
        B[:, 0] = 2 * np.pi * cross_1 / (np.dot(v[:, 0], cross_1))
        B[:, 1] = 2 * np.pi * cross_2 / (np.dot(v[:, 1], cross_2))
        B[:, 2] = 2 * np.pi * cross_3 / (np.dot(v[:, 2], cross_3))

        return B

    def get_kpoint_grid(self, separation):
        """
        Get the MP kpoint grid size for a given kpoint separation.

        Parameters
        ----------
        separation : float or int or ndarray of shape (3, )
            Maximum separation between kpoints, in units of inverse Angstroms.
            If an array, this is the separations in each reciprocal supercell
            direction.

        Returns
        -------
        ndarray of int of shape (3, )
            MP kpoint grid dimensions along each reciprocal supercell
            direction.

        """

        recip = self.reciprocal_supercell
        grid = np.ceil(np.round(
            np.linalg.norm(recip, axis=0) / (separation * 2 * np.pi),
            decimals=8)
        ).astype(int)

        return grid

    def get_kpoint_spacing(self, grid):
        """
        Get the kpoint spacing given an MP kpoint grid size.

        Parameters
        ----------
        grid : list of length 3
            Grid size in each of the reciprocal supercell directions.

        Returns
        -------
        ndarray of shape (3, )
            Separation between kpoints in each of the reciprocal supercell
            directions.

        """

        recip = self.reciprocal_supercell
        seps = np.linalg.norm(recip, axis=0) / (np.array(grid) * 2 * np.pi)

        return seps

    # def __str__(self):
    #     pass

    # def __repr__(self):
    #     pass


class BulkCrystal(AtomisticStructure):
    """

    Attributes
    ----------
    crystal_structure : CrystalStructure

    TODO:
    -   Add proper support for cs_orientation and cs_origin. Maybe allow one of
        `box_lat` or `box_std` for the more general case.

    """

    def __init__(self, crystal_structure, box_lat):
        """Constructor method for BulkCrystal object."""

        supercell = np.dot(crystal_structure.bravais_lattice.vecs, box_lat)
        cb = CrystalBox(crystal_structure, supercell)
        atom_sites = cb.atom_sites_std
        lattice_sites = cb.lat_sites_std
        crystal_idx = np.zeros(atom_sites.shape[1])
        lat_crystal_idx = np.zeros(lattice_sites.shape[1])
        crystals = [{
            'crystal': supercell,
            'origin': np.zeros((3, 1)),
            'cs_idx': 0,
            'cs_orientation': np.eye(3),
            'cs_origin': [0, 0, 0]
        }]

        super().__init__(atom_sites,
                         supercell,
                         lattice_sites=lattice_sites,
                         crystals=crystals,
                         crystal_structures=[crystal_structure],
                         crystal_idx=crystal_idx,
                         lat_crystal_idx=lat_crystal_idx,
                         species_idx=cb.species_idx,
                         motif_idx=cb.motif_idx)

    # def __str__(self):
    #     pass

    # def __repr__(self):
    #     pass


class CSLBicrystal(AtomisticStructure):
    """
    Class to represent a bicrystal supercell constructed using CSL vectors.

    Attributes
    ----------
    GB_TYPES : dict of str : ndarray of shape (3, 3)
        Some
    atoms_gb_dist : ndarray of shape (N,)
        Perpendicular distances of each atom from the origin boundary plane

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
        If True, methods acting on the `CSLBicrystal` object will maintain
        inversion symmetry of the bicrystal.
    reorient : bool, optional
        If True, after construction of the boundary, reorient_to_lammps() is
        invoked. Default is True.
    boundary_vac_args : dict, optional 
        If not None, after construction of the boundary, apply_boundary_vac()
        is invoked with this dict as keyword arguments. Default is None.
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
    -   Clarify the docstring about `maintain_inv_sym`; inversion symmetry
        will be maintained about what point?
    -   Sort out lattice sites in apply_boundary_vac() & apply_relative_shift()
    -   Rename wrap_atoms_to_supercell to wrap_to_supercell and apply wrapping
        to lattice sites, and crystal boxes as well.

    """

    GB_TYPES = {
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

    def __init__(self, crystal_structure, csl_vecs, box_csl=None,
                 gb_type=None, gb_size=None, edge_conditions=None,
                 maintain_inv_sym=False, reorient=True,
                 boundary_vac_args=None, relative_shift_args=None,
                 wrap=True):
        """Constructor method for CSLBicrystal object."""

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

            if gb_type not in CSLBicrystal.GB_TYPES:
                raise ValueError(
                    'Invalid `gb_type`: {}. Must be one of {}'.format(
                        gb_type, list(CSLBicrystal.GB_TYPES.keys())))

            box_csl = CSLBicrystal.GB_TYPES.get(gb_type) * gb_size

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
            rot_angle = 0
            rot_mat = np.eye(3)

        else:
            rot_angle = vectors.col_wise_angles(
                csl_vecs_std[0], csl_vecs_std[1])[0]

            rot_mat = vectors.rotation_matrix(rot_ax_std[:, 0], rot_angle)[0]

        rot_angle_deg = np.rad2deg(rot_angle)

        # Check if grain A forms a right-handed coordinate system:
        grn_vol = np.dot(np.cross(grn_a_std[:, 0],
                                  grn_a_std[:, 1]), grn_a_std[:, 2])

        if grn_vol < 0:
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
        sup_std[:, NBI] *= 2

        # Boundary normal vector:
        n = np.cross(sup_std[:, BI[0]], sup_std[:, BI[1]])[:, np.newaxis]
        n_unit = n / np.linalg.norm(n)

        # Non-boundary supercell unit vector
        u = sup_std[:, NBI:NBI + 1]
        u_unit = u / np.linalg.norm(u)

        # Set instance CSLBicrystal-specific attributes:
        self.maintain_inv_sym = maintain_inv_sym
        self.n_unit = n_unit
        self.u_unit = u_unit
        self.non_boundary_idx = NBI
        self.boundary_idx = BI

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

        # Call parent constructor
        super().__init__(atom_sites, sup_std,
                         lattice_sites=lattice_sites,
                         crystals=crystals,
                         crystal_structures=[crystal_structure],
                         crystal_idx=crystal_idx,
                         lat_crystal_idx=lat_crystal_idx,
                         species_idx=species_idx,
                         motif_idx=motif_idx)

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
        -   Understand/fix behaviour for negative vac_thick
        -   Also apply to lattice sites

        """

        vt = vac_thickness
        if vt < 0:
            raise NotImplementedError('`vt` must be a positive number.')

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

    def apply_relative_shift(self, shift):
        """
        Apply in-boundary-plane shifts to grain_a to explore the microscopic
        degrees of freedom.

        `shift` is a 2 element array whose elements are the
        relative shift in fractional coords of the boundary area.

        TODO:
        -   Also apply to lattice sites

        """

        shift = np.array(shift)
        if any(shift <= -1) or any(shift >= 1):
            raise ValueError('Elements of `shift` should be between -1 and 1.')

        # Convenience:
        grn_a = self.crystals[0]
        nbi = self.non_boundary_idx
        bi = self.boundary_idx

        shift_gb = np.zeros((3, 1))
        shift_gb[bi] = shift[:, np.newaxis]
        shift_std = np.dot(grn_a['crystal'], shift_gb)

        # Translate grain A atoms:
        as_shift = np.copy(self.atom_sites)
        as_shift[:, np.where(self.crystal_idx == 0)[0]] += shift_std

        # Translate grain A origin:
        grn_a_org_shift = grn_a['origin'] + shift_std

        # Update attributes:
        self.atom_sites = as_shift
        self.crystals[0].update({
            'origin': grn_a_org_shift
        })

        if self.maintain_inv_sym:
            # Modify out-of-boundary supercell vector
            sup_shift = np.copy(self.supercell)
            sup_shift[:, nbi:nbi + 1] += (2 * shift_std)

            # Update attribute:
            self.supercell = sup_shift

    def wrap_atoms_to_supercell(self):
        """
        Wrap atoms to within the boundary plane as defined by the supercell.

        """

        super().wrap_atoms_to_supercell(dirs=self.boundary_idx)


class CSLBulkCrystal(CSLBicrystal):
    """
    Class to represent a bulk crystal constructed in a similar way to a
    CSLBicrystal object.

    This is a convenience class to generate complimentary bulk supercells for a
    given CSLBicrystal supercell.

    """

    def __init__(self, crystal_structure, csl_vecs, box_csl=None,
                 gb_type=None, gb_size=None, edge_conditions=None,
                 reorient=True):
        """Constructor method for CSLBulkCrystal object."""

        super().__init__(crystal_structure, [csl_vecs, csl_vecs],
                         box_csl=box_csl, gb_type=gb_type, gb_size=gb_size,
                         edge_conditions=edge_conditions, reorient=reorient,
                         wrap=False)

    def apply_boundary_vac(self, *args, **kwargs):

        raise NotImplementedError(
            'Cannot apply boundary vacuum to a CSLBulkCrystal.')

    def apply_relative_shift(self, *args, **kwargs):

        raise NotImplementedError(
            'Cannot apply relative shift to a CSLBulkCrystal.')

    def wrap_atoms_to_supercell(self, *args, **kwargs):

        raise NotImplementedError(
            'Cannot wrap atoms within supercell in a CSLBulkCrystal.')


class CSLSurfaceCrystal(CSLBicrystal):
    """
    Class to represent a surface crystal constructed in a similar way to a
    CSLBicrystal object.

    This is a convenience class to generate complimentary surface supercells
    for a given CSLBicrystal supercell.

    """

    def __init__(self, crystal_structure, csl_vecs, box_csl=None,
                 gb_type=None, gb_size=None, edge_conditions=None,
                 maintain_inv_sym=False, reorient=True, boundary_vac_args=None,
                 relative_shift_args=None, wrap=True, surface_idx=0):
        """Constructor method for CSLSurfaceCrystal object."""

        super().__init__(crystal_structure, [csl_vecs, csl_vecs],
                         box_csl=box_csl, gb_type=gb_type, gb_size=gb_size,
                         edge_conditions=edge_conditions,
                         maintain_inv_sym=maintain_inv_sym,
                         reorient=reorient,
                         boundary_vac_args=boundary_vac_args,
                         relative_shift_args=relative_shift_args,
                         wrap=wrap)

        # Remove atoms from removed crystal
        atoms_keep = np.where(self.crystal_idx == surface_idx)[0]
        self.atom_sites = self.atom_sites[:, atoms_keep]
        self.species_idx = self.species_idx[atoms_keep]
        self.motif_idx = self.motif_idx[atoms_keep]
        self.crystal_idx = self.crystal_idx[atoms_keep]

        # Remove lattice sites from removed crystal
        lat_keep = np.where(self.lat_crystal_idx == surface_idx)[0]
        self.lattice_sites = self.lattice_sites[:, lat_keep]
        self.lat_crystal_idx = self.lat_crystal_idx[lat_keep]


class CrystalBox(object):
    """
    Class to represent a parallelopiped filled with a crystal structure.

    Attributes
    ----------
    crystal_structure : CrystalStructure
    box_vecs : ndarray of shape (3, 3)
        Array of column vectors representing the edge vectors of the
        parallelopiped to fill with crystal.
    bounding_box : dict of str : ndarray
        Dict of arrays defining the bounding box used to generate an array of
        candidate lattice and atom sites.
    lat_sites_std : ndarray of shape (3, N)
        Array of column vectors representing the lattice sites in Cartesian
        coordinates.
    lat_sites_frac : ndarray of shape (3, N)
        Array of column vectors representing the lattice sites in fractional
        coordinates of the crystal box.
    atom_sites_std : ndarray of shape (3, M)
        Array of column vectors representing the atoms in Cartesian
        coordinates.
    atom_sites_frac : ndarray of shape (3, M)
        Array of column vectors representing the atoms in fractional
        coordinates of the crystal box.
    species_idx : ndarray of shape (M)
        Identifies the species for each of the M atoms.
    species : list of str
        List of element symbols for species present in the crystal.

    """

    def __init__(self, crystal_structure, box_vecs, edge_conditions=None):
        """
        Fill a parallelopiped with atoms belonging to a given crystal
        structure.

        Parameters
        ----------
        crystal_structure : CrystalStructure
        box_vecs : ndarray of shape (3, 3)
            Array of column vectors representing the edge vectors of the
            parallelopiped to fill with crystal.
        edge_conditions : list of str, optional
            Determines if atom and lattice sites on the edges of the `box_vecs`
            parallelopiped should be included. It is a list of three
            two-character strings, each being a `1` or `0`. These refer to
            whether atoms are included (`1`) or not (`0`) for the near and far
            boundary along the dimension given by the position in the list. The
            near boundary is the boundary of the crystal box which intercepts
            the crystal box origin. Default is None, in which case it will be
            set to ['10', '10', '10']. For a given component, say x, the
            strings are decoded in the following way:
                '00': 0 <  x <  1
                '01': 0 <  x <= 1
                '10': 0 <= x <  1
                '11': 0 <= x <= 1

        Notes
        -----
        Algorithm proceeds as follows:
        1.  Form a bounding parallelopiped around the parallelopiped defined by
            `box_vecs`, whose edge vectors are parallel to the lattice vectors.
        2.  Find all lattice and atom sites within and on the edges/corners of
            that bounding box.
        3.  Transform sites to the box basis.
        4.  Find valid lattice and atom sites, whcih have vector components in
            the interval [0, 1] in the box basis, where the interval may be
            (half-)closed/open depending on the specified edge conditions.

        TODO:
        -   Check that plotting atom indices is correct, think it's not.
        -   Add proper support for cs_orientation and cs_origin.

        """

        if edge_conditions is None:
            edge_conditions = ['10', '10', '10']

        lat_vecs = crystal_structure.bravais_lattice.vecs

        # Get the bounding box of box_vecs whose vectors are parallel to the
        # crystal lattice. Use padding to catch edge atoms which aren't on
        # lattice sites.
        bounding_box = geometry.get_bounding_box(
            box_vecs, bound_vecs=lat_vecs, padding=1)
        box_vecs_inv = np.linalg.inv(box_vecs)

        b_box = bounding_box['bound_box'][0]
        b_box_origin = bounding_box['bound_box_origin'][:, 0]
        b_box_bv = bounding_box['bound_box_bv'][:, 0]
        b_box_origin_bv = bounding_box['bound_box_origin_bv'][:, 0]

        self.bounding_box = bounding_box
        self.crystal_structure = crystal_structure

        # Get all lattice sites within the bounding box, as column vectors:
        unit_cell_origins = np.vstack(np.meshgrid(
            range(b_box_origin_bv[0],
                  b_box_origin_bv[0] + b_box_bv[0] + 1),
            range(b_box_origin_bv[1],
                  b_box_origin_bv[1] + b_box_bv[1] + 1),
            range(b_box_origin_bv[2],
                  b_box_origin_bv[2] + b_box_bv[2] + 1))).reshape((3, -1))

        # Consider all lattice sites within each unit cell origin:
        ls_lat = np.concatenate(
            unit_cell_origins + crystal_structure.lat_sites_frac.T.reshape(
                (-1, 3, 1)), axis=1)

        # Transform lattice sites to original box basis
        ls_std = np.dot(lat_vecs, ls_lat)
        ls_box = np.dot(box_vecs_inv, ls_std)

        # Get all atom sites within the bounding box, as column vectors:
        as_lat = np.concatenate(
            ls_lat + crystal_structure.motif['atom_sites'].T.reshape(
                (-1, 3, 1)), axis=1)

        as_std = np.dot(lat_vecs, as_lat)
        as_box = np.dot(box_vecs_inv, as_std)

        species_idx = np.repeat(crystal_structure.lat_site_species_idx,
                                ls_lat.shape[1])
        motif_idx = np.repeat(crystal_structure.lat_site_motif_idx,
                              ls_lat.shape[1])

        tol = 1e-14
        ls_box = vectors.snap_arr_to_val(ls_box, 0, tol)
        ls_box = vectors.snap_arr_to_val(ls_box, 1, tol)
        as_box = vectors.snap_arr_to_val(as_box, 0, tol)
        as_box = vectors.snap_arr_to_val(as_box, 1, tol)

        # Form a boolean edge condition array based on `edge_condtions`. Start
        # by allowing all sites:
        cnd_lat = np.ones(ls_box.shape[1], dtype=bool)
        cnd_atm = np.ones(as_box.shape[1], dtype=bool)

        for dir_idx, pt in enumerate(edge_conditions):

            if pt[0] == '1':
                cnd_lat = np.logical_and(cnd_lat, ls_box[dir_idx] >= 0)
                cnd_atm = np.logical_and(cnd_atm, as_box[dir_idx] >= 0)

            elif pt[0] == '0':
                cnd_lat = np.logical_and(cnd_lat, ls_box[dir_idx] > 0)
                cnd_atm = np.logical_and(cnd_atm, as_box[dir_idx] > 0)

            if pt[1] == '1':
                cnd_lat = np.logical_and(cnd_lat, ls_box[dir_idx] <= 1)
                cnd_atm = np.logical_and(cnd_atm, as_box[dir_idx] <= 1)

            elif pt[1] == '0':
                cnd_lat = np.logical_and(cnd_lat, ls_box[dir_idx] < 1)
                cnd_atm = np.logical_and(cnd_atm, as_box[dir_idx] < 1)

        inbox_lat_idx = np.where(cnd_lat)[0]
        ls_box_in = ls_box[:, inbox_lat_idx]
        ls_std_in = ls_std[:, inbox_lat_idx]

        inbox_atm_idx = np.where(cnd_atm)[0]
        as_box_in = as_box[:, inbox_atm_idx]
        as_std_in = as_std[:, inbox_atm_idx]

        species_idx = species_idx[inbox_atm_idx]
        motif_idx = motif_idx[inbox_atm_idx]

        ls_std_in = vectors.snap_arr_to_val(ls_std_in, 0, tol)
        as_std_in = vectors.snap_arr_to_val(as_std_in, 0, tol)

        self.box_vecs = box_vecs
        self.lat_sites_std = ls_std_in
        self.lat_sites_frac = ls_box_in
        self.atom_sites_std = as_std_in
        self.atom_sites_frac = as_box_in
        self.species_idx = species_idx
        self.motif_idx = motif_idx
        self.species_set = crystal_structure.species_set
        self.species_motif = crystal_structure.species_motif

    def visualise(self):
        """
        Plot the crystal structure using Plotly.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        box_xyz = geometry.get_box_xyz(self.box_vecs)[0]

        lattice_site_props = {
            'mode': 'markers',
            'marker': {
                'color': 'rgb(100,100,100)',
                'symbol': 'x',
                'size': 5
            },
            'name': 'Lattice sites',
            'legendgroup': 'Lattice sites'
        }

        data = [
            graph_objs.Scatter3d(
                x=box_xyz[0],
                y=box_xyz[1],
                z=box_xyz[2],
                mode='lines',
                name='Crystal box'
            ),
            graph_objs.Scatter3d(
                x=self.lat_sites_std[0],
                y=self.lat_sites_std[1],
                z=self.lat_sites_std[2],
                **lattice_site_props
            )
        ]

        bound_box_xyz = geometry.get_box_xyz(
            self.bounding_box['bound_box'][0],
            origin=self.bounding_box['bound_box_origin'])[0]

        # Add the bounding box trace:
        data.append(
            graph_objs.Scatter3d(
                x=bound_box_xyz[0],
                y=bound_box_xyz[1],
                z=bound_box_xyz[2],
                mode='lines',
                marker={
                    'color': 'red'
                },
                name='Bounding box',
                visible='legendonly'
            )
        )

        # Get colours for atom species:
        atom_cols = readwrite.read_pickle(
            os.path.join(REF_PATH, 'jmol_colours.pickle'))

        for sp_idx, sp_name in enumerate(self.species_motif):

            atom_idx = np.where(self.motif_idx == sp_idx)[0]
            atom_sites_sp = self.atom_sites_std[:, atom_idx]
            sp_col = str(
                atom_cols[self.crystal_structure.motif['species'][sp_idx]])

            atom_site_props = {
                'mode': 'markers',
                'marker': {
                    'symbol': 'o',
                    'size': 7,
                    'color': 'rgb' + sp_col
                },
                'name': sp_name,
                'legendgroup': sp_name,
            }

            # Add traces for this atom species to the Bravais lattice data:
            data.append(
                graph_objs.Scatter3d(
                    x=self.atom_sites_std[0, atom_idx],
                    y=self.atom_sites_std[1, atom_idx],
                    z=self.atom_sites_std[2, atom_idx],
                    **atom_site_props
                )
            )

            # Add traces for atom numbers
            data.append(
                graph_objs.Scatter3d(
                    x=self.atom_sites_std[0, atom_idx],
                    y=self.atom_sites_std[1, atom_idx],
                    z=self.atom_sites_std[2, atom_idx],
                    **{
                        'mode': 'text',
                        'text': [str(i) for i in atom_idx],
                        'name': 'Atom index',
                        'legendgroup': 'Atom index',
                        'showlegend': True if sp_idx == 0 else False,
                        'visible': 'legendonly'
                    }
                )
            )

        layout = graph_objs.Layout(
            width=650,
            scene={
                'aspectmode': 'data'
            }
        )

        fig = graph_objs.Figure(data=data, layout=layout)
        iplot(fig)


class CrystalStructure(object):
    """
    Class to represent a crystal structure with a lattice and atomic motif.

    Parameters
    ----------
    bravais_lattice : BravaisLattice
    motif : dict
        atom_sites : ndarray of shape (3, P)
            Array of column vectors representing positions of the atoms
            associated with each lattice site. Given in fractional coordinates
            of the lattice unit cell.
        species : list of length P of str
            Species names associated with each atom site.

    Attributes
    ----------
    bravais_lattice: BravaisLattice
    motif: dict
    lat_sites_std : ndarray of shape (3, N)
    lat_sites_frac : ndarray of shape (3, N)
    atom_sites_std : ndarray of shape (3, M)
    atom_sites_frac : ndarray of shape (3, M)

    """

    def __init__(self, bravais_lattice, motif):
        """
        Constructor method for CrystalStructure object.
        """

        self.bravais_lattice = bravais_lattice
        self.motif = motif
        self.lat_sites_frac = bravais_lattice.lat_sites_frac
        self.lat_sites_std = bravais_lattice.lat_sites_std

        # Add atomic motif to each lattice site
        self.atom_sites_frac = np.concatenate(
            self.lat_sites_frac + motif['atom_sites'].T.reshape(
                (-1, 3, 1)), axis=1)

        # Get unique species in the motif
        species_set = []
        for m in motif['species']:
            if m not in species_set:
                species_set.append(m)

        # Label species by their repetition in the motif
        species_motif = []
        species_map = []
        species_count = [0] * len(species_set)
        for m in motif['species']:
            set_idx = species_set.index(m)
            species_map.append(set_idx)
            i = species_count[set_idx]
            species_count[set_idx] += 1
            species_motif.append(m + ' #{}'.format(i + 1))

        # motif index (indexes species_motif) for a single lattice site:
        self.lat_site_motif_idx = np.arange(self.motif['atom_sites'].shape[1])

        # motif index (indexes species_motif) for all lattice sites:
        self.motif_idx = np.repeat(self.lat_site_motif_idx,
                                   self.lat_sites_frac.shape[1])

        # species index (indexes species_set) for a single lattice site:
        lat_site_species_idx = np.copy(self.lat_site_motif_idx)
        for sm_idx, sm in enumerate(species_map):
            if sm_idx != sm:
                lat_site_species_idx[lat_site_species_idx == sm_idx] = sm

        self.lat_site_species_idx = lat_site_species_idx

        # species index (indexes species_set) for all lattice sites:
        self.species_idx = np.repeat(self.lat_site_species_idx,
                                     self.lat_sites_frac.shape[1])

        self.species_set = species_set
        self.species_motif = species_motif

        self.atom_sites_std = np.dot(
            self.bravais_lattice.vecs, self.atom_sites_frac)

    def visualise(self, periodic_sites=False):
        """
        Plot the crystal structure using Plotly.

        Parameters
        ----------
        periodic_sites : bool, optional
            If True, show atom and lattice sites in the unit cell which are
            periodically equivalent. Helpful for visualising lattices. Default
            is False.

        Returns
        -------
        None

        """

        # Get the data for the Bravais lattice:
        b_data = self.bravais_lattice.get_fig_data(periodic_sites)

        # Get colours for atom species:
        atom_cols = readwrite.read_pickle(
            os.path.join(REF_PATH, 'jmol_colours.pickle'))

        for sp_idx, sp_name in enumerate(self.species_motif):

            atom_idx = np.where(self.motif_idx == sp_idx)[0]
            atom_sites_sp = self.atom_sites_std[:, atom_idx]
            sp_col = str(atom_cols[self.motif['species'][sp_idx]])

            atom_site_props = {
                'mode': 'markers',
                'marker': {
                    'symbol': 'o',
                    'size': 7,
                    'color': 'rgb' + sp_col
                },
                'name': sp_name,
                'legendgroup': sp_name,
            }

            # Add traces for this atom species to the Bravais lattice data:
            b_data.append(
                graph_objs.Scatter3d(
                    x=self.atom_sites_std[0, atom_idx],
                    y=self.atom_sites_std[1, atom_idx],
                    z=self.atom_sites_std[2, atom_idx],
                    **atom_site_props
                )
            )

            # Add traces for atom numbers
            b_data.append(
                graph_objs.Scatter3d(
                    x=self.atom_sites_std[0, atom_idx],
                    y=self.atom_sites_std[1, atom_idx],
                    z=self.atom_sites_std[2, atom_idx],
                    **{
                        'mode': 'text',
                        'text': [str(i) for i in atom_idx],
                        'name': 'Atom index',
                        'legendgroup': 'Atom index',
                        'showlegend': True if sp_idx == 0 else False,
                        'visible': 'legendonly'
                    }
                )
            )

        layout = graph_objs.Layout(
            width=650,
            scene={
                'aspectmode': 'data'
            }
        )

        fig = graph_objs.Figure(data=b_data, layout=layout)
        iplot(fig)

    def __repr__(self):

        return ('CrystalStructure(\n'
                '\t' + self.bravais_lattice.__repr__() + '\n'
                '\t' + '{!r}'.format(self.motif) + '\n'
                ')')

    def __str__(self):

        motif_str = ''
        for sp_idx, sp in enumerate(self.species_motif):
            motif_str += sp + ' @ ' + str(
                self.motif['atom_sites'][:, sp_idx]) + '\n'

        return ('{!s}-{!s} Bravais lattice + {!s}-atom motif\n\n'
                'Lattice parameters:\n'
                'a = {!s}\nb = {!s}\nc = {!s}\n'
                'α = {!s}°\nβ = {!s}°\nγ = {!s}°\n'
                '\nLattice vectors = \n{!s}\n'
                '\nLattice sites (fractional) = \n{!s}\n'
                '\nLattice sites (Cartesian) = \n{!s}\n'
                '\nMotif in fractional coordinates of '
                'unit cell = \n{!s}\n').format(
                    self.bravais_lattice.lattice_system,
                    self.bravais_lattice.centring_type,
                    self.motif['atom_sites'].shape[1],
                    self.bravais_lattice.a, self.bravais_lattice.b,
                    self.bravais_lattice.c, self.bravais_lattice.α,
                    self.bravais_lattice.β, self.bravais_lattice.γ,
                    self.bravais_lattice.vecs,
                    self.bravais_lattice.lat_sites_frac,
                    self.bravais_lattice.lat_sites_std,
                    motif_str)


class BravaisLattice(object):
    """
    Class to represent a Bravais lattice unit cell.

    Parameters
    ----------
    lattice_system : str
        Lattice system is one of: cubic, rhombohedral, orthorhombic,
        tetragonal, monoclinic, triclinic, hexagonal.
    centring_type : str, optional
        The centring type of the lattice, also known as the lattice type, is
        one of P (primitive), B (base-centred), I (body-centred), F
        (face-centred) or R (rhombohedrally-centred). Not all centring types
        are compatible with all lattice systems. Default is None, in which case
        the rhomboherally-centred centring type (for the rhombohedral lattice
        system) or primitive centring type (for all other lattice systems) will
        be chosen.
    a : float, optional
        Lattice parameter, magnitude of the first unit cell edge vector.
    b : float, optional
        Lattice parameter, magnitude of the second unit cell edge vector.
    c : float, optional
        Lattice parameter, magnitude of the third unit cell edge vector.
    α : float, optional
        Lattice parameter, angle in degrees between the second and third unit
        cell edge vectors.
    β : float, optional
        Lattice parameter, angle in degrees between the first and third unit
        cell edge vectors.
    γ : float, optional
        Lattice parameter, angle in degrees between the first and second unit
        cell edge vectors.

    Attributes
    ----------
    lattice_system : str
        Lattice system
    centring_type : str
        Centring type also known as lattice type.
    a : float
        Lattice parameter, magnitude of the first unit cell edge vector.
    b : float
        Lattice parameter, magnitude of the second unit cell edge vector.
    c : float
        Lattice parameter, magnitude of the third unit cell edge vector.
    α : float
        Lattice parameter, angle in degrees between the second and third unit
        cell edge vectors.
    β : float
        Lattice parameter, angle in degrees between the first and third unit
        cell edge vectors.
    γ : float
        Lattice parameter, angle in degrees between the first and second unit
        cell edge vectors.
    vecs : ndarray of shape (3, 3)
        Array of column vectors defining the lattice unit cell
    lat_sites_std: ndarray of shape (3, N)
        Array of column vectors defining the Cartesian positions of lattice
        sites within the lattice unit cell.
    lat_sites_frac: ndarray of shape (3, N)
        Array of column vectors defining the fractional positions of lattice
        sites within the lattice unit cell.

    Methods
    -------
    plot()
        Plot the lattice using Plotly

    Notes
    -----
    Lattice vectors are formed by aligning the crystallographic x-axis (i.e
    with magnitude `a`) along the Cartesian x-axis and aligning the
    crystallographic xy-plane (i.e. the vectors with magnitudes `a` and `b`)
    parallel to the Cartesian xy-plane.

    Conventional unit cells are generated and expected in the lattice parameter
    parameters. For instance the rhombohedral lattice system is represented
    with a rhombohedrally-centred hexagonal unit cell, rather than a primitive
    rhombohedral cell.

    References
    ----------
    <Insert reference here to the 14 Bravais lattices in 3D.>

    Examples
    --------
    <Insert example here>

    TODO:
    -   Add primitive centring type to allowed centring types for rhombohedral
        lattice system.
    -   Fix bug: if we set hex a to be equal to the default for c, then we get
        an error: BravaisLattice('hexagonal', a=2)
    -   Regarding resrictions/validations on lattice types, need to allow
        restrictions to be "any n parameters must be..." rather than "the first
        two parameters must be". E.g. monoclinic: "two of the angle parameters
        must be 90 deg" rather than "parameters 3 and 5 must be 90 deg".

    """

    def __init__(self, lattice_system, centring_type=None,
                 a=None, b=None, c=None, α=None, β=None, γ=None):
        """Constructor method for BravaisLattice object."""

        if centring_type is None:
            if lattice_system == 'rhombohedral':
                centring_type = 'R'
            else:
                centring_type = 'P'

        # List valid Bravais lattice systems and centering types:
        all_lattice_systems = ['triclinic', 'monoclinic', 'orthorhombic',
                               'tetragonal', 'rhombohedral', 'hexagonal',
                               'cubic']
        all_centring_types = ['P', 'B', 'I', 'F', 'R']

        # Check specified lattice system and centering type are compatible:
        if lattice_system not in all_lattice_systems:
            raise ValueError('"{}" is not a valid lattice system. '
                             '`lattice_system` must be one of: {}.'.format(
                                 lattice_system, all_lattice_systems))

        if centring_type not in all_centring_types:
            raise ValueError('"{}" is not a valid centering type. '
                             'centring_type must be one of {}.'.format(
                                 centring_type, all_centring_types))

        if (
            (lattice_system not in ['orthorhombic', 'cubic'] and
             centring_type == 'F')
            or (lattice_system not in ['orthorhombic', 'cubic', 'tetragonal']
                and centring_type == 'I')
            or (lattice_system not in ['orthorhombic', 'monoclinic']
                and centring_type == 'B')
            or (lattice_system == 'rhombohedral' and centring_type != 'R')
            or (lattice_system != 'rhombohedral' and centring_type == 'R')
        ):
            raise ValueError('Lattice system {} and centering type {} are not '
                             'compatible.'.format(
                                 lattice_system, centring_type))

        # Check specified lattice parameters are compatible with specified
        # lattice system:

        if lattice_system == 'triclinic':
            equal_to = {}
            not_equal_to = {3: 90, 4: 90, 5: 90}
            equal_groups = []
            unique_groups = [[0, 1, 2], [3, 4, 5]]
            defaults = {0: 1, 1: 2, 2: 3, 3: 60, 4: 80, 5: 50}

        elif lattice_system == 'monoclinic':
            equal_to = {3: 90}
            not_equal_to = {4: 90}
            equal_groups = [[3, 5]]
            unique_groups = [[0, 1, 2], [3, 4], [4, 5]]
            defaults = {0: 1, 1: 2, 2: 3, 3: 90, 4: 100, 5: 90}

        elif lattice_system == 'orthorhombic':
            equal_to = {3: 90, 4: 90, 5: 90}
            not_equal_to = {}
            equal_groups = [[3, 4, 5]]
            unique_groups = [[0, 1, 2]]
            defaults = {0: 1, 1: 2, 2: 3, 3: 90, 4: 90, 5: 90}

        elif lattice_system == 'tetragonal':
            equal_to = {3: 90, 4: 90, 5: 90}
            not_equal_to = {}
            equal_groups = [[0, 1], [3, 4, 5]]
            unique_groups = [[0, 2]]
            defaults = {0: 1, 1: 1, 2: 2, 3: 90, 4: 90, 5: 90}

        elif lattice_system == 'rhombohedral':
            equal_to = {3: 90, 5: 120}
            not_equal_to = {}
            equal_groups = [[0, 1], [3, 4]]
            unique_groups = [[0, 2]]
            defaults = {0: 1, 1: 1, 2: 2, 3: 90, 4: 90, 5: 120}

        elif lattice_system == 'hexagonal':
            equal_to = {3: 90, 5: 120}
            not_equal_to = {}
            equal_groups = [[0, 1], [3, 4]]
            unique_groups = [[0, 2]]
            defaults = {0: 1, 1: 1, 2: 2, 3: 90, 4: 90, 5: 120}

        elif lattice_system == 'cubic':
            equal_to = {3: 90, 4: 90, 5: 90}
            not_equal_to = {}
            equal_groups = [[0, 1, 2], [3, 4, 5]]
            unique_groups = []
            defaults = {0: 1, 1: 1, 2: 1, 3: 90, 4: 90, 5: 90}

        a, b, c, α, β, γ = utils.validate_numeric_params(
            [a, b, c, α, β, γ],
            equal_to=equal_to,
            not_equal_to=not_equal_to,
            equal_groups=equal_groups,
            unique_groups=unique_groups,
            defaults=defaults)

        self.a = a
        self.b = b
        self.c = c
        self.α = α
        self.β = β
        self.γ = γ
        self.lattice_system = lattice_system
        self.centring_type = centring_type

        # Form lattice column vectors from lattice parameters by aligining `a`
        # along x and `b` in the xy plane:

        α_rad = np.deg2rad(α)
        β_rad = np.deg2rad(β)
        γ_rad = np.deg2rad(γ)

        a_x = self.a
        b_x = self.b * np.cos(γ_rad)
        b_y = self.b * np.sin(γ_rad)
        c_x = self.c * np.cos(β_rad)
        c_y = (abs(self.c) * abs(self.b) * np.cos(α_rad) - b_x * c_x) / b_y
        c_z = np.sqrt(c**2 - c_x**2 - c_y**2)

        vecs = np.array([
            [a_x,   0,   0],
            [b_x, b_y,   0],
            [c_x, c_y, c_z]
        ]).T

        self.vecs = vectors.snap_arr_to_val(vecs, 0, 1e-14)

        # Set lattice sites for the specified centring type:
        if centring_type == 'P':  # Primitive
            lat_sites_frac = np.array([[0, 0, 0]], dtype=float).T

        elif centring_type == 'B':  # Base-centred
            lat_sites_frac = np.array([
                [0, 0, 0],
                [0.5, 0.5, 0]
            ]).T

        elif centring_type == 'I':  # Body-centred
            lat_sites_frac = np.array([
                [0, 0, 0],
                [0.5, 0.5, 0.5]
            ]).T

        elif centring_type == 'F':  # Face-centred
            lat_sites_frac = np.array([
                [0, 0, 0],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5]
            ]).T

        elif centring_type == 'R':  # Rhombohedrally-centred
            lat_sites_frac = np.array([
                [0, 0, 0],
                [1 / 3, 2 / 3, 1 / 3],
                [2 / 3, 1 / 3, 2 / 3],
            ]).T

        # Find lattice sites in Cartesian basis:
        lat_sites_std = np.dot(vecs, lat_sites_frac)

        self.lat_sites_frac = lat_sites_frac
        self.lat_sites_std = vectors.snap_arr_to_val(lat_sites_std, 0, 1e-15)

    def visualise(self, periodic_sites=False):
        """
        Plot the lattice unit cell with lattice points using Plotly.

        Parameters
        ----------
        periodic_sites : bool, optional
            If True, show lattice sites in the unit cell which are periodically
            equivalent. Helpful for visualising lattices. Default is False.

        Returns
        -------
        None

        """

        data = self.get_fig_data(periodic_sites)
        layout = graph_objs.Layout(
            width=650,
            scene={
                'aspectmode': 'data'
            }
        )
        fig = graph_objs.Figure(data=data, layout=layout)
        iplot(fig)

    def get_fig_data(self, periodic_sites):
        """
        Get the Plotly traces for visualising the Bravais lattice.

        Parameters
        ----------
        periodic_sites : bool, optional
            If True, show lattice sites in the unit cell which are periodically
            equivalent. Helpful for visualising lattices. Default is False.

        Returns
        -------
        list of Plotly trace objects

        TODO:
        -   Fix bug: if `periodic_sites` is True for face-centred, we don't get
            all the expected lattice sites shown. Need to change the method for
            doing this.
        -   Add plotting of specified crystallographic planes using Plotly's
            mesh3D trace type.
        -   Add some arrows to show the unit cell vectors.

        """

        lat_xyz = geometry.get_box_xyz(self.vecs)[0]

        lattice_site_props = {
            'mode': 'markers',
            'marker': {
                'color': 'rgb(100,100,100)',
                'symbol': 'x',
                'size': 5
            },
            'name': 'Lattice sites',
            'legendgroup': 'Lattice sites'
        }

        data = [
            graph_objs.Scatter3d(
                x=lat_xyz[0],
                y=lat_xyz[1],
                z=lat_xyz[2],
                mode='lines',
                name='Unit cell'
            ),
            graph_objs.Scatter3d(
                x=self.lat_sites_std[0],
                y=self.lat_sites_std[1],
                z=self.lat_sites_std[2],
                **lattice_site_props
            )
        ]

        if periodic_sites:
            periodic_edge_sites = np.array([
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]
            ])

            periodic_edge_sites_std = np.dot(self.vecs,
                                             periodic_edge_sites.T)

            for i in self.lat_sites_frac.T:
                if i in periodic_edge_sites:
                    data.append(
                        graph_objs.Scatter3d(
                            x=periodic_edge_sites_std[0],
                            y=periodic_edge_sites_std[1],
                            z=periodic_edge_sites_std[2],
                            showlegend=False,
                            **lattice_site_props,
                        )
                    )
                    break

        return data

    def __repr__(self):

        return ('BravaisLattice({!r}, centring_type={!r}, a={!r}, '
                'b={!r}, c={!r}, α={!r}, β={!r}, γ={!r})').format(
                    self.lattice_system, self.centring_type,
                    self.a, self.b, self.c, self.α, self.β, self.γ)

    def __str__(self):

        return ('{!s}-{!s} Bravais lattice\n\n'
                'Lattice parameters:\n'
                'a = {!s}\nb = {!s}\nc = {!s}\n'
                'α = {!s}°\nβ = {!s}°\nγ = {!s}°\n'
                '\nLattice vectors = \n{!s}\n'
                '\nLattice sites (fractional) = \n{!s}\n'
                '\nLattice sites (Cartesian) = \n{!s}\n').format(
                    self.lattice_system, self.centring_type,
                    self.a, self.b, self.c, self.α, self.β, self.γ, self.vecs,
                    self.lat_sites_frac, self.lat_sites_std)
