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
from plotly.offline import plot, iplot
from crystal import CrystalBox, CrystalStructure
import simsio
import geometry
import vectors
import mathsutils
import readwrite

REF_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref')


class AtomisticSimulation(object):

    def __init__(self, atomistic_structure, options):

        self.structure = atomistic_structure
        self.options = options

    def write_input_files(self):

        if self.options['method'] == 'castep':

            cst_opt = self.options['castep']
            set_opt = self.options['set_up']

            cst_in_params = {
                'supercell': self.structure.supercell,
                'atom_sites': self.structure.atom_sites,
                'species': self.structure.all_species,
                'species_idx': self.structure.all_species_idx,
                'path': set_opt['stage_series_path'],
                'seedname': cst_opt['seedname'],
                'cell': cst_opt['cell'],
                'param': cst_opt['param'],
                'cell_constraints': cst_opt['cell_constraints'],
                'atom_constraints': cst_opt['atom_constraints'],
            }

            simsio.write_castep_inputs(**cst_in_params)


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
        Either specify (`all_species` and `all_species_idx`) or (`species_idx`
        and `motif_idx`), but not both.
    motif_idx : ndarray of shape (N,), optional
        Defines to which motif atom each atom belongs, indexed within the
        atom's crystal_structure. For atom index `i`, this indexes
        `crystal_structures[
            crystals[crystal_idx[i]]['cs_idx']]['species_motif']`
        Either specify (`all_species` and `all_species_idx`) or (`species_idx`
        and `motif_idx`), but not both.
    all_species : ndarray of str, optional
        1D array of strings representing the distinct species. Either specify
        (`all_species` and `all_species_idx`) or (`species_idx` and
        `motif_idx`), but not both.
    all_species_idx : ndarray of shape (N, ), optional
        Defines to which species each atom belongs, indexed over the whole
        AtomisticStructure. This indexes `all_species`. Either specify
        (`all_species` and `all_species_idx`) or (`species_idx` and
        `motif_idx`), but not both.

    num_atoms_per_crystal
    num_atoms
    num_crystals
    reciprocal_supercell

    Methods
    -------
    todo

    TODO:
    -   Think about how some of the methods would work if __init__() takes the
        bare minimum: supercell and atom_sites, or with some additional params
        like lattice_sites.

    """

    def __init__(self, atom_sites, supercell, lattice_sites=None,
                 crystals=None, crystal_structures=None, crystal_idx=None,
                 lat_crystal_idx=None, species_idx=None, motif_idx=None,
                 all_species=None, all_species_idx=None):
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

        if [i is None for i in [all_species, all_species_idx]].count(True) == 1:
            raise ValueError('Must specify both `all_species` and '
                             '`all_species_idx`.')

        if [i is None for i in [species_idx, motif_idx]].count(True) == 1:
            raise ValueError('Must specify both `species_idx` and '
                             '`motif_idx`.')

        if [i is None for i in [species_idx, all_species_idx]].count(True) != 1:
            raise ValueError('Either specify (`all_species` and '
                             '`all_species_idx`) or (`species_idx` and '
                             '`motif_idx`), but not both.')

        if species_idx is not None:
            if len(species_idx) != atom_sites.shape[1]:
                raise ValueError('Length of `species_idx` must match number '
                                 'of atoms specified as column vectors in '
                                 '`atom_sites`.')

        if motif_idx is not None:
            if len(motif_idx) != atom_sites.shape[1]:
                raise ValueError('Length of `motif_idx` must match number '
                                 'of atoms specified as column vectors in '
                                 '`atom_sites`.')

        if all_species_idx is not None:
            if len(all_species_idx) != atom_sites.shape[1]:
                raise ValueError('Length of `all_species_idx` must match '
                                 'number of atoms specified as column vectors '
                                 'in `atom_sites`.')

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
        self._all_species = all_species
        self._all_species_idx = all_species_idx

        self.check_overlapping_atoms()

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

        if self.crystals is None:
            # Plot all atoms by species:

            for sp_idx, sp in enumerate(self.all_species):

                atom_idx = np.where(self.all_species_idx == sp_idx)[0]
                atom_sites_sp = self.atom_sites[:, atom_idx]
                sp_col = str(atom_cols[sp])

                trace_name = sp

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

        else:

            ccent = self.crystal_centres

            # Plot atoms by crystal and motif
            # Crystal boxes and atoms
            for c_idx, c in enumerate(self.crystals):

                # Crystal centre
                data.append(
                    graph_objs.Scatter3d(
                        x=ccent[c_idx][0],
                        y=ccent[c_idx][1],
                        z=ccent[c_idx][2],
                        name='Crystal #{} centre'.format(c_idx + 1),
                        mode='markers',
                        marker={
                            'color': 'red',
                            'symbol': 'x',
                            'size': 3
                        }
                    )
                )

                # Crystal box
                c_xyz = geometry.get_box_xyz(
                    c['crystal'], origin=c['origin'])[0]
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
                unit_cell = np.dot(c['cs_orientation'],
                                   cs.bravais_lattice.vecs)

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
                # TODO: Add traces for atom numbers
                for sp_idx, sp_name in enumerate(sp_motif):

                    atom_idx = np.where(
                        self.motif_idx[crys_atm_idx] == sp_idx)[0]
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

        # Snap to 0:
        as_sup_wrp = vectors.snap_arr_to_val(as_sup_wrp, 0, 1e-12)

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

    def get_all_species(self):

        all_sp = []
        all_sp_idx = []
        all_sp_count = 0

        for c_idx in range(len(self.crystals)):

            cs = self.crystal_structures[self.crystals[c_idx]['cs_idx']]

            # Local species for this crystal:
            cs_sp = cs.motif['species']

            # Local species index for this crystal:
            c_sp_idx = np.copy(
                self.species_idx[np.where(self.crystal_idx == c_idx)[0]])

            # Need to map the indices from local CrystalStructure to global
            # AtomisticStructure
            for sp_idx, sp in enumerate(cs_sp):

                if sp not in all_sp:
                    new_sp_idx = all_sp_count
                    all_sp.append(sp)
                    all_sp_count += 1

                else:
                    new_sp_idx = all_sp.index(sp)

                c_sp_idx[np.where(c_sp_idx == sp_idx)[0]] = new_sp_idx

            all_sp_idx.extend(c_sp_idx)

        all_sp = np.array(all_sp)
        all_sp_idx = np.array(all_sp_idx)

        return all_sp, all_sp_idx

    @property
    def all_species(self):
        """"""

        if self.species_idx is None:
            return self._all_species

        else:
            all_sp, _ = self.get_all_species()
            return all_sp

    @property
    def all_species_idx(self):
        """"""

        if self.species_idx is None:
            return self._all_species_idx

        else:
            _, all_sp_idx = self.get_all_species()
            return all_sp_idx

    @property
    def crystal_centres(self):
        """Get the midpoints of each crystal in the structure."""

        return [geometry.get_box_centre(c['crystal'], origin=c['origin'])
                for c in self.crystals]

    def get_tiled_atoms(self, tiles):
        """
        Get atom sites tiled by some integer factors in each supercell
        direction.

        Atoms are tiled in the positive supercell directions.

        Parameters
        ----------
        tiles : tuple or list of length 3
            Number of repeats in each supercell direction.        

        Returns
        -------
        ndarray

        """

        invalid_msg = ('`tiles` must be a tuple or list of 3 integers greater'
                       ' than 0.')
        if len(tiles) != 3:
            raise ValueError(invalid_msg)

        as_tiled = np.copy(self.atom_sites)
        for t_idx, t in enumerate(tiles):

            if t == 1:
                continue

            if not isinstance(t, int) or t == 0:
                raise ValueError(invalid_msg)

            v = self.supercell[:, t_idx:t_idx + 1]
            all_t = (v * np.arange(1, t)).T[:, :, np.newaxis]
            as_tiled_t = np.hstack(all_t + as_tiled)
            as_tiled = np.hstack([as_tiled, as_tiled_t])

        return as_tiled

    def get_interatomic_dist(self, periodic=True):
        """
        Find the distances between unique atom pairs across the whole 
        structure.

        Parameters
        ----------
        periodic : bool
            If True, the atom sites are first tiled in each supercell direction
            to ensure that distances between periodic cells are considered. 
            Currently, this is crude, and so produces interatomic distances 
            between like atoms (i.e. of one supercell vector length).

        Returns
        ------
        ndarray of shape (N,)

        TODO: 
        -   Improve consideration of periodicity. Maybe instead have a function
            `get_min_interatomic_dist` which gets the minimum distances of each
            atom and every other atom, given periodicity.

        """
        if periodic:
            atms = self.get_tiled_atoms([2, 2, 2])
        else:
            atms = self.atom_sites

        return vectors.get_vec_distances(atms)

    def check_overlapping_atoms(self, tol=1):
        """
        Returns True if any atoms are overlapping within a tolerance.abs

        Parameters
        ----------
        tol : float
            Distance below which atoms are considered to be overlapping.abs

        Raises
        ------
        ValueError
            If any atoms are found to overlap.

        Returns
        -------
        None

        """
        dist = self.get_interatomic_dist()
        if np.any(dist < tol):
            raise ValueError('Found overlapping atoms.')

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

        self.check_overlapping_atoms()
        self.check_inv_symmetry()

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

        self.check_overlapping_atoms()
        self.check_inv_symmetry()

    def wrap_atoms_to_supercell(self):
        """
        Wrap atoms to within the boundary plane as defined by the supercell.

        """

        super().wrap_atoms_to_supercell(dirs=self.boundary_idx)
        self.check_inv_symmetry()

    def check_inv_symmetry(self):
        """
        Check atoms exhibit inversion symmetry through the two crystal centres,
        if `self.maintain_inv_sym` is True.

        """

        if self.maintain_inv_sym:
            for cc_idx, cc in enumerate(self.crystal_centres):
                if not geometry.check_centrosymmetry(
                        self.atom_sites, cc, periodic_box=self.supercell):
                    raise ValueError('The bicrystal does not have inversion '
                                     'symmetry through the crystral centres.')


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
