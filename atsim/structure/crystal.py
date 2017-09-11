import numpy as np
import os
from plotly import graph_objs
from plotly.offline import plot, iplot
from atsim.structure.bravais import BravaisLattice
from atsim import geometry, vectors, readwrite
from atsim.simsio import castep

REF_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref')


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
    @classmethod
    def from_file(cls, path, lattice_system, centring_type=None, filetype='.cell'):
        """
        Get bravais_lattice and motif from a file.

        Parameters
        ----------
        path : string
            Path to input file.
        lattice_system : string
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
        filetype : string
            Type of file provided [default: .cell from castep]

        Notes
        -----
        Only works for .cell files.

        """
        if filetype == '.cell':
            latt_data = castep.read_cell_file(path)
        else:
            raise NotImplementedError(
                'File type "{}" is not supported.'.format(filetype))

        params = dict(zip(['a', 'b', 'c', 'α', 'β', 'γ'],
                          latt_data['latt_params']))

        bl = BravaisLattice(lattice_system, centring_type=centring_type,
                            **params, degrees=False)
        motif = latt_data['motif']

        return cls(bl, motif)

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
