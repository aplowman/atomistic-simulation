import numpy as np
import os
from plotly import graph_objs
from plotly.offline import plot, iplot
from atsim.structure.bravais import BravaisLattice
from atsim import geometry, vectors, readwrite, REF_PATH, plotting, utils
from atsim.simsio import castep
from beautifultable import BeautifulTable


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
    motif : dict with the following keys:
        atom_sites : ndarray of shape (3, N)
            Array of column vectors representing positions of the atoms
            associated with each lattice site. Given in fractional coordinates
            of the lattice unit cell.
        species : ndarray or list of length P of str
            Species names associated with each atom site.
        species_idx : ndarray or list of length N of int
            Array which maps each atom site to a chemical symbol in `species`.

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

        # Validation:

        allowed_motif_keys = [
            'atom_sites',
            'species',
            'species_idx',
            'atom_labels',
            'bulk_interstitials',
            'bulk_interstitials_idx',
            'bulk_interstitials_names',
        ]
        required_motif_keys = [
            allowed_motif_keys[0],
            allowed_motif_keys[1],
            allowed_motif_keys[2],
        ]
        for k in required_motif_keys:
            if k not in motif:
                raise ValueError('Motif key: `{}` is required.'.format(k))

        for k, v in motif.items():
            if k not in allowed_motif_keys:
                raise ValueError('Motif key: "{}" is not allowed.'.format(k))

        num_motif_atom_sites = motif['atom_sites'].shape[1]
        num_species_idx = len(motif['species_idx'])
        if num_species_idx != num_motif_atom_sites:
            raise ValueError('`species_idx` has length: {}, '
                             'but number of atom sites in the motif '
                             'is: {}.'.format(num_species_idx, num_motif_atom_sites))

        if (np.max(motif['species_idx']) > len(motif['species']) - 1):
            raise ValueError(
                'Motif key `species_idx` should index motif key `species`.')

        bulk_interstitials = None
        bulk_interstitials_idx = None
        bulk_interstitials_names = None

        if motif.get('bulk_interstitials') is not None:

            bkii = 'bulk_interstitials_idx'
            bkin = 'bulk_interstitials_names'

            bulk_int_info = [i in motif for i in [bkii, bkin]]

            missing_msg = ('Either specify both `{}` and `{}` in the motif or '
                           'neither.'.format(bkii, bkii))

            if any(bulk_int_info):

                if not all(bulk_int_info):
                    raise ValueError(missing_msg)

                else:
                    if (np.max(motif[bkii]) > len(motif[bkin]) - 1):
                        raise ValueError('Motif key `{}` should index motif '
                                         'key `{}`.'.format(bkii, bkin))

                    num_bkii = motif['bulk_interstitials'].shape[1]
                    num_bkii_idx = len(motif[bkii])
                    if num_bkii != num_bkii_idx:
                        raise ValueError('`{}` has length: {}, but number of '
                                         'bulk interstitial sites in the '
                                         'motif is: {}.'.format(bkii, num_bkii_idx, num_bkii))

                    bulk_interstitials_idx = np.array(motif[bkii])
                    bulk_interstitials_names = np.array(motif[bkin])

            bulk_interstitials = np.array(motif['bulk_interstitials'])

        self.bulk_interstitials_frac = bulk_interstitials
        self.bulk_interstitials = np.dot(
            bravais_lattice.vecs, bulk_interstitials)
        self.bulk_interstitials_idx = bulk_interstitials_idx
        self.bulk_interstitials_names = bulk_interstitials_names

        self.bravais_lattice = bravais_lattice
        self.motif = motif

        # Lattice sites
        lat_sites_frac = bravais_lattice.lat_sites_frac
        lat_sites_std = bravais_lattice.lat_sites_std
        num_lat_sites = lat_sites_frac.shape[1]
        self.lattice_sites_frac = lat_sites_frac
        self.lattice_sites_std = lat_sites_std

        # Atom sites: add atomic motif to each lattice site to get
        motif_rs = motif['atom_sites'].T.reshape((-1, 3, 1))
        utils.prt(motif_rs, 'motif_rs')
        atom_sites_frac = np.concatenate(lat_sites_frac + motif_rs, axis=1)
        utils.prt(atom_sites_frac, 'atom_sites_frac')
        atom_sites_std = np.dot(self.bravais_lattice.vecs, atom_sites_frac)
        num_atom_sites = atom_sites_frac.shape[1]
        self.atom_sites = atom_sites_std

        # Map atom sites to species
        species = np.array(motif['species'])
        # utils.prt(species, 'species')
        motif_species_idx = np.array(motif['species_idx'])
        # utils.prt(motif_species_idx, 'motif_species_idx')
        species_idx = np.repeat(motif_species_idx, num_lat_sites)
        # utils.prt(species_idx, 'species_idx')

        species_count = np.ones(len(motif_species_idx)) * np.nan

        for i in range(len(species)):
            w = np.where(motif_species_idx == i)[0]
            species_count[w] = np.arange(len(w))

        species_count = species_count.astype(int)
        species_count = np.tile(species_count.astype(int), num_lat_sites)

        self.species = species
        self.species_idx = species_idx

        # Atom labels: additional optional per-atom info
        atom_labels = {
            'species_count': species_count.astype(int)
        }

        if motif.get('atom_labels') is not None:

            allowed_labels = ['bulk_coord_num', ]

            for k, v in motif['atom_labels'].items():

                # Validation:
                if k not in allowed_labels:
                    raise ValueError(
                        'Atom label: "{}" is not allowed.'.format(k))

                if len(v) != num_motif_atom_sites:
                    raise ValueError('Motif atom label: "{}" has length: {}, '
                                     'but number of atom sites in the motif '
                                     'is: {}.'.format(k, len(v), num_motif_atom_sites))

                atom_labels.update({k: np.tile(v, num_lat_sites)})

        self.atom_labels = atom_labels

    @property
    def atom_sites_frac(self):
        return np.dot(np.linalg.inv(self.bravais_lattice.vecs), self.atom_sites)

    def visualise(self, show_iplot=False, plot_2d='xyz', use_interstitial_names=False,
                  atom_label=None):
        """
        Parameters
        ----------
        use_interstitial_names : bool, optional
            If True, bulk interstitial sites are plotted by names given in 
            `bulk_interstials_names` according to `bulk_interstitials_idx`.
        atom_label : str, optional
            If True, atoms are grouped according to one of their atom labels.
            For instance, if set to `species_count`, which is an atom label
            that is automatically added to the CrystalStructure, atoms will be
            grouped by their position in the motif within their species. So for
            a motif which has two X atoms, these atoms will be plotted on
            separate traces: "X (#1)" and "X (#2)".

        """

        # Get colours for atom species:
        atom_cols = readwrite.read_pickle(
            os.path.join(REF_PATH, 'jmol_colours.pickle'))

        points = []

        for i in range(len(self.species)):

            atom_idx = np.where(self.species_idx == i)[0]
            sp = self.species[i]
            sp_col = 'rgb' + str(atom_cols[sp])

            # Atoms
            if atom_label:

                lab_vals = self.atom_labels[atom_label]
                utils.prt(lab_vals, 'lab_vals')

                if atom_label not in self.atom_labels:
                    raise ValueError('Atom label "{}" does not exist for this '
                                     'CrystalStructure.'.format(atom_label))

                unique_vals = np.unique(lab_vals)
                utils.prt(unique_vals, 'unique_vals')

                for i in unique_vals:
                    w = np.where(lab_vals == i)[0]
                    utils.prt(w, 'w')

                    atom_idx = np.intersect1d(atom_idx, w)
                    utils.prt(atom_idx, 'atom_idx')

                    # Special treatment for `species_count` atom label:
                    if atom_label == 'species_count':
                        atom_name = '{} (#{})'.format(sp, i + 1)
                    else:
                        atom_name = '{} ({}: {})'.format(sp, atom_label, i)

                    points.append({
                        'data': self.atom_sites[:, atom_idx],
                        'colour': sp_col,
                        'symbol': 'o',
                        'name': atom_name,
                    })

            else:
                points.append({
                    'data': self.atom_sites[:, atom_idx],
                    'colour': sp_col,
                    'symbol': 'o',
                    'name': '{}'.format(sp),
                })

        # Lattice sites:
        points.append({
            'data': self.lattice_sites_std,
            'colour': 'gray',
            'symbol': 'x',
            'name': 'Lattice sites'
        })

        # Bulk interstitials
        if self.bulk_interstitials is not None:

            if use_interstitial_names:

                if self.bulk_interstitials_names is None:
                    raise ValueError('Cannot plot bulk interstitials by name '
                                     ' when `bulk_interstials_names` is not assigned.')

                for i in range(self.bulk_interstitials_names.shape[0]):

                    w = np.where(self.bulk_interstitials_idx == i)[0]

                    bi_sites = self.bulk_interstitials[:, w]
                    bi_name = self.bulk_interstitials_names[i]

                    points.append({
                        'data': bi_sites,
                        'colour': 'orange',
                        'symbol': 'x',
                        'name': '{} bulk interstitials'.format(bi_name),
                    })

            else:

                points.append({
                    'data': self.bulk_interstitials,
                    'colour': 'orange',
                    'symbol': 'x',
                    'name': 'Bulk interstitials',
                })

        boxes = [
            {
                'edges': self.bravais_lattice.vecs,
                'name': 'Unit cell',
                'colour': 'navy'
            }
        ]

        f3d, f2d = plotting.plot_geometry_plotly(points, boxes)
        if show_iplot:
            iplot(f3d)
            iplot(f2d)

    def __repr__(self):

        return ('CrystalStructure(\n'
                '\t' + self.bravais_lattice.__repr__() + '\n'
                '\t' + '{!r}'.format(self.motif) + '\n'
                ')')

    def __str__(self):

        atoms_str = BeautifulTable()
        atoms_str.numeric_precision = 4
        atoms_str.intersection_char = ''
        column_headers = ['Number', 'Species',  'x', 'y', 'z']

        for i in self.atom_labels.keys():
            column_headers.append(i)

        atoms_str.column_headers = column_headers

        atom_sites_frac = self.atom_sites_frac
        for idx, si in enumerate(self.species_idx):
            row = [
                idx, self.species[si],
                *(atom_sites_frac[:, idx]),
                *[v[idx] for k, v in self.atom_labels.items()]
            ]
            atoms_str.append_row(row)

        ret = ('{!s}-{!s} Bravais lattice + {!s}-atom motif\n\n'
               'Lattice parameters:\n'
               'a = {!s}\nb = {!s}\nc = {!s}\n'
               'α = {!s}°\nβ = {!s}°\nγ = {!s}°\n'
               '\nLattice vectors = \n{!s}\n'
               '\nLattice sites (fractional) = \n{!s}\n'
               '\nLattice sites (Cartesian) = \n{!s}\n'
               '\nAtoms (fractional coordinates of '
               'unit cell) = \n{!s}\n').format(
            self.bravais_lattice.lattice_system,
            self.bravais_lattice.centring_type,
            self.motif['atom_sites'].shape[1],
            self.bravais_lattice.a, self.bravais_lattice.b,
            self.bravais_lattice.c, self.bravais_lattice.α,
            self.bravais_lattice.β, self.bravais_lattice.γ,
            self.bravais_lattice.vecs,
            self.bravais_lattice.lat_sites_frac,
            self.bravais_lattice.lat_sites_std,
            atoms_str)

        return ret
