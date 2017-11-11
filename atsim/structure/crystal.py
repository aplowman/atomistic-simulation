import numpy as np
import os
from atsim.structure.bravais import BravaisLattice
from atsim.structure.visualise import visualise as struct_visualise
from atsim import geometry, vectors, readwrite, REF_PATH, plotting, utils
from atsim.simsio import castep
from beautifultable import BeautifulTable
from atsim.utils import prt


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

    TODO: correct docstring

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

        prt(bounding_box, 'bounding_box')

        self.crystal_structure = crystal_structure

        # Get all lattice sites within the bounding box, as column vectors:
        unit_cell_origins = np.vstack(np.meshgrid(
            range(b_box_origin_bv[0],
                  b_box_origin_bv[0] + b_box_bv[0] + 1),
            range(b_box_origin_bv[1],
                  b_box_origin_bv[1] + b_box_bv[1] + 1),
            range(b_box_origin_bv[2],
                  b_box_origin_bv[2] + b_box_bv[2] + 1))).reshape((3, -1))

        tol = 1e-14

        # Consider all lattice sites within each unit cell origin:
        ls_rs = crystal_structure.lattice_sites_frac.T.reshape((-1, 3, 1))
        ls_lat = np.concatenate(unit_cell_origins + ls_rs, axis=1)
        ls_std = np.dot(lat_vecs, ls_lat)
        ls_box = np.dot(box_vecs_inv, ls_std)
        ls_box = vectors.snap_arr_to_val(ls_box, 0, tol)
        ls_box = vectors.snap_arr_to_val(ls_box, 1, tol)

        # Get all atom sites within the bounding box, as column vectors:
        as_rs = crystal_structure.atom_sites_frac.T.reshape((-1, 3, 1))
        as_lat = np.concatenate(unit_cell_origins + as_rs, axis=1)
        as_std = np.dot(lat_vecs, as_lat)
        as_box = np.dot(box_vecs_inv, as_std)
        as_box = vectors.snap_arr_to_val(as_box, 0, tol)
        as_box = vectors.snap_arr_to_val(as_box, 1, tol)

        compute_bis = False
        bulk_interstitials = None
        bulk_interstitials_idx = None
        if crystal_structure.bulk_interstitials is not None:
            compute_bis = True
            bi_rs = crystal_structure.bulk_interstitials_frac.T.reshape(
                (-1, 3, 1))
            bi_lat = np.concatenate(unit_cell_origins + bi_rs, axis=1)
            bi_std = np.dot(lat_vecs, bi_lat)
            bi_box = np.dot(box_vecs_inv, bi_std)
            bi_box = vectors.snap_arr_to_val(bi_box, 0, tol)
            bi_box = vectors.snap_arr_to_val(bi_box, 1, tol)

            bulk_interstitials_idx = np.repeat(
                crystal_structure.bulk_interstitials_idx,
                unit_cell_origins.shape[1])

            cnd_bi = np.ones(bi_box.shape[1], dtype=bool)

        species_idx = np.repeat(
            crystal_structure.species_idx, unit_cell_origins.shape[1])

        # Form a boolean edge condition array based on `edge_condtions`. Start
        # by allowing all sites:
        cnd_lat = np.ones(ls_box.shape[1], dtype=bool)
        cnd_atm = np.ones(as_box.shape[1], dtype=bool)

        for dir_idx, pt in enumerate(edge_conditions):

            if pt[0] == '1':
                cnd_lat = np.logical_and(cnd_lat, ls_box[dir_idx] >= 0)
                cnd_atm = np.logical_and(cnd_atm, as_box[dir_idx] >= 0)
                if compute_bis:
                    cnd_bi = np.logical_and(cnd_bi, bi_box[dir_idx] >= 0)

            elif pt[0] == '0':
                cnd_lat = np.logical_and(cnd_lat, ls_box[dir_idx] > 0)
                cnd_atm = np.logical_and(cnd_atm, as_box[dir_idx] > 0)
                if compute_bis:
                    cnd_bi = np.logical_and(cnd_bi, bi_box[dir_idx] > 0)

            if pt[1] == '1':
                cnd_lat = np.logical_and(cnd_lat, ls_box[dir_idx] <= 1)
                cnd_atm = np.logical_and(cnd_atm, as_box[dir_idx] <= 1)
                if compute_bis:
                    cnd_bi = np.logical_and(cnd_bi, bi_box[dir_idx] <= 1)

            elif pt[1] == '0':
                cnd_lat = np.logical_and(cnd_lat, ls_box[dir_idx] < 1)
                cnd_atm = np.logical_and(cnd_atm, as_box[dir_idx] < 1)
                if compute_bis:
                    cnd_bi = np.logical_and(cnd_bi, bi_box[dir_idx] < 1)

        inbox_lat_idx = np.where(cnd_lat)[0]
        ls_box_in = ls_box[:, inbox_lat_idx]
        ls_std_in = ls_std[:, inbox_lat_idx]
        ls_std_in = vectors.snap_arr_to_val(ls_std_in, 0, tol)

        inbox_atm_idx = np.where(cnd_atm)[0]
        as_box_in = as_box[:, inbox_atm_idx]
        as_std_in = as_std[:, inbox_atm_idx]
        as_std_in = vectors.snap_arr_to_val(as_std_in, 0, tol)

        species_idx = species_idx[inbox_atm_idx]
        atom_labels = {}
        for k, v in crystal_structure.atom_labels.items():
            atom_labels.update({
                k: np.repeat(v, unit_cell_origins.shape[1])[inbox_atm_idx]
            })

        self.box_vecs = box_vecs
        self.lattice_sites = ls_std_in
        self.lattice_sites_frac = ls_box_in
        self.atom_sites = as_std_in
        self.atom_sites_frac = as_box_in
        self.species = crystal_structure.species
        self.species_idx = species_idx
        self.atom_labels = atom_labels

        if compute_bis:
            inbox_bi_idx = np.where(cnd_bi)[0]
            bi_box_in = bi_box[:, inbox_bi_idx]
            bi_std_in = bi_std[:, inbox_bi_idx]
            bi_std_in = vectors.snap_arr_to_val(bi_std_in, 0, tol)
            bulk_interstitials = bi_std_in
            bulk_interstitials_idx = bulk_interstitials_idx[inbox_bi_idx]

        self.bulk_interstitials = bulk_interstitials
        self.bulk_interstitials_idx = bulk_interstitials_idx
        self.bulk_interstitials_names = crystal_structure.bulk_interstitials_names

    def visualise(self, **kwargs):
        struct_visualise(self, **kwargs)


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

    TODO: finish docstring
    TODO: update from_file to work with species/species_idx

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
        lat_sites_frac = bravais_lattice.lattice_sites_frac
        num_lat_sites = lat_sites_frac.shape[1]
        self.lattice_sites = bravais_lattice.lattice_sites
        self.lattice_sites_frac = lat_sites_frac

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

    def visualise(self, **kwargs):
        struct_visualise(self, **kwargs)

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
            self.bravais_lattice.lattice_sites_frac,
            self.bravais_lattice.lattice_sites,
            atoms_str)

        return ret
