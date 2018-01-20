"""matsim.atomistic.structure.crystal.py"""

import copy

import numpy as np
from beautifultable import BeautifulTable

from atsim import geometry, vectors, utils, mathsutils
from atsim.utils import prt, mut_exc_args
from atsim.atomistic.software import castep
from atsim.atomistic.structure import site_labs_to_jsonable, site_labs_from_jsonable
from atsim.atomistic.structure.bravais import BravaisLattice
from atsim.atomistic.structure.visualise import visualise as struct_visualise


class Crystal(object):
    """
    Class to represent a bounded volume filled with a given CrystalStructure.

    """

    def translate(self, shift):
        """
        Translate the crystal.

        Parameters
        ----------
        shift : list or ndarray of size 3

        """

        shift = utils.to_col_vec(shift)
        self.origin += shift
        self.atom_sites += shift

        if self.lattice_sites is not None:
            self.lattice_sites += shift

        if self.interstice_sites is not None:
            self.interstice_sites += shift

    def rotate(self, rot_mat):
        """
        Rotate the crystal about its origin according to a rotation matrix.

        Parameters
        ----------
        rot_mat : ndarray of shape (3, 3)
            Rotation matrix that pre-multiplies column vectors in order to 
            rotate them about a particular axis and angle.

        """

        origin = np.copy(self.origin)
        self.translate(-origin)

        self.atom_sites = np.dot(rot_mat, self.atom_sites)

        if self.lattice_sites is not None:
            self.lattice_sites = np.dot(rot_mat, self.lattice_sites)

        if self.interstice_sites is not None:
            self.interstice_sites = np.dot(rot_mat, self.interstice_sites)

        self.translate(origin)

    def to_jsonable(self):
        """Generate a dict representation which can be serialised to JSON."""

        ret = {
            'origin': self.origin,
            'atom_sites': self.atom_sites.tolist(),
            'atom_labels': site_labs_to_jsonable(self.atom_labels),
        }

        if self.lattice_sites:
            ret.update({
                'lattice_sites': self.lattice_sites.tolist(),
                'lattice_labels': site_labs_to_jsonable(self.lattice_labels),
            })

        if self.interstice_sites:
            ret.update({
                'interstice_sites': self.interstice_sites.tolist(),
                'interstice_labels': site_labs_to_jsonable(self.interstice_labels)
            })

        return ret

    @property
    def atom_sites_frac(self):
        return np.dot(np.linalg.inv(self.box_vecs.vecs), self.atom_sites)

    @property
    def lattice_sites_frac(self):
        if self.lattice_sites is not None:
            return np.dot(np.linalg.inv(self.box_vecs.vecs), self.lattice_sites)
        else:
            return None

    @property
    def interstice_sites_frac(self):
        if self.interstice_sites is not None:
            return np.dot(np.linalg.inv(self.box_vecs.vecs),
                          self.interstice_sites)
        else:
            return None

    @property
    def species(self):
        return self.atom_labels['species'][0]

    @property
    def species_idx(self):
        return self.atom_labels['species'][1]

    @property
    def all_species(self):
        return self.species[self.species_idx]


class CrystalBox(Crystal):
    """
    Class to represent a parallelopiped filled with a CrystalStructure

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

    def translate(self, shift):
        """
        Translate the crystal box.

        Parameters
        ----------
        shift : list or ndarray of size 3

        """
        shift = utils.to_col_vec(shift)
        super().translate(shift)

        self.bounding_box['bound_box_origin'] += shift

    def rotate(self, rot_mat):
        """
        Rotate the crystal box about its origin according to a rotation matrix.

        Parameters
        ----------
        rot_mat : ndarray of shape (3, 3)
            Rotation matrix that pre-multiplies column vectors in order to 
            rotate them about a particular axis and angle.

        """

        super().rotate(rot_mat)

        self.box_vecs = np.dot(rot_mat, self.box_vecs)
        self.bounding_box['bound_box'][0] = np.dot(
            rot_mat, self.bounding_box['bound_box'][0])

    def _find_valid_sites(self, sites_frac, labels, unit_cell_origins, lat_vecs,
                          box_vecs_inv, edge_conditions):

        tol = 1e-14

        sts_frac_rs = sites_frac.T.reshape((-1, 3, 1))
        sts_lat = np.concatenate(unit_cell_origins + sts_frac_rs, axis=1)
        sts_std = np.dot(lat_vecs, sts_lat)
        sts_box = np.dot(box_vecs_inv, sts_std)
        sts_box = vectors.snap_arr_to_val(sts_box, 0, tol)
        sts_box = vectors.snap_arr_to_val(sts_box, 1, tol)

        # Form a boolean edge condition array based on `edge_condtions`.
        # Start by allowing all sites:
        cnd = np.ones(sts_box.shape[1], dtype=bool)

        for dir_idx, pt in enumerate(edge_conditions):

            if pt[0] == '1':
                cnd = np.logical_and(cnd, sts_box[dir_idx] >= 0)
            elif pt[0] == '0':
                cnd = np.logical_and(cnd, sts_box[dir_idx] > 0)

            if pt[1] == '1':
                cnd = np.logical_and(cnd, sts_box[dir_idx] <= 1)
            elif pt[1] == '0':
                cnd = np.logical_and(cnd, sts_box[dir_idx] < 1)

        in_idx = np.where(cnd)[0]
        sts_box_in = sts_box[:, in_idx]
        sts_std_in = sts_std[:, in_idx]
        sts_std_in = vectors.snap_arr_to_val(sts_std_in, 0, tol)

        labels_in = {}
        for k, v in labels.items():
            labels_in.update({
                k: (v[0],
                    np.repeat(v[1], unit_cell_origins.shape[1])[in_idx])
            })

        return (sts_std_in, sts_box_in, labels_in)

    def __init__(self, crystal_structure=None, box_vecs=None, edge_conditions=None,
                 origin=None, state=None):
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
        origin : list or ndarray of size 3, optional
            Origin of the crystal box. By default, set to [0, 0, 0].

        Notes
        -----
        Algorithm proceeds as follows:
        1.  Form a bounding parallelopiped around the parallelopiped defined by
            `box_vecs`, whose edge vectors are parallel to the lattice vectors.
        2.  Find all sites within and on the edges/corners of that bounding
            box.
        3.  Transform sites to the box basis.
        4.  Find valid sites, which have vector components in the interval 
            [0, 1] in the box basis, where the interval may be (half-)closed
            /open depending on the specified edge conditions.

        """

        mut_exc_args(
            {'crystal_structure': crystal_structure, 'box_vecs': box_vecs},
            {'state': state}
        )

        if state:

            self.lattice_sites = state['lattice_sites']
            self.lattice_labels = state['lattice_labels']
            self.atom_sites = state['atom_sites']
            self.atom_labels = state['atom_labels']
            self.interstice_sites = state['interstice_sites']
            self.interstice_labels = state['interstice_labels']
            self.bounding_box = state['bounding_box']
            self.crystal_structure = state['crystal_structure']
            self.box_vecs = state['box_vecs']
            self.origin = state['origin']

        else:

            if edge_conditions is None:
                edge_conditions = ['10', '10', '10']

            # Convenience:
            cs = crystal_structure
            lat_vecs = cs.bravais_lattice.vecs

            # Get the bounding box of box_vecs whose vectors are parallel to the
            # crystal lattice. Use padding to catch edge atoms which aren't on
            # lattice sites.
            bounding_box = geometry.get_bounding_box(
                box_vecs, bound_vecs=lat_vecs, padding=1)
            box_vecs_inv = np.linalg.inv(box_vecs)

            bb = bounding_box['bound_box'][0]
            bb_org = bounding_box['bound_box_origin'][:, 0]
            bb_bv = bounding_box['bound_box_bv'][:, 0]
            bb_org_bv = bounding_box['bound_box_origin_bv'][:, 0]

            # Get all lattice sites within the bounding box, as column vectors:
            grid = [range(bb_org_bv[i], bb_org_bv[i] + bb_bv[i] + 1)
                    for i in [0, 1, 2]]
            unit_cell_origins = np.vstack(np.meshgrid(*grid)).reshape((3, -1))

            com_params = [
                unit_cell_origins,
                lat_vecs,
                box_vecs_inv,
                edge_conditions,
            ]

            (lat_std, lat_box,
             lat_labs) = self._find_valid_sites(cs.lattice_sites_frac,
                                                cs.lattice_labels, *com_params)

            (at_std, at_box,
             at_labs) = self._find_valid_sites(cs.atom_sites_frac,
                                               cs.atom_labels, *com_params)

            int_std, int_box, int_labs = None, None, None
            if cs.interstice_sites is not None:
                (int_std, int_box,
                 int_labs) = self._find_valid_sites(cs.interstice_sites_frac,
                                                    cs.interstice_labels,
                                                    *com_params)

            self.lattice_sites = lat_std
            self.lattice_labels = lat_labs

            self.atom_sites = at_std
            self.atom_labels = at_labs

            self.interstice_sites = int_std
            self.interstice_labels = int_labs

            self.bounding_box = bounding_box
            self.crystal_structure = cs
            self.box_vecs = box_vecs
            self.origin = np.zeros((3, 1))

            if origin is not None:
                self.translate(origin)

    def to_jsonable(self):

        ret = super().to_jsonable()

        # Jsonify bounding box which has form dict of {str: ndarray}
        bb_js = {key: val.tolist() for key, val in self.bounding_box.items()}

        # Add bounding box, crystal structure, box vecs:
        ret.update({
            'bounding_box': bb_js,
            'box_vecs': self.box_vecs.tolist(),
            'crystal_structure': self.crystal_structure.to_jsonable(),
        })

        return ret

    @classmethod
    def from_jsonable(cls, state):
        """Instantiate from a JSONable dict."""

        bb_native = {key: np.array(val)
                     for key, val in state['bounding_box'].items()}

        state.update({
            'atom_sites': np.array(state['atom_sites']),
            'bounding_box': bb_native,
            'crystal_structure': state['crystal_structure'].from_jsonable(),
            'box_vecs': np.array(state['box_vecs']),
            'origin': np.array(state['origin']),
        })

        if state['interstice_sites']:
            state.update({
                'interstice_sites': np.array(state['interstice_sites']),
                'interstice_labels': site_labs_from_jsonable(state['interstice_labels']),
            })

        if state['lattice_sites']:
            state.update({
                'lattice_sites': np.array(state['lattice_sites']),
                'lattice_labels': site_labs_from_jsonable(state['lattice_labels']),
            })

        return cls(state=state)

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

    TODO: finish/correct docstring

    """
    @classmethod
    def from_file(cls, path, lattice_system, centring_type=None, motif=None,
                  filetype='.cell', align='cz'):
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
        motif : dict, optional
            Additional sites and optional labels to add to the CrystalStructure.
            For instance:
            `interstices`: {
                `sites`: <array of column vectors of fractional coordinates of sites>,
                `labels`: {
                    <label_name>: (
                        <array of unique label values>, 
                        <array of label value indices>,
                    )
                }
            }
        filetype : string
            Type of file provided [default: .cell from castep]
        align : string
            Alignment of a crystallographic axis with a Cartesian axis. 
            Implemented options: 'ax' and 'cz'.

        Notes
        -----
        Only works for .cell files.

        """
        if filetype == '.cell':
            file_data = castep.read_castep_cell_file(path, ret_frac=True)
        else:
            raise NotImplementedError(
                'File type "{}" is not supported.'.format(filetype))

        cell_params = mathsutils.get_cell_parameters(file_data['supercell'])
        bl_params = {
            'lattice_system': lattice_system,
            'centring_type': centring_type,
            'degrees': False,
            'align': align,
            **cell_params,
        }
        bl = BravaisLattice(**bl_params)

        all_motif = {
            'atoms': {
                'sites': file_data['atom_sites'],
                'labels': {
                    'species': (file_data['species'], file_data['species_idx'])
                }
            }
        }
        if motif is not None:
            all_motif.update(**motif)

        return cls(bl, all_motif)

    def _validate_motif(self, motif):
        """Validate the motif dict."""

        allowed_sites_label_keys = {
            'atoms': ['species', 'bulk_coord_num'],
            'interstices': ['bulk_name']
        }

        req_sites_keys = ['labels', 'sites']
        mtf_fail_msg = 'Motif failed validation: '

        for k, v in motif.items():
            if k not in allowed_sites_label_keys.keys():
                raise ValueError(
                    mtf_fail_msg +
                    '"{}" is not an allowed sites name.'.format(k)
                )

            found_sites_keys = list(sorted(v.keys()))
            if found_sites_keys != req_sites_keys:
                raise ValueError(
                    mtf_fail_msg +
                    'required sites keys are {}, but found keys: {}'.format(
                        req_sites_keys, found_sites_keys)
                )

            for lab_key, lab_val in v['labels'].items():

                lab_val_fail_msg = (mtf_fail_msg + 'each sites label key must '
                                    'be a tuple of length two, but found label'
                                    ' value: {}'.format(lab_val))

                if not isinstance(lab_val, tuple):
                    raise ValueError(lab_val_fail_msg)

                if len(lab_val) != 2:
                    raise ValueError(lab_val_fail_msg)

                if lab_key not in allowed_sites_label_keys[k]:
                    raise ValueError(
                        '"{}" is not an allowed label name for {}.'.format(
                            lab_key, k
                        )
                    )

                utils.check_indices(lab_val[0], lab_val[1])

    def __init__(self, bravais_lattice=None, motif=None, state=None):
        """
        Instantiate a CrystalStructure object. Use parameters `bravais_lattice`
        and `motif` if generating a new CrystalStructure, or parameter `state`
        if loading from a saved state.

        """

        mut_exc_args({'bravais_lattice': bravais_lattice,
                      'motif': motif},
                     {'state': state})

        if state:
            self.bravais_lattice = state['bravais_lattice']
            self.motif = state['motif']
            self.lattice_sites = state['lattice_sites']
            self.lattice_sites_frac = state['lattice_sites_frac']
            self.lattice_labels = state['lattice_labels']
            self.interstice_sites = state['interstice_sites']
            self.interstice_sites_frac = state['interstice_sites_frac']
            self.atom_sites = state['atom_sites']
            self.atom_labels = state['atom_labels']

        else:

            self._validate_motif(motif)

            # Set some attributes directly from BravaisLattice:
            lat_sites_frac = bravais_lattice.lattice_sites_frac
            num_lat_sites = lat_sites_frac.shape[1]
            self.bravais_lattice = bravais_lattice
            self.motif = motif
            self.lattice_sites = bravais_lattice.lattice_sites
            self.lattice_sites_frac = lat_sites_frac

            # Set labels for lattice sites:
            self.lattice_labels = {}

            # Set atom sites: add atomic motif to each lattice site to get
            motif_rs = motif['atoms']['sites'].T.reshape((-1, 3, 1))

            atom_sites_frac = np.concatenate(lat_sites_frac + motif_rs, axis=1)
            atom_sites_std = np.dot(bravais_lattice.vecs, atom_sites_frac)
            self.atom_sites = atom_sites_std

            # Set labels for atom sites:
            mt_atm_labs = motif['atoms']['labels']
            self.atom_labels = {}
            for lab_name, valsidx in mt_atm_labs.items():

                lab_vals = np.array(valsidx[0])
                lab_idx = np.array(valsidx[1])
                lab_idx_rp = np.repeat(lab_idx, num_lat_sites)
                self.atom_labels.update({lab_name: (lab_vals, lab_idx_rp)})

            # Generate a special atom label which tells us, for motifs with more
            # than one atom of the same species, which within-species motif atom
            # number a given atom maps to:

            motif_species = np.array(mt_atm_labs['species'][0])
            motif_species_idx = np.array(mt_atm_labs['species'][1])
            species_count = np.ones(len(motif_species_idx)) * np.nan
            for i in range(len(motif_species)):
                where_sp_idx = np.where(motif_species_idx == i)[0]
                species_count[where_sp_idx] = np.arange(len(where_sp_idx))

            species_count = species_count.astype(int)
            species_count = np.tile(species_count.astype(int), num_lat_sites)

            self.atom_labels.update({
                'species_count': (species_count, np.arange(len(species_count)))
            })

            # Set interstice sites:
            interstice_sites = None
            interstice_sites_frac = None
            interstice_labels = None

            if motif.get('interstices') is not None:
                int_sites = motif['interstices']['sites']
                int_sites_labs = {}
                for lab_name, valsidx in motif['interstices']['labels'].items():
                    int_sites_labs.update({
                        lab_name: (np.array(valsidx[0]), np.array(valsidx[1]))
                    })
                interstice_sites_frac = int_sites
                interstice_sites = np.dot(bravais_lattice.vecs, int_sites)
                interstice_labels = int_sites_labs

            self.interstice_sites = interstice_sites
            self.interstice_sites_frac = interstice_sites_frac
            self.interstice_labels = interstice_labels

            # prt(self.motif, 'self.motif')

    @classmethod
    def from_jsonable(cls, state):
        """Instantiate CrystalStructure from a JSONable dict."""

        motif_ntv = {}
        for motkey, motval in state['motif'].items():
            motval_ntv = copy.deepcopy(motval)
            motval_ntv['sites'] = np.array(motval_ntv['sites'])
            motval_ntv['labels'] = site_labs_from_jsonable(
                motval_ntv['labels'])
            motif_ntv.update({motkey: motval_ntv})

        state.update({
            'bravais_lattice': BravaisLattice.from_jsonable(state['bravais_lattice']),
            'motif': motif_ntv,
            'lattice_sites': np.array(state['lattice_sites']),
            'lattice_sites_frac': np.array(state['lattice_sites_frac']),
            'lattice_labels': state['lattice_labels'],
            'interstice_sites': state['interstice_sites'],
            'interstice_sites_frac': state['interstice_sites_frac'],
            'atom_sites': np.array(state['atom_sites']),
            'atom_labels': site_labs_from_jsonable(state['atom_labels']),
        })

        return cls(state=state)

    def to_jsonable(self):
        """Generate a dict representation that can be JSON serialised."""

        # JSONify motif which has form dict of
        # {str: dict of {
        #   "sites": ndarray,
        #   "labels": dict of {str : tuple of (ndarray, ndarray) }}}

        motif_js = {}
        for motkey, motval in self.motif.items():

            motval_js = copy.deepcopy(motval)
            motval_js['sites'] = motval_js['sites'].tolist()
            motval_js['labels'] = site_labs_to_jsonable(motval_js['labels'])

            motif_js.update({motkey: motval_js})

        ret = {
            'bravais_lattice': self.bravais_lattice.to_jsonable(),
            'motif': motif_js,
            'lattice_sites': self.lattice_sites.tolist(),
            'lattice_sites_frac': self.lattice_sites_frac.tolist(),
            'lattice_labels': self.lattice_labels,
            'interstice_sites': self.interstice_sites,
            'interstice_sites_frac': self.interstice_sites_frac,
            'atom_sites': self.atom_sites.tolist(),
            'atom_labels': site_labs_to_jsonable(self.atom_labels),
        }

        return ret

    @property
    def atom_sites_frac(self):
        return np.dot(np.linalg.inv(self.bravais_lattice.vecs), self.atom_sites)

    @property
    def species(self):
        return self.atom_labels['species'][0]

    @property
    def species_idx(self):
        return self.atom_labels['species'][1]

    @property
    def all_species(self):
        return self.species[self.species_idx]

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
        column_headers = ['Number', 'x', 'y', 'z']

        for i in self.atom_labels.keys():
            column_headers.append(i)

        atoms_str.column_headers = column_headers
        atom_sites_frac = self.atom_sites_frac

        for atom_idx in range(atom_sites_frac.shape[1]):

            row = [
                atom_idx,
                *(atom_sites_frac[:, atom_idx]),
                *[v[0][v[1]][atom_idx] for _, v in self.atom_labels.items()]
            ]
            prt(row, 'row')
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
            self.motif['atoms']['sites'].shape[1],
            self.bravais_lattice.a, self.bravais_lattice.b,
            self.bravais_lattice.c, self.bravais_lattice.α,
            self.bravais_lattice.β, self.bravais_lattice.γ,
            self.bravais_lattice.vecs,
            self.bravais_lattice.lattice_sites_frac,
            self.bravais_lattice.lattice_sites,
            atoms_str)

        return ret
