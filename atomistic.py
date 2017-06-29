import numpy as np
import os
from plotly import graph_objs
from plotly.offline import init_notebook_mode, plot, iplot
import geometry
import vectors
import utils
import readwrite

REF_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref')


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

        self.species_idx = np.repeat(
            np.arange(self.motif['atom_sites'].shape[1]),
            self.lat_sites_frac.shape[1])

        self.species = motif['species']

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

        for sp_idx in set(self.species_idx):

            atom_idx = np.where(self.species_idx == sp_idx)[0]
            atom_sites_sp = self.atom_sites_std[:, atom_idx]
            sp_col = str(atom_cols[self.species[sp_idx]])

            atom_site_props = {
                'mode': 'markers',
                'marker': {
                    'symbol': 'o',
                    'size': 7,
                    'color': 'rgb' + sp_col
                },
                'name': self.species[sp_idx],
                'legendgroup': self.species[sp_idx]
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
        for sp_idx, sp in enumerate(self.species):
            motif_str += sp + ' @ ' + str(
                self.motif['atom_sites'][:, sp_idx]) + '\n'

        return ('{!s}-{!s} Bravais lattice + {!s}-atom motif\n\n'
                'Lattice parameters:\n'
                'a = {!s}\nb = {!s}\nc = {!s}\n'
                'α = {!s}°\nβ = {!s}°\nγ = {!s}°\n'
                '\nLattice vectors = \n{!s}\n'
                '\nLattice sites (fractional) = \n{!s}\n'
                '\nLattice sites (Cartesian) = \n{!s}\n'
                '\nMotif in fractional coordinates of unit cell = \n{!s}\n').format(
                    self.bravais_lattice.lattice_system,
                    self.bravais_lattice.centring_type,
                    self.atom_sites_frac.shape[1],
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
    vectors : ndarray of shape (3, 3)
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
        Add primitive centring type to allowed centring types for rhombohedral
        lattice system.
        Fix bug: if we set hex a to be equal to the default for c, then we get
        an error: BravaisLattice('hexagonal', a=2)

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
            equal_to = {3: 90, 5: 90}
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
