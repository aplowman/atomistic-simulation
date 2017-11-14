import numpy as np
import os
import atsim.plotting as plotting
from plotly import graph_objs
from plotly.offline import plot, iplot, init_notebook_mode
from atsim.structure.crystal import CrystalBox, CrystalStructure
from atsim.structure import REF_PATH
from atsim import geometry, vectors, readwrite, utils, mathsutils
from atsim.simsio.lammps import get_LAMMPS_compatible_box
from mendeleev import element

from functools import reduce
import warnings
import spglib


class AtomisticStructureException(Exception):
    pass


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

    atom_sites_frac
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

    def __init__(self, supercell, atom_sites, atom_labels, origin=None,
                 lattice_sites=None, lattice_labels=None, interstice_sites=None,
                 interstice_labels=None, crystals=None, crystal_structures=None,
                 overlap_tol=1):
        """Constructor method for AtomisticStructure object."""

        # Input validation
        # ----------------
        # 1.    Check length of `crystal_idx`, `species_idx`, and `motif_idx`
        #       match number of atoms in `atom_sites`.
        # 2.    Check length of 'lat_crystal_idx' matches number of lattice
        #       sites in `lattice_sites'.
        # 3.    Check set of indices in `crystal_idx` resolve in `crystals`.

        # if crystal_idx is not None:
        #     if len(crystal_idx) != atom_sites.shape[1]:
        #         raise ValueError('Length of `crystal_idx` must match number '
        #                          'of atoms specified as column vectors in '
        #                          '`atom_sites`.')

        #     c_idx_set = sorted(list(set(crystal_idx)))
        #     if c_idx_set[0] < 0 or c_idx_set[-1] >= len(crystals):
        #         raise ValueError('Indices in `crystal_idx` must index elements'
        #                          ' in `crystals`.')

        # if lat_crystal_idx is not None:
        #     if len(lat_crystal_idx) != lattice_sites.shape[1]:
        #         raise ValueError('Length of `lat_crystal_idx` must match '
        #                          'number of lattice sites specified as column '
        #                          'vectors in `lattice_sites`.')

        # if [i is None for i in [all_species, all_species_idx]].count(True) == 1:
        #     raise ValueError('Must specify both `all_species` and '
        #                      '`all_species_idx`.')

        # if [i is None for i in [species_idx, motif_idx]].count(True) == 1:
        #     raise ValueError('Must specify both `species_idx` and '
        #                      '`motif_idx`.')

        # if [i is None for i in [species_idx, all_species_idx]].count(True) != 1:
        #     raise ValueError('Either specify (`all_species` and '
        #                      '`all_species_idx`) or (`species_idx` and '
        #                      '`motif_idx`), but not both.')

        # if species_idx is not None:
        #     if len(species_idx) != atom_sites.shape[1]:
        #         raise ValueError('Length of `species_idx` must match number '
        #                          'of atoms specified as column vectors in '
        #                          '`atom_sites`.')

        # if motif_idx is not None:
        #     if len(motif_idx) != atom_sites.shape[1]:
        #         raise ValueError('Length of `motif_idx` must match number '
        #                          'of atoms specified as column vectors in '
        #                          '`atom_sites`.')

        # if all_species_idx is not None:
        #     if len(all_species_idx) != atom_sites.shape[1]:
        #         raise ValueError('Length of `all_species_idx` ({}) must match '
        #                          'number of atoms specified as column vectors '
        #                          'in `atom_sites` ({}).'.format(len(all_species_idx), atom_sites.shape[1]))

        # Set attributes
        # --------------

        if origin is None:
            origin = np.zeros((3, 1))

        self.origin = utils.to_col_vec(origin)

        self.atom_sites = atom_sites
        self.atom_labels = atom_labels
        self.supercell = supercell
        self.meta = {}

        self.lattice_sites = lattice_sites
        self.lattice_labels = lattice_labels
        self.interstice_sites = interstice_sites
        self.interstice_labels = interstice_labels
        self.crystals = crystals
        self.crystal_structures = crystal_structures
        self._overlap_tol = overlap_tol

        self.check_overlapping_atoms(overlap_tol)

    def translate(self, shift):
        """
        Translate the AtomisticStructure.

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

        if self.crystals is not None:
            for c_idx in range(len(self.crystals)):
                self.crystals[c_idx]['origin'] += shift

    def rotate(self, rot_mat):
        """
        Rotate the AtomisticStructure about its origin according to a rotation
        matrix.

        Parameters
        ----------
        rot_mat : ndarray of shape (3, 3)
            Rotation matrix that pre-multiplies column vectors in order to 
            rotate them about a particular axis and angle.

        """

        origin = np.copy(self.origin)
        self.translate(-origin)

        self.supercell = np.dot(rot_mat, self.supercell)
        self.atom_sites = np.dot(rot_mat, self.atom_sites)

        if self.lattice_sites is not None:
            self.lattice_sites = np.dot(rot_mat, self.lattice_sites)

        if self.interstice_sites is not None:
            self.interstice_sites = np.dot(rot_mat, self.interstice_sites)

        if self.crystals is not None:

            for c_idx in range(len(self.crystals)):

                c = self.crystals[c_idx]

                c['crystal'] = np.dot(rot_mat, c['crystal'])
                c['origin'] = np.dot(rot_mat, c['origin'])

                if 'cs_orientation' in c.keys():
                    c['cs_orientation'] = np.dot(rot_mat, c['cs_orientation'])

        self.translate(origin)

    def visualise(self, proj_2d=False, show_iplot=True, save=False,
                  save_args=None, sym_op=None, wrap_sym_op=False,
                  atoms_3d=False, ret_fig=False):
        """
        Parameters
        ----------
        proj_2d : bool
            If True, 2D plots are also drawn.
        sym_op : list of ndarrays
        atoms_3d : bool
            If True, plots atoms as appropriately sized spheres instead of
            markers. Not recommended for more than a handful of atoms!


        TODO:
        -   Add 3D arrows/cones to supercell vectors using mesh3D.
        -   Add lattice vectors/arrows to lattice unit cells
        -   Add minimums lengths to 2D projection subplots so for very slim
            supercells, we can still see the shorter directions.

        """

        # Validation:
        if not show_iplot and not save and not ret_fig:
            raise ValueError('Visualisation will not be displayed or saved!')

        crystal_cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        if show_iplot:
            init_notebook_mode()

        # Get colours for atom species:
        atom_cols = readwrite.read_pickle(
            os.path.join(REF_PATH, 'jmol_colours.pickle'))

        if save:
            if save_args is None:
                save_args = {
                    'filename': 'plots.html',
                    'auto_open': False
                }
            elif save_args.get('filename') is None:
                save_args.update({'filename': 'plots.html'})

        data = []
        div_2d = ''
        if proj_2d:

            data_2d = []
            proj_2d_dirs = [0, 1, 2]
            vert_space = 0.050
            hori_space = 0.050

            # Tuples for each projection direction (x, y, z):
            dirs_2d = ((1, 2), (0, 2), (1, 0))
            ax_lab_2d = (('x', 'y'), ('x2', 'y'), ('x', 'y3'))
            show_leg_2d = (True, False, False)

        # Sym ops to atom sites:
        if sym_op is not None:
            atom_sites_sym = np.dot(sym_op[0], self.atom_sites)
            atom_sites_sym += np.dot(self.supercell, sym_op[1])

            if wrap_sym_op:
                as_sym_frac = np.dot(self.supercell_inv, atom_sites_sym)
                as_sym_frac -= np.floor(as_sym_frac)
                atom_sites_sym = np.dot(self.supercell, as_sym_frac)

            sym_atom_site_props = {
                'mode': 'text+markers',
                'text': np.arange(self.num_atoms),
                'textposition': 'bottom center',
                'textfont': {
                    'color': 'purple',
                },
                'marker': {
                    'symbol': 'x',
                    'size': 5,
                    'color': 'purple'
                },
                'name': 'Sym op',
                'legendgroup': 'Sym op',
            }

            data.append(
                graph_objs.Scatter3d(
                    x=atom_sites_sym[0],
                    y=atom_sites_sym[1],
                    z=atom_sites_sym[2],
                    **sym_atom_site_props
                )
            )

            # Add lines mapping symmetrically connected atoms:
            for a_idx, a in enumerate(atom_sites_sym.T):
                data.append({
                    'type': 'scatter3d',
                    'x': [a[0], self.atom_sites.T[a_idx][0]],
                    'y': [a[1], self.atom_sites.T[a_idx][1]],
                    'z': [a[2], self.atom_sites.T[a_idx][2]],
                    'mode': 'lines',
                    'name': 'Sym op',
                    'legendgroup': 'Sym op',
                    'showlegend': False,
                    'line': {
                        'color': 'purple',
                    },
                })

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

        if proj_2d:
            for i in proj_2d_dirs:
                data_2d.append({
                    'type': 'scatter',
                    'x': sup_xyz[dirs_2d[i][0]],
                    'y': sup_xyz[dirs_2d[i][1]],
                    'xaxis': ax_lab_2d[i][0],
                    'yaxis': ax_lab_2d[i][1],
                    'name': 'Supercell',
                    'legendgroup': 'Supercell',
                    'showlegend': show_leg_2d[i],
                    **sup_props
                })

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
            sup_vecs_lab_props = {
                'mode': 'text',
                'text': [sup_vec_labs[i]],
                'legendgroup': 'Supercell vectors',
                'showlegend': False,
                'textfont': {
                    'size': 20
                }
            }
            data.append(
                graph_objs.Scatter3d(
                    x=[self.supercell[0, i]],
                    y=[self.supercell[1, i]],
                    z=[self.supercell[2, i]],
                    **sup_vecs_lab_props
                )
            )
            if proj_2d:
                for j in proj_2d_dirs:
                    data_2d.append({
                        'type': 'scatter',
                        'x': [0, self.supercell[dirs_2d[j][0], i]],
                        'y': [0, self.supercell[dirs_2d[j][1], i]],
                        'xaxis': ax_lab_2d[j][0],
                        'yaxis': ax_lab_2d[j][1],
                        'name': 'Supercell vectors',
                        'legendgroup': 'Supercell vectors',
                        'showlegend': show_leg_2d[j] if i == 0 else False,
                        'visible': 'legendonly',
                        **sup_edge_props
                    })
                    if i == j:
                        continue
                    data_2d.append({
                        'type': 'scatter',
                        'x': [self.supercell[dirs_2d[j][0], i]],
                        'y': [self.supercell[dirs_2d[j][1], i]],
                        'xaxis': ax_lab_2d[j][0],
                        'yaxis': ax_lab_2d[j][1],
                        'visible': 'legendonly',
                        **sup_vecs_lab_props
                    })

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

                if atoms_3d:
                    rad = element(sp).vdw_radius * 0.25 * 1e-2  # in Ang
                    # Get sphere trace for each atom:
                    for i in atom_sites_sp.T:
                        sph_args = {
                            'radius': rad,
                            'colour': sp_col,
                            'origin': i[:, np.newaxis],
                            'n': 10
                        }
                        sph = plotting.get_sphere_plotly(**sph_args)
                        sph[0].update({
                            'name': trace_name,
                            'legendgroup': trace_name,
                        })
                        data.append(sph[0])

                else:
                    data.append(
                        graph_objs.Scatter3d(
                            x=atom_sites_sp[0],
                            y=atom_sites_sp[1],
                            z=atom_sites_sp[2],
                            **atom_site_props
                        )
                    )

                if proj_2d:
                    for i in proj_2d_dirs:
                        data_2d.append({
                            'type': 'scatter',
                            'x': atom_sites_sp[dirs_2d[i][0]],
                            'y': atom_sites_sp[dirs_2d[i][1]],
                            'xaxis': ax_lab_2d[i][0],
                            'yaxis': ax_lab_2d[i][1],
                            'name': trace_name,
                            'legendgroup': trace_name,
                            'showlegend': show_leg_2d[i],
                            **atom_site_props
                        })

        else:

            ccent = self.crystal_centres

            # Plot atoms by crystal and motif
            # Crystal boxes and atoms
            c_prev_num = 0  # Number of atoms in previous crystal
            for c_idx, c in enumerate(self.crystals):

                # Crystal centres
                cc_trce_nm = 'Crystal #{} centre'.format(c_idx + 1)
                cc_props = {
                    'mode': 'markers',
                    'marker': {
                        'color': 'red',
                        'symbol': 'x',
                        'size': 3
                    }
                }
                data.append(
                    graph_objs.Scatter3d(
                        x=ccent[c_idx][0],
                        y=ccent[c_idx][1],
                        z=ccent[c_idx][2],
                        name=cc_trce_nm,
                        **cc_props
                    )
                )
                if proj_2d:
                    for i in proj_2d_dirs:
                        data_2d.append({
                            'type': 'scatter',
                            'x': ccent[c_idx][dirs_2d[i][0]],
                            'y': ccent[c_idx][dirs_2d[i][1]],
                            'xaxis': ax_lab_2d[i][0],
                            'yaxis': ax_lab_2d[i][1],
                            'name': cc_trce_nm,
                            'legendgroup': cc_trce_nm,
                            'showlegend': show_leg_2d[i],
                            **cc_props
                        })

                # Crystal boxes
                c_xyz = geometry.get_box_xyz(
                    c['crystal'], origin=c['origin'])[0]
                c_props = {
                    'mode': 'lines',
                    'line': {
                        'color': crystal_cols[c_idx]
                    }
                }
                c_trce_nm = 'Crystal #{}'.format(c_idx + 1)
                data.append(
                    graph_objs.Scatter3d(
                        x=c_xyz[0],
                        y=c_xyz[1],
                        z=c_xyz[2],
                        name=c_trce_nm,
                        **c_props
                    )
                )
                if proj_2d:
                    for i in proj_2d_dirs:
                        data_2d.append({
                            'type': 'scatter',
                            'x': c_xyz[dirs_2d[i][0]],
                            'y': c_xyz[dirs_2d[i][1]],
                            'xaxis': ax_lab_2d[i][0],
                            'yaxis': ax_lab_2d[i][1],
                            'name': c_trce_nm,
                            'legendgroup': c_trce_nm,
                            'showlegend': show_leg_2d[i],
                            **c_props
                        })

                # Get CrystalStructure associated with this crystal:
                has_cs = False
                if c.get('cs_idx') is not None:
                    has_cs = True

                if has_cs:
                    cs = self.crystal_structures[c['cs_idx']]

                # Lattice unit cell, need to rotate by given orientation
                if ((c.get('cs_orientation') is not None) and
                        (c.get('cs_origin') is not None)):

                    unit_cell = np.dot(c['cs_orientation'],
                                       cs.bravais_lattice.vecs)

                    cs_origin = np.dot(unit_cell, c['cs_origin'])
                    uc_origin = c['origin'] + cs_origin[:, np.newaxis]
                    uc_xyz = geometry.get_box_xyz(
                        unit_cell, origin=uc_origin)[0]

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
                    if proj_2d:
                        for i in proj_2d_dirs:
                            data_2d.append({
                                'type': 'scatter',
                                'x': uc_xyz[dirs_2d[i][0]],
                                'y': uc_xyz[dirs_2d[i][1]],
                                'xaxis': ax_lab_2d[i][0],
                                'yaxis': ax_lab_2d[i][1],
                                'name': uc_trace_name,
                                'legendgroup': uc_trace_name,
                                'showlegend': show_leg_2d[i],
                                **uc_props
                            })

                # Lattice sites
                if self.lattice_sites is not None:
                    ls_idx = np.where(self.lat_crystal_idx == c_idx)[0]
                    ls = self.lattice_sites[:, ls_idx]
                    ls_trace_name = 'Lattice sites (crystal #{})'.format(
                        c_idx + 1)
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
                    if proj_2d:
                        for i in proj_2d_dirs:
                            data_2d.append({
                                'type': 'scatter',
                                'x': ls[dirs_2d[i][0]],
                                'y': ls[dirs_2d[i][1]],
                                'xaxis': ax_lab_2d[i][0],
                                'yaxis': ax_lab_2d[i][1],
                                'showlegend': show_leg_2d[i],
                                **lat_site_props
                            })

                # Get indices of atoms in this crystal
                crys_atm_idx = np.where(self.crystal_idx == c_idx)[0]

                if has_cs:
                    # Get motif associated with this crystal:
                    sp_motif = cs.species_motif

                    # Atoms by species
                    # TODO: Add traces for atom numbers
                    for sp_idx, sp_name in enumerate(sp_motif):

                        atom_idx = np.where(
                            self.motif_idx[crys_atm_idx] == sp_idx)[0]
                        atom_sites_sp = self.atom_sites[:,
                                                        crys_atm_idx[atom_idx]]
                        sp_i = cs.motif['species'][sp_idx]
                        sp_col = 'rgb' + str(atom_cols[sp_i])

                        trace_name = sp_name + \
                            ' (crystal #{})'.format(c_idx + 1)
                        num_trace_name = 'Atom index (crystal #{})'.format(
                            c_idx + 1)

                        atom_site_props = {
                            'mode': 'markers',
                            'marker': {
                                'symbol': 'o',
                                'size': 7,
                                'color': sp_col
                            },
                            'name': trace_name,
                            'legendgroup': trace_name,
                        }
                        # Add traces for atom numbers
                        data.append(
                            graph_objs.Scatter3d({
                                'x': atom_sites_sp[0],
                                'y': atom_sites_sp[1],
                                'z': atom_sites_sp[2],
                                'mode': 'text',
                                'text': [str(i + c_prev_num) for i in atom_idx],
                                'name': num_trace_name,
                                'legendgroup': num_trace_name,
                                'showlegend': True if sp_idx == 0 else False,
                                'visible': 'legendonly',
                            })
                        )
                        if atoms_3d:
                            rad = element(sp_i).vdw_radius * \
                                0.25 * 1e-2  # in Ang
                            # Get sphere trace for each atom:
                            for i in atom_sites_sp.T:
                                sph_args = {
                                    'radius': rad,
                                    'colour': sp_col,
                                    'origin': i[:, np.newaxis],
                                    'n': 10
                                }
                                sph = plotting.get_sphere_plotly(**sph_args)
                                sph[0].update({
                                    'name': trace_name,
                                    'legendgroup': trace_name,
                                })
                                data.append(sph[0])

                        else:
                            data.append(
                                graph_objs.Scatter3d(
                                    x=atom_sites_sp[0],
                                    y=atom_sites_sp[1],
                                    z=atom_sites_sp[2],
                                    **atom_site_props
                                )
                            )
                        if proj_2d:
                            for i in proj_2d_dirs:
                                data_2d.append({
                                    'type': 'scatter',
                                    'x': atom_sites_sp[dirs_2d[i][0]],
                                    'y': atom_sites_sp[dirs_2d[i][1]],
                                    'xaxis': ax_lab_2d[i][0],
                                    'yaxis': ax_lab_2d[i][1],
                                    'showlegend': show_leg_2d[i],
                                    **atom_site_props
                                })
                else:
                    # crystals but no crystal structure
                    for sp_idx, sp in enumerate(self.all_species):

                        atom_idx = np.where(
                            self.all_species_idx[crys_atm_idx] == sp_idx)[0]
                        atom_sites_sp = self.atom_sites[:,
                                                        crys_atm_idx[atom_idx]]
                        sp_col = str(atom_cols[sp])
                        trace_name = sp + ' (Crystal #{})'.format(c_idx + 1)

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
                        if proj_2d:
                            for i in proj_2d_dirs:
                                data_2d.append({
                                    'type': 'scatter',
                                    'x': atom_sites_sp[dirs_2d[i][0]],
                                    'y': atom_sites_sp[dirs_2d[i][1]],
                                    'xaxis': ax_lab_2d[i][0],
                                    'yaxis': ax_lab_2d[i][1],
                                    'showlegend': show_leg_2d[i],
                                    **atom_site_props
                                })

                c_prev_num = len(crys_atm_idx)

        layout = graph_objs.Layout(
            width=1000,
            height=800,
            scene={
                'aspectmode': 'data'
            }
        )

        fig_2d = None
        if proj_2d:
            # Get approximate ratio of y1 : y3
            sup_z_rn = np.max(sup_xyz[2]) - np.min(sup_xyz[2])
            sup_x_rn = np.max(sup_xyz[0]) - np.min(sup_xyz[0])
            sup_tot_rn_vert = sup_z_rn + sup_x_rn

            # Get ratio of x1 : x2
            sup_y_rn = np.max(sup_xyz[1]) - np.min(sup_xyz[1])
            sup_tot_rn_hori = sup_y_rn + sup_x_rn
            y1_frac = sup_z_rn / sup_tot_rn_vert
            y3_frac = sup_x_rn / sup_tot_rn_vert
            x1_frac = sup_y_rn / sup_tot_rn_hori
            x2_frac = sup_x_rn / sup_tot_rn_hori

            layout_2d = {
                'height': 1000,
                'width': 1000,

                'xaxis': {
                    'domain': [0, x1_frac - hori_space / 2],
                    'anchor': 'y',
                    'scaleanchor': 'y',
                    'side': 'top',
                    'title': 'y',
                },
                'yaxis': {
                    'domain': [y3_frac + vert_space / 2, 1],
                    'anchor': 'x',
                    'title': 'z',
                },

                'xaxis2': {
                    'domain': [x1_frac + hori_space / 2, 1],
                    'anchor': 'y',
                    'side': 'top',
                    'scaleanchor': 'y',
                    'title': 'x',
                },
                'yaxis3': {
                    'domain': [0, y3_frac - vert_space / 2],
                    'anchor': 'x',
                    'scaleanchor': 'x',
                    'title': 'x',
                },
            }

            fig_2d = graph_objs.Figure(data=data_2d, layout=layout_2d)
            if show_iplot:
                iplot(fig_2d)
            if save:
                div_2d = plot(fig_2d, **save_args,
                              output_type='div', include_plotlyjs=False)

        fig = graph_objs.Figure(data=data, layout=layout)

        if show_iplot:
            iplot(fig)
        if save:
            div_3d = plot(fig, **save_args, output_type='div',
                          include_plotlyjs=True)

            html_all = div_3d + div_2d
            with open(save_args.get('filename'), 'w') as plt_file:
                plt_file.write(html_all)

        if ret_fig:
            return (fig, fig_2d)

    def reorient_to_lammps(self):
        """
        Reorient the supercell and its contents to a LAMMPS-compatible
        orientation. Also translate the origin to (0,0,0).

        Returns
        -------
        ndarray of shape (3, 3)
            Rotation matrix used to reorient the supercell and its contents

        """

        # Find rotation matrix which rotates to a LAMMPS compatible orientation
        sup_lmps = get_LAMMPS_compatible_box(self.supercell)
        R = np.dot(sup_lmps, self.supercell_inv)

        # Move the origin to (0,0,0): (not sure if this is necessary for LAMMPS)
        self.translate(-self.origin)

        # Rotate the supercell and its contents by R
        self.rotate(R)

        return R

    def wrap_sites_to_supercell(self, sites='all', dirs=None):
        """
        Wrap sites to within the supercell.

        Parameters
        ----------
        sites : str
            One of "atom", "lattice", "interstice" or "all".
        dirs : list of int, optional
            Supercell direction indices to apply wrapping. Default is None, in
            which case atoms are wrapped in all directions.            

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

        allowed_sites_str = [
            'atom',
            'lattice',
            'interstice',
            'all',
        ]

        if not isinstance(sites, str) or sites not in allowed_sites_str:
            raise ValueError('`sites` must be a string and one of: "atom", '
                             '"lattice", "interstice" or "all".')

        if sites == 'all':
            sites_arr = [
                self.atom_sites,
                self.lattice_sites,
                self.interstice_sites
            ]
        elif sites == 'atom':
            sites_arr = [self.atom_sites]
        elif sites == 'lattice':
            sites_arr = [self.lattice_sites]
        elif sites == 'interstice':
            sites_arr = [self.interstice_sites]

        for s in sites_arr:

            # Get sites in supercell basis:
            s_sup = np.dot(self.supercell_inv, s)

            # Wrap atoms:
            s_sup_wrp = np.copy(s_sup)
            s_sup_wrp[dirs] -= np.floor(s_sup_wrp[dirs])

            # Snap to 0:
            s_sup_wrp = vectors.snap_arr_to_val(s_sup_wrp, 0, 1e-12)

            # Convert back to Cartesian basis
            s_std_wrp = np.dot(self.supercell, s_sup_wrp)

            # Update attributes:
            s = s_std_wrp

    def add_point_defects(self, point_defects):
        """
        Add point defects to the structure.

        Parameters
        ----------
        point_defects : list of PointDefect objects

        """
        pass

    def add_atom(self, coords, species, crystal_idx, is_frac_coords=False):
        """Add an atom to the structure."""
        pass

    def remove_atom(self, atom_idx):
        """Remove an atom from the structure."""
        pass

    @property
    def supercell_inv(self):
        return np.linalg.inv(self.supercell)

    @property
    def atom_sites_frac(self):
        return np.dot(self.supercell_inv, self.atom_sites)

    @property
    def lattice_sites_frac(self):
        if self.lattice_sites is not None:
            return np.dot(self.supercell_inv, self.lattice_sites)
        else:
            return None

    @property
    def interstice_sites_frac(self):
        if self.interstice_sites is not None:
            return np.dot(self.supercell_inv, self.interstice_sites)
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
        """Get the species of each atom as a string array."""
        return self.species[self.species_idx]

    @property
    def spglib_cell(self):
        """Returns a tuple representing valid input for the spglib library."""

        cell = (self.supercell.T,
                self.atom_sites_frac.T,
                [element(i).atomic_number for i in self.all_species])
        return cell

    @property
    def num_atoms_per_crystal(self):
        """Computes number of atoms in each crystal, returns a list."""

        if self.crystals is None:
            return None

        na = []
        for c_idx in range(len(self.crystals)):
            crystal_idx_tup = self.atom_labels['crystal_idx']
            crystal_idx = crystal_idx_tup[0][crystal_idx_tup[1]]
            na.append(np.where(crystal_idx == c_idx)[0].shape[0])

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

    @property
    def crystal_centres(self):
        """Get the midpoints of each crystal in the structure."""

        return [geometry.get_box_centre(c['crystal'], origin=c['origin'])
                for c in self.crystals]

    def tile_supercell(self, tiles):
        """
        Tile supercell and its sites by some integer factors in each supercell 
        direction.

        Parameters
        ----------
        tiles : tuple or list of length 3 or ndarray of size 3
            Number of repeats in each supercell direction.

        """
        invalid_msg = ('`tiles` must be a tuple or list of three integers '
                       'greater than 0.')

        if isinstance(tiles, np.ndarray):
            tiles = np.squeeze(tiles).tolist()

        if len(tiles) != 3:
            raise ValueError(invalid_msg)

        for t in tiles:
            if not isinstance(t, int) or t < 1:
                raise ValueError(invalid_msg)

        tl_atm, tl_atm_lb = self.get_tiled_sites(
            self.atom_sites, self.atom_labels, tiles)

        self.atom_sites = tl_atm
        self.atom_labels = tl_atm_lb

        if self.lattice_sites is not None:
            tl_lat, tl_lat_lb = self.get_tiled_sites(
                self.lattice_sites, self.lattice_labels, tiles)

            self.lattice_sites = tl_lat
            self.lattice_labels = tl_lat_lb

        if self.interstice_sites is not None:
            tl_int, tl_int_lb = self.get_tiled_sites(
                self.interstice_sites, self.interstice_labels, tiles)

            self.interstice_sites = tl_int
            self.interstice_labels = tl_int_lb

        self.supercell *= tiles

    def get_tiled_sites(self, sites, site_labels, tiles):
        """
        Get sites (atoms, lattice, interstice) and site labels tiled by some
        integer factors in each supercell direction.

        Sites are tiled in the positive supercell directions.

        Parameters
        ----------
        tiles : tuple or list of length 3 or ndarray of size 3
            Number of repeats in each supercell direction.

        Returns
        -------
        sites_tiled : ndarray
        labels_tiled : dict

        """

        invalid_msg = ('`tiles` must be a tuple or list of three integers '
                       'greater than 0.')

        if isinstance(tiles, np.ndarray):
            tiles = np.squeeze(tiles).tolist()

        if len(tiles) != 3:
            raise ValueError(invalid_msg)

        sites_tiled = np.copy(sites)
        labels_tiled = {k: tuple(np.copy(i) for i in v)
                        for k, v in site_labels.items()}

        for t_idx, t in enumerate(tiles):

            if t == 1:
                continue

            if not isinstance(t, int) or t < 1:
                raise ValueError(invalid_msg)

            v = self.supercell[:, t_idx:t_idx + 1]
            v_range = v * np.arange(1, t)

            all_t = v_range.T[:, :, np.newaxis]

            sites_stack = all_t + sites_tiled
            add_sites = np.hstack(sites_stack)
            sites_tiled = np.hstack([sites_tiled, add_sites])

            labels_tiled_new = {}
            for k, v in labels_tiled.items():

                add_label_idx = np.tile(v[1], t - 1)
                new_label_idx = np.concatenate((v[1], add_label_idx))
                labels_tiled_new.update({
                    k: (v[0], new_label_idx)
                })

            labels_tiled = labels_tiled_new

        return sites_tiled, labels_tiled

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
            atms = self.get_tiled_sites(
                self.atom_sites, self.atom_labels, [2, 2, 2])[0]
        else:
            atms = self.atom_sites

        return vectors.get_vec_distances(atms)

    def check_overlapping_atoms(self, tol=None):
        """
        Checks if any atoms are overlapping within a tolerance.

        Parameters
        ----------
        tol : float, optional
            Distance below which atoms are considered to be overlapping. By,
            default uses the value assigned on object initialisation as
            `_overlap_tol`.

        Raises
        ------
        AtomisticStructureException
            If any atoms are found to overlap within `tol`.

        """
        if tol is None:
            tol = self._overlap_tol

        dist = self.get_interatomic_dist()
        if np.any(dist < tol):
            raise AtomisticStructureException('Found overlapping atoms. '
                                              'Minimum separation: '
                                              '{:.3f}'.format(np.min(dist)))

    def get_sym_ops(self):
        return spglib.get_symmetry(self.spglib_cell)

    def shift_atoms(self, shift, wrap=False):
        """
        Perform a rigid shift on all atoms, in fractional supercell coordinates.

        Parameters
        ----------
        shift : list or tuple of length three or ndarry of shape (3,) of float
            Fractional supercell coordinates to translate all atoms by.
        wrap : bool
            If True, wrap atoms to within the supercell edges after shift.
        """

        shift = np.array(shift)[:, np.newaxis]
        shift_std = np.dot(self.supercell, shift)
        self.atom_sites += shift_std

        if wrap:
            self.wrap_sites_to_supercell(sites='atom')

    def add_vac(self, thickness, dir_idx, position=1):
        """
        Extend the supercell in a given direction.

        Supercell vector given by direction index `dir_idx` is extended such
        that it's component in the direction normal to the other two supercell
        vectors is a particular `thickness`.

        Parameters
        ----------
        thickness : float
            Thickness of vacuum to add
        dir_idx : int 0, 1 or 2
            Supercell direction in which to add vacuum
        position : float
            Fractional coordinate along supercell vector given by `dir_idx` at
            which to add the vacuum. By default, adds vacuum to the far face of
            the supercell, such that atom Cartesian coordinates are not
            affected. Must be between 0 (inclusive) and 1 (inclusive).
        """

        # TODO: validate it does what we want. Maybe revert back to calling it
        # `add_surface_vac`.

        warnings.warn('!! Untested function... !!')

        if dir_idx not in [0, 1, 2]:
            raise ValueError('`dir_idx` must be 0, 1 or 2.')

        if position < 0 or position > 1:
            raise ValueError('`position` must be between 0 (inclusive) and 1 '
                             '(inclusive).')

        non_dir_idx = [i for i in [0, 1, 2] if i != dir_idx]
        v1v2 = self.supercell[:, non_dir_idx]
        v3 = self.supercell[:, dir_idx]

        n = np.cross(v1v2[:, 0], v1v2[:, 1])
        n_unit = n / np.linalg.norm(n)
        v3_mag = np.linalg.norm(v3)
        v3_unit = v3 / v3_mag
        d = thickness / np.dot(n_unit, v3_unit)

        v3_mag_new = v3_mag + d
        v3_new = v3_unit * v3_mag_new

        self.supercell[:, dir_idx] = v3_new

        asf = self.atom_sites_frac
        shift_idx = np.where(asf[dir_idx] > position)[0]

        self.atom_sites[:, shift_idx] += (n_unit * thickness)

    def check_atomic_environment(self, checks_list):
        """Invoke checks of the atomic environment."""

        allowed_checks = {
            'atoms_overlap': self.check_overlapping_atoms,
        }

        for chk, func in allowed_checks.items():
            if chk in checks_list:
                func()


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

        # Validation
        if any([i in vectors.num_equal_cols(box_lat) for i in [2, 3]]):
            raise ValueError(
                'Identical columns found in box_lat: \n{}\n'.format(box_lat))

        supercell = np.dot(crystal_structure.bravais_lattice.vecs, box_lat)

        cb = CrystalBox(crystal_structure, supercell)
        atom_sites = cb.atom_sites
        lattice_sites = cb.lat_sites
        interstice_sites = cb.interstice_sites

        crystal_idx_lab = {
            'crystal_idx': (
                np.array([0]),
                np.zeros(atom_sites.shape[1])
            ),
        }

        atom_labels = copy.deepcopy(cb.atom_labels)
        atom_labels.update({**crystal_idx_lab})

        lattice_labels = copy.deepcopy(cb.lattice_labels)
        lattice_labels.update({**crystal_idx_lab})

        interstice_labels = copy.deepcopy(cb.interstice_labels)
        interstice_labels.update({**crystal_idx_lab})

        crystals = [{
            'crystal': supercell,
            'origin': np.zeros((3, 1)),
            'cs_idx': 0,
            'cs_orientation': np.eye(3),
            'cs_origin': [0, 0, 0]
        }]

        super().__init__(supercell,
                         atom_sites,
                         atom_labels,
                         lattice_sites=lattice_sites,
                         lattice_labels=lattice_labels,
                         interstice_sites=interstice_sites,
                         interstice_labels=interstice_labels,
                         crystals=crystals,
                         crystal_structures=[crystal_structure])

        self.meta.update({'supercell_type': ['bulk']})


class PointDefect(object):
    """
    Class to represent a point defect embedded within an AtomisticStructure

    Attributes
    ----------
    defect_species : str
        Chemical symbol of the defect species or "v" for vacancy
    host_species : str
        Chemical symbol of the species which this defect replaces or "i" for
        interstitial.
    index : int
        The atom or interstitial site index within the AtomisticStructure.
    charge : float
        The defect's electronic charge relative to that of the site it occupies.
    interstice_type : str
        Set to "tetrahedral" or "octahedral" if `host_species` is "i".

    """

    def __init__(self, defect_species, host_species, index=None, charge=0,
                 interstice_type=None):

        # Validation
        if interstice_type not in [None, 'tetrahedral', 'octahedral']:
            raise ValueError('Interstice type "{}" not understood.'.format(
                interstice_type))

        if host_species != 'i' and interstice_type is not None:
            raise ValueError('Non-interstitial defect specified but '
                             '`interstice_type` also specified.')

        if defect_species == 'v' and host_species == 'i':
            raise ValueError('Cannot add a vacancy defect to an '
                             'interstitial site!')

        if host_species == 'i' and interstice_type is None:
            raise ValueError('`interstice_type` must be specified for '
                             'interstitial point defect.')

        self.defect_species = defect_species
        self.host_species = host_species
        self.index = index
        self.charge = charge
        self.interstice_type = interstice_type

    def __str__(self):
        """
        References
        ----------
        https://en.wikipedia.org/wiki/Kr%C3%B6ger%E2%80%93Vink_notation

        """
        # String representation of the charge in Kroger-Vink notation
        if self.charge == 0:
            charge_str = 'x'
        elif self.charge > 0:
            charge_str = '•' * abs(self.charge)
        elif self.charge < 0:
            charge_str = '′' * abs(self.charge)

        out = '{}_{}^{}'.format(self.defect_species, self.host_species, charge_str,
                                self.index)

        if self.index is not None:
            idx_str_int = 'interstitial' if self.host_species == 'i' else 'atom'
            idx_str = 'at {} index {}'.format(idx_str_int, self.index)
        else:
            idx_str = ''

        if self.interstice_type is not None:
            out += ' ({}'.format(self.interstice_type)
            out += ' ' + idx_str + ')'
        else:
            out += ' (' + idx_str + ')'

        return out

    def __repr__(self):
        return ('PointDefect({!r}, {!r}, index={!r}, charge={!r}, '
                'interstice_type={!r})').format(
                    self.defect_species, self.atom_site, self.index, self.charge,
                    self.interstice_type)
