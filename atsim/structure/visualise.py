"""
Module containing code to visualise structures: CrystalStructure, CrystalBox,
AtomisticStructure.

Each of these classes share common instance attributes:
    `atom_sites`, `species`, `species_idx`, `lattice_sites` (optional),
    `bulk_interstitial_sites` (optional), `atom_labels` (optional)

"""

from atsim import readwrite, REF_PATH, plotting, utils
import os
import numpy as np
from plotly import graph_objs
from plotly.offline import plot, iplot


def visualise(structure, show_iplot=False, plot_2d='xyz', use_interstitial_names=False,
              use_atom_labels=None):
    """
    Parameters
    ----------
    structure : one of CrystalStructure, CrystalBox or AtomisticStructure.
    use_interstitial_names : bool, optional
        If True, bulk interstitial sites are plotted by names given in
        `bulk_interstials_names` according to `bulk_interstitials_idx`.
    use_atom_labels : list of str, optional
        If set, atoms are grouped according to one or more of their atom
        labels. For instance, if set to `species_count`, which is an atom
        label that is automatically added to the CrystalStructure, atoms
        will be grouped by their position in the motif within their
        species. So for a motif which has two X atoms, these atoms will be
        plotted on separate traces: "X (#1)" and "X (#2)". Note that atoms
        are always grouped by species (e.g. "X").

    TODO: change `use_atom_labels` to `group_atoms_by` (can also have 
    `group_interstices_by` and `group_lattice_sites_by`).

    """

    if use_atom_labels is None:
        use_atom_labels = []

    for lab in use_atom_labels:
        if lab not in structure.atom_labels.keys():
            raise ValueError('"{}" is not a valid atom label.'.format(lab))

    # Get colours for atom species:
    atom_cols = readwrite.read_pickle(
        os.path.join(REF_PATH, 'jmol_colours.pickle'))

    points = []

    # Plot atoms by groupings
    atom_groups_names = ['species']
    atom_groups = [structure.species[structure.species_idx]]
    for k, v in structure.atom_labels.items():
        if k in use_atom_labels:
            atom_groups_names.append(k)
            atom_groups.append(v)

    atom_combs, atom_combs_idx = utils.combination_idx(*atom_groups)

    for i in range(len(atom_combs)):

        c = atom_combs[i]
        c_idx = atom_combs_idx[i]
        sp_group_name_idx = atom_groups_names.index('species')
        sp = c[sp_group_name_idx]
        sp_col = 'rgb' + str(atom_cols[sp])

        # Special treatment for species and (optionally) species_count:
        atoms_name = '{}'.format(sp)
        skip_idx = [sp_group_name_idx]

        if 'species_count' in atom_groups_names:
            sp_ct_group_name_idx = atom_groups_names.index('species_count')
            atoms_name += ' #{}'.format(c[sp_ct_group_name_idx] + 1)
            skip_idx.append(sp_ct_group_name_idx)

        for idx, (i, j) in enumerate(zip(atom_groups_names, c)):
            if idx in skip_idx:
                continue
            atoms_name += '; {}: {}'.format(i, j)

        points.append({
            'data': structure.atom_sites[:, c_idx],
            'symbol': 'o',
            'colour': sp_col,
            'name': '{}'.format(atoms_name),
        })

    # Lattice sites:
    points.append({
        'data': structure.lattice_sites,
        'colour': 'gray',
        'symbol': 'x',
        'name': 'Lattice sites'
    })

    # Bulk interstitials
    if structure.bulk_interstitials is not None:

        if use_interstitial_names:

            if structure.bulk_interstitials_names is None:
                raise ValueError('Cannot plot bulk interstitials by name '
                                 ' when `bulk_interstials_names` is not assigned.')

            for i in range(structure.bulk_interstitials_names.shape[0]):

                w = np.where(structure.bulk_interstitials_idx == i)[0]

                bi_sites = structure.bulk_interstitials[:, w]
                bi_name = structure.bulk_interstitials_names[i]

                points.append({
                    'data': bi_sites,
                    'colour': 'orange',
                    'symbol': 'x',
                    'name': '{} bulk interstitials'.format(bi_name),
                })

        else:

            points.append({
                'data': structure.bulk_interstitials,
                'colour': 'orange',
                'symbol': 'x',
                'name': 'Bulk interstitials',
            })

    boxes = []

    if hasattr(structure, 'bravais_lattice'):
        # CrystalStructure
        boxes.append({
            'edges': structure.bravais_lattice.vecs,
            'name': 'Unit cell',
            'colour': 'navy'
        })

    if hasattr(structure, 'box_vecs'):
        # CrystalBox
        boxes.append({
            'edges': structure.box_vecs,
            'name': 'Crystal box',
            'colour': 'green',
        })

        # Add the bounding box trace:
        boxes.append({
            'edges': structure.bounding_box['bound_box'][0],
            'origin': structure.bounding_box['bound_box_origin'],
            'name': 'Bounding box',
            'colour': 'red',
        })

    f3d, f2d = plotting.plot_geometry_plotly(points, boxes)
    if show_iplot:
        iplot(f3d)
        iplot(f2d)
