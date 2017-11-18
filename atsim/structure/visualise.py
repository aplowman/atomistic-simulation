"""
Module containing code to visualise structures: CrystalStructure, CrystalBox,
AtomisticStructure.

Each of these classes share common instance attributes:
    `atom_sites`, `species`, `species_idx`, `lattice_sites` (optional),
    `bulk_interstitial_sites` (optional), `atom_labels` (optional)

"""
import os
import numpy as np
from atsim import readwrite, REF_PATH, plotting, utils
from atsim.utils import prt
from plotly.offline import plot, iplot


def visualise(structure, show_iplot=False, save=False, save_args=None, plot_2d='xyz', ret_fig=False,
              group_atoms_by=None, group_lattice_sites_by=None, group_interstices_by=None):
    """
    Parameters
    ----------
    structure : one of CrystalStructure, CrystalBox or AtomisticStructure.
    use_interstice_names : bool, optional
        If True, bulk interstices are plotted by names given in
        `interstice_names` according to `interstice_names_idx`.
    group_atoms_by : list of str, optional
        If set, atoms are grouped according to one or more of their labels.
        For instance, if set to `species_count`, which is an atom label that is
        automatically added to the CrystalStructure, atoms will be grouped by
        their position in the motif within their species. So for a motif which
        has two X atoms, these atoms will be plotted on separate traces:
        "X (#1)" and "X (#2)". Note that atoms are grouped by species
        (e.g. "X") by default.
    group_lattice_sites_by : list of str, optional
        If set, lattice sites are grouped according to one or more of their
        labels.
    group_interstices_by : list of str, optional
        If set, interstices are grouped according to one or more of their
        labels.

    TODO: add `colour_{atoms, lattice_sites, interstices}_by` parameters which 
          will be a string that must be in the corresponding group_{}_by list
          Or maybe don't have this restriction, would ideally want to be able
          to colour according to a colourscale e.g. by volume per atom, bond
          order parameter, etc. Can do this in Plotly by setting 
          marker.colorscale to an array of the same length as the number of 
          markers. And for Matplotlib: https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter

    TODO: consider merging parameters into a dict: 
          `group_sites_by` = {
              atoms: [...], lattice_sites: [...], interstices: [...]} etc.

    """

    if save:
        if save_args is None:
            save_args = {
                'filename': 'plots.html',
                'auto_open': False
            }
        elif save_args.get('filename') is None:
            save_args.update({'filename': 'plots.html'})

    if group_atoms_by is None:
        group_atoms_by = []

    if group_lattice_sites_by is None:
        group_lattice_sites_by = []

    if group_interstices_by is None:
        group_interstices_by = []

    for lab in group_atoms_by:
        if lab not in structure.atom_labels.keys():
            raise ValueError(
                '"{}" is not a valid atom label.'.format(lab)
            )

    for lab in group_lattice_sites_by:
        if lab not in structure.lattice_labels.keys():
            raise ValueError(
                '"{}" is not a valid lattice site label.'.format(lab)
            )

    for lab in group_interstices_by:
        if lab not in structure.interstice_labels.keys():
            raise ValueError(
                '"{}" is not a valid interstice label.'.format(lab)
            )

    # Get colours for atom species:
    atom_cols = readwrite.read_pickle(
        os.path.join(REF_PATH, 'jmol_colours.pickle'))

    # Add atom number labels:
    text = []
    text.append({
        'data': structure.atom_sites,
        'text': list(range(structure.atom_sites.shape[1])),
        'position': 'top',
        'colour': 'gray',
        'name': 'Atom labels',
        'visible': 'legendonly',
    })

    points = []

    # Add atoms by groupings
    atom_groups_names = []
    atom_groups = []
    for k, v in structure.atom_labels.items():
        if k in group_atoms_by:
            atom_groups_names.append(k)
            atom_groups.append(v[0][v[1]])

    atm_col = 'black'
    atm_sym = 'o'

    if len(atom_groups) > 0:
        atom_combs, atom_combs_idx = utils.combination_idx(*atom_groups)

        for ac_idx in range(len(atom_combs)):

            c = atom_combs[ac_idx]
            c_idx = atom_combs_idx[ac_idx]
            skip_idx = []
            atoms_name = 'Atoms'

            # Special treatment for species and species_count if grouping requested:
            if 'species' in atom_groups_names:
                sp_group_name_idx = atom_groups_names.index('species')
                sp = c[sp_group_name_idx]
                atm_col = 'rgb' + str(atom_cols[sp])

                atoms_name += ': {}'.format(sp)
                skip_idx = [sp_group_name_idx]

                if 'species_count' in atom_groups_names:
                    sp_ct_group_name_idx = atom_groups_names.index(
                        'species_count')
                    atoms_name += ' #{}'.format(c[sp_ct_group_name_idx] + 1)
                    skip_idx.append(sp_ct_group_name_idx)

            for idx, (i, j) in enumerate(zip(atom_groups_names, c)):
                if idx in skip_idx:
                    continue
                atoms_name += '; {}: {}'.format(i, j)

            points.append({
                'data': structure.atom_sites[:, c_idx],
                'symbol': atm_sym,
                'colour': atm_col,
                'name': atoms_name,
            })

    else:
        points.append({
            'data': structure.atom_sites,
            'symbol': atm_sym,
            'colour': atm_col,
            'name': 'Atoms',
        })

    # Add lattice sites by groupings
    if structure.lattice_sites is not None:
        lat_groups_names = []
        lat_groups = []
        for k, v in structure.lattice_labels.items():
            if k in group_lattice_sites_by:
                lat_groups_names.append(k)
                lat_groups.append(v[0][v[1]])

        lat_col = 'grey'
        lat_sym = 'x'

        if len(lat_groups) > 0:
            lat_combs, lat_combs_idx = utils.combination_idx(*lat_groups)

            for lc_idx in range(len(lat_combs)):
                c = lat_combs[lc_idx]
                c_idx = lat_combs_idx[lc_idx]
                skip_idx = []
                lats_name = 'Lattice sites'

                for idx, (i, j) in enumerate(zip(lat_groups_names, c)):
                    lats_name += '; {}: {}'.format(i, j)

                points.append({
                    'data': structure.lattice_sites[:, c_idx],
                    'symbol': lat_sym,
                    'colour': lat_col,
                    'name': lats_name,
                    'visible': 'legendonly',
                })

        else:
            points.append({
                'data': structure.lattice_sites,
                'symbol': lat_sym,
                'colour': lat_col,
                'name': 'Lattice sites',
                'visible': 'legendonly',
            })

    # Add interstices by groupings
    if structure.interstice_sites is not None:
        int_groups_names = []
        int_groups = []
        for k, v in structure.interstice_labels.items():
            if k in group_interstices_by:
                int_groups_names.append(k)
                int_groups.append(v[0][v[1]])

        int_col = 'orange'
        int_sym = 'x'

        if len(int_groups) > 0:
            int_combs, int_combs_idx = utils.combination_idx(*int_groups)

            for ic_idx in range(len(int_combs)):
                c = int_combs[ic_idx]
                c_idx = int_combs_idx[ic_idx]
                skip_idx = []
                ints_name = 'Interstices'

                for idx, (i, j) in enumerate(zip(int_groups_names, c)):
                    ints_name += '; {}: {}'.format(i, j)

                points.append({
                    'data': structure.interstice_sites[:, c_idx],
                    'symbol': int_sym,
                    'colour': int_col,
                    'name': ints_name,
                })

        else:
            points.append({
                'data': structure.interstice_sites,
                'symbol': int_sym,
                'colour': int_col,
                'name': 'Interstices',
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
            'origin': structure.origin,
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

    if hasattr(structure, 'supercell'):

        # AtomisticStructure
        boxes.append({
            'edges': structure.supercell,
            'origin': structure.origin,
            'name': 'Supercell',
            'colour': 'green',
        })

        crystal_cols = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for c_idx, c in enumerate(structure.crystals):

            boxes.append({
                'edges': c['crystal'],
                'origin': c['origin'],
                'name': 'Crystal #{}'.format(c_idx + 1),
                'colour': crystal_cols[c_idx],
            })

    f3d, f2d = plotting.plot_geometry_plotly(points, boxes, text)

    if show_iplot:
        iplot(f3d)
        iplot(f2d)

    if save:
        if plot_2d != '':
            div_2d = plot(f2d, **save_args, output_type='div',
                          include_plotlyjs=False)

        div_3d = plot(f3d, **save_args, output_type='div',
                      include_plotlyjs=True)

        html_all = div_3d + div_2d
        with open(save_args.get('filename'), 'w') as plt_file:
            plt_file.write(html_all)

    if ret_fig:
        return (f3d, f2d)
