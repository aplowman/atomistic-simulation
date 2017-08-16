from readwrite import read_pickle, write_pickle, format_list, format_dict
import simsio
import dict_parser
import os
import numpy as np
import utils
import shutil
from set_up.opt_results import RES_OPT

SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
REF_PATH = os.path.join(SCRIPTS_PATH, 'ref')
SU_PATH = os.path.join(SCRIPTS_PATH, 'set_up')
HOME_PATH = r'C:\Users\{}\Dropbox (Research Group)\calcs'.format(os.getlogin())


def collate_results(sid, skip_idx=None, query_all=False):
    """
    """

    sid_path = os.path.join(HOME_PATH, sid)
    sims = read_pickle(os.path.join(sid_path, 'sims.pickle'))
    method = sims['base_options']['method']

    s_count = 0
    for s_idx, sim_i in enumerate(sims['all_sims']):

        if skip_idx is not None and s_idx in skip_idx:
            continue

        s_count += 1
        srs_paths = []
        srs_id = sim_i.options.get('series_id')
        if srs_id is not None:
            for srs_id_lst in srs_id:
                srs_paths.append('_'.join([i['path'] for i in srs_id_lst]))

        calc_path = os.path.join(sid_path, 'calcs', *srs_paths)

        if method == 'castep':
            out = simsio.read_castep_output(calc_path)

        elif method == 'lammps':
            out = simsio.read_lammps_output(calc_path)

        query_i = False
        if query_all:
            query_i = True
        elif not query_all and s_count == 1:
            query_i = True

        save_res = True
        if query_i and hasattr(sim_i, 'results'):
            save_res = False
            msg = 'Results already collated for: {}'.format(sid)
            if query_all:
                msg += ' : {}'.format(s_idx)
            msg += '. Overwrite?'
            if utils.confirm(msg):
                save_res = True

        if save_res:
            sims['all_sims'][s_idx].results = out

    pick_path = os.path.join(sid_path, 'sims.pickle')
    write_pickle(sims, pick_path)


def compute_results(res_opt):
    """

    For combining a series of GB/Bulk/Surface calcs (e.g. size convergence),
    first combine all session IDs to form a list of successful calcs for
    each supercell type, then try to match series elements for the supercell
    types neccessary for a computed quantity. E.g. GB energy requires GB and
    bulk(s). Surface energy requries surface and bulk.

    May also have to "broadcast" one supercell - need rules for when this is
    allowed. E.g. GB energy for a γ surface may use the same bulk energy
    for each GB energy on the γ surface.

    """

    def compute_time_fmt(results):
        return utils.format_time(results['tot_time'])

    def compute_per_atom_energies(results, energy_idx):

        n = results['num_ions']
        final_energy_pa = results['final_energy'] / n
        final_fenergy_pa = results['final_fenergy'] / n
        final_zenergy_pa = results['final_zenergy'] / n

        return (final_energy_pa, final_fenergy_pa,
                final_zenergy_pa)[energy_idx]

    def get_rms_force(forces):
        """
        Parameters
        ----------
        forces : ndarray of shape (M, 3, N)
            Array representing the force components on N atoms
            for M steps.
        """
        if len(forces) == 0:
            return None

        forces_rshp = forces.reshape(forces.shape[0], -1)
        forces_rms = np.sqrt(np.mean(forces_rshp ** 2, axis=1))

        return forces_rms

    def compute_rms_force(results, force_idx):
        if force_idx == 0:
            return get_rms_force(results['forces_constrained'])
        elif force_idx == 1:
            return get_rms_force(results['forces_unconstrained'])
        elif force_idx == 2:
            return get_rms_force(results['forces_constrained_sym'])

    # Computed quantities which are dependent on exactly one simulation:
    SINGLE_COMPUTES = {
        'time_fmt': (compute_time_fmt,),
        'final_energy_pa': (compute_per_atom_energies, 0),
        'final_fenergy_pa': (compute_per_atom_energies, 1),
        'final_zenergy_pa': (compute_per_atom_energies, 2),
        'forces_cons_rms':  (compute_rms_force, 0),
        'forces_uncons_rms':  (compute_rms_force, 1),
        'forces_cons_sym_rms':  (compute_rms_force, 2),
    }

    # Computed quantities which are dependent on more than one simulation:
    MULTI_COMPUTES = {}

    # roughly same as in save_results - refactor.
    sids = res_opt['sid']
    skip_idx = res_opt.get('skip_idx')
    computes = res_opt.get('computes')

    if computes is None:
        return

    if skip_idx is None:
        skip_idx = [[] * len(sids)]

    # TODO: parse list datatypes!
    for i_idx, i in enumerate(skip_idx):
        for j_idx, j in enumerate(i):
            skip_idx[i_idx][j_idx] = int(j)

    # Combine multiple sims series into one all_sims list
    sim_paths = [os.path.join(HOME_PATH, s, 'sims.pickle') for s in sids]
    all_sims = [read_pickle(p)['all_sims'] for p in sim_paths]

    # Remove sims according to `skip_idx` and flatten:
    all_sims_flt = []
    sids_idx = []
    for i_idx, (i, j) in enumerate(zip(all_sims, skip_idx)):
        trmd_sims = [k for k_idx, k in enumerate(i) if k_idx not in j]
        all_sims_flt.extend(trmd_sims)
        sids_idx.extend([i_idx] * len(trmd_sims))

    ########
    all_computed_results = []
    for s_idx, sim_i in enumerate(all_sims_flt):
        computed_results = {}
        for cmpt in computes:
            nm = cmpt['name']
            cmpt_props = SINGLE_COMPUTES.get(nm)
            if cmpt_props is not None:
                computed_results.update(
                    {nm: cmpt_props[0](sim_i.results, *cmpt_props[1:])})

        all_computed_results.append(computed_results)

    return all_computed_results


def save_results(opt, computed_results):
    """
    TODO:
    -   Fix issue with different sims having output with different lengths,
        i.e. the header needs to print the maximum length.
    """

    # TODO: only allow combining results if `series` dicts have identical names.

    sids = opt['sid']
    skip_idx = opt.get('skip_idx')

    if skip_idx is None:
        skip_idx = [[] * len(sids)]

    # TODO: parse list datatypes!
    for i_idx, i in enumerate(skip_idx):
        for j_idx, j in enumerate(i):
            skip_idx[i_idx][j_idx] = int(j)

    results = opt['results']
    for i_idx in range(len(results)):
        r_idx = results[i_idx].get('idx')
        if r_idx is not None:  # TODO: parse tuples in dict parser
            new_ridx = []
            for j in results[i_idx]['idx']:
                try:
                    j_int = int(j)
                except:
                    j_int = j
                new_ridx.append(j_int)
            results[i_idx]['idx'] = tuple(new_ridx)

    # Must be a key value pair for each possible series type
    series_id_formatting = {
        'cut_off_energy': {
            'fmt': '{:.1f}',
            'print_path': False,
        },
        'kpoint': {
            'fmt': '{:.3f}',
            'print_path': False,
        },
        'smearing_width': {
            'fmt': '{:.2f}',
            'print_path': False
        },
        'box_lat': {
            'fmt': '{:.0f}',
            'print_path': False
        },
        'gb_size': {
            'fmt': '{}',
            'print_path': False,
        },
        'nextra_bands': {
            'fmt': '{:d}',
            'print_path': False,
        },
        'geom_energy_tol': {
            'fmt': '{:.1e}',
            'print_path': False
        },
        'geom_stress_tol': {
            'fmt': '{:.1e}',
            'print_path': False
        }
    }

    # Combine multiple sims series into one all_sims list
    sim_paths = [os.path.join(HOME_PATH, s, 'sims.pickle') for s in sids]
    all_sims = [read_pickle(p)['all_sims'] for p in sim_paths]

    # Remove sims according to `skip_idx` and flatten:
    all_sims_flt = []
    sids_idx = []
    for i_idx, (i, j) in enumerate(zip(all_sims, skip_idx)):
        trmd_sims = [k for k_idx, k in enumerate(i) if k_idx not in j]
        all_sims_flt.extend(trmd_sims)
        sids_idx.extend([i_idx] * len(trmd_sims))

    txt_lns = []
    hdr_ln = []

    for s_idx, sim_i in enumerate(all_sims_flt):

        itms = []
        srs_paths = []
        srs_vals = []
        srs_id = sim_i.options.get('series_id')
        if srs_id is not None:

            for srs_id_lst in srs_id:
                srs_paths.extend([i['path'] for i in srs_id_lst])
                srs_vals.extend([i['val'] for i in srs_id_lst])

                for i in srs_id_lst:
                    hdr_itm = series_id_formatting[i['name']]
                    hdr_itm.update({'name': i['name']})
                    itms.append(hdr_itm)

        itms = itms + results
        out = sim_i.results

        # Integrate computed results into `out`:
        out.update(computed_results[s_idx])

        for i_idx in range(len(srs_vals)):
            itms[i_idx].update({'val': srs_vals[i_idx]})
            if itms[i_idx]['print_path']:
                itms[i_idx].update({'path': srs_paths[i_idx]})

        # if len(sids) > 1:
        sids_fmt = [
            {'name': 'sid', 'fmt': '{}', 'val': sids[sids_idx[s_idx]]}]
        itms = sids_fmt + itms

        srs_ln = []
        for ri in itms:

            ri_nm = ri['name']
            ri_val = ri.get('val')
            ri_fmt = ri.get('fmt')
            ri_pp = ri.get('print_path')
            ri_pth = ri.get('path')
            ri_unt = ri.get('unit')
            ri_idx = ri.get('idx')
            ri_dnm = ri.get('display_name')

            if ri_val is not None:
                v = ri_val
            else:
                v = out[ri_nm]

            if ri_idx:
                for j in ri_idx:
                    v = v[j]

            if isinstance(v, (list, np.ndarray)):
                if isinstance(v, list):
                    v = np.array(v)
                v_fmt = [ri_fmt.format(i) for i in v.flatten()]
                srs_ln.extend(v_fmt)
            else:
                srs_ln.append(ri_fmt.format(v))

            if ri_pp:
                srs_ln.append(ri_pth)

            # Header line:
            if s_idx == 0:

                if ri_dnm is not None:
                    hdr_name = ri_dnm
                else:
                    hdr_name = ri_nm

                hdr_name_str = hdr_name

                if ri_pp:
                    hdr_name_str = hdr_name + '[val]'
                if ri_dnm is None and ri_idx:
                    hdr_name_str += ''.join(['[{}]'.format(i) for i in ri_idx])

                if isinstance(v, (list, np.ndarray)):
                    for i in utils.get_idx_arr(v).T:
                        hdr_ro_i = hdr_name_str
                        hdr_ro_i += ''.join(['[' + str(j) + ']'for j in i])
                        if ri_unt:
                            hdr_ro_i += ' {}'.format(ri_unt)
                        hdr_ln.append(hdr_ro_i)
                else:
                    if ri_unt:
                        hdr_name_str += ' {}'.format(ri_unt)

                    hdr_ln.append(hdr_name_str)

                if ri_pp:
                    hdr_ln.append(hdr_name + '[path]')

        if s_idx == 0:
            txt_lns.append(', '.join(hdr_ln))

        txt_lns.append(', '.join(srs_ln))

    res_path = os.path.join(HOME_PATH, sids[0], 'results')
    res_fn = os.path.join(res_path, 'results.csv')

    overwrite = True
    if os.path.isfile(res_fn):
        overwrite = utils.confirm('`results.csv` already exists. Overwrite?')

    if overwrite:
        os.makedirs(res_path, exist_ok=True)
        with open(res_fn, mode='w', encoding='utf-8', newline='') as rf:
            rf.writelines([i + '\n' for i in txt_lns])

        # Save the results options file:
        # TODO: parse tuples in dict-parser! (the saved results options file
        # below has results `idx` keys as str representations of tuples, rather
        # than lists; so the written file couldn't be used as input again.)
        opt_dst_path = os.path.join(HOME_PATH, sids[0], 'results')
        opt_dst_fn = os.path.join(opt_dst_path, 'opt_results.txt')
        with open(opt_dst_fn, 'w', encoding='utf-8', newline='') as opt_f:
            opt_f.write(format_dict(opt))
        print('Results file saved.')

    else:
        print('Exiting.')


def main():

    sids = RES_OPT['sid']
    skip_idx = RES_OPT['skip_idx']
    computes = RES_OPT['computes']

    # TODO: parse list datatypes!
    for i_idx, i in enumerate(skip_idx):
        for j_idx, j in enumerate(i):
            skip_idx[i_idx][j_idx] = int(j)

    for s_idx, s in enumerate(sids):
        collate_results(s, skip_idx=skip_idx[s_idx])

    # Compute additional properties
    cmpt_res = compute_results(RES_OPT)
    save_results(RES_OPT, cmpt_res)


if __name__ == '__main__':
    main()
