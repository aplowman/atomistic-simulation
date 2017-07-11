import os
import numpy as np
import shutil
import dict_parser
import utils
import atomistic


SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
REF_PATH = os.path.join(SCRIPTS_PATH, 'ref')
SU_PATH = os.path.join(SCRIPTS_PATH, 'set_up')


def main():
    """
    Read the options file and generate a simulation (series).

    TODO:
    -   Validation:
        -   check all ints in cs_idx resolve in crystal_structures
    -   Allow datatype parsing on list elements so specifying crystal structure
        index when forming struct_opt is cleaner.
    -   Allow dict_parser to parse other files so don't need csl_lookup (can
        have a file in /ref: csl_hex_[0001].txt, which can provide csl_vecs for
        a given sigma value.)
    -   Also allow dict_parser to have variables so crystal structure can be
        reference as a variable instead of an index: instead of in opt.txt:
        base_structure --> cs_idx = [0] we could have base_structure 
        --> crystal_structure = <crystal_structures>[0] where 
        crystal_structures is also defined in the opt.txt

    """

    struct_lookup = {
        'BulkCrystal': atomistic.BulkCrystal,
        'CSLBicrystal': atomistic.CSLBicrystal,
        'CSLBulkCrystal': atomistic.CSLBulkCrystal,
        'CSLSurfaceCrystal': atomistic.CSLSurfaceCrystal
    }
    csl_lookup = {
        7: [
            np.array([
                [3, 2, 0],
                [1, 3, 0],
                [0, 0, 1]
            ]),
            np.array([
                [2, 3, 0],
                [-1, 2, 0],
                [0, 0, 1]
            ]),
        ],
        13: [
            np.array([
                [4, 3, 0],
                [1, 4, 0],
                [0, 0, 1]
            ]),
            np.array([
                [3, 4, 0],
                [-1, 3, 0],
                [0, 0, 1]
            ]),
        ],
        19: [
            np.array([
                [5, 2, 0],
                [3, 5, 0],
                [0, 0, 1]
            ]),
            np.array([
                [5, 3, 0],
                [2, 5, 0],
                [0, 0, 1]
            ]),
        ],
        31: [
            np.array([
                [6, -1, 0],
                [1, 5, 0],
                [0, 0, 1]
            ]),
            np.array([
                [5, 1, 0],
                [-1, 6, 0],
                [0, 0, 1]
            ]),
        ],
    }
    log = []

    # Read the options file
    log.append('Reading log file.')
    opt_path = os.path.join(SU_PATH, 'opt.txt')
    opt = dict_parser.parse_dict_file(opt_path)
    print(opt)

    # Modify options dictionary to include additional info

    s_date, s_num = utils.get_date_time_stamp(split=True)
    s_id = s_date + '_' + s_num
    opt['set_up']['session_id'] = s_id
    opt['set_up']['job_name'] = "j_" + s_num

    scratch_os = opt['set_up']['scratch_os']
    local_os = os.name

    if scratch_os == 'nt':
        scratch_path_sep = '\\'
    elif scratch_os == 'posix':
        scratch_path_sep = '/'

    if local_os == 'nt':
        local_path_sep = '\\'
    elif local_os == 'posix':
        local_path_sep = '/'

    stage_path = os.path.join(opt['set_up']['stage_path'], s_id)
    print('stage_path: {}'.format(stage_path))

    log.append('Making stage directory at: {}.'.format(stage_path))
    os.makedirs(stage_path, exist_ok=False)

    # Generate CrystalStructure objects:
    log.append('Generating CrystalStructure objects.')
    cs = []
    for cs_opt in opt['crystal_structures']:
        brav_lat = atomistic.BravaisLattice(**cs_opt['lattice'])
        cs.append(atomistic.CrystalStructure(brav_lat, cs_opt['motif']))

    # print('cs: \n{}\n'.format(cs))

    # Generate base structure
    log.append('Generating base AtomisticStructure object.')
    struct_opt = {}
    base_as_opt = opt['base_structure']
    for k, v in base_as_opt.items():
        if k == 'type':
            continue
        elif k == 'cs_idx':
            struct_opt.update({'crystal_structure': cs[v[0][0]]})
        elif k == 'sigma':
            struct_opt.update({'csl_vecs': csl_lookup[v]})
        else:
            struct_opt.update({k: v})

    base_as = struct_lookup[base_as_opt['type']](**struct_opt)
    print('base_as: \n{}\n'.format(base_as))

    # Visualise base AtomisticStructure:
    save_args = {
        'filename': os.path.join(stage_path, 'base_structure.html'),
        'auto_open': False
    }
    base_as.visualise(show_iplot=False, save=True, save_args=save_args)

    # Save original options file
    opt_src_path = os.path.join(SU_PATH, 'opt.txt')
    opt_dst_path = os.path.join(stage_path, 'opt_in.txt')
    shutil.copy(opt_src_path, opt_dst_path)

    # Save current options dict
    opt_p_str_path = os.path.join(stage_path, 'opt_processed.txt')
    with open(opt_p_str_path, mode='w', encoding='utf-8') as f:
        f.write(dict_parser.formatting.format_dict(opt))


if __name__ == '__main__':
    main()
