import os
import sys
import traceback
import subprocess

import uproot as up
from rich import print as pprint

import python.calibrations as calibs
import python.file_manager as fm
import python.histos as Histos
import python.tree_reader as treereader
from python import collections, timecounter

# @profile
analyze_counter = 1


def get_file_size_xrootd(server, filepath):
    cmd = ["xrdfs", server, "stat", filepath]
    try:
        output = subprocess.check_output(cmd, universal_newlines=True)
        for line in output.splitlines():
            if line.startswith("Size:"):
                return int(line.split()[1])
    except subprocess.CalledProcessError as e:
        print(f"Error accessing {filepath}: {e}")
        return 0

def total_xrootd_file_size(server, filepaths):
    total = 0
    for path in filepaths:
        size = get_file_size_xrootd(server, path)
        # print(f"{path}: {size} bytes")
        total += size
    return total

def cleanup(params, batch_idx=-1):
    # print(params)
    
    debug = int(params.debug)

    input_files = []
    range_ev = (0, params.maxEvents)

    input_files = fm.get_files_for_processing(
        input_dir=os.path.join(params.input_base_dir, params.input_sample_dir),
        tree=params.tree_name,
        nev_toprocess=params.maxEvents,
        debug=debug,
    )
    ifs = [f for f in input_files if f.endswith('.root')]

    # print(ifs)
    input_dir=os.path.join(params.input_base_dir, params.input_sample_dir)
    pprint(f'\n- Sample has {len(input_files)} files from dir {input_dir}:')
    
    
    # for file_name in input_files:
    #     pprint(f'        - {file_name}')
    # pprint('')


    tot_size = total_xrootd_file_size(fm.get_eos_protocol(ifs[0]), ifs)
    # print(f'Total size of files: {tot_size} bytes')
    print(f"\Sample size: {tot_size / 1e9:.3f} GB")
    # files_with_protocol = [fm.get_eos_protocol(file_name) + file_name for file_name in input_files]

    # for file_name in files_with_protocol:
        
    #     try:
    #         with up.open(file_name) as f:
    #             if params.tree_name not in f.keys():
    #                 pprint(f'ERROR: tree {params.tree_name} not found in file {file_name}')
    #                 continue
    #             tree = f[params.tree_name]
    #             if not tree:
    #                 pprint(f'ERROR: tree {params.tree_name} is empty in file {file_name}')
    #                 continue
    #             treereader.read_tree(tree, params.eventsToDump, debug=debug)
    #     except Exception as e:
    #         pprint(f'ERROR: failed to read file {file_name}: {e}')
    #         traceback.print_exc()
    #         continue
    return f'eos root://eoscms.cern.ch rm -r {input_dir}'
    # return 0
