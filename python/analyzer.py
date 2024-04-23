from calendar import c
import os
import sys
import traceback

import uproot as up
from rich import print as pprint

import python.calibrations as calibs
import python.file_manager as fm
import python.l1THistos as Histos
import python.tree_reader as treereader
from python import collections, timecounter
import dask
import random
import time
from dask.distributed import Client, progress

# @profile
analyze_counter = 1

import threading

def analyze(params, batch_idx=-1):
    params.print()
    debug = int(params.debug)
    input_files = []
    range_ev = (0, params.maxEvents)
    
    # ------------------------- PRINT KEYS ------------------------------

    for key, value in params.items():
        print("KEY:", key , " VALUE: ", value)
    
    # ------------------------- READ FILES ------------------------------

    if params.events_per_job == -1:
        pprint('This is interactive processing...')
        input_files = fm.get_files_for_processing(
            input_dir=os.path.join(params.input_base_dir, params.input_sample_dir),
            tree=params.tree_name,
            nev_toprocess=params.maxEvents,
            debug=debug,
        )
    else:
        pprint('This is batch processing...')
        input_files, range_ev = fm.get_files_and_events_for_batchprocessing(
            input_dir=os.path.join(params.input_base_dir, params.input_sample_dir),
            tree=params.tree_name,
            nev_toprocess=params.maxEvents,
            nev_perjob=params.events_per_job,
            batch_id=batch_idx,
            debug=debug,
        )

    pprint(f'\n- will read {len(input_files)} files from dir {params.input_sample_dir}:')
    for file_name in input_files:
        pprint(f'        - {file_name}')
    pprint('')

    files_with_protocol = [fm.get_eos_protocol(file_name) + file_name for file_name in input_files]
    
    client = Client(threads_per_worker=4, n_workers=1)

    # -------------------------CALIBRATIONS ------------------------------

    calib_manager = calibs.CalibManager()
    calib_manager.set_calibration_version(params.calib_version)
    if params.rate_pt_wps:
        calib_manager.set_pt_wps_version(params.rate_pt_wps)
    
    # -------------------------BOOK HISTOS------------------------------

    output = up.recreate(params.output_filename)
    hm = Histos.HistoManager()
    hm.file = output
    
    plotter_collection = []
    plotter_collection.extend(params.plotters)
    
    for plotter in plotter_collection:
        plotter.book_histos()

    collection_manager = collections.EventManager()

    if params.weight_file is not None:
        collection_manager.read_weight_file(params.weight_file)

    # -------------------------EVENT LOOP--------------------------------

    pprint('')
    pprint(f"{'events_per_job':<15}: {params.events_per_job}")
    pprint(f"{'maxEvents':<15}: {params.maxEvents}")
    pprint(f"{'range_ev':<15}: {range_ev}")
    pprint('')

    print("Creating dask-delayed objects for file reading...")
    files_dask_delayed = []
    tree_reader_instances = []
    for file in files_with_protocol:      
        single_tree_reader = treereader.TreeReader(range_ev, params.maxEvents)
        single_file_dd = dask.delayed(process_file(file, params, single_tree_reader, debug, collection_manager, plotter_collection, hm))
        files_dask_delayed.append(single_file_dd)
        tree_reader_instances.append(single_tree_reader)

    start_time_reading_files = time.time()
    print("Reading .ROOT files in parallel...")
    dask.compute(files_dask_delayed)
    finish_time_reading_files = time.time()
    print("Finished reading .ROOT files in parallel! Took: ", finish_time_reading_files - start_time_reading_files, " s.")

    # ------------------------- WRITING HISTOGRAMS --------------------------------

    pprint(f'Writing histos to file {params.output_filename}')
    start_time_writing_histograms = time.time()

    hm.writeHistos()
    output.close()
    
    finish_time_writing_histograms = time.time()
    print("Writing histos to file FINISHED! Took: ", finish_time_writing_histograms - start_time_writing_histograms, " s.")

    # ------------------------- TOTAL ENTRIES OUTPUT --------------------------------

    total_entries = 0
    for tree_reader_instance in tree_reader_instances:
        total_entries += tree_reader_instance.n_tot_entries

    return total_entries

def process_file(file_name, params, tree_reader, debug, collection_manager, plotter_collection, hm):    
    tree_file = up.open(file_name, num_workers=1)
    pprint(f'[process_file] opening file: {file_name}')
    pprint(f'[process_file] . tree name: {params.tree_name}')

    ttree = tree_file[params.tree_name]
    tree_reader.setTree(ttree)

    while tree_reader.next(debug):
        try:
            collection_manager.read(tree_reader, debug)

            for plotter in plotter_collection:
                plotter.fill_histos_event(tree_reader.file_entry, debug=debug)

        except Exception as inst:
            pprint(f'[EXCEPTION OCCURRED:] {inst!s}')
            pprint('Unexpected error:', sys.exc_info()[0])
            return 1

    tree_file.close()
        
    return 0