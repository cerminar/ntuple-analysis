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
    
    #client = Client(threads_per_worker=4, n_workers=10)

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

    # ------------------------- READ .ROOT FILES --------------------------------

    start_time_reading_files = time.time()
    pprint('')
    pprint(f"{'events_per_job':<15}: {params.events_per_job}")
    pprint(f"{'maxEvents':<15}: {params.maxEvents}")
    pprint(f"{'range_ev':<15}: {range_ev}")
    pprint('')

    print("Creating dask-delayed objects for file reading...")
    print("Reading .ROOT files sequentially...")
  
    tree_reader = treereader.TreeReader(range_ev, params.maxEvents)

    for tree_file_name in files_with_protocol:
        tree_file = up.open(tree_file_name, num_workers=1)
        pprint(f'opening file: {tree_file_name}')
        pprint(f' . tree name: {params.tree_name}')

        ttree = tree_file[params.tree_name]

        tree_reader.setTree(ttree)

        while tree_reader.next(debug):
            try:
                collection_manager.read(tree_reader, debug)

                for plotter in plotter_collection:
                    plotter.fill_histos_event(tree_reader.file_entry, debug=debug)

                if (
                    batch_idx != -1
                    and timecounter.counter.started()
                    and tree_reader.global_entry % 100 == 0
                    and timecounter.counter.job_flavor_time_left(params.htc_jobflavor) < 5 * 60
                ):
                    tree_reader.printEntry()
                    pprint('    less than 5 min left for batch slot: exit event loop!')
                    timecounter.counter.job_flavor_time_perc(params.htc_jobflavor)
                    break

            except Exception as inst:
                tree_reader.printEntry()
                pprint(f'[EXCEPTION OCCURRED:] {inst!s}')
                pprint('Unexpected error:', sys.exc_info()[0])
                traceback.print_exc()
                tree_file.close()
                sys.exit(200)

        tree_file.close()

    finish_time_reading_files = time.time()
    print("Finished reading .ROOT files in sequentially! Took: ", finish_time_reading_files - start_time_reading_files, "s.")

    # ------------------------- COMPUTE HISTOGRAMS --------------------------------
    pprint(f'Computing histograms... {params.output_filename}')

    start_time_computing = time.time()

    all_histos = hm.computeHistos()

    finish_time_computing = time.time()
    pprint(f'Computing histograms is FINISHED. Took: {finish_time_computing - start_time_computing} s.')
    
    # ------------------------- WRITE HISTOGRAMS --------------------------------  
    pprint(f'Writing histos to file {params.output_filename}...')

    start_time_writing = time.time()

    hm.writeHistos()
    output.close()
    
    finish_time_writing= time.time()
    print("Writing histos to file is FINISHED. Took: ", finish_time_writing - start_time_writing, " s.")

    # ------------------------- TOTAL ENTRIES OUTPUT --------------------------------

    return tree_reader.n_tot_entries