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

# @profile
analyze_counter = 1


def analyze(params, batch_idx=-1):
    params.print()
    debug = int(params.debug)

    input_files = []
    range_ev = (0, params.maxEvents)

    if params.events_per_job == -1:
        pprint("This is interactive processing...")
        input_files = fm.get_files_for_processing(
            input_dir=os.path.join(params.input_base_dir, params.input_sample_dir),
            tree=params.tree_name,
            nev_toprocess=params.maxEvents,
            debug=debug,
        )
    else:
        pprint("This is batch processing...")
        input_files, range_ev = fm.get_files_and_events_for_batchprocessing(
            input_dir=os.path.join(params.input_base_dir, params.input_sample_dir),
            tree=params.tree_name,
            nev_toprocess=params.maxEvents,
            nev_perjob=params.events_per_job,
            batch_id=batch_idx,
            debug=debug,
        )

    pprint(f"\n- will read {len(input_files)} files from dir {params.input_sample_dir}:")
    for file_name in input_files:
        pprint(f"        - {file_name}")
    pprint("")

    files_with_protocol = [fm.get_eos_protocol(file_name) + file_name for file_name in input_files]

    calib_manager = calibs.CalibManager()
    calib_manager.set_calibration_version(params.calib_version)
    if params.rate_pt_wps:
        calib_manager.set_pt_wps_version(params.rate_pt_wps)

    output = up.recreate(params.output_filename)
    hm = Histos.HistoManager()
    hm.file = output

    # instantiate all the plotters
    plotter_collection = []
    plotter_collection.extend(params.plotters)

    # -------------------------BOOK HISTOS------------------------------

    for plotter in plotter_collection:
        plotter.book_histos()

    collection_manager = collections.EventManager()

    if params.weight_file is not None:
        collection_manager.read_weight_file(params.weight_file)

    # -------------------------EVENT LOOP--------------------------------

    tree_reader = treereader.TreeReader(range_ev, params.maxEvents)
    pprint("")
    pprint(f"{'events_per_job':<15}: {params.events_per_job}")
    pprint(f"{'maxEvents':<15}: {params.maxEvents}")
    pprint(f"{'range_ev':<15}: {range_ev}")
    pprint("")

    for tree_file_name in files_with_protocol:
        tree_file = up.open(tree_file_name, num_workers=1)
        pprint(f"opening file: {tree_file_name}")
        pprint(f" . tree name: {params.tree_name}")

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
                    tree_reader.pprintEntry()
                    pprint("    less than 5 min left for batch slot: exit event loop!")
                    timecounter.counter.job_flavor_time_perc(params.htc_jobflavor)
                    break

            except Exception as inst:
                tree_reader.pprintEntry()
                pprint(f"[EXCEPTION OCCURRED:] {inst!s}")
                pprint("Unexpected error:", sys.exc_info()[0])
                traceback.pprint_exc()
                tree_file.close()
                sys.exit(200)

        tree_file.close()

    pprint(f"Writing histos to file {params.output_filename}")
    hm.writeHistos()
    output.close()

    return tree_reader.n_tot_entries
