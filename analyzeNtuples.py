#!/usr/bin/env python

"""
Main script for L1 TP analysis.

The script reads the configuration, opens the input and output files for the given sample,
runs the event loop and saves histograms to disk.
All the analysis logic is anyhow elsewhere:

Data:
    which data are potentially read is handled in the `collections` module.
    How to select the data is handled in the `selections` module.
Plotters:
    what to do with the data is handled in the `plotters` module
Histograms:
    which histograms are produced is handled in the `l1THistos` module (and the plotters).
"""
import argparse
import sys
import os
import traceback
import platform
import uproot as up
import ROOT

from python.main import main
import python.l1THistos as histos
import python.file_manager as fm
import python.collections as collections
import python.calibrations as calibs
import python.timecounter as timecounter
import python.tree_reader as treereader




# @profile
def analyze(params, batch_idx=-1):
    print(params)
    debug = int(params.debug)

    input_files = []
    range_ev = (0, params.maxEvents)

    if params.events_per_job == -1:
        print('This is interactive processing...')
        input_files = fm.get_files_for_processing(input_dir=os.path.join(params.input_base_dir,
                                                                         params.input_sample_dir),
                                                  tree=params.tree_name,
                                                  nev_toprocess=params.maxEvents,
                                                  debug=debug)
    else:
        print('This is batch processing...')
        input_files, range_ev = fm.get_files_and_events_for_batchprocessing(input_dir=os.path.join(params.input_base_dir,
                                                                                                   params.input_sample_dir),
                                                                            tree=params.tree_name,
                                                                            nev_toprocess=params.maxEvents,
                                                                            nev_perjob=params.events_per_job,
                                                                            batch_id=batch_idx,
                                                                            debug=debug)

    # print ('- dir {} contains {} files.'.format(params.input_sample_dir, len(input_files)))
    print('- will read {} files from dir {}:'.format(len(input_files), params.input_sample_dir))
    for file_name in input_files:
        print('        - {}'.format(file_name))

    files_with_protocol = [fm.get_eos_protocol(file_name)+file_name for file_name in input_files]


    calib_manager = calibs.CalibManager()
    calib_manager.set_calibration_version(params.calib_version)
    if params.rate_pt_wps:
        calib_manager.set_pt_wps_version(params.rate_pt_wps)

    output = up.recreate(params.output_filename)
    hm = histos.HistoManager()
    hm.file = output

    # instantiate all the plotters
    plotter_collection = []
    plotter_collection.extend(params.plotters)
    # print(plotter_collection)

    # -------------------------------------------------------
    # book histos
    for plotter in plotter_collection:
        plotter.book_histos()

    collection_manager = collections.EventManager()

    if params.weight_file is not None:
        collection_manager.read_weight_file(params.weight_file)

    # -------------------------------------------------------
    # event loop

    tree_reader = treereader.TreeReader(range_ev, params.maxEvents)
    print('events_per_job: {}'.format(params.events_per_job))
    print('maxEvents: {}'.format(params.maxEvents))
    print('range_ev: {}'.format(range_ev))

    # tr = Tracer()

    break_file_loop = False
    for tree_file_name in files_with_protocol:
        if break_file_loop:
            break
        # tree_file = up.open(tree_file_name, num_workers=2)
        tree_file = up.open(tree_file_name, num_workers=1)
        print(f'opening file: {tree_file_name}')
        print(f' . tree name: {params.tree_name}')

        def getUpTree(uprobj, name):
            parts = name.split('/')
            if len(parts) > 1:
                return getUpTree(uprobj, '/'.join(parts[1:]))
            return uprobj[name]

        ttree = getUpTree(tree_file, params.tree_name)

        tree_reader.setTree(ttree)

        while tree_reader.next(debug):

            try:
                collection_manager.read(tree_reader, debug)
                # processes = []
                for plotter in plotter_collection:
                    plotter.fill_histos_event(tree_reader.file_entry, debug=debug)

                # if tree_reader.global_entry % 100 == 0:
                #     tr.collect_stats()

                if batch_idx != -1 and timecounter.counter.started() and tree_reader.global_entry % 100 == 0:
                    # when in batch mode, if < 5min are left we stop the event loop
                    if timecounter.counter.job_flavor_time_left(params.htc_jobflavor) < 5*60:
                        tree_reader.printEntry()
                        print('    less than 5 min left for batch slot: exit event loop!')
                        timecounter.counter.job_flavor_time_perc(params.htc_jobflavor)
                        break_file_loop = True
                        break

            except Exception as inst:
                tree_reader.printEntry()
                print(f"[EXCEPTION OCCURRED:] {str(inst)}")
                print("Unexpected error:", sys.exc_info()[0])
                traceback.print_exc()
                tree_file.close()
                sys.exit(200)

        tree_file.close()
    # print("Processed {} events/{} TOT events".format(nev, ntuple.nevents()))

    print("Writing histos to file {}".format(params.output_filename))
    hm.writeHistos()
    output.close()
    # ROOT.ROOT.DisableImplicitMT()

    return tree_reader.n_tot_entries




if __name__ == "__main__":

    tic = 0
    if int(platform.python_version().split('.')[1]) >= 8:
        timecounter.counter.start()

    nevents = 0
    try:
        nevents += main(analyze=analyze)
    except Exception as inst:
        print(str(inst))
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc()
        sys.exit(100)

    if timecounter.counter.started():
        analysis_time, time_per_event = timecounter.counter.time_per_event(nevents)
        print('Analyzed {} events in {:.2f} s ({:.2f} s/ev)'.format(
            nevents, analysis_time, time_per_event))
        # print (' real time: {:.2f} s'.format(timecounter.counter.real_time()))
        timecounter.counter.print_nevent_per_jobflavor(time_per_event)
