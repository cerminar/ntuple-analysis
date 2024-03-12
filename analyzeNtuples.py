#!/usr/bin/env python

"""
Main script for L1 TP analysis.

The script reads the configuration,
opens the input and output files for the given sample,
runs the event loop and saves histograms to disk.
All the analysis logic is anyhow elsewhere:

Data:
    which data are potentially read is handled in the `collections` module.
    How to select the data is handled in the `selections` module.
Plotters:
    what to do with the data is handled in the `plotters` module
Histograms:
    which histograms are produced is handled in the
    `l1THistos` module (and the plotters).
"""

import sys
import traceback

from python.analyzer import analyze
from python.main import main
from python.timecounter import TimeCounter

if __name__ == "__main__":
    counter = TimeCounter()
    counter.start()

    nevents = 0
    try:
        nevents += main(analyze=analyze)
    except Exception as inst:  # noqa: BLE001
        print(str(inst))
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc()
        sys.exit(100)

    if counter.started():
        analysis_time, time_per_event = counter.time_per_event(nevents)
        print(f"\nAnalyzed {nevents} events in {analysis_time:.2f} s ({time_per_event:.2f} s/ev)")
        counter.print_nevent_per_jobflavor(time_per_event)