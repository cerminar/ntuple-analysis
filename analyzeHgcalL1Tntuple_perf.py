#!/usr/bin/env python
# import ROOT

# usage: python -m cProfile -s 'cumtime' analyzeHgcalL1Tntuple_perf.py
from __future__ import print_function
from NtupleDataFormat import HGCalNtuple, Event
import sys


# The purpose of this file is to demonstrate mainly the objects
# that are in the HGCalNtuple
import ROOT
import os
import math

import l1THistos as histos


def listFiles(input_dir):
    onlyfiles = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    return onlyfiles


def getChain(name, files):
    chain = ROOT.TChain(name)
    for file_name in files:
        chain.Add(file_name)
    return chain


def printTC(tc):
    print ("TC: id: {}, subdet: {}, zside: {}, layer: {}, wafer: {}, wafertype: {}, cell: {}, data: {}, energy: {}, eta: {}, phi: {}, z: {}"
           .format(tc.id(), tc.zside(), tc.subdet(), tc.layer(), tc.wafer(), tc.wafertype(), tc.cell(), tc.data(), tc.energy(), tc.eta(), tc.phi(), tc.z()))


def main():
    # ============================================
    # configuration bit
    maxEvents = 100
    debug = 1
    input_base_dir = '/Users/cerminar/cernbox/hgcal/CMSSW932/'
    #input_sample_dir = 'FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU0_20171005/NTUP/'
    #output_filename = 'histos_EleE50_PU0.root'

    # input_sample_dir = 'FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005/NTUP/'
    # output_filename = 'histos_EleE50_PU50.root'


    input_sample_dir = 'FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU200_20171005/NTUP/'
    output_filename = 'histos_EleE50_PU200.root'

    # ============================================

    input_files = listFiles(os.path.join(input_base_dir, input_sample_dir))
    print ('- dir {} contains {} files.'.format(input_sample_dir, len(input_files)))

    chain = getChain('hgcalTriggerNtuplizer/HGCalTriggerNtuple', input_files)
    print ('- created TChain containing {} events'.format(chain.GetEntries()))

    ntuple = HGCalNtuple(input_files, tree='hgcalTriggerNtuplizer/HGCalTriggerNtuple')

    output = ROOT.TFile(output_filename, "RECREATE")
    output.cd()
    hgen = histos.GenPartHistos('h_genAll')
    htc = histos.TCHistos('h_tcAll')
    h2dcl = histos.ClusterHistos('h_clAll')
    h3dcl = histos.Cluster3DHistos('h_cl3dAll')


    import time
    import numpy as np

    def loopNP(narray, buff):
        for e in range(0, len(narray)):
            buff += narray[e]
        return buff

    def loopROOT(narray, buff):
        for e in range(0, len(narray)):
            buff += narray[e]
        return buff

    def loopCMG(narray, buff):
        for row in narray['energy']:
            buff += row

        buff = np.sum(narray['energy'])

#             # print

#         for e in range(0, len(narray._dataframe)):
#             buff += narray._dataframe.loc[e].tc_energy
        return buff

    for ientry, entry in enumerate(chain):

        print(ientry)
        if ientry == 2:
            break
        start = time.clock()
        narray_id = np.array(entry.tc_id)
        narray_subdet = np.array(entry.tc_subdet)
        narray_zside = np.array(entry.tc_zside)
        narray_wafer = np.array(entry.tc_wafer)
        narray_wafertype = np.array(entry.tc_wafertype)
        narray_energy = np.array(entry.tc_energy)
        narray_eta = np.array(entry.tc_eta)
        narray_phi = np.array(entry.tc_phi)
        narray_z = np.array(entry.tc_z)
        narray_cell = np.array(entry.tc_cell)
        narray_data = np.array(entry.tc_data)
        narray_layer = np.array(entry.tc_layer)

        buff = 0
        buff = loopNP(narray_energy, buff)
        end = time.clock()
        print("LEN NP: {}".format(len(narray_energy)))
        print("PERF Numpy: {}".format(end-start))
        print("SUM: {}".format(buff))

        buff = 0
        print("LEN PY: {}".format(len(entry.tc_energy)))
        start = time.clock()
        buff = loopROOT(entry.tc_energy, buff)
        end = time.clock()
        print("PERF py: {}".format(end-start))
        print("SUM: {}".format(buff))

        buff = 0


        start = time.clock()
        event = Event(chain, ientry)
        triggerCells = event.getDataFrame(prefix='tc')
        buff = loopCMG(triggerCells, buff)
        end = time.clock()
        print("LEN CMG: {}".format(len(triggerCells.energy)))
        print("PERF CMG: {}".format(end-start))
        print("SUM: {}".format(buff))

    sys.exit(0)


if __name__ == "__main__":
    main()
