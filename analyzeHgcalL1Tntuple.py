#!/usr/bin/env python
# import ROOT
from __future__ import print_function
from NtupleDataFormat import HGCalNtuple, Event
import sys
import root_numpy as rnp
import pandas as pd
import numpy as np

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


class Parameters:
    def __init__(cls,
                 input_base_dir,
                 input_sample_dir,
                 output_filename,
                 maxEvents=-1,
                 debug=0):
        cls.maxEvents = maxEvents
        cls.debug = debug
        cls.input_base_dir = input_base_dir
        cls.input_sample_dir = input_sample_dir
        cls.output_filename = output_filename


def analyze(params):
    debug = params.debug

    input_files = listFiles(os.path.join(params.input_base_dir, params.input_sample_dir))
    print ('- dir {} contains {} files.'.format(params.input_sample_dir, len(input_files)))

    chain = getChain('hgcalTriggerNtuplizer/HGCalTriggerNtuple', input_files)
    print ('- created TChain containing {} events'.format(chain.GetEntries()))

    ntuple = HGCalNtuple(input_files, tree='hgcalTriggerNtuplizer/HGCalTriggerNtuple')

    output = ROOT.TFile(params.output_filename, "RECREATE")
    output.cd()
    hgen = histos.GenPartHistos('h_genAll')
    htc = histos.TCHistos('h_tcAll')
    h2dcl = histos.ClusterHistos('h_clAll')
    h3dcl = histos.Cluster3DHistos('h_cl3dAll')

    for event in ntuple:
        if (params.maxEvents != -1 and event.entry() >= params.maxEvents):
            break
        if debug >= 2 or event.entry() % 100 == 0:
            print ("--- Event", event.entry())

        genParts = event.getDataFrame(prefix='gen')
        triggerCells = event.getDataFrame(prefix='tc')
        triggerClusters = event.getDataFrame(prefix='cl')
        trigger3DClusters = event.getDataFrame(prefix='cl3d')

        if debug >= 2:
            print ("# gen parts: {}".format(len(genParts)))
        if debug >= 3:
            print(genParts.iloc[:3])

        hgen.fill(genParts)

        if(debug >= 2):
            print ("# of TC: {}".format(len(triggerCells)))

        # if(debug == 10):
        #     print(triggerCells.index)
        #     print(triggerCells.columns)
        #     print(triggerCells.size)
        #     print(triggerCells.energy)
        #     print(triggerCells.iloc[:3])
        #     print(triggerCells[(triggerCells.subdet > 3) & (triggerCells.wafer == 9)])
        #     print(triggerCells[1:3])
        #     print(triggerCells[['energy', 'layer']].iloc[:3])
        #     print(triggerCells[['energy', 'layer']].iloc[:3].shape)

        if(debug >= 3):
            print(triggerCells.iloc[:3])
        htc.fill(triggerCells)

        if(debug >= 2):
            print('# of clusters: {}'.format(len(triggerClusters)))

        if(debug >= 3):
            print(triggerClusters.iloc[:3])
        #     print(triggerClusters.cells.iloc[:3])
        #     # these are all the trigger-cells used in the first 3 2D clusters
        #     print(triggerCells[triggerCells.index.isin(np.concatenate(triggerClusters.cells.iloc[:3]))])

        h2dcl.fill(triggerClusters)

        # clusters3d = event.trigger3DClusters()
        # print('# 3D clusters old style: {}'.format(len(clusters3d)))
        # for cluster in clusters3d:
        #     print(len(cluster.clusters()))


        if(debug >= 2):
            print('# of 3D clusters: {}'.format(len(trigger3DClusters)))
        if(debug >= 3):
            print(trigger3DClusters.iloc[:3])
        h3dcl.fill(trigger3DClusters)

        # FIXME: plot resolution

    print ("Processed {} events/{} TOT events".format(event.entry(), ntuple.nevents()))
    print ("Writing histos to file {}".format(params.output_filename))

    output.cd()
    hgen.write()
    htc.write()
    h2dcl.write()
    h3dcl.write()

    return




def main():
    # ============================================
    # configuration bit

    #input_sample_dir = 'FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU0_20171005/NTUP/'
    #output_filename = 'histos_EleE50_PU0.root'

    # input_sample_dir = 'FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005/NTUP/'
    # output_filename = 'histos_EleE50_PU50.root'


    # ============================================
    singleEleE50_PU200 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                                    input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU200_20171005/NTUP/',
                                    output_filename='histos_EleE50_PU200.root')

    singleEleE50_PU0 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                                  input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU0_20171005/NTUP/',
                                  output_filename='histos_EleE50_PU0.root')

    singleEleE50_PU50 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                                   input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005/NTUP/',
                                   output_filename='histos_EleE50_PU50.root')
    #test = singleEleE50_PU0
    #test.debug = 3
    #test.maxEvents = 10

    analyze(singleEleE50_PU200)


if __name__ == "__main__":
    main()
