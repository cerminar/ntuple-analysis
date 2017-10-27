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
import utils as utils


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

    # -------------------------------------------------------
    # book histos
    hgen = histos.GenPartHistos('h_genAll')
    hdigis = histos.DigiHistos('h_hgcDigisAll')
    htc = histos.TCHistos('h_tcAll')
    h2dcl = histos.ClusterHistos('h_clAll')
    h3dcl = histos.Cluster3DHistos('h_cl3dAll')
    htcMatch = histos.TCHistos('h_tcMatch')
    h2dclMatch = histos.ClusterHistos('h_clMatch')
    h3dclMatch = histos.Cluster3DHistos('h_cl3dMatch')

    hreso = histos.ResoHistos('h_EleReso')

    for event in ntuple:
        if (params.maxEvents != -1 and event.entry() >= params.maxEvents):
            break
        if debug >= 2 or event.entry() % 100 == 0:
            print ("--- Event", event.entry())


        # -------------------------------------------------------
        # --- GenParticles
        genParts = event.getDataFrame(prefix='gen')
        if debug >= 2:
            print ("# gen parts: {}".format(len(genParts)))
        if debug >= 3:
            print(genParts.iloc[:3])

        hgen.fill(genParts)


        # -------------------------------------------------------
        # --- Digis
        hgcDigis = event.getDataFrame(prefix='hgcdigi')
        if debug >= 2:
            print ('# HGCAL digis: {}'.format(len(hgcDigis)))
        if debug >=3:
            print (hgcDigis.iloc[:3])
        hdigis.fill(hgcDigis)

        # -------------------------------------------------------
        # --- Trigger Cells
        triggerCells = event.getDataFrame(prefix='tc')
        if(debug >= 2):
            print ("# of TC: {}".format(len(triggerCells)))

        # if(debug == 10):
        #     print(triggerCells.index)
        #     print(triggerCells.columns)
        #     print(triggerCells.size)
        #     print(triggerCells.energy)
        #     print(triggerCells.iloc[:3])
        # print(triggerCells[(triggerCells.subdet > 3) & (triggerCells.wafer == 9)])
        # slicing and selection
        # print(triggerCells[(triggerCells.layer >= 1) & (triggerCells.layer <= 5)][['layer', 'energy']])

        #     print(triggerCells[1:3])
        #     print(triggerCells[['energy', 'layer']].iloc[:3])
        #     print(triggerCells[['energy', 'layer']].iloc[:3].shape)

        if(debug >= 3):
            print(triggerCells.iloc[:3])
        htc.fill(triggerCells)

        triggerClusters = event.getDataFrame(prefix='cl')
        trigger3DClusters = event.getDataFrame(prefix='cl3d')

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

        # resolution study
        electron_PID = 11
        genElectrons = genParts[(abs(genParts.id) == electron_PID)]
        matched_idx = utils.match_etaphi(genElectrons[['eta', 'phi']], trigger3DClusters[['eta', 'phi']], trigger3DClusters['pt'], deltaR=0.2)

        for idx, genElectron in genElectrons.iterrows():
            matched3DCluster = trigger3DClusters.iloc[[matched_idx[idx]]]
            matchedClusters = triggerClusters.iloc[matched3DCluster.clusters.item()]
            matchedTriggerCells = triggerCells.iloc[np.concatenate(matchedClusters.cells.values)]

            if debug >= 4:
                print ('GEN electron:')
                print (genElectron)
                print ('Matched to 3D cluster:')
                print (matched3DCluster)
                print ('Matched 2D clusters:')
                print (matchedClusters)
                print ('matched cells:')
                print (matchedTriggerCells)

            print ('3D cluster energy: {}'.format(matched3DCluster.energy.sum()))
            print ('3D cluster pt: {}'.format(matched3DCluster.pt.sum()))
            calib_factor = 1.084
            print ('sum 2D cluster energy: {}'.format(matchedClusters.energy.sum()*calib_factor))
            print ('sum 2D cluster pt: {}'.format(matchedClusters.pt.sum()*calib_factor))

            print ('sum TC energy: {}'.format(matchedTriggerCells.energy.sum()))
            h3dclMatch.fill(matched3DCluster)
            h2dclMatch.fill(matchedClusters)
            htcMatch.fill(matchedTriggerCells)

            hreso.fill(reference=genElectron, target=matched3DCluster.iloc[0])


    print ("Processed {} events/{} TOT events".format(event.entry(), ntuple.nevents()))
    print ("Writing histos to file {}".format(params.output_filename))

    output.cd()
    hgen.write()
    hdigis.write()
    htc.write()
    h2dcl.write()
    h3dcl.write()
    htcMatch.write()
    h2dclMatch.write()
    h3dclMatch.write()
    hreso.write()
    output.Close()
    return


from multiprocessing import Pool


def main():
    # ============================================
    # configuration bit

    #input_sample_dir = 'FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU0_20171005/NTUP/'
    #output_filename = 'histos_EleE50_PU0.root'

    # input_sample_dir = 'FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005/NTUP/'
    # output_filename = 'histos_EleE50_PU50.root'

    ntuple_version = 'NTUP'

    # ============================================
    singleEleE50_PU200 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                                    input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU200_20171005/{}/'.format(ntuple_version),
                                    output_filename='histos_EleE50_PU200.root')

    singleEleE50_PU0 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                                  input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU0_20171005/{}/'.format(ntuple_version),
                                  output_filename='histos_EleE50_PU0.root')

    singleEleE50_PU50 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                                   input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005/{}/'.format(ntuple_version),
                                   output_filename='histos_EleE50_PU50.root')

    singleEleE50_PU100 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                                    input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU100_20171005/{}/'.format(ntuple_version),
                                    output_filename='histos_EleE50_PU100.root')

    electron_samples = [singleEleE50_PU0, singleEleE50_PU50, singleEleE50_PU100, singleEleE50_PU200]

    test = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                      input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005/{}/'.format(ntuple_version),
                      output_filename='test.root',
                      maxEvents=10,
                      debug=2)

    test_sample = [test]

    nuGun_PU50 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                            input_sample_dir='FlatRandomPtGunProducer_NuGunPU50_20171005/{}/'.format(ntuple_version),
                            output_filename='histos_NuGun_PU50.root')

    nuGun_PU100 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                             input_sample_dir='FlatRandomPtGunProducer_NuGunPU100_20171005/{}/'.format(ntuple_version),
                             output_filename='histos_NuGun_PU100.root')

    nuGun_PU140 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                             input_sample_dir='FlatRandomPtGunProducer_NuGunPU140_20171005/{}/'.format(ntuple_version),
                             output_filename='histos_NuGun_PU140.root')

    nuGun_PU200 = Parameters(input_base_dir='/Users/cerminar/cernbox/hgcal/CMSSW932/',
                             input_sample_dir='FlatRandomPtGunProducer_NuGunPU200_20171006/{}/'.format(ntuple_version),
                             output_filename='histos_NuGun_PU200.root')

    nugun_samples = [nuGun_PU50, nuGun_PU100, nuGun_PU140, nuGun_PU200]
#
# FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU0_20171005   FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU200_20171005 FlatRandomPtGunProducer_NuGunPU140_20171005
# FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU100_20171005 FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005  FlatRandomPtGunProducer_NuGunPU200_20171006
# FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU140_20171006 FlatRandomPtGunProducer_NuGunPU100_20171005             FlatRandomPtGunProducer_NuGunPU50_20171005


    pool = Pool(3)
    #pool.map(analyze, nugun_samples)
    #pool.map(analyze, test_sample)
    #pool.map(analyze, electron_samples)
    #pool.map(analyze, [singleEleE50_PU200])
    analyze(test)

if __name__ == "__main__":
    main()
