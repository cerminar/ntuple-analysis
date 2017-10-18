#!/usr/bin/env python
# import ROOT
from __future__ import print_function
from NtupleDataFormat import HGCalNtuple, Event
import sys
import root_numpy as rnp


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



def main():
    # ============================================
    # configuration bit
    maxEvents = 1000
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

    for event in ntuple:
        if event.entry() >= maxEvents:
            break
        if debug == 1 or event.entry() % 100 == 0:
            print ("--- Event", event.entry()+1)

        genParts = event.getDataFrame(prefix='gen')
        triggerCells = event.getDataFrame(prefix='tc')
        triggerClusters = event.getDataFrame(prefix='cl')
        trigger3DClusters = event.getDataFrame(prefix='cl3d')

        if debug == 1:
            print ("# gen parts: {}".format(len(genParts)))
        for genpart in genParts:
            hgen.fill(genpart.pt(), genpart.energy())

        if(debug == 1):
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

        if(debug == 3):
            print(triggerCells.iloc[:3])
        htc.fill(triggerCells)

        break
        if(debug == 1):
            print ('# of clusters: {}'.format(len(triggerClusters)))
        for cluster in triggerClusters:
            if(debug == 3):
                print('Cluster: pt: {}, energy: {}, eta: {}, phi: {}, layer: {}, ncells: {}'
                      .format(cluster.pt(), cluster.energy(), cluster.eta(), cluster.phi(), cluster.layer(), cluster.ncells(), cluster.cells()))
                for cell in cluster.cells():
                    print (cell)
                    printTC(triggerCells[cell])
            h2dcl.fill(cluster.energy(), cluster.layer(), cluster.ncells())

        for cl3d in trigger3DClusters:
            h3dcl.fill(cl3d.pt(),
                       cl3d.energy(),
                       cl3d.nclu(),
                       cl3d.showerlength(),
                       cl3d.firstlayer(),
                       cl3d.seetot(),
                       cl3d.seemax(),
                       cl3d.spptot(),
                       cl3d.sppmax(),
                       cl3d.szz(),
                       cl3d.emaxe())

        # FIXME: plot resolution


    print ("Processed {} events/{} TOT events".format(maxEvents, ntuple.nevents()))
    print ("Writing histos to file {}".format(output_filename))

    output.cd()
    hgen.write()
    htc.write()
    h2dcl.write()
    h3dcl.write()

    return

    #
    #
    # inFile = sys.argv[1]
    # ntuple = HGCalNtuple(inFile)
    #
    # maxEvents = 10
    #
    # tot_nevents = 0
    # tot_genpart = 0
    # tot_rechit = 0
    # tot_cluster2d = 0
    # tot_multiclus = 0
    # tot_simcluster = 0
    # tot_pfcluster = 0
    # tot_calopart = 0
    # tot_track = 0
    #
    # for event in ntuple:
    #     if event.entry() >= maxEvents:
    #         break
    #     print "Event", event.entry()+1
    #     tot_nevents += 1
    #     genParts = event.genParticles()
    #     tot_genpart += len(genParts)
    #     recHits = event.recHits()
    #     tot_rechit += len(recHits)
    #     layerClusters = event.layerClusters()
    #     tot_cluster2d += len(layerClusters)
    #     multiClusters = event.multiClusters()
    #     tot_multiclus += len(multiClusters)
    #     simClusters = event.simClusters()
    #     tot_simcluster += len(simClusters)
    #     pfClusters = event.pfClusters()
    #     tot_pfcluster += len(pfClusters)
    #     pfClusters = event.pfClusters()
    #     tot_pfcluster += len(pfClusters)
    #     caloParts = event.caloParticles()
    #     tot_calopart += len(caloParts)
    #     tracks = event.tracks()
    #     tot_track += len(tracks)
    #
    #     # for genPart in genParts:
    #     #     print tot_nevents, "genPart pt:", genPart.pt()
    #
    # print "Processed %d events" % tot_nevents
    # print "On average %f generator particles" % (float(tot_genpart) / tot_nevents)
    # print "On average %f reconstructed hits" % (float(tot_rechit) / tot_nevents)
    # print "On average %f layer clusters" % (float(tot_cluster2d) / tot_nevents)
    # print "On average %f multi-clusters" % (float(tot_multiclus) / tot_nevents)
    # print "On average %f sim-clusters" % (float(tot_simcluster) / tot_nevents)
    # print "On average %f PF clusters" % (float(tot_pfcluster) / tot_nevents)
    # print "On average %f calo particles" % (float(tot_calopart) / tot_nevents)
    # print "On average %f tracks" % (float(tot_track) / tot_nevents)


if __name__ == "__main__":
    main()
