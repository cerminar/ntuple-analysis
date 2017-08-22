#!/usr/bin/env python
# import ROOT
from __future__ import print_function
from NtupleDataFormat import HGCalNtuple
import sys


# The purpose of this file is to demonstrate mainly the objects
# that are in the HGCalNtuple


def main():
    inFile = sys.argv[1]
    ntuple = HGCalNtuple(inFile)

    maxEvents = 10

    tot_nevents = 0
    tot_genpart = 0
    tot_rechit = 0
    tot_cluster2d = 0
    tot_multiclus = 0
    tot_simcluster = 0
    tot_pfcluster = 0
    tot_calopart = 0
    tot_track = 0

    for event in ntuple:
        if event.entry() >= maxEvents:
            break
        print "Event", event.entry()+1
        tot_nevents += 1
        genParts = event.genParticles()
        tot_genpart += len(genParts)
        recHits = event.recHits()
        tot_rechit += len(recHits)
        layerClusters = event.layerClusters()
        tot_cluster2d += len(layerClusters)
        multiClusters = event.multiClusters()
        tot_multiclus += len(multiClusters)
        simClusters = event.simClusters()
        tot_simcluster += len(simClusters)
        pfClusters = event.pfClusters()
        tot_pfcluster += len(pfClusters)
        pfClusters = event.pfClusters()
        tot_pfcluster += len(pfClusters)
        caloParts = event.caloParticles()
        tot_calopart += len(caloParts)
        tracks = event.tracks()
        tot_track += len(tracks)

        # for genPart in genParts:
        #     print tot_nevents, "genPart pt:", genPart.pt()

    print "Processed %d events" % tot_nevents
    print "On average %f generator particles" % (float(tot_genpart) / tot_nevents)
    print "On average %f reconstructed hits" % (float(tot_rechit) / tot_nevents)
    print "On average %f layer clusters" % (float(tot_cluster2d) / tot_nevents)
    print "On average %f multi-clusters" % (float(tot_multiclus) / tot_nevents)
    print "On average %f sim-clusters" % (float(tot_simcluster) / tot_nevents)
    print "On average %f PF clusters" % (float(tot_pfcluster) / tot_nevents)
    print "On average %f calo particles" % (float(tot_calopart) / tot_nevents)
    print "On average %f tracks" % (float(tot_track) / tot_nevents)


if __name__ == "__main__":
    main()
