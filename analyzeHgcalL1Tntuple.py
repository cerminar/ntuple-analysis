#!/usr/bin/env python
# import ROOT
from __future__ import print_function
from NtupleDataFormat import HGCalNtuple, Event
import sys
import root_numpy as rnp
import pandas as pd
import numpy as np
from multiprocessing import Pool

# The purpose of this file is to demonstrate mainly the objects
# that are in the HGCalNtuple
import ROOT
import os
import math
import copy
import socket
import datetime

import l1THistos as histos
import utils as utils
import clusterTools as clAlgo


def listFiles(input_dir):
    onlyfiles = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    return onlyfiles


def getChain(name, files):
    chain = ROOT.TChain(name)
    for file_name in files:
        chain.Add(file_name)
    return chain


class Parameters:
    def __init__(self,
                 input_base_dir,
                 input_sample_dir,
                 output_filename,
                 clusterize,
                 eventsToDump,
                 maxEvents=-1,
                 computeDensity=False,
                 debug=0,
                 name=''):
        self.name=name
        self.maxEvents = maxEvents
        self.debug = debug
        self.input_base_dir = input_base_dir
        self.input_sample_dir = input_sample_dir
        self.output_filename = output_filename
        self.clusterize = clusterize
        self.eventsToDump = eventsToDump
        self.computeDensity = computeDensity

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


def convertGeomTreeToDF(tree):
    branches = [br.GetName() for br in tree.GetListOfBranches() if not br.GetName().startswith('c_')]
    cell_array = rnp.tree2array(tree, branches=branches)
    cell_df = pd.DataFrame()
    for idx in range(0, len(branches)):
        cell_df[branches[idx]] = cell_array[branches[idx]]
    return cell_df


def dumpFrame2JSON(filename, frame):
    with open(filename, 'w') as f:
        f.write(frame.to_json())


def plot3DClusterMatch(genParticles,
                       trigger3DClusters,
                       triggerClusters,
                       triggerCells,
                       histoTCMatch,
                       histoClMatch,
                       histo3DClMatch,
                       histoReso,
                       histoReso2D,
                       histoReso2D_1t6,
                       histoReso2D_10t20,
                       histoReso2D_20t28,
                       algoname,
                       debug):

    matched_idx = {}
    if trigger3DClusters.shape[0] != 0:
        matched_idx, allmatches = utils.match_etaphi(genParticles[['eta', 'phi']],
                                                     trigger3DClusters[['eta', 'phi']],
                                                     trigger3DClusters['pt'],
                                                     deltaR=0.2)
        # print ('-----------------------')
        # print (matched_idx)
    for idx, genParticle in genParticles.iterrows():
        if idx in matched_idx.keys():
            # print ('-----------------------')
            # print(genParticle)
            matched3DCluster = trigger3DClusters.iloc[[matched_idx[idx]]]
            # print (matched3DCluster)
            # allMatches = trigger3DClusters.iloc[allmatches[idx]]
            # print ('--')
            # print (allMatches)
            matchedClusters = triggerClusters.iloc[matched3DCluster.clusters.item()]
            matchedTriggerCells = triggerCells.iloc[np.concatenate(matchedClusters.cells.values)]

            if 'energyCentral' not in matched3DCluster.columns:
                calib_factor = 1.084
                matched3DCluster['energyCentral'] = [matchedClusters[(matchedClusters.layer > 9) & (matchedClusters.layer < 21)].energy.sum()*calib_factor]

            # fill the plots
            histoTCMatch.fill(matchedTriggerCells)
            histoClMatch.fill(matchedClusters)
            histo3DClMatch.fill(matched3DCluster)

            histoReso2D.fill(reference=genParticle, target=matchedClusters)
            histoReso2D_1t6.fill(reference=genParticle, target=matchedClusters[matchedClusters.layer < 7])
            histoReso2D_10t20.fill(reference=genParticle, target=matchedClusters[(matchedClusters.layer > 9) & (matchedClusters.layer < 21)])
            histoReso2D_20t28.fill(reference=genParticle, target=matchedClusters[matchedClusters.layer > 20])

            histoReso.fill(reference=genParticle, target=matched3DCluster.iloc[0])

            if debug >= 4:
                print ('--- Dump match for algo {} ---------------'.format(algoname))
                print ('GEN particle: idx: {}'.format(idx))
                print (genParticle)
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

        else:
            print ('==== Warning no match found for algo {}, idx {} ======================'.format(algoname, idx))
            print (genParticle)
            print (trigger3DClusters)


def build3DClusters(name, algorithm, triggerClusters, pool, debug):
    trigger3DClusters = pd.DataFrame()
    clusterSides = [triggerClusters[triggerClusters.eta > 0], triggerClusters[triggerClusters.eta < 0]]
    results3Dcl = pool.map(algorithm, clusterSides)
    for res3D in results3Dcl:
        trigger3DClusters = trigger3DClusters.append(res3D, ignore_index=True)

    if(debug >= 2):
        print('# of 3D clusters {}: {}'.format(name, len(trigger3DClusters)))
    if(debug >= 3):
        print(trigger3DClusters.iloc[:3])
    return trigger3DClusters


def analyze(params):
    debug = params.debug
    computeDensity = params.computeDensity
    plot2DCLDR = False

    pool = Pool(5)

    # read the geometry dump
    geom_file = params.input_base_dir+'/geom/test_triggergeom.root'

    tc_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeTriggerCells')
    print ('read TC GEOM tree with # events: {}'.format(tc_geom_tree.nevents()))
    tc_geom_df = convertGeomTreeToDF(tc_geom_tree._tree)
    tc_geom_df['radius'] = np.sqrt(tc_geom_df['x']**2+tc_geom_df['y']**2)
    tc_geom_df['eta'] = np.arcsinh(tc_geom_df.z/tc_geom_df.radius)

    # print (tc_geom_df[:3])
    # tc_geom_df['max_neigh_dist'] = 1
    # a5 = tc_geom_df[tc_geom_df.neighbor_n == 5]
    # a5['max_neigh_dist'] =  a5['neighbor_distance'].max()
    # a6 = tc_geom_df[tc_geom_df.neighbor_n == 6]
    # a6['max_neigh_dist'] =  a6['neighbor_distance'].max()

    # for index, tc_geom in tc_geom_df.iterrows():
    #     tc_geom.max_dist_neigh = np.max(tc_geom.neighbor_distance)

    # print (tc_geom_df[:10])

    # treeTriggerCells = inputFile.Get("hgcaltriggergeomtester/TreeTriggerCells")
    # treeCells        = inputFile.Get("hgcaltriggergeomtester/TreeCells")

    input_files = listFiles(os.path.join(params.input_base_dir, params.input_sample_dir))
    print ('- dir {} contains {} files.'.format(params.input_sample_dir, len(input_files)))

    ntuple = HGCalNtuple(input_files, tree='hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    print ('- created TChain containing {} events'.format(ntuple.nevents()))

    output = ROOT.TFile(params.output_filename, "RECREATE")
    output.cd()

    if False:
        hTCGeom = histos.GeomHistos('hTCGeom')
        hTCGeom.fill(tc_geom_df[(np.abs(tc_geom_df.eta) > 1.65) & (np.abs(tc_geom_df.eta) < 2.85)])

# for index, tc_geom in tc_geom_df.iterrows():
#     tc_geom.max_dist_neigh = np.max(tc_geom.neighbor_distance)

    # -------------------------------------------------------
    # book histos
    hgen = histos.GenPartHistos('h_genAll')
    hGenPartsGammas = histos.GenParticleHistos('h_genPartsGammas')

    hdigis = histos.DigiHistos('h_hgcDigisAll')
    htc = histos.TCHistos('h_tcAll')

    h2dcl = histos.ClusterHistos('h_clAll')
    h3dcl = histos.Cluster3DHistos('h_cl3dAll')

    h2dclGEO = histos.ClusterHistos('h_clGEOAll')
    h3dclGEO = histos.Cluster3DHistos('h_cl3dGEOAll')


    h2dclDBS = histos.ClusterHistos('h_clDBSAll')
    h3dclDBS = histos.Cluster3DHistos('h_cl3dDBSAll')

    h3dclDBSp = histos.Cluster3DHistos('h_cl3dDBSpAll')

    htcMatch = histos.TCHistos('h_tcMatch')
    h2dclMatch = histos.ClusterHistos('h_clMatch')
    h3dclMatch = histos.Cluster3DHistos('h_cl3dMatch')

    htcMatchGEO = histos.TCHistos('h_tcMatchGEO')
    h2dclMatchGEO = histos.ClusterHistos('h_clMatchGEO')
    h3dclMatchGEO = histos.Cluster3DHistos('h_cl3dMatchGEO')


    htcMatchDBS = histos.TCHistos('h_tcMatchDBS')
    h2dclMatchDBS = histos.ClusterHistos('h_clMatchDBS')
    h3dclMatchDBS = histos.Cluster3DHistos('h_cl3dMatchDBS')

    htcMatchDBSp = histos.TCHistos('h_tcMatchDBSp')
    h2dclMatchDBSp = histos.ClusterHistos('h_clMatchDBSp')
    h3dclMatchDBSp = histos.Cluster3DHistos('h_cl3dMatchDBSp')

    hreso = histos.ResoHistos('h_EleReso')
    hreso2D = histos.Reso2DHistos('h_ClReso')
    hreso2D_1t6 = histos.Reso2DHistos('h_ClReso1t6')
    hreso2D_10t20 = histos.Reso2DHistos('h_ClReso10t20')
    hreso2D_20t28 = histos.Reso2DHistos('h_ClReso20t28')

    hresoGEO = histos.ResoHistos('h_EleResoGEO')
    hreso2DGEO = histos.Reso2DHistos('h_ClResoGEO')
    hreso2DGEO_1t6 = histos.Reso2DHistos('h_ClResoGEO1t6')
    hreso2DGEO_10t20 = histos.Reso2DHistos('h_ClResoGEO10t20')
    hreso2DGEO_20t28 = histos.Reso2DHistos('h_ClResoGEO20t28')

    hresoDBS = histos.ResoHistos('h_EleResoDBS')
    hreso2DDBS = histos.Reso2DHistos('h_ClResoDBS')
    hreso2DDBS_1t6 = histos.Reso2DHistos('h_ClResoDBS1t6')
    hreso2DDBS_10t20 = histos.Reso2DHistos('h_ClResoDBS10t20')
    hreso2DDBS_20t28 = histos.Reso2DHistos('h_ClResoDBS20t28')

    hresoDBSp = histos.ResoHistos('h_EleResoDBSp')
    hreso2DDBSp = histos.Reso2DHistos('h_ClResoDBSp')
    hreso2DDBSp_1t6 = histos.Reso2DHistos('h_ClResoDBSp1t6')
    hreso2DDBSp_10t20 = histos.Reso2DHistos('h_ClResoDBSp10t20')
    hreso2DDBSp_20t28 = histos.Reso2DHistos('h_ClResoDBSp20t28')

    hDensity_3p6 = histos.DensityHistos('h_dens3p6')
    hDensityClus_3p6 = histos.DensityHistos('h_densClus3p6')
    hDensity_2p5 = histos.DensityHistos('h_dens2p5')
    hDensityClus_2p5 = histos.DensityHistos('h_densClus2p5')

    hDR = ROOT.TH1F('hDR', 'DR 2D clusters', 100, 0, 1)
    dump = False

    for event in ntuple:
        if (params.maxEvents != -1 and event.entry() >= params.maxEvents):
            break
        if debug >= 2 or event.entry() % 100 == 0:
            print ("--- Event", event.entry())

        if event.entry() in params.eventsToDump:
            dump = True
        else:
            dump = False
        # -------------------------------------------------------
        # --- GenParticles
        genParts = event.getDataFrame(prefix='gen')
        if debug >= 2:
            print ("# gen parts: {}".format(len(genParts)))
        if debug >= 3:
            print(genParts.iloc[:3])

        hgen.fill(genParts)

        genParticles = event.getDataFrame(prefix='genpart')
        if debug >= 2:
            print ("# gen particles: {}".format(len(genParticles)))
        if debug >= 3:
            print(genParticles[['eta', 'phi', 'pt', 'energy', 'mother', 'gen', 'pid', 'reachedEE']])

        # we find the genparticles matched to the GEN info
        genPartGenerator = genParticles[genParticles.gen > 0]
        if debug >= 3:
            print(genPartGenerator)
        #hgen.fill(genParts)
        hGenPartsGammas.fill(genParticles[(genParticles.gen > 0) & (genParticles.pid == 22)])


        # -------------------------------------------------------
        # --- Digis
        hgcDigis = event.getDataFrame(prefix='hgcdigi')
        if debug >= 2:
            print ('# HGCAL digis: {}'.format(len(hgcDigis)))
        if debug >= 3:
            print (hgcDigis.iloc[:3])
        hdigis.fill(hgcDigis)

        # -------------------------------------------------------
        # --- Trigger Cells
        triggerCells = event.getDataFrame(prefix='tc')
        if(debug >= 2):
            print ("# of TC: {}".format(len(triggerCells)))

        tcsWithPos = pd.merge(triggerCells, tc_geom_df[['id', 'x', 'y']], on='id')

        # json.dump(data, f)test_sample
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

        if dump:
            js_filename = 'tc_dump_ev_{}.json'.format(event.entry())
            dumpFrame2JSON(js_filename, tcsWithPos)
            js_2dc_filename = '2dc_dump_ev_{}.json'.format(event.entry())
            dumpFrame2JSON(js_2dc_filename, triggerClusters)

        if(debug >= 2):
            print('# of NN clusters: {}'.format(len(triggerClusters)))

        if(debug >= 3):
            print(triggerClusters.iloc[:3])
        #     print(triggerClusters.cells.iloc[:3])
        #     # these are all the trigger-cells used in the first 3 2D clusters
        #     print(triggerCells[triggerCells.index.isin(np.concatenate(triggerClusters.cells.iloc[:3]))])

        h2dcl.fill(triggerClusters)

        triggerClustersGEO = event.getDataFrame(prefix='clGEO')
        trigger3DClustersGEO = event.getDataFrame(prefix='cl3dGEO')

        if(debug >= 2):
            print('# of GEO clusters: {}'.format(len(triggerClustersGEO)))

        if(debug >= 3):
            print(triggerClustersGEO.iloc[:3])

        h2dclGEO.fill(triggerClustersGEO)

        if computeDensity:
            # def computeDensity(tcs):
            #     eps = 3.5
            #     for idx, tc in tcsWithPos_ee_layer.iterrows():
            #         energy_list = list()
            #         tcsinradius = tcs[((tcs.x-tc.x)**2+(tcs.y-tc.y)**2) < eps**2]
            #         totE = np.sum(tcsinradius.energy)
            #
            tcsWithPos_ee = tcsWithPos[tcsWithPos.subdet == 3]
            # triggerClusters_ee = triggerClusters[triggerClusters.subdet == 3]

            def getEnergyAndTCsInRadius(tc, tcs, radius):
                tcs_in_radius = tcs[((tcs.x-tc.x)**2+(tcs.y-tc.y)**2) < radius**2]
                e_in_radius = np.sum(tcs_in_radius.energy)
                ntc_in_radius = tcs_in_radius.shape[0]
                return e_in_radius, ntc_in_radius

            for layer in range(1, 29):
                # print ('------- Layer {}'.format(layer))
                tcsWithPos_ee_layer = tcsWithPos_ee[tcsWithPos_ee.layer == layer]
                # print ('   --- Cells: ')
                # print (tcsWithPos_ee_layer)
                triggerClusters_ee_layer = triggerClusters[triggerClusters.layer == layer]

                for eps in [3.6, 2.5]:
                    hDensity = hDensity_3p6
                    hDensityClus = hDensityClus_3p6
                    if eps == 2.5:
                        hDensity = hDensity_2p5
                        hDensityClus = hDensityClus_2p5

                    energy_list_layer = list()
                    ntcs_list_layer = list()
                    for idx, tc in tcsWithPos_ee_layer.iterrows():
                        en_in_radius, ntcs_in_radius = getEnergyAndTCsInRadius(tc, tcsWithPos_ee_layer, eps)
                        energy_list_layer.append(en_in_radius)
                        ntcs_list_layer.append(ntcs_in_radius)
                    if(len(energy_list_layer) != 0):
                        hDensity.fill(layer, max(energy_list_layer), max(ntcs_list_layer))

                    for idx, tcl in triggerClusters_ee_layer.iterrows():
                        tcsInCl = tcsWithPos_ee_layer.loc[tcl.cells]
                        energy_list = list()
                        ntcs_list = list()
                        for idc, tc in tcsInCl.iterrows():
                            en_in_radius, ntcs_in_radius = getEnergyAndTCsInRadius(tc, tcsInCl, eps)
                            energy_list.append(en_in_radius)
                            ntcs_list.append(ntcs_in_radius)
                        if(len(energy_list) != 0):
                            hDensityClus.fill(layer, max(energy_list), max(ntcs_list))

        # Now build DBSCAN 2D clusters
        triggerClustersDBS = pd.DataFrame()
        if params.clusterize:
            for zside in [-1, 1]:
                arg = [(layer, zside, tcsWithPos) for layer in range(0, 29)]
                results = pool.map(clAlgo.buildDBSCANClustersUnpack, arg)
                for clres in results:
                    triggerClustersDBS = triggerClustersDBS.append(clres, ignore_index=True)
                if plot2DCLDR:
                    for idx, cl in triggerClustersDBS[triggerClustersDBS.zside == zside].iterrows():
                        for idx2 in range(idx+1, triggerClustersDBS[triggerClustersDBS.zside == zside].shape[0]):
                            hDR.Fill(math.sqrt((cl.eta-triggerClustersDBS[triggerClustersDBS.zside == zside].loc[idx2].eta)**2+(cl.phi-triggerClustersDBS[triggerClustersDBS.zside == zside].loc[idx2].phi)**2))
            # for layer in range(0, 29):
            #     triggerClustersDBS = triggerClustersDBS.append(clAlgo.buildDBSCANClusters(layer, zside, tcsWithPos), ignore_index=True)
        if(debug >= 2):
            print('# of DBS clusters: {}'.format(len(triggerClustersDBS)))

        if(debug >= 3):
            print(triggerClustersDBS.iloc[:3])
        if(triggerClustersDBS.shape[0] != 0):
            h2dclDBS.fill(triggerClustersDBS)

        # clusters3d = event.trigger3DClusters()
        # print('# 3D clusters old style: {}'.format(len(clusters3d)))
        # for cluster in clusters3d:
        #     print(len(cluster.clusters()))

        if(debug >= 2):
            print('# of NN 3D clusters: {}'.format(len(trigger3DClusters)))
        if(debug >= 3):
            print(trigger3DClusters.iloc[:3])
        h3dcl.fill(trigger3DClusters)

        if(debug >= 2):
            print('# of GEO 3D clusters: {}'.format(len(trigger3DClustersGEO)))
        if(debug >= 3):
            print(trigger3DClustersGEO.iloc[:3])
        h3dclGEO.fill(trigger3DClustersGEO)


        trigger3DClustersDBS = pd.DataFrame()
        if params.clusterize:
            trigger3DClustersDBS = build3DClusters('DBSCAN', clAlgo.build3DClustersEtaPhi, triggerClustersDBS, pool, debug)

        if(trigger3DClustersDBS.shape[0] != 0):
            h3dclDBS.fill(trigger3DClustersDBS)

        trigger3DClustersDBSp = pd.DataFrame()
        if params.clusterize:
            trigger3DClustersDBSp = build3DClusters('DBSCANp', clAlgo.build3DClustersProj, triggerClustersDBS, pool, debug)

        if(trigger3DClustersDBSp.shape[0] != 0):
            h3dclDBSp.fill(trigger3DClustersDBSp)

        # resolution study
        electron_PID = 11
        genElectrons = genParts[(abs(genParts.pdgid) == electron_PID)]

        plot3DClusterMatch(genElectrons,
                           trigger3DClusters,
                           triggerClusters,
                           tcsWithPos,
                           htcMatch,
                           h2dclMatch,
                           h3dclMatch,
                           hreso,
                           hreso2D,
                           hreso2D_1t6,
                           hreso2D_10t20,
                           hreso2D_20t28,
                           'NN',
                           debug)

        plot3DClusterMatch(genElectrons,
                           trigger3DClustersGEO,
                           triggerClustersGEO,
                           tcsWithPos,
                           htcMatchGEO,
                           h2dclMatchGEO,
                           h3dclMatchGEO,
                           hresoGEO,
                           hreso2DGEO,
                           hreso2DGEO_1t6,
                           hreso2DGEO_10t20,
                           hreso2DGEO_20t28,
                           'GEO',
                           debug)

        if params.clusterize:
            plot3DClusterMatch(genElectrons,
                               trigger3DClustersDBS,
                               triggerClustersDBS,
                               tcsWithPos,
                               htcMatchDBS,
                               h2dclMatchDBS,
                               h3dclMatchDBS,
                               hresoDBS,
                               hreso2DDBS,
                               hreso2DDBS_1t6,
                               hreso2DDBS_10t20,
                               hreso2DDBS_20t28,
                               'DBSCAN',
                               debug)


            plot3DClusterMatch(genElectrons,
                               trigger3DClustersDBSp,
                               triggerClustersDBS,
                               tcsWithPos,
                               htcMatchDBSp,
                               h2dclMatchDBSp,
                               h3dclMatchDBSp,
                               hresoDBSp,
                               hreso2DDBSp,
                               hreso2DDBSp_1t6,
                               hreso2DDBSp_10t20,
                               hreso2DDBSp_20t28,
                               'DBSCANp',
                               debug)

    print ("Processed {} events/{} TOT events".format(event.entry(), ntuple.nevents()))
    print ("Writing histos to file {}".format(params.output_filename))

    output.cd()
    hm = histos.HistoManager()
    hm.writeHistos()

    hDR.Write()
    output.Close()

    return



import optparse
import ConfigParser

def main():
    # ============================================
    # configuration bit

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)
    parser.add_option('-f', '--file', dest='CONFIGFILE', help='specify the ini configuration file')
    parser.add_option('-c', '--collection', dest='COLLECTION', help='specify the collection to be processed')
    parser.add_option('-s', '--sample', dest='SAMPLE', help='specify the sample (within the collection) to be processed (-1 for all)')
    parser.add_option('-d', '--debug', dest='DEBUG', help='debug level (default is 0)', default=0)
    parser.add_option('-n', '--nevents', dest='NEVENTS', help='# of events to process per sample (default is 10)', default=10)
    parser.add_option("-b", "--batch", action="store_true", dest="BATCH", default=False, help="submit the jobs via CONDOR")


    global opt, args
    (opt, args) = parser.parse_args()

    cfgfile = ConfigParser.ConfigParser()
    cfgfile.optionxform = str

    cfgfile.read(opt.CONFIGFILE)

    collection_dict = {}
    collections = cfgfile.get('common', 'collections').split(',')
    basedir = cfgfile.get('common', 'input_dir_lx')
    hostname = socket.gethostname()
    if 'matterhorn' in hostname or 'Matterhorn' in hostname:
            basedir = cfgfile.get('common', 'input_dir_local')
    plot_version = cfgfile.get('common', 'plot_version')
    run_clustering = False
    if  cfgfile.get('common', 'run_clustering') == 'True':
        run_clustering = True
    run_density_computation = False
    if  cfgfile.get('common', 'run_density_computation') == 'True':
        run_density_computation = True

    events_to_dump = cfgfile.get('common', 'events_to_dump').split(',')

    for collection in collections:
        samples = cfgfile.get(collection, 'samples').split(',')
        print ('--- Collection: {} with samples: {}'.format(collection, samples))
        sample_list = list()
        for sample in samples:
            params = Parameters(input_base_dir=basedir,
                                input_sample_dir=cfgfile.get(sample, 'input_sample_dir'),
                                output_filename='{}/plots/histos_{}_{}.root'.format(basedir, sample, plot_version),
                                clusterize=run_clustering,
                                eventsToDump=events_to_dump,
                                maxEvents=opt.NEVENTS,
                                debug=opt.DEBUG,
                                computeDensity=run_density_computation,
                                name=sample)
            sample_list.append(params)
        collection_dict[collection]=sample_list
        #collection_dict

    samples_to_process = list()
    if opt.COLLECTION:
        if opt.COLLECTION in collection_dict.keys():
            if opt.SAMPLE:
                if int(opt.SAMPLE) == -1:
                    samples_to_process.extend(collection_dict[opt.COLLECTION])
                else:
                    samples_to_process.append(collection_dict[opt.COLLECTION][int(opt.SAMPLE)])
            else:
                print ('Collection: {}, available samples: {}'.format(opt.COLLECTION, collection_dict[opt.COLLECTION]))
                sys.exit(0)
        else:
            print ('ERROR: collection {} not in the cfg file'.format(opt.COLLECTION))
            sys.exit(10)
    else:
        print ('\nAvailable collections: {}'.format(collection_dict.keys()))
        sys.exit(0)

    print ('About to process samples: {}'.format(samples_to_process))

    if opt.BATCH:
        samp = opt.SAMPLE
        if samp == '-1':
            samp = 'all'
        batch_dir = 'batch_{}_{}_{}'.format(opt.COLLECTION, samp, plot_version)
        os.mkdir(batch_dir)
        os.mkdir(batch_dir+'/conf/')
        os.mkdir(batch_dir+'/logs/')

        # prepare the CONDOR .sub file
        condor_template_name = 'templates/batch.sub'
        condor_template_file = open(condor_template_name)
        condor_template = condor_template_file.read()
        condor_template_file.close()

        condor_template = condor_template.replace('TEMPL_TASKDIR', batch_dir)
        condor_template = condor_template.replace('TEMPL_NJOBS', str(len(samples_to_process)))

        condor_file_path = os.path.join(batch_dir, 'batch.sub')
        condor_file = open(condor_file_path, 'w')
        condor_file.write(condor_template)
        condor_file.close()

        # prepare the exec script
        script_template_name = 'templates/run_batch.sh'
        script_template_file = open(script_template_name)
        script_template = script_template_file.read()
        script_template_file.close()

        script_template = script_template.replace('TEMPL_WORKDIR', os.environ["PWD"])
        script_template = script_template.replace('TEMPL_CFG', opt.CONFIGFILE)
        script_template = script_template.replace('TEMPL_COLL', opt.COLLECTION)

        script_file_path = os.path.join(batch_dir, 'run_batch.sh')
        script_file = open(script_file_path, 'w')
        script_file.write(script_template)
        script_file.close()

        print('Ready for submission please run the following commands:')
        print('condor_submit {}'.format(condor_file_path))
        sys.exit(0)



    # test = copy.deepcopy(singleEleE50_PU0)
    # #test.output_filename = 'test2222.root'
    # test.maxEvents = 5
    # test.debug = 6
    # test.eventsToDump = [1, 2, 3, 4]
    # test.clusterize = False
    # test.computeDensity = True
    #
    # test_sample = [test]

    # pool = Pool(1)
    # pool.map(analyze, nugun_samples)
    # pool.map(analyze, test_sample)
    # pool.map(analyze, electron_samples)
    # pool.map(analyze, [singleEleE50_PU200])

    #samples = test_sample
    for sample in samples_to_process:
        analyze(sample)

if __name__ == "__main__":
    main()
