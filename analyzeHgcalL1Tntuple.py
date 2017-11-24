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
                 debug=0):
        self.maxEvents = maxEvents
        self.debug = debug
        self.input_base_dir = input_base_dir
        self.input_sample_dir = input_sample_dir
        self.output_filename = output_filename
        self.clusterize = clusterize
        self.eventsToDump = eventsToDump


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
        matched_idx = utils.match_etaphi(genParticles[['eta', 'phi']],
                                         trigger3DClusters[['eta', 'phi']],
                                         trigger3DClusters['pt'],
                                         deltaR=0.2)
        # print (matched_idx)
    for idx, genParticle in genParticles.iterrows():
        if idx in matched_idx.keys():
            matched3DCluster = trigger3DClusters.iloc[[matched_idx[idx]]]
            matchedClusters = triggerClusters.iloc[matched3DCluster.clusters.item()]
            matchedTriggerCells = triggerCells.iloc[np.concatenate(matchedClusters.cells.values)]
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
    computeDensity = False
    plot2DCLDR = False

    pool = Pool(5)

    # read the geometry dump
    geom_file = 'test_triggergeom.root'

    tc_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeTriggerCells')
    print ('read TC GEOM tree with # events: {}'.format(tc_geom_tree.nevents()))
    tc_geom_df = convertGeomTreeToDF(tc_geom_tree._tree)
    tc_geom_df['radius'] = np.sqrt(tc_geom_df['x']**2+tc_geom_df['y']**2)
    # tc_geom_df['max_neigh_dist'] = 1
    # a5 = tc_geom_df[tc_geom_df.neighbor_n == 5]
    # a5['max_neigh_dist'] =  a5['neighbor_distance'].max()
    # a6 = tc_geom_df[tc_geom_df.neighbor_n == 6]
    # a6['max_neigh_dist'] =  a6['neighbor_distance'].max()

    # for index, tc_geom in tc_geom_df.iterrows():
    #     tc_geom.max_dist_neigh = np.max(tc_geom.neighbor_distance)

    # print (tc_geom_df[:10])

    #treeTriggerCells = inputFile.Get("hgcaltriggergeomtester/TreeTriggerCells")
    #treeCells        = inputFile.Get("hgcaltriggergeomtester/TreeCells")


    input_files = listFiles(os.path.join(params.input_base_dir, params.input_sample_dir))
    print ('- dir {} contains {} files.'.format(params.input_sample_dir, len(input_files)))

    ntuple = HGCalNtuple(input_files, tree='hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    print ('- created TChain containing {} events'.format(ntuple.nevents()))

    output = ROOT.TFile(params.output_filename, "RECREATE")
    output.cd()

    hTCGeom = histos.GeomHistos('hTCGeom')
    hTCGeom.fill(tc_geom_df)

# for index, tc_geom in tc_geom_df.iterrows():
#     tc_geom.max_dist_neigh = np.max(tc_geom.neighbor_distance)

    # -------------------------------------------------------
    # book histos
    hgen = histos.GenPartHistos('h_genAll')
    hdigis = histos.DigiHistos('h_hgcDigisAll')
    htc = histos.TCHistos('h_tcAll')
    h2dcl = histos.ClusterHistos('h_clAll')
    h2dclDBS = histos.ClusterHistos('h_clDBSAll')

    h3dcl = histos.Cluster3DHistos('h_cl3dAll')
    h3dclDBS = histos.Cluster3DHistos('h_cl3dDBSAll')
    h3dclDBSp = histos.Cluster3DHistos('h_cl3dDBSpAll')

    htcMatch = histos.TCHistos('h_tcMatch')
    h2dclMatch = histos.ClusterHistos('h_clMatch')
    h3dclMatch = histos.Cluster3DHistos('h_cl3dMatch')

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


    hDensity = ROOT.TH2F('hDensity', 'E (GeV) Density per layer', 60, 0, 60, 200, 0, 10)
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

        tcsWithPos = pd.merge(triggerCells, tc_geom_df[['id', 'x', 'y', 'radius']], on='id')
        if computeDensity:
            # def computeDensity(tcs):
            #     eps = 3.5
            #     for idx, tc in tcsWithPos_ee_layer.iterrows():
            #         energy_list = list()
            #         tcsinradius = tcs[((tcs.x-tc.x)**2+(tcs.y-tc.y)**2) < eps**2]
            #         totE = np.sum(tcsinradius.energy)
            #
            tcsWithPos_ee = tcsWithPos[tcsWithPos.subdet == 3]
            eps = 3.5
            for layer in range(1, 29):
                tcsWithPos_ee_layer = tcsWithPos_ee[tcsWithPos_ee.layer == layer]

                for idx, tc in tcsWithPos_ee_layer.iterrows():
                    energy_list = list()

                    # get all TCs within eps radius from the current one
                    tcsinradius = tcsWithPos_ee_layer[((tcsWithPos_ee_layer.x-tc.x)**2+(tcsWithPos_ee_layer.y-tc.y)**2) < eps**2]
                    totE = np.sum(tcsinradius.energy)
                    energy_list.append(totE)
                    if(len(energy_list) != 0):
                        hDensity.Fill(layer, max(energy_list))

                # json.dump(data, f)
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
            print('# of clusters: {}'.format(len(triggerClusters)))

        if(debug >= 3):
            print(triggerClusters.iloc[:3])
        #     print(triggerClusters.cells.iloc[:3])
        #     # these are all the trigger-cells used in the first 3 2D clusters
        #     print(triggerCells[triggerCells.index.isin(np.concatenate(triggerClusters.cells.iloc[:3]))])

        h2dcl.fill(triggerClusters)

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
            print('# of 3D clusters: {}'.format(len(trigger3DClusters)))
        if(debug >= 3):
            print(trigger3DClusters.iloc[:3])
        h3dcl.fill(trigger3DClusters)

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
        genElectrons = genParts[(abs(genParts.id) == electron_PID)]

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
    hgen.write()
    hdigis.write()

    htc.write()
    h2dcl.write()
    h2dclDBS.write()

    h3dcl.write()
    htcMatch.write()
    h2dclMatch.write()
    h3dclMatch.write()

    htcMatchDBS.write()
    h2dclMatchDBS.write()
    h3dclMatchDBS.write()

    h3dclDBS.write()

    htcMatchDBSp.write()
    h2dclMatchDBSp.write()
    h3dclMatchDBSp.write()

    h3dclDBSp.write()


    hreso.write()
    hreso2D.write()

    hresoDBS.write()
    hreso2DDBS.write()

    hresoDBSp.write()
    hreso2DDBSp.write()

    hreso2D_1t6.write()
    hreso2D_10t20.write()
    hreso2D_20t28.write()

    hreso2DDBS_1t6.write()
    hreso2DDBS_10t20.write()
    hreso2DDBS_20t28.write()

    hreso2DDBSp_1t6.write()
    hreso2DDBSp_10t20.write()
    hreso2DDBSp_20t28.write()

    hTCGeom.write()
    hDensity.Write()
    hDR.Write()
    output.Close()

    return




def main():
    # ============================================
    # configuration bit

    #input_sample_dir = 'FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU0_20171005/NTUP/'
    #output_filename = 'histos_EleE50_PU0.root'

    # input_sample_dir = 'FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005/NTUP/'
    # output_filename = 'histos_EleE50_PU50.root'

    ntuple_version = 'NTUP'
    run_clustering = True
    plot_version = 'v3'
    # ============================================
    basedir = '/eos/user/c/cerminar/hgcal/CMSSW932'
    hostname = socket.gethostname()
    if 'matterhorn' in hostname:
            basedir = '/Users/cerminar/cernbox/hgcal/CMSSW932/'
    #
    singleEleE50_PU200 = Parameters(input_base_dir=basedir,
                                    input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU200_20171005/{}/'.format(ntuple_version),
                                    output_filename='histos_EleE50_PU200_{}.root'.format(plot_version),
                                    clusterize=run_clustering,
                                    eventsToDump=[])

    singleEleE50_PU0 = Parameters(input_base_dir=basedir,
                                  input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU0_20171005/{}/'.format(ntuple_version),
                                  output_filename='histos_EleE50_PU0_{}.root'.format(plot_version),
                                  clusterize=run_clustering,
                                  eventsToDump=[])

    singleEleE50_PU50 = Parameters(input_base_dir=basedir,
                                   input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005/{}/'.format(ntuple_version),
                                   output_filename='histos_EleE50_PU50_{}.root'.format(plot_version),
                                   clusterize=run_clustering,
                                   eventsToDump=[])

    singleEleE50_PU100 = Parameters(input_base_dir=basedir,
                                    input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU100_20171005/{}/'.format(ntuple_version),
                                    output_filename='histos_EleE50_PU100_{}.root'.format(plot_version),
                                    clusterize=run_clustering,
                                    eventsToDump=[])

    electron_samples = [singleEleE50_PU0, singleEleE50_PU200, singleEleE50_PU50, singleEleE50_PU100 ]


    nuGun_PU50 = Parameters(input_base_dir=basedir,
                            input_sample_dir='FlatRandomPtGunProducer_NuGunPU50_20171005/{}/'.format(ntuple_version),
                            output_filename='histos_NuGun_PU50_{}.root'.format(plot_version),
                            clusterize=run_clustering,
                            eventsToDump=[])

    nuGun_PU100 = Parameters(input_base_dir=basedir,
                             input_sample_dir='FlatRandomPtGunProducer_NuGunPU100_20171005/{}/'.format(ntuple_version),
                             output_filename='histos_NuGun_PU100_{}.root'.format(plot_version),
                             clusterize=run_clustering,
                             eventsToDump=[])

    nuGun_PU140 = Parameters(input_base_dir=basedir,
                             input_sample_dir='FlatRandomPtGunProducer_NuGunPU140_20171005/{}/'.format(ntuple_version),
                             output_filename='histos_NuGun_PU140_{}.root'.format(plot_version),
                             clusterize=run_clustering,
                             eventsToDump=[])

    nuGun_PU200 = Parameters(input_base_dir=basedir,
                             input_sample_dir='FlatRandomPtGunProducer_NuGunPU200_20171006/{}/'.format(ntuple_version),
                             output_filename='histos_NuGun_PU200_{}.root'.format(plot_version),
                             clusterize=run_clustering,
                             eventsToDump=[])

    nugun_samples = [nuGun_PU50, nuGun_PU100, nuGun_PU140, nuGun_PU200]
#
    test = copy.deepcopy(singleEleE50_PU0)
    test.output_filename = 'test44.root'
    test.maxEvents = 100
    test.debug = 2
    test.eventsToDump = [1, 2, 3, 4]
    test.clusterize = True

    test_sample = [test]

    # pool = Pool(1)
    # pool.map(analyze, nugun_samples)
    # pool.map(analyze, test_sample)
    # pool.map(analyze, electron_samples)
    # pool.map(analyze, [singleEleE50_PU200])

    samples = test_sample
    for sample in samples:
        analyze(sample)

if __name__ == "__main__":
    main()
