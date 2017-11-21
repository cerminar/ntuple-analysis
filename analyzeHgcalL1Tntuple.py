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


def analyze(params):
    debug = params.debug
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

    print (tc_geom_df[:10])

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

    htcMatch = histos.TCHistos('h_tcMatch')
    h2dclMatch = histos.ClusterHistos('h_clMatch')
    h3dclMatch = histos.Cluster3DHistos('h_cl3dMatch')

    hreso = histos.ResoHistos('h_EleReso')
    hresoDBS = histos.ResoHistos('h_EleResoDBS')

    hDensity = ROOT.TH2F('hDensity', 'E (GeV) Density per layer', 60, 0, 60, 100, 0, 20)

    dump = False

    eventToDump = [1,2,3,4]

    for event in ntuple:
        if (params.maxEvents != -1 and event.entry() >= params.maxEvents):
            break
        if debug >= 2 or event.entry() % 100 == 0:
            print ("--- Event", event.entry())

        if event.entry() in eventToDump:
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
        computeDensity = False
        if computeDensity:
            tcsWithPos_ee = tcsWithPos[tcsWithPos.subdet == 3]
            eps = 3.5
            for layer in range(1, 29):
                tcsWithPos_ee_layer = tcsWithPos_ee[tcsWithPos_ee.layer == layer]
                energy_list = list()

                for idx, tc in tcsWithPos_ee_layer.iterrows():
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
        for zside in [-1, 1]:
            arg = [(layer, zside, tcsWithPos) for layer in range(0, 29)]
            results = pool.map(clAlgo.buildDBSCANClustersUnpack, arg)
            for clres in results:
                triggerClustersDBS = triggerClustersDBS.append(clres, ignore_index=True)
            # for layer in range(0, 29):
            #     triggerClustersDBS = triggerClustersDBS.append(clAlgo.buildDBSCANClusters(layer, zside, tcsWithPos), ignore_index=True)
        if(debug >= 2):
            print('# of DBS clusters: {}'.format(len(triggerClustersDBS)))

        if(debug >= 3):
            print(triggerClustersDBS.iloc[:3])

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
        clusterSides = [triggerClustersDBS[triggerClustersDBS.eta > 0], triggerClustersDBS[triggerClustersDBS.eta < 0]]
        results3Dcl = pool.map(clAlgo.build3DClusters, clusterSides)
        for res3D in results3Dcl:
            trigger3DClustersDBS = trigger3DClustersDBS.append(res3D, ignore_index=True)

        if(debug >= 2):
            print('# of DBS 3D clusters: {}'.format(len(trigger3DClustersDBS)))
        if(debug >= 3):
            print(trigger3DClustersDBS.iloc[:3])
        h3dclDBS.fill(trigger3DClustersDBS)

        # resolution study
        electron_PID = 11
        genElectrons = genParts[(abs(genParts.id) == electron_PID)]
        matched_idx = utils.match_etaphi(genElectrons[['eta', 'phi']], trigger3DClusters[['eta', 'phi']], trigger3DClusters['pt'], deltaR=0.2)

        matchedDBS_idx = utils.match_etaphi(genElectrons[['eta', 'phi']], trigger3DClustersDBS[['eta', 'phi']], trigger3DClustersDBS['pt'], deltaR=0.2)

        for idx, genElectron in genElectrons.iterrows():
            matched3DCluster = trigger3DClusters.iloc[[matched_idx[idx]]]
            matchedClusters = triggerClusters.iloc[matched3DCluster.clusters.item()]
            matchedTriggerCells = triggerCells.iloc[np.concatenate(matchedClusters.cells.values)]

            if idx in matchedDBS_idx.keys():
                matched3DClusterDBS = trigger3DClustersDBS.iloc[[matchedDBS_idx[idx]]]
                matchedClustersDBS = triggerClustersDBS.iloc[matched3DClusterDBS.clusters.item()]
                matchedTriggerCellsDBS = triggerCells.iloc[np.concatenate(matchedClustersDBS.cells.values)]
                hresoDBS.fill(reference=genElectron, target=matched3DClusterDBS.iloc[0])

            else:
                print (genElectrons[['eta', 'phi', 'energy']])
                print (trigger3DClusters[['eta', 'phi', 'energy']])
                print (trigger3DClustersDBS[['eta', 'phi', 'energy']])
                print("ERROR: no match found for DBS!!!")
            #FIXME: understand why this is not the case: there is a problem with phi definition for eta < 0???

            if dump:
                if genElectron.eta > 0:
                    js_2dc_filename = 'm2dc_dump_ev_{}.json'.format(event.entry())
                    dumpFrame2JSON(js_2dc_filename, matchedClusters)

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
    h2dclDBS.write()
    h3dcl.write()
    htcMatch.write()
    h2dclMatch.write()
    h3dclMatch.write()
    h3dclDBS.write()
    hreso.write()
    hresoDBS.write()
    hTCGeom.write()
    hDensity.Write()
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
    basedir = '/eos/user/c/cerminar/hgcal/CMSSW932'
    # basedir = '/Users/cerminar/cernbox/hgcal/CMSSW932/'
    singleEleE50_PU200 = Parameters(input_base_dir=basedir,
                                    input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU200_20171005/{}/'.format(ntuple_version),
                                    output_filename='histos_EleE50_PU200.root')

    singleEleE50_PU0 = Parameters(input_base_dir=basedir,
                                  input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU0_20171005/{}/'.format(ntuple_version),
                                  output_filename='histos_EleE50_PU0.root')

    singleEleE50_PU50 = Parameters(input_base_dir=basedir,
                                   input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005/{}/'.format(ntuple_version),
                                   output_filename='histos_EleE50_PU50.root')

    singleEleE50_PU100 = Parameters(input_base_dir=basedir,
                                    input_sample_dir='FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU100_20171005/{}/'.format(ntuple_version),
                                    output_filename='histos_EleE50_PU100.root')

    electron_samples = [singleEleE50_PU0, singleEleE50_PU50, singleEleE50_PU100, singleEleE50_PU200]

    test = singleEleE50_PU200
    test.output_filename = 'testa.root'
    test.maxEvents = 10
    test.debug = 2

    test_sample = [test]

    nuGun_PU50 = Parameters(input_base_dir=basedir,
                            input_sample_dir='FlatRandomPtGunProducer_NuGunPU50_20171005/{}/'.format(ntuple_version),
                            output_filename='histos_NuGun_PU50.root')

    nuGun_PU100 = Parameters(input_base_dir=basedir,
                             input_sample_dir='FlatRandomPtGunProducer_NuGunPU100_20171005/{}/'.format(ntuple_version),
                             output_filename='histos_NuGun_PU100.root')

    nuGun_PU140 = Parameters(input_base_dir=basedir,
                             input_sample_dir='FlatRandomPtGunProducer_NuGunPU140_20171005/{}/'.format(ntuple_version),
                             output_filename='histos_NuGun_PU140.root')

    nuGun_PU200 = Parameters(input_base_dir=basedir,
                             input_sample_dir='FlatRandomPtGunProducer_NuGunPU200_20171006/{}/'.format(ntuple_version),
                             output_filename='histos_NuGun_PU200.root')

    nugun_samples = [nuGun_PU50, nuGun_PU100, nuGun_PU140, nuGun_PU200]
#
# FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU0_20171005   FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU200_20171005 FlatRandomPtGunProducer_NuGunPU140_20171005
# FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU100_20171005 FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU50_20171005  FlatRandomPtGunProducer_NuGunPU200_20171006
# FlatRandomEGunProducer_EleGunE50_1p7_2p8_PU140_20171006 FlatRandomPtGunProducer_NuGunPU100_20171005             FlatRandomPtGunProducer_NuGunPU50_20171005


    pool = Pool(1)
    #pool.map(analyze, nugun_samples)
    #pool.map(analyze, test_sample)
    #pool.map(analyze, electron_samples)
    #pool.map(analyze, [singleEleE50_PU200])
    analyze(test)

if __name__ == "__main__":
    main()
