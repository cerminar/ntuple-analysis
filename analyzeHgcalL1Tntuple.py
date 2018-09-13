#!/usr/bin/env python
# import ROOT
# from __future__ import print_function
from NtupleDataFormat import HGCalNtuple, Event
import sys
import root_numpy as rnp
import pandas as pd
import numpy as np
from multiprocessing import Pool
from shutil import copyfile

# The purpose of this file is to demonstrate mainly the objects
# that are in the HGCalNtuple
import ROOT
import os
import math
import copy
import socket
import datetime
import optparse
import ConfigParser

import l1THistos as histos
import utils as utils
import clusterTools as clAlgo
import traceback
import subprocess32
from utils import debugPrintOut

import file_manager as fm

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
                 output_dir,
                 clusterize,
                 eventsToDump,
                 events_per_job,
                 version,
                 maxEvents=-1,
                 computeDensity=False,
                 debug=0,
                 name=''):
        self.name = name
        self.maxEvents = maxEvents
        self.debug = debug
        self.input_base_dir = input_base_dir
        self.input_sample_dir = input_sample_dir
        self.output_filename = output_filename
        self.output_dir = output_dir
        self.clusterize = clusterize
        self.eventsToDump = eventsToDump
        self.computeDensity = computeDensity
        self.events_per_job = events_per_job
        self.version = version

    def __str__(self):
        return 'Name: {},\n \
                clusterize: {}\n \
                compute density: {}\n \
                maxEvents: {}\n \
                output file: {}\n \
                events per job: {}\n \
                debug: {}'.format(self.name,
                                  self.clusterize,
                                  self.computeDensity,
                                  self.maxEvents,
                                  self.output_filename,
                                  self.events_per_job,
                                  self.debug)

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


def computeIsolation(all3DClusters, idx_best_match, idx_incone, dr):
    ret = pd.DataFrame()
    # print 'index best match: {}'.format(idx_best_match)
    # print 'indexes all in cone: {}'.format(idx_incone)
    components = all3DClusters[(all3DClusters.index.isin(idx_incone)) & ~(all3DClusters.index == idx_best_match)]
    # print 'components indexes: {}'.format(components.index)
    compindr = components[np.sqrt((components.eta-all3DClusters.loc[idx_best_match].eta)**2 + (components.phi-all3DClusters.loc[idx_best_match].phi)**2) < dr]
    if not compindr.empty:
        # print 'components indexes in dr: {}'.format(compindr.index)
        ret['energy'] = [compindr.energy.sum()]
        ret['eta'] = [np.sum(compindr.eta*compindr.energy)/compindr.energy.sum()]
        ret['pt'] = [(ret.energy/np.cosh(ret.eta)).values[0]]
    else:
        ret['energy'] = [0.]
        ret['eta'] = [0.]
        ret['pt'] = [0.]
    return ret


def sumClustersInCone(all3DClusters, idx_incone):
    ret = pd.DataFrame()
    components = all3DClusters[all3DClusters.index.isin(idx_incone)]
    ret['energy'] = [components.energy.sum()]
    # FIXME: this needs to be better defined
    ret['energyCore'] = [components.energy.sum()]
    ret['energyCentral'] = [components.energy.sum()]

    ret['eta'] = [np.sum(components.eta*components.energy)/components.energy.sum()]
    ret['phi'] = [np.sum(components.phi*components.energy)/components.energy.sum()]
    ret['pt'] = [(ret.energy/np.cosh(ret.eta)).values[0]]
    ret['ptCore'] = [(ret.energyCore/np.cosh(ret.eta)).values[0]]
    # ret['layers'] = [np.unique(np.concatenate(components.layers.values))]
    ret['clusters'] = [np.concatenate(components.clusters.values)]
    ret['nclu'] = [components.nclu.sum()]
    ret['firstlayer'] = [np.min(components.firstlayer.values)]
    # FIXME: placeholder
    ret['showerlength'] = [1]
    ret['seetot'] = [1]
    ret['seemax'] = [1]
    ret['spptot'] = [1]
    ret['sppmax'] = [1]
    ret['szz'] = [1]
    ret['emaxe'] = [1]
    ret['id'] = [1]
    ret['n010'] = len(components[components.pt > 0.1])
    ret['n025'] = len(components[components.pt > 0.25])

    return ret


def buildTriggerTowerCluster(allTowers, seedTower, debug):
    eta_seed = seedTower.eta.values[0]
    iEta_seed = seedTower.iEta.values[0]
    iPhi_seed = seedTower.iPhi.values[0]
    clusterTowers = allTowers[(allTowers.eta*eta_seed > 0) &
                              (allTowers.iEta <= (iEta_seed + 1)) &
                              (allTowers.iEta >= (iEta_seed - 1)) &
                              (allTowers.iPhi <= (iPhi_seed + 1)) &
                              (allTowers.iPhi >= (iPhi_seed - 1))]
    clusterTowers.loc[clusterTowers.index, 'logEnergy'] = np.log(clusterTowers.energy)
    if debug >= 5:
        print '---- SEED:'
        print seedTower
        print 'Cluster components:'
        print clusterTowers
    ret = pd.DataFrame(columns=['energy', 'eta', 'phi', 'pt'])
    ret['energy'] = [clusterTowers.energy.sum()]
    ret['logEnergy'] = np.log(ret.energy)
    ret['eta'] = [np.sum(clusterTowers.eta*clusterTowers.energy)/ret.energy.values[0]]
    ret['phi'] = [np.sum(clusterTowers.phi*clusterTowers.energy)/ret.energy.values[0]]
    ret['etalw'] = [np.sum(clusterTowers.eta*clusterTowers.logEnergy)/np.sum(clusterTowers.logEnergy)]
    ret['philw'] = [np.sum(clusterTowers.phi*clusterTowers.logEnergy)/np.sum(clusterTowers.logEnergy)]
    ret['pt'] = [(ret.energy / np.cosh(ret.eta)).values[0]]
    return ret


def plotTriggerTowerMatch(genParticles,
                          histoGen,
                          triggerTowers,
                          histoTowersMatch,
                          histoTowersReso,
                          histoTowersResoCl,
                          algoname,
                          debug):

    best_match_indexes = {}
    if triggerTowers.shape[0] != 0:
        best_match_indexes, allmatches = utils.match_etaphi(genParticles[['eta', 'phi']],
                                                            triggerTowers[['eta', 'phi']],
                                                            triggerTowers['pt'],
                                                            deltaR=0.2)
        # print ('-----------------------')
        # print (best_match_indexes)
    # print ('------ best match: ')
    # print (best_match_indexes)
    # print ('------ all matches:')
    # print (allmatches)

    if histoGen is not None:
        histoGen.fill(genParticles)

    for idx, genParticle in genParticles.iterrows():
        if idx in best_match_indexes.keys():
            # print ('-----------------------')
            #  print(genParticle)
            matchedTower = triggerTowers.loc[[best_match_indexes[idx]]]
            # print (matched3DCluster)
            # allMatches = trigger3DClusters.iloc[allmatches[idx]]
            # print ('--')
            # print (allMatches)
            # print (matched3DCluster.clusters.item())
            # print (type(matched3DCluster.clusters.item()))
            # matchedClusters = triggerClusters[ [x in matched3DCluster.clusters.item() for x in triggerClusters.id]]

            # fill the plots
            histoTowersMatch.fill(matchedTower)
            histoTowersReso.fill(reference=genParticle, target=matchedTower)

            ttCluster = buildTriggerTowerCluster(triggerTowers, matchedTower, debug)
            histoTowersResoCl.fill(reference=genParticle, target=ttCluster)

            # clustersInCone = sumClustersInCone(trigger3DClusters, allmatches[idx])
            # print ('----- in cone sum:')
            # print (clustersInCone)
            # histoResoCone.fill(reference=genParticle, target=clustersInCone.iloc[0])

            if debug >= 4:
                print ('--- Dump match for algo {} ---------------'.format(algoname))
                print ('GEN particle: idx: {}'.format(idx))
                print (genParticle)
                print ('Matched Trigger Tower:')
                print (matchedTower)
        else:
            if debug >= 0:
                print ('==== Warning no match found for algo {}, idx {} ======================'.format(algoname, idx))
                if debug >= 2:
                    print (genParticle)


def unpack(mytuple):
    return mytuple[0].getDataFrame(mytuple[1])


def computeClusterRodSharing(cl2ds, tcs):
    cl2ds['rod_bin_max'] = pd.Series(index=cl2ds.index, dtype=object)
    cl2ds['rod_bin_shares'] = pd.Series(index=cl2ds.index, dtype=object)
    cl2ds['rod_bins'] = pd.Series(index=cl2ds.index, dtype=object)

    for index, cl2d in cl2ds.iterrows():
        matchedTriggerCells = tcs[tcs.id.isin(cl2d.cells)]
        energy_sums_byRod = matchedTriggerCells.groupby(by='rod_bin', axis=0).sum()
        bin_max = energy_sums_byRod[['energy']].idxmax()[0]
        cl2ds.set_value(index, 'rod_bin_max', bin_max)
        cl2ds.set_value(index, 'rod_bins', energy_sums_byRod.index.values)

        shares = []
        for iy in range(bin_max[1]-1, bin_max[1]+2):
            for ix in range(bin_max[0]-1, bin_max[0]+2):
                bin = (ix, iy)
                energy = 0.
                if bin in energy_sums_byRod.index:
                    energy = energy_sums_byRod.loc[[bin]].energy[0]
                shares.append(energy)
        cl2ds.set_value(index, 'rod_bin_shares', shares)


def plot3DClusterMatch(genParticles,
                       trigger3DClusters,
                       triggerClusters,
                       triggerCells,
                       histoGen,
                       histoGenMatched,
                       histoTCMatch,
                       histoClMatch,
                       histo3DClMatch,
                       histoReso,
                       histoResoCone,
                       histoReso2D,
                       algoname,
                       debug):

    best_match_indexes = {}
    if not trigger3DClusters.empty:
        best_match_indexes, allmatches = utils.match_etaphi(genParticles[['eta', 'phi']],
                                                            trigger3DClusters[['eta', 'phi']],
                                                            trigger3DClusters['pt'],
                                                            deltaR=0.1)
    # print ('------ best match: ')
    # print (best_match_indexes)
    # print ('------ all matches:')
    # print (allmatches)

    # allmatched2Dclusters = list()
    # matchedClustersAll = pd.DataFrame()
    if histoGen is not None:
        histoGen.fill(genParticles)

    for idx, genParticle in genParticles.iterrows():
        if idx in best_match_indexes.keys():
            # print ('-----------------------')
            #  print(genParticle)
            matched3DCluster = trigger3DClusters.loc[[best_match_indexes[idx]]]
            # print (matched3DCluster)
            # allMatches = trigger3DClusters.iloc[allmatches[idx]]
            # print ('--')
            # print (allMatches)
            # print (matched3DCluster.clusters.item())
            # print (type(matched3DCluster.clusters.item()))
            # matchedClusters = triggerClusters[ [x in matched3DCluster.clusters.item() for x in triggerClusters.id]]
            matchedClusters = triggerClusters[triggerClusters.id.isin(matched3DCluster.clusters.item())]
            # print (matchedClusters)
            matchedTriggerCells = triggerCells[triggerCells.id.isin(np.concatenate(matchedClusters.cells.values))]
            # allmatched2Dclusters. append(matchedClusters)

            if 'energyCentral' not in matched3DCluster.columns:
                calib_factor = 1.084
                matched3DCluster['energyCentral'] = [matchedClusters[(matchedClusters.layer > 9) & (matchedClusters.layer < 21)].energy.sum()*calib_factor]

            iso_df = computeIsolation(trigger3DClusters, idx_best_match=best_match_indexes[idx], idx_incone=allmatches[idx], dr=0.2)
            matched3DCluster['iso0p2'] = iso_df.energy
            matched3DCluster['isoRel0p2'] = iso_df.pt/matched3DCluster.pt

            # fill the plots
            histoTCMatch.fill(matchedTriggerCells)
            histoClMatch.fill(matchedClusters)
            histo3DClMatch.fill(matched3DCluster)
            histoReso2D.fill(reference=genParticle, target=matchedClusters)
            histoReso.fill(reference=genParticle, target=matched3DCluster.iloc[0])

            # now we fill the reso plot for all the clusters in the cone
            clustersInCone = sumClustersInCone(trigger3DClusters, allmatches[idx])

            # print ('----- in cone sum:')
            # print (clustersInCone)
            histoResoCone.fill(reference=genParticle, target=clustersInCone.iloc[0])
            if histoGenMatched is not None:
                histoGenMatched.fill(genParticles.loc[[idx]])

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
                # print ('sum 2D cluster pt: {}'.format(matchedClusters.pt.sum()*calib_factor))
                print ('sum TC energy: {}'.format(matchedTriggerCells.energy.sum()))
                print ('Sum of matched clusters in cone:')
                print (clustersInCone)
        else:
            if debug >= 5:
                print ('==== Warning no match found for algo {}, idx {} ======================'.format(algoname, idx))
                if debug >= 2:
                    print (genParticle)
                    print (trigger3DClusters)

    # if len(allmatched2Dclusters) != 0:
    #     matchedClustersAll = pd.concat(allmatched2Dclusters)
    # return matchedClustersAll


def build3DClusters(name, algorithm, triggerClusters, pool, debug):
    trigger3DClusters = pd.DataFrame()
    if triggerClusters.empty:
        return trigger3DClusters
    clusterSides = [x for x in [triggerClusters[triggerClusters.eta > 0], triggerClusters[triggerClusters.eta < 0]] if not x.empty]
    results3Dcl = pool.map(algorithm, clusterSides)
    for res3D in results3Dcl:
        trigger3DClusters = trigger3DClusters.append(res3D, ignore_index=True)

    debugPrintOut(debug, name='{} 3D clusters'.format(name),
                  toCount=trigger3DClusters,
                  toPrint=trigger3DClusters.iloc[:3])
    return trigger3DClusters


class PID:
    electron = 11
    photon = 22
    pizero = 111
    pion = 211
    kzero = 130


class Particle:
    def __init__(self,
                 name,
                 pdgid,
                 selection=None):
        self.name = name
        self.pdgid = pdgid
        self.sel = selection


class TPSet:
    def __init__(self,
                 name,
                 tc_sel=None,
                 cl2D_sel=None,
                 cl3D_sel=None,
                 particles=None):
        self.name = name
        # selection on the TPs
        self.tc_sel = tc_sel
        self.cl2D_sel = cl2D_sel
        self.cl3D_sel = cl3D_sel
        # particles for the matching
        self.particles = particles
        # histogram sets
        self.h_tpset = {}
        self.h_resoset = {}
        self.h_effset = {}
        self.rate_selections = {}
        self.h_rate = {}

    def book_histos(self):
        for particle in self.particles:
            histo_name = '{}_{}'.format(self.name, particle.name)
            # we book the histos for TCs, 2Dcl and 3Dcl
            self.h_tpset[particle.name] = histos.HistoSetClusters(histo_name)
            if particle.name != 'nomatch' and particle.pdgid != 0:
                # we also book the resolution plots
                self.h_resoset[particle.name] = histos.HistoSetReso(histo_name)
                self.h_effset[particle.name] = histos.HistoSetEff(histo_name)

    def book_rate_histos(self, selections):
        self.rate_selections = selections
        for name, selection in self.rate_selections.iteritems():
            self.h_rate[name] = histos.RateHistos(name='{}_{}'.format(self.name, name))

    def fill_histos(self, all_tcs, all_cl2Ds, all_cl3Ds, all_genParticles, debug):
        tcs = all_tcs
        cl2Ds = all_cl2Ds
        cl3Ds = all_cl3Ds
        if self.tc_sel:
            tcs = all_tcs.query(self.tc_sel)
        if self.cl2D_sel:
            cl2Ds = all_cl2Ds.query(self.cl2D_sel)
        if self.cl3D_sel:
            # print 'APPLY 3D cl SELECTION: {}'.format(self.cl3D_sel)
            cl3Ds = all_cl3Ds.query(self.cl3D_sel)
        debugPrintOut(debug, '{}_{}'.format(self.name, 'TCs'), tcs, tcs[:3])
        debugPrintOut(debug, '{}_{}'.format(self.name, 'CL2D'), cl2Ds, cl2Ds[:3])
        debugPrintOut(debug, '{}_{}'.format(self.name, 'CL3D'), cl3Ds, cl3Ds[:3])

        for particle in self.particles:
            if particle.pdgid == 0:
                # we just fill without any matching
                self.h_tpset[particle.name].fill(tcs, cl2Ds, cl3Ds)
            else:
                genReference = all_genParticles[(all_genParticles.gen > 0) & (np.abs(all_genParticles.pid) == particle.pdgid)]
                # FIXME: eta range selection was 1.7 2.8
                if particle.sel:
                    genReference = genReference.query(particle.sel)
                    # FIXME: this doesn't work for pizeros since they are never listed in the genParticles...we need a working solution
                    # elif  particle.pdgid == PID.pizero:
                    #     genReference = genParts[(genParts.pid == particle.pdgid)]
                hsetMatchAlgoPart = self.h_tpset[particle.name]
                hsetResoAlgoPart = self.h_resoset[particle.name]
                hgenseleff = self.h_effset[particle.name]
                plot3DClusterMatch(genReference,
                                   cl3Ds,
                                   cl2Ds,
                                   tcs,
                                   hgenseleff.h_den,
                                   hgenseleff.h_num,
                                   hsetMatchAlgoPart.htc,
                                   hsetMatchAlgoPart.hcl2d,
                                   hsetMatchAlgoPart.hcl3d,
                                   hsetResoAlgoPart.hreso,
                                   hsetResoAlgoPart.hresoCone,
                                   hsetResoAlgoPart.hreso2D,
                                   self.name,
                                   debug)

        for name, h_rate in self.h_rate.iteritems():
            selection = self.rate_selections[name]
            sel_clusters = cl3Ds.query(selection)
            # print '--- ALL: ------------------'
            # print cl3Ds
            # print '--- SEL {}: {} ------------'.format(name, selection)
            # print sel_clusters
            trigger_clusters = sel_clusters[['pt', 'eta']].sort_values(by='pt', ascending=False)
            # print '--- SORTED ----------------'
            # print trigger_clusters
            if not trigger_clusters.empty:
                h_rate.fill(trigger_clusters.iloc[0].pt, trigger_clusters.iloc[0].eta)
            h_rate.fill_norm()


def analyze(params, batch_idx=0):
    print (params)
    doAlternative = False

    debug = int(params.debug)
    computeDensity = params.computeDensity
    plot2DCLDR = False

    pool = Pool(5)

    tc_geom_df = pd.DataFrame()
    tc_rod_bins = pd.DataFrame()
    if True:
        # read the geometry dump
        geom_file = os.path.join(params.input_base_dir, 'geom/test_triggergeom.root')
        tc_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeTriggerCells')
        tc_geom_tree.setCache(learn_events=100)
        print ('read TC GEOM tree with # events: {}'.format(tc_geom_tree.nevents()))
        tc_geom_df = convertGeomTreeToDF(tc_geom_tree._tree)
        tc_geom_df['radius'] = np.sqrt(tc_geom_df['x']**2+tc_geom_df['y']**2)
        tc_geom_df['eta'] = np.arcsinh(tc_geom_df.z/tc_geom_df.radius)

        if False:
            tc_rod_bins = pd.read_csv(filepath_or_buffer='data/TCmapping_v2.txt',
                                      sep=' ',
                                      names=['id', 'rod_x', 'rod_y'],
                                      index_col=False)
            tc_rod_bins['rod_bin'] = tc_rod_bins.apply(func=lambda cell: (int(cell.rod_x), int(cell.rod_y)), axis=1)

            tc_geom_df = pd.merge(tc_geom_df, tc_rod_bins, on='id')

        # print (tc_geom_df[:3])
        # print (tc_geom_df[tc_geom_df.id == 1712072976])
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
        if debug == -4:
            tc_geom_tree.PrintCacheStats()
        print ('...done')

    tree_name = 'hgcalTriggerNtuplizer/HGCalTriggerNtuple'
    input_files = []
    range_ev = (0, params.maxEvents)

    if params.events_per_job == -1:
        print 'This is interactive processing...'
        input_files = fm.get_files_for_processing(input_dir=os.path.join(params.input_base_dir, params.input_sample_dir),
                                                  tree=tree_name,
                                                  nev_toprocess=params.maxEvents,
                                                  debug=debug)
    else:
        print 'This is batch processing...'
        input_files, range_ev = fm.get_files_and_events_for_batchprocessing(input_dir=os.path.join(params.input_base_dir, params.input_sample_dir),
                                                                            tree=tree_name,
                                                                            nev_toprocess=params.maxEvents,
                                                                            nev_perjob=params.events_per_job,
                                                                            batch_id=batch_idx,
                                                                            debug=debug)

    # print ('- dir {} contains {} files.'.format(params.input_sample_dir, len(input_files)))
    print '- will read {} files from dir {}:'.format(len(input_files), params.input_sample_dir)
    for file_name in input_files:
        print '        - {}'.format(file_name)

    ntuple = HGCalNtuple(input_files, tree=tree_name)
    if params.events_per_job == -1:
        if params.maxEvents == -1:
            range_ev = (0, ntuple.nevents())

    print ('- created TChain containing {} events'.format(ntuple.nevents()))
    print ('- reading from event: {} to event {}'.format(range_ev[0], range_ev[1]))

    ntuple.setCache(learn_events=1, entry_range=range_ev)
    output = ROOT.TFile(params.output_filename, "RECREATE")
    output.cd()

    if False:
        hTCGeom = histos.GeomHistos('hTCGeom')
        hTCGeom.fill(tc_geom_df[(np.abs(tc_geom_df.eta) > 1.65) & (np.abs(tc_geom_df.eta) < 2.85)])

# for index, tc_geom in tc_geom_df.iterrows():
#     tc_geom.max_dist_neigh = np.max(tc_geom.neighbor_distance)

    particles = [Particle('nomatch', 0),
                 Particle('ele', PID.electron, 'reachedEE == 2'),
                 Particle('elePt20', PID.electron, '(reachedEE == 2) & (pt > 20)'),
                 Particle('elePt30', PID.electron, '(reachedEE == 2) & (pt > 30)'),
                 Particle('elePt40', PID.electron, '(reachedEE == 2) & (pt > 40)'),
                 Particle('eleA', PID.electron, '(abseta <= 1.52) & (reachedEE == 2)'),
                 Particle('eleB', PID.electron, '(1.52 < abseta <= 1.7) & (reachedEE == 2)'),
                 Particle('eleC', PID.electron, '(1.7 < abseta <= 2.4) & (reachedEE == 2)'),
                 Particle('eleD', PID.electron, '(2.4 < abseta <= 2.8) & (reachedEE == 2)'),
                 Particle('eleE', PID.electron, '(abseta > 2.8) & (reachedEE == 2)'),
                 Particle('eleAB', PID.electron, '(abseta <= 1.7) & (reachedEE == 2)'),
                 Particle('eleABC', PID.electron, '(abseta <= 2.4) & (reachedEE == 2)'),
                 Particle('eleBC', PID.electron, '(1.52 < abseta <= 2.4) & (reachedEE == 2)'),
                 Particle('eleBCD', PID.electron, '(1.52 < abseta <= 2.8) & (reachedEE == 2)'),
                 Particle('eleBCDE', PID.electron, '(abseta > 1.52) & (reachedEE == 2)'),
                 Particle('photon', PID.photon, '(reachedEE == 2)'),
                 Particle('photonA', PID.photon, '(1.4 < abseta < 1.7) & (reachedEE == 2)'),
                 Particle('photonB', PID.photon, '(1.7 <= abseta <= 2.8) & (reachedEE == 2)'),
                 Particle('photonC', PID.photon, '(abseta > 2.8) & (reachedEE == 2)'),
                 Particle('photonD', PID.photon, '(abseta < 2.4) & (reachedEE == 2)'),
                 Particle('pion', PID.pion)]

    tp_sets = []
    tps_DEF = TPSet('DEF',
                    particles=particles)

    tps_DEFem = TPSet('DEF_em',
                      particles=particles,
                      cl3D_sel='quality > 0')

    tps_DEF_pt10 = TPSet('DEF_pt10',
                         particles=particles,
                         cl3D_sel='pt > 10')

    tps_DEF_pt10_em = TPSet('DEF_pt10_em',
                            particles=particles,
                            cl3D_sel='(quality > 0) & (pt > 10)')

    tps_DEF_pt20 = TPSet('DEF_pt20',
                         particles=particles,
                         cl3D_sel='pt > 20')

    tps_DEF_pt20_em = TPSet('DEF_pt20_em',
                            particles=particles,
                            cl3D_sel='(quality > 0) & (pt > 20)')


    tps_DEF_pt25 = TPSet('DEF_pt25',
                         particles=particles,
                         cl3D_sel='pt > 25')

    tps_DEF_pt25_em = TPSet('DEF_pt25_em',
                            particles=particles,
                            cl3D_sel='(quality > 0) & (pt > 25)')

    tps_DEF_pt30 = TPSet('DEF_pt30',
                         particles=particles,
                         cl3D_sel='pt > 30')

    tps_DEF_pt30_em = TPSet('DEF_pt30_em',
                            particles=particles,
                            cl3D_sel='(quality > 0) & (pt > 30)')

    tp_sets.append(tps_DEF)
    tp_sets.append(tps_DEFem)
    tp_sets.append(tps_DEF_pt10)
    tp_sets.append(tps_DEF_pt10_em)
    tp_sets.append(tps_DEF_pt20)
    tp_sets.append(tps_DEF_pt20_em)
    tp_sets.append(tps_DEF_pt25)
    tp_sets.append(tps_DEF_pt25_em)
    tp_sets.append(tps_DEF_pt30)
    tp_sets.append(tps_DEF_pt30_em)

    # -------------------------------------------------------
    # book histos
    hgen = histos.GenPartHistos('h_genAll')

    hGenParts = {}
    for particle in particles:
        hGenParts[particle] = histos.GenParticleHistos('h_genParts_{}'.format(particle.name))

    hGenPartsSel = {}
    for particle in particles:
        hGenPartsSel[particle] = histos.GenParticleHistos('h_genPartsSel_{}'.format(particle.name))

    # hdigis = histos.DigiHistos('h_hgcDigisAll')

    for tp_set in tp_sets:
        tp_set.book_histos()

    rate_selections = {'all': 'pt >= 0',
                       'etaA': 'abs(eta) <= 1.52',
                       'etaB': '(1.52 < abs(eta) <= 1.7)',
                       'etaC': '(1.7 < abs(eta) <= 2.4)',
                       'etaD': '(2.4 < abs(eta) <= 2.8)',
                       'etaE': '(abs(eta) > 2.8)',
                       'etaAB': 'abs(eta) <= 1.7',
                       'etaABC': 'abs(eta) <= 2.4',
                       'etaBC': '(1.52 < abs(eta) <= 2.4)',
                       'etaBCD': '(1.52 < abs(eta) <= 2.8)',
                       'etaBCDE': '(1.52 < abs(eta))'}

    tps_DEF.book_rate_histos(rate_selections)
    tps_DEFem.book_rate_histos(rate_selections)

    hTT_all = histos.TriggerTowerHistos('h_TT_all')
    # TT_algos = ['TTMATCH']
    # hsetTTMatched
    #
    hTT_matched = {}
    h_reso_TT = {}
    h_reso_TTCL = {}
    for particle in particles:
        hTT_matched[particle.name] = histos.TriggerTowerHistos('h_TT_{}'.format(particle.name))
        h_reso_TT[particle.name] = histos.TriggerTowerResoHistos('h_reso_TT_{}'.format(particle.name))
        h_reso_TTCL[particle.name] = histos.TriggerTowerResoHistos('h_reso_TTCl_{}'.format(particle.name))

    # htcMatchGEO = histos.TCHistos('h_tcMatchGEO')
    # h2dclMatchGEO = histos.ClusterHistos('h_clMatchGEO')
    # h3dclMatchGEO = histos.Cluster3DHistos('h_cl3dMatchGEO')

    hDensity_3p6 = histos.DensityHistos('h_dens3p6')
    hDensityClus_3p6 = histos.DensityHistos('h_densClus3p6')
    hDensity_2p5 = histos.DensityHistos('h_dens2p5')
    hDensityClus_2p5 = histos.DensityHistos('h_densClus2p5')

    hDR = ROOT.TH1F('hDR', 'DR 2D clusters', 100, 0, 1)
    dump = False
    # print (range_ev)

    # -------------------------------------------------------
    # event loop

    nev = 0
    for evt_idx in range(range_ev[0], range_ev[1]+1):
        # print(evt_idx)
        event = ntuple.getEvent(evt_idx)
        if (params.maxEvents != -1 and nev >= params.maxEvents):
            break
        if debug >= 2 or event.entry() % 100 == 0:
            print ("--- Event {}, @ {}".format(event.entry(), datetime.datetime.now()))
            print ('    run: {}, lumi: {}, event: {}'.format(event.run(), event.lumi(), event.event()))

        nev += 1
        if event.entry() in params.eventsToDump:
            dump = True
        else:
            dump = False

        # get the interesting data-frames
        genParts = event.getDataFrame(prefix='gen')

        # FIXME: we remove this preselection for now paying the price of reading all branches also
        # for non interesting events, is this a good idea?
        # if len(genParts[(genParts.eta > 1.7) & (genParts.eta < 2.5)]) == 0:
        #     continue

        branches = [(event, 'genpart'),
                    # (event, 'hgcdigi'),
                    (event, 'tc'),
                    (event, 'cl'),
                    (event, 'cl3d'),
                    (event, 'tower')]

        # dataframes = pool.map(unpack, branches)

        dataframes = []
        for idx, branch in enumerate(branches):
            dataframes.append(unpack(branch))

        genParticles = dataframes[0]
        # hgcDigis = dataframes[1]
        triggerCells = dataframes[1]
        triggerClusters = dataframes[2]
        trigger3DClusters = dataframes[3]
        triggerTowers = dataframes[4]

        puInfo = event.getPUInfo()
        debugPrintOut(debug, 'PU', toCount=puInfo, toPrint=puInfo)

        # ----------------------------------
        if not tc_rod_bins.empty:
            triggerCells = pd.merge(triggerCells,
                                    tc_rod_bins,
                                    on='id')

        genParticles['pdgid'] = genParticles.pid
        genParticles['abseta'] = np.abs(genParticles.eta)

        # this is not needed anymore in recent versions of the ntuples
        # tcsWithPos = pd.merge(triggerCells, tc_geom_df[['id', 'x', 'y']], on='id')
        triggerClusters['ncells'] = [len(x) for x in triggerClusters.cells]
        # if 'x' not in triggerClusters.columns:
        #     triggerClusters = pd.merge(triggerClusters, tc_geom_df[['z', 'id']], on='id')
        #     triggerClusters['R'] = triggerClusters.z/np.sinh(triggerClusters.eta)
        #     triggerClusters['x'] = triggerClusters.R*np.cos(triggerClusters.phi)
        #     triggerClusters['y'] = triggerClusters.R*np.sin(triggerClusters.phi)

        trigger3DClusters['nclu'] = [len(x) for x in trigger3DClusters.clusters]
        trigger3DClustersP = pd.DataFrame()
        triggerClustersGEO = pd.DataFrame()
        trigger3DClustersGEO = pd.DataFrame()
        triggerClustersDBS = pd.DataFrame()
        trigger3DClustersDBS = pd.DataFrame()
        trigger3DClustersDBSp = pd.DataFrame()

        triggerTowers.eval('HoE = etHad/etEm', inplace=True)
        # triggerTowers['HoE'] = triggerTowers.etHad/triggerTowers.etEm
        # if 'iX' not in triggerTowers.columns:
        #     triggerTowers['iX'] = triggerTowers.hwEta
        #     triggerTowers['iY'] = triggerTowers.hwPhi

        if not tc_rod_bins.empty:
            clAlgo.computeClusterRodSharing(triggerClusters, triggerCells)

        debugPrintOut(debug, 'gen parts', toCount=genParts, toPrint=genParts)
        debugPrintOut(debug, 'gen particles',
                      toCount=genParticles,
                      toPrint=genParticles[['eta', 'phi', 'pt', 'energy', 'mother', 'fbrem', 'pid', 'gen', 'reachedEE', 'fromBeamPipe']])
        # print genParticles.columns
        # debugPrintOut(debug, 'digis',
        #               toCount=hgcDigis,
        #               toPrint=hgcDigis.iloc[:3])
        debugPrintOut(debug, 'Trigger Cells',
                      toCount=triggerCells,
                      toPrint=triggerCells.iloc[:3])
        debugPrintOut(debug, '2D clusters',
                      toCount=triggerClusters,
                      toPrint=triggerClusters.iloc[:3])
        debugPrintOut(debug, '3D clusters',
                      toCount=trigger3DClusters,
                      toPrint=trigger3DClusters.iloc[:3])
        debugPrintOut(debug, 'Trigger Towers',
                      toCount=triggerTowers,
                      toPrint=triggerTowers.sort_values(by='pt', ascending=False).iloc[:10])
        # print '# towers eta >0 {}'.format(len(triggerTowers[triggerTowers.eta > 0]))
        # print '# towers eta <0 {}'.format(len(triggerTowers[triggerTowers.eta < 0]))

        if params.clusterize:
            # Now build DBSCAN 2D clusters
            for zside in [-1, 1]:
                arg = [(layer, zside, triggerCells) for layer in range(0, 53)]
                results = pool.map(clAlgo.buildDBSCANClustersUnpack, arg)
                for clres in results:
                    triggerClustersDBS = triggerClustersDBS.append(clres, ignore_index=True)

            if not tc_rod_bins.empty:
                clAlgo.computeClusterRodSharing(triggerClustersDBS, triggerCells)

            debugPrintOut(debug, 'DBS 2D clusters',
                          toCount=triggerClustersDBS,
                          toPrint=triggerClustersDBS.iloc[:3])

            trigger3DClustersDBS = build3DClusters('DBS', clAlgo.build3DClustersEtaPhi, triggerClustersDBS, pool, debug)
            trigger3DClustersDBSp = build3DClusters('DBSp', clAlgo.build3DClustersProjTowers, triggerClustersDBS, pool, debug)
            trigger3DClustersP = build3DClusters('DEFp', clAlgo.build3DClustersProjTowers, triggerClusters, pool, debug)
        # if doAlternative:
        #     triggerClustersGEO = event.getDataFrame(prefix='clGEO')
        #     trigger3DClustersGEO = event.getDataFrame(prefix='cl3dGEO')
        #     debugPrintOut(debug, 'GEO 2D clusters',
        #                   toCount=triggerClustersGEO,
        #                   toPrint=triggerClustersGEO.loc[:3])
        #     debugPrintOut(debug, 'GEO 3D clusters',
        #                   toCount=trigger3DClustersGEO,
        #                   toPrint=trigger3DClustersGEO.loc[:3])
        #     print(triggerCells[triggerCells.index.isin(np.concatenate(triggerClusters.cells.iloc[:3]))])

        # fill histograms
        hgen.fill(genParts)

        # we find the genparticles matched to the GEN info
        for particle in particles:
            # if particle.pdgid != PID.pizero:
            # FIXME: this doesn't work for pizeros since they are never listed in the genParticles...we need a working solution
            hGenParts[particle].fill(genParticles[(genParticles.gen > 0) & (np.abs(genParticles.pid) == particle.pdgid)])
            # else:
            #     hGenParts[particle].fill(genParts[(genParts.pid == particle.pdgid)])

        # hdigis.fill(hgcDigis)

        tps_DEF.fill_histos(triggerCells, triggerClusters, trigger3DClusters, genParticles, debug)
        tps_DEFem.fill_histos(triggerCells, triggerClusters, trigger3DClusters, genParticles, debug)
        tps_DEF_pt10.fill_histos(triggerCells, triggerClusters, trigger3DClusters, genParticles, debug)
        tps_DEF_pt10_em.fill_histos(triggerCells, triggerClusters, trigger3DClusters, genParticles, debug)
        tps_DEF_pt20.fill_histos(triggerCells, triggerClusters, trigger3DClusters, genParticles, debug)
        tps_DEF_pt20_em.fill_histos(triggerCells, triggerClusters, trigger3DClusters, genParticles, debug)
        tps_DEF_pt25.fill_histos(triggerCells, triggerClusters, trigger3DClusters, genParticles, debug)
        tps_DEF_pt25_em.fill_histos(triggerCells, triggerClusters, trigger3DClusters, genParticles, debug)
        tps_DEF_pt30.fill_histos(triggerCells, triggerClusters, trigger3DClusters, genParticles, debug)
        tps_DEF_pt30_em.fill_histos(triggerCells, triggerClusters, trigger3DClusters, genParticles, debug)

        hTT_all.fill(triggerTowers)

        if True:
            # now we try to match the Clusters to the GEN particles of various types
            for particle in particles:
                genReference = genParticles[(genParticles.gen > 0) & (np.abs(genParticles.pid) == particle.pdgid) & (np.abs(genParticles.eta) < 2.8) & (np.abs(genParticles.eta) > 1.7)]
                # for the photons we add a further selection
                if particle.pdgid == PID.photon:
                    genReference = genParticles[(genParticles.gen > 0) & (genParticles.pid == PID.photon) & (genParticles.reachedEE == 2) & (np.abs(genParticles.eta) < 2.8) & (np.abs(genParticles.eta) > 1.7)]
                # FIXME: this doesn't work for pizeros since they are never listed in the genParticles...we need a working solution
                # elif  particle.pdgid == PID.pizero:
                #     genReference = genParts[(genParts.pid == particle.pdgid)]
                plotTriggerTowerMatch(genReference,
                                      None,
                                      triggerTowers,
                                      hTT_matched[particle.name],
                                      h_reso_TT[particle.name],
                                      h_reso_TTCL[particle.name],
                                      "TThighestPt",
                                      debug)

        # dump the data-frames to JSON if needed
        if dump:
            js_filename = 'tc_dump_ev_{}.json'.format(event.entry())
            dumpFrame2JSON(js_filename, triggerCells)
            js_2dc_filename = '2dc_dump_ev_{}.json'.format(event.entry())
            dumpFrame2JSON(js_2dc_filename, triggerClusters)

        if computeDensity:
            # def computeDensity(tcs):
            #     eps = 3.5
            #     for idx, tc in tcsWithPos_ee_layer.iterrows():
            #         energy_list = list()
            #         tcsinradius = tcs[((tcs.x-tc.x)**2+(tcs.y-tc.y)**2) < eps**2]
            #         totE = np.sum(tcsinradius.energy)
            #
            tcsWithPos_ee = triggerCells[triggerCells.subdet == 3]
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

                # if plot2DCLDR:
                #     for idx, cl in triggerClustersDBS[triggerClustersDBS.zside == zside].iterrows():
                #         for idx2 in range(idx+1, triggerClustersDBS[triggerClustersDBS.zside == zside].shape[0]):
                #             hDR.Fill(math.sqrt((cl.eta-triggerClustersDBS[triggerClustersDBS.zside == zside].loc[idx2].eta)**2+(cl.phi-triggerClustersDBS[triggerClustersDBS.zside == zside].loc[idx2].phi)**2))
            # for layer in range(0, 29):
            #     triggerClustersDBS = triggerClustersDBS.append(clAlgo.buildDBSCANClusters(layer, zside, tcsWithPos), ignore_index=True)

    print ("Processed {} events/{} TOT events".format(nev, ntuple.nevents()))
    print ("Writing histos to file {}".format(params.output_filename))

    lastfile = ntuple.tree().GetFile()
    print 'Read bytes: {}, # of transaction: {}'.format(lastfile.GetBytesRead(),  lastfile.GetReadCalls())
    if debug == -4:
        ntuple.PrintCacheStats()

    output.cd()
    hm = histos.HistoManager()
    hm.writeHistos()

    hDR.Write()
    output.Close()

    return


def editTemplate(infile, outfile, params):
    template_file = open(infile)
    template = template_file.read()
    template_file.close()

    for param in params.keys():
        template = template.replace(param, params[param])

    out_file = open(outfile, 'w')
    out_file.write(template)
    out_file.close()


def main(analyze):
    # ============================================
    # configuration bit

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)
    parser.add_option('-f', '--file', dest='CONFIGFILE', help='specify the ini configuration file')
    parser.add_option('-c', '--collection', dest='COLLECTION', help='specify the collection to be processed')
    parser.add_option('-s', '--sample', dest='SAMPLE', help='specify the sample (within the collection) to be processed ("all" to run the full collection)')
    parser.add_option('-d', '--debug', dest='DEBUG', help='debug level (default is 0)', default=0)
    parser.add_option('-n', '--nevents', dest='NEVENTS', help='# of events to process per sample (default is 10)', default=10)
    parser.add_option("-b", "--batch", action="store_true", dest="BATCH", default=False, help="submit the jobs via CONDOR")
    parser.add_option("-r", "--run", dest="RUN", default=None, help="the batch_id to run (need to be used with the option -b)")
    parser.add_option("-o", "--outdir", dest="OUTDIR", default=None, help="override the output directory for the files")
    # parser.add_option("-i", "--inputJson", dest="INPUT", default='input.json', help="list of input files and properties in JSON format")

    global opt, args
    (opt, args) = parser.parse_args()

    # read the config file
    cfgfile = ConfigParser.ConfigParser()
    cfgfile.optionxform = str
    cfgfile.read(opt.CONFIGFILE)

    collection_dict = {}
    collections = [coll.strip() for coll in cfgfile.get('common', 'collections').split(',')]
    basedir = cfgfile.get('common', 'input_dir_lx')
    outdir = cfgfile.get('common', 'output_dir_lx')
    hostname = socket.gethostname()
    if 'matterhorn' in hostname or 'Matterhorn' in hostname:
            basedir = cfgfile.get('common', 'input_dir_local')
            outdir = cfgfile.get('common', 'output_dir_local')
    plot_version = cfgfile.get('common', 'plot_version')
    run_clustering = False
    if cfgfile.get('common', 'run_clustering') == 'True':
        run_clustering = True
    run_density_computation = False
    if cfgfile.get('common', 'run_density_computation') == 'True':
        run_density_computation = True

    events_to_dump = []
    if cfgfile.has_option('common', "events_to_dump"):
        events_to_dump = [int(num) for num in cfgfile.get('common', 'events_to_dump').split(',')]

    for collection in collections:
        samples = cfgfile.get(collection, 'samples').split(',')
        print ('--- Collection: {} with samples: {}'.format(collection, samples))
        sample_list = list()
        for sample in samples:
            events_per_job = -1
            out_file_name = 'histos_{}_{}.root'.format(sample, plot_version)
            if opt.BATCH:
                events_per_job = int(cfgfile.get(sample, 'events_per_job'))
                if opt.RUN:
                    out_file_name = 'histos_{}_{}_{}.root'.format(sample, plot_version, opt.RUN)

            if opt.OUTDIR:
                outdir = opt.OUTDIR

            out_file = os.path.join(outdir, out_file_name)

            params = Parameters(input_base_dir=basedir,
                                input_sample_dir=cfgfile.get(sample, 'input_sample_dir'),
                                output_filename=out_file,
                                output_dir=outdir,
                                clusterize=run_clustering,
                                eventsToDump=events_to_dump,
                                version=plot_version,
                                maxEvents=int(opt.NEVENTS),
                                events_per_job=events_per_job,
                                debug=opt.DEBUG,
                                computeDensity=run_density_computation,
                                name=sample)
            sample_list.append(params)
        collection_dict[collection] = sample_list

    samples_to_process = list()
    if opt.COLLECTION:
        if opt.COLLECTION in collection_dict.keys():
            if opt.SAMPLE:
                if opt.SAMPLE == 'all':
                    samples_to_process.extend(collection_dict[opt.COLLECTION])
                else:
                    sel_sample = [sample for sample in collection_dict[opt.COLLECTION] if sample.name == opt.SAMPLE]
                    samples_to_process.append(sel_sample[0])
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

    if opt.BATCH and not opt.RUN:
        batch_dir = 'batch_{}_{}'.format(opt.COLLECTION, plot_version)
        if not os.path.exists(batch_dir):
            os.mkdir(batch_dir)
            os.mkdir(batch_dir+'/conf/')
            os.mkdir(batch_dir+'/logs/')

        dagman_sub = ''
        dagman_dep = ''
        dagman_ret = ''
        for sample in samples_to_process:
            dagman_spl = ''
            dagman_spl_retry = ''
            sample_batch_dir = os.path.join(batch_dir, sample.name)
            sample_batch_dir_logs = os.path.join(sample_batch_dir, 'logs')
            os.mkdir(sample_batch_dir)
            os.mkdir(sample_batch_dir_logs)
            print(sample)
            nevents = int(opt.NEVENTS)
            n_jobs = fm.get_number_of_jobs_for_batchprocessing(input_dir=os.path.join(sample.input_base_dir, sample.input_sample_dir),
                                                               tree='hgcalTriggerNtuplizer/HGCalTriggerNtuple',
                                                               nev_toprocess=nevents,
                                                               nev_perjob=sample.events_per_job,
                                                               debug=int(opt.DEBUG))
            print ('Total # of events to be processed: {}'.format(nevents))
            print ('# of events per job: {}'.format(sample.events_per_job))
            if n_jobs == 0:
                n_jobs = 1
            print ('# of jobs to be submitted: {}'.format(n_jobs))

            params = {}
            params['TEMPL_TASKDIR'] = sample_batch_dir
            params['TEMPL_NJOBS'] = str(n_jobs)
            params['TEMPL_WORKDIR'] = os.environ["PWD"]
            params['TEMPL_CFG'] = opt.CONFIGFILE
            params['TEMPL_COLL'] = opt.COLLECTION
            params['TEMPL_SAMPLE'] = sample.name
            params['TEMPL_OUTFILE'] = 'histos_{}_{}.root'.format(sample.name, sample.version)
            unmerged_files = [os.path.join(sample.output_dir, 'histos_{}_{}_{}.root'.format(sample.name, sample.version, job)) for job in range(0, n_jobs)]
            # protocol = ''
            # if '/eos/user/' in sample.output_dir:
            #     protocol = 'root://eosuser.cern.ch/'
            # elif '/eos/cms/' in sample.output_dir:
            #     protocol = 'root://eoscms.cern.ch/'
            params['TEMPL_INFILES'] = ' '.join(unmerged_files)
            params['TEMPL_OUTDIR'] = sample.output_dir
            params['TEMPL_VIRTUALENV'] = os.path.basename(os.environ['VIRTUAL_ENV'])

            editTemplate(infile='templates/batch.sub',
                         outfile=os.path.join(sample_batch_dir, 'batch.sub'),
                         params=params)

            editTemplate(infile='templates/run_batch.sh',
                         outfile=os.path.join(sample_batch_dir, 'run_batch.sh'),
                         params=params)

            editTemplate(infile='templates/copy_files.sh',
                         outfile=os.path.join(sample_batch_dir, 'copy_files.sh'),
                         params=params)
            os.chmod(os.path.join(sample_batch_dir, 'copy_files.sh'),  0754)

            editTemplate(infile='templates/batch_hadd.sub',
                         outfile=os.path.join(sample_batch_dir, 'batch_hadd.sub'),
                         params=params)

            editTemplate(infile='templates/run_batch_hadd.sh',
                         outfile=os.path.join(sample_batch_dir, 'run_batch_hadd.sh'),
                         params=params)

            editTemplate(infile='templates/batch_cleanup.sub',
                         outfile=os.path.join(sample_batch_dir, 'batch_cleanup.sub'),
                         params=params)

            editTemplate(infile='templates/run_batch_cleanup.sh',
                         outfile=os.path.join(sample_batch_dir, 'run_batch_cleanup.sh'),
                         params=params)

            for jid in range(0, n_jobs):
                dagman_spl += 'JOB Job_{} batch.sub\n'.format(jid)
                dagman_spl += 'VARS Job_{} JOB_ID="{}"\n'.format(jid, jid)
                dagman_spl_retry += 'Retry Job_{} 3\n'.format(jid)

            dagman_sub += 'SPLICE {} {}.spl DIR {}\n'.format(sample.name, sample.name, sample_batch_dir)
            dagman_sub += 'JOB {} {}/batch_hadd.sub\n'.format(sample.name+'_hadd', sample_batch_dir)
            dagman_sub += 'JOB {} {}/batch_cleanup.sub\n'.format(sample.name+'_cleanup', sample_batch_dir)

            dagman_dep += 'PARENT {} CHILD {}\n'.format(sample.name, sample.name+'_hadd')
            dagman_dep += 'PARENT {} CHILD {}\n'.format(sample.name+'_hadd', sample.name+'_cleanup')

            # dagman_ret += 'Retry {} 3\n'.format(sample.name)
            dagman_ret += 'Retry {} 3\n'.format(sample.name+'_hadd')

            dagman_splice = open(os.path.join(sample_batch_dir, '{}.spl'.format(sample.name)), 'w')
            dagman_splice.write(dagman_spl)
            dagman_splice.write(dagman_spl_retry)
            dagman_splice.close()

            # copy the config file in the batch directory
            copyfile(opt.CONFIGFILE, os.path.join(sample_batch_dir, opt.CONFIGFILE))

        dagman_file_name = os.path.join(batch_dir, 'dagman.dag')
        dagman_file = open(dagman_file_name, 'w')
        dagman_file.write(dagman_sub)
        dagman_file.write(dagman_dep)
        dagman_file.write(dagman_ret)
        dagman_file.close()

        #cp TEMPL_TASKDIR/TEMPL_CFG
        print('Ready for submission please run the following commands:')
        # print('condor_submit {}'.format(condor_file_path))
        print('condor_submit_dag {}'.format(dagman_file_name))
        sys.exit(0)

    batch_idx = 0
    if opt.BATCH and opt.RUN:
        batch_idx = int(opt.RUN)

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

    # samples = test_sample
    for sample in samples_to_process:
        analyze(sample, batch_idx=batch_idx)


if __name__ == "__main__":
    try:
        main(analyze=analyze)
    except Exception as inst:
        print (str(inst))
        print ("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc()
        sys.exit(100)
