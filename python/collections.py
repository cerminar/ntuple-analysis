"""
Manger of the ntuple data collections.

This module manages the code and the instances of objects actually
accessing the ntuple data.

Classes:
    EventManager
    DFCollection
    TPSet

Objects:
    all collections (DFCollection and TPSet instances) that
    can be used by plotters.
"""

import pandas as pd
import numpy as np
import ROOT
import math

import root_numpy.tmva as rnptmva

from utils import debugPrintOut, match_etaphi
import python.clusterTools as clAlgo
from python.mp_pool import POOL
import python.classifiers as classifiers
import python.calibrations as calib


class EventManager(object):
    """
    EventManager.

    Manages the registration of collections and reads them when needed,
    i.e. only if they have been activated (typically by a plotter).

    It is a singleton.
    """

    class __TheManager:
        def __init__(self):
            self.collections = []
            self.active_collections = []

        def registerCollection(self, collection):
            print '[EventManager] registering collection: {}'.format(collection.name)
            self.collections.append(collection)

        def registerActiveCollection(self, collection):
            print '[EventManager] registering collection as active: {}'.format(collection.name)
            self.active_collections.append(collection)

        def read(self, event, debug):
            for collection in self.active_collections:
                if debug >= 3:
                    print '[EventManager] filling collection: {}'.format(collection.name)
                collection.fill(event, debug)

    instance = None

    def __new__(cls):
        if not EventManager.instance:
            EventManager.instance = EventManager.__TheManager()
        return EventManager.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


class DFCollection(object):
    """
    [DFCollection]: collection of objects consumed by plotters.
    This class represents the DataFrame of the objects which need to be plotted.
    The objects are registered with the EventManager at creation time but they
    are actually created/read event by event only if one plotter object activates
    them (or anotrher DFCollection depending on them) at booking time.
    As a result you can instantiate as many objects as needed and only those
    actually consumed by a plotter will be read.

    Args:
        name (string): name which enters the histo name
        label (string): drawn on plots or legends
        filler_function (callable): function accepting event
                                    as parameter and returning the DataFrame filled
        fixture_function (callable): adds columns to the data-frame
        depends_on (list): specify the dependencies (typically called as argument by the filler_function at filling time)
        debug (int): specify the debug level object by object (also the global one is used)
    """

    def __init__(self, name, label, filler_function, fixture_function=None, depends_on=[], debug=0):
        self.df = None
        self.name = name
        self.label = label
        self.is_active = False
        self.filler_function = filler_function
        self.fixture_function = fixture_function
        self.depends_on = depends_on
        self.debug = debug
        self.register()

    def register(self):
        event_manager = EventManager()
        event_manager.registerCollection(self)

    def activate(self):
        if not self.is_active:
            for dep in self.depends_on:
                dep.activate()
            self.is_active = True
            event_manager = EventManager()
            event_manager.registerActiveCollection(self)
        return self.is_active

    def fill(self, event, debug):
        self.df = self.filler_function(event)
        if self.fixture_function is not None:
            self.fixture_function(self.df)
        if not self.df.empty:
            debugPrintOut(max(debug, self.debug), self.label,
                          toCount=self.df,
                          toPrint=self.df)


def cl3d_fixtures(clusters):
    clusters['nclu'] = [len(x) for x in clusters.clusters]
    clusters['ptem'] = clusters.pt/(1+clusters.hoe)
    clusters['eem'] = clusters.energy/(1+clusters.hoe)
    if False:
        clusters['bdt_pu'] = rnptmva.evaluate_reader(
            classifiers.mva_pu_classifier, 'BDT', clusters[['pt', 'eta', 'maxlayer', 'hoe', 'emaxe', 'szz']])

        clusters['bdt_pi'] = rnptmva.evaluate_reader(
            classifiers.mva_pi_classifier, 'BDT', clusters[['pt', 'eta', 'maxlayer', 'hoe', 'emaxe', 'szz']])


def gen_fixtures(particles, mc_particles):
    particles['pdgid'] = particles.pid
    particles['abseta'] = np.abs(particles.eta)

    def get_mother_pdgid(particle, mc_particles):
        if particle.gen == -1:
            return -1
        return mc_particles.df.loc[particle.gen-1].firstmother_pdgid
    particles['firstmother_pdgid'] = particles.apply(func=lambda x: get_mother_pdgid(x, mc_particles), axis=1)


def mc_fixtures(particles):
    particles['firstmother'] = particles.index
    particles['firstmother_pdgid'] = particles.pdgid

    for particle in particles.itertuples():
        # print particle.Index
        particles.loc[particle.daughters, 'firstmother'] = particle.Index
        particles.loc[particle.daughters, 'firstmother_pdgid'] = particle.pdgid
        # print particles.loc[particle.daughters]['firstmother']


def tc_fixtures(tcs):
    tcs['ncells'] = 1
    if not tcs.empty:
        tcs['cells'] = tcs.apply(func=lambda x: [int(x.id)], axis=1)


def cl2d_fixtures(clusters):
    clusters['ncells'] = [len(x) for x in clusters.cells]


def tower_fixtures(towers):
    if towers.empty:
        # print '***[compute_tower_data]:WARNING input data-frame is empty'
        return towers

    towers.eval('HoE = etHad/etEm', inplace=True)

    def fill_momentum(tower):
        vector = ROOT.TLorentzVector()
        vector.SetPtEtaPhiE(tower.pt, tower.eta, tower.phi, tower.energy)
        tower.momentum = vector
        # print tower.pt, tower.momentum.Pt()
        return tower

    towers['momentum'] = ROOT.TLorentzVector()
    towers = towers.apply(fill_momentum, axis=1)


def get_merged_cl3d(triggerClusters, pool, debug=0):
    merged_clusters = pd.DataFrame(columns=triggerClusters.columns)
    if triggerClusters.empty:
        return merged_clusters
    # FIXME: filter only interesting clusters
    clusterSides = [x for x in [triggerClusters[triggerClusters.eta > 0],
                                triggerClusters[triggerClusters.eta < 0]] if not x.empty]

    results3Dcl = pool.map(clAlgo.merge3DClustersEtaPhi, clusterSides)
    for res3D in results3Dcl:
        merged_clusters = merged_clusters.append(res3D, ignore_index=True, sort=False)
    return merged_clusters


def get_trackmatched_egs(egs, tracks, debug=0):
    newcolumns = ['pt', 'energy', 'eta', 'phi', 'hwQual']
    newcolumns.extend(['tkpt', 'tketa', 'tkphi', 'tkz0', 'tkchi2', 'tkchi2Red', 'tknstubs', 'deta', 'dphi', 'dr'])
    matched_egs = pd.DataFrame(columns=newcolumns)
    if egs.empty or tracks.empty:
        return matched_egs
    best_match_indexes, allmatches = match_etaphi(egs[['eta', 'phi']],
                                                  tracks[['caloeta', 'calophi']],
                                                  tracks['pt'],
                                                  deltaR=0.1)
    for bestmatch_idxes in best_match_indexes.iteritems():
        bestmatch_eg = egs.loc[bestmatch_idxes[0]]
        bestmatch_tk = tracks.loc[bestmatch_idxes[1]]
        matched_egs = matched_egs.append({'pt': bestmatch_eg.pt,
                                          'energy': bestmatch_eg.energy,
                                          'eta': bestmatch_eg.eta,
                                          'phi': bestmatch_eg.phi,
                                          'hwQual': bestmatch_eg.hwQual,
                                          'tkpt': bestmatch_tk.pt,
                                          'tketa': bestmatch_tk.eta,
                                          'tkphi': bestmatch_tk.phi,
                                          'tkz0': bestmatch_tk.z0,
                                          'tkchi2': bestmatch_tk.chi2,
                                          'tkchi2Red': bestmatch_tk.chi2Red,
                                          'tknstubs': bestmatch_tk.nStubs,
                                          'deta': bestmatch_tk.eta - bestmatch_eg.eta,
                                          'dphi': bestmatch_tk.phi - bestmatch_eg.phi,
                                          'dr': math.sqrt((bestmatch_tk.phi-bestmatch_eg.phi)**2+(bestmatch_tk.eta-bestmatch_eg.eta)**2)},
                                         ignore_index=True, sort=False)
    return matched_egs


def get_calibrated_clusters(calib_factors, input_3Dclusters):
    calibrated_clusters = input_3Dclusters.copy(deep=True)

    def apply_calibration(cluster):
        calib_factor = 1
        calib_factor_tmp = calib_factors[(calib_factors.eta_l < abs(cluster.eta)) &
                                         (calib_factors.eta_h >= abs(cluster.eta)) &
                                         (calib_factors.pt_l <= cluster.pt) &
                                         (calib_factors.pt_h > cluster.pt)]
        if not calib_factor_tmp.empty:
            # print 'cluster pt: {}, eta: {}, calib_factor: {}'.format(cluster.pt, cluster.eta, calib_factor_tmp.calib.values[0])
            # print calib_factor_tmp
            calib_factor = 1./calib_factor_tmp.calib.values[0]
        # print cluster
        # else:
            # if cluster.eta <= 2.8 and cluster.eta > 1.52 and cluster.pt > 4 and cluster.pt <= 100:
            #     print cluster[['pt', 'eta']]
        cluster.pt = cluster.pt*calib_factor
        # cluster['pt1'] = cluster.pt*calib_factor
        # cluster['cf'] = 1./calib_factor
        return cluster
        # input_3Dclusters[(input_3Dclusters.eta_l > abs(cluster.eta)) & ()]
    calibrated_clusters = calibrated_clusters.apply(apply_calibration, axis=1)
    return calibrated_clusters


def build3DClusters(name, algorithm, triggerClusters, pool, debug):
    trigger3DClusters = pd.DataFrame()
    if triggerClusters.empty:
        return trigger3DClusters
    clusterSides = [x for x in [triggerClusters[triggerClusters.eta > 0],
                                triggerClusters[triggerClusters.eta < 0]] if not x.empty]
    results3Dcl = pool.map(algorithm, clusterSides)
    for res3D in results3Dcl:
        trigger3DClusters = trigger3DClusters.append(res3D, ignore_index=True, sort=False)

    debugPrintOut(debug, name='{} 3D clusters'.format(name),
                  toCount=trigger3DClusters,
                  toPrint=trigger3DClusters.iloc[:3])
    return trigger3DClusters
#     trigger3DClustersDBS = build3DClusters(
#         'DBS', clAlgo.build3DClustersEtaPhi, triggerClustersDBS, pool, debug)
#     trigger3DClustersDBSp = build3DClusters(
#         'DBSp', clAlgo.build3DClustersProjTowers, triggerClustersDBS, pool, debug)
#     trigger3DClustersP = build3DClusters(
#         'DEFp', clAlgo.build3DClustersProjTowers, triggerClusters, pool, debug)


def get_calibrated_clusters2(calib_factors, input_3Dclusters):
    calibrated_clusters = input_3Dclusters.copy(deep=True)

    def apply_calibration(cluster):
        calib_factor = 1.
        calib_factor_tmp = calib_factors[(calib_factors.eta_l < abs(cluster.eta)) &
                                         (calib_factors.eta_h >= abs(cluster.eta)) &
                                         (calib_factors.pt_l < cluster.pt) &
                                         (calib_factors.pt_h >= cluster.pt)]
        if not calib_factor_tmp.empty:
            # print 'cluster pt: {}, eta: {}, calib_factor: {}'.format(cluster.pt, cluster.eta, calib_factor_tmp.calib.values[0])
            # print calib_factor_tmp
            calib_factor = 1./calib_factor_tmp.calib.values[0]
        # print cluster
        else:
            if cluster.eta <= 2.8 and cluster.eta > 1.52 and cluster.pt > 4 and cluster.pt <= 100:
                print cluster[['pt', 'eta']]

        cluster['pt2'] = cluster.pt*calib_factor
        return cluster
        # input_3Dclusters[(input_3Dclusters.eta_l > abs(cluster.eta)) & ()]
    calibrated_clusters = calibrated_clusters.apply(apply_calibration, axis=1)
    return calibrated_clusters


gen = DFCollection(name='MC', label='MC particles',
                         filler_function=lambda event: event.getDataFrame(prefix='gen'),
                         fixture_function=mc_fixtures, debug=0)


gen_parts = DFCollection(name='GEN', label='GEN particles',
                         filler_function=lambda event: event.getDataFrame(prefix='genpart'),
                         fixture_function=lambda gen_parts: gen_fixtures(gen_parts, gen),
                         depends_on=[gen], debug=0)

tcs = DFCollection(name='TC', label='Trigger Cells',
                   filler_function=lambda event: event.getDataFrame(prefix='tc'),
                   fixture_function=tc_fixtures)

cl2d_def = DFCollection(name='DEF2D', label='dRC2d',
                        filler_function=lambda event: event.getDataFrame(prefix='cl'),
                        fixture_function=cl2d_fixtures)

cl3d_def = DFCollection(name='DEF', label='dRC3d',
                        filler_function=lambda event: event.getDataFrame(prefix='cl3d'),
                        fixture_function=cl3d_fixtures)

cl3d_def_nc = DFCollection(name='DEFNC', label='dRC3d NewTh',
                           filler_function=lambda event: event.getDataFrame(prefix='cl3dNC'),
                           fixture_function=cl3d_fixtures)

cl3d_hm = DFCollection(name='HMvDR', label='HM+dR(layer) Cl3d',
                       filler_function=lambda event: event.getDataFrame(prefix='hmVRcl3d'),
                       fixture_function=cl3d_fixtures)

cl3d_hm_nc0 = DFCollection(name='HMvDRNC0', label='HM+dR(layer) Cl3d + NewTh0',
                           filler_function=lambda event: event.getDataFrame(prefix='hmVRcl3dNC0'),
                           fixture_function=cl3d_fixtures)

cl3d_hm_nc1 = DFCollection(name='HMvDRNC1', label='HM+dR(layer) Cl3d + NewTh1',
                           filler_function=lambda event: event.getDataFrame(prefix='hmVRcl3dNC1'),
                           fixture_function=cl3d_fixtures)

cl3d_def_merged = DFCollection(name='DEFMerged', label='dRC3d merged',
                               filler_function=lambda event: get_merged_cl3d(cl3d_def.df[cl3d_def.df.quality > 0], POOL),
                               depends_on=[cl3d_def])

cl3d_def_calib = DFCollection(name='DEFCalib', label='dRC3d calib.',
                              filler_function=lambda event: get_calibrated_clusters(calib.calib_factors, cl3d_def.df),
                              depends_on=[cl3d_def])

cl3d_hm_merged = DFCollection(name='HMvDRMerged', label='HM+dR(layer) merged',
                              filler_function=lambda event: get_merged_cl3d(cl3d_hm.df[cl3d_hm.df.quality > 0], POOL),
                              depends_on=[cl3d_hm])

towers_tcs = DFCollection(name='TT', label='TT (TC)',
                          filler_function=lambda event: event.getDataFrame(prefix='tower'),
                          fixture_function=tower_fixtures)

towers_sim = DFCollection(name='SimTT', label='TT (sim)',
                          filler_function=lambda event: event.getDataFrame(prefix='simTower'),
                          fixture_function=tower_fixtures)

towers_hgcroc = DFCollection(name='HgcrocTT', label='TT (HGCROC)',
                             filler_function=lambda event: event.getDataFrame(prefix='hgcrocTower'),
                             fixture_function=tower_fixtures)

towers_wafer = DFCollection(name='WaferTT', label='TT (Wafer)',
                            filler_function=lambda event: event.getDataFrame(prefix='waferTower'),
                            fixture_function=tower_fixtures)

egs = DFCollection(name='EG', label='EG',
                   filler_function=lambda event: event.getDataFrame(prefix='egammaEE'))

tracks = DFCollection(name='L1Trk', label='L1Track',
                      filler_function=lambda event: event.getDataFrame(prefix='l1track'))

tkeles = DFCollection(name='TkEle', label='TkEle',
                      filler_function=lambda event: event.getDataFrame(prefix='tkEle'))

tkisoeles = DFCollection(name='TkIsoEle', label='TkIsoEle',
                         filler_function=lambda event: event.getDataFrame(prefix='tkIsoEle'))

tkegs = DFCollection(name='TKEG', label='TkEG',
                     filler_function=lambda event: get_trackmatched_egs(egs=egs.df, tracks=tracks.df),
                     depends_on=[egs, tracks])


class TPSet:
    """
    [TPSet] Represents a complet set of TPs and components (3D clusters + 2D and TC).

    Doesn't actually provide any functionality beyond an interface to the
    corresponding DFCollection objects.
    """

    def __init__(self, tcs, cl2ds, cl3ds):
        self.tcs = tcs
        self.cl2ds = cl2ds
        self.cl3ds = cl3ds

    @property
    def name(self):
        return self.cl3ds.name

    @property
    def label(self):
        return self.cl3ds.label

    @property
    def tc_df(self):
        return self.tcs.df

    @property
    def cl2d_df(self):
        return self.cl2ds.df

    @property
    def cl3d_df(self):
        return self.cl3ds.df

    @property
    def df(self):
        return self.cl3d_df

    def activate(self):
        self.tcs.activate()
        self.cl2ds.activate()
        self.cl3ds.activate()


tp_def = TPSet(tcs, cl2d_def, cl3d_def)
tp_def_nc = TPSet(tcs, cl2d_def, cl3d_def_nc)
tp_def_merged = TPSet(tcs, cl2d_def, cl3d_def_merged)
tp_def_calib = TPSet(tcs, cl2d_def, cl3d_def_calib)
tp_hm_vdr = TPSet(tcs, tcs, cl3d_hm)
tp_hm_vdr_nc0 = TPSet(tcs, tcs, cl3d_hm_nc0)
tp_hm_vdr_nc1 = TPSet(tcs, tcs, cl3d_hm_nc1)
tp_hm_vdr_merged = TPSet(tcs, tcs, cl3d_hm_merged)
