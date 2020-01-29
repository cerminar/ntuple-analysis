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
            # print '[EventManager] registering collection: {}'.format(collection.name)
            self.collections.append(collection)

        def registerActiveCollection(self, collection):
            print '[EventManager] registering collection as active: {}'.format(collection.name)
            self.active_collections.append(collection)

        def read(self, event, debug):
            for collection in self.active_collections:
                if debug >= 3:
                    print '[EventManager] filling collection: {}'.format(collection.name)
                collection.fill(event, debug)

        def get_labels(self):
            label_dict = {}
            for col in self.collections:
                label_dict[col.name] = col.label
            return label_dict

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

    def __init__(self, name, label,
                 filler_function,
                 fixture_function=None,
                 depends_on=[],
                 debug=0,
                 print_function=lambda df: df):
        self.df = None
        self.name = name
        self.label = label
        self.is_active = False
        self.filler_function = filler_function
        self.fixture_function = fixture_function
        self.depends_on = depends_on
        self.debug = debug
        self.print_function = print_function
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
            self.df = self.fixture_function(self.df)
        if not self.df.empty:
            debugPrintOut(max(debug, self.debug), self.label,
                          toCount=self.df,
                          toPrint=self.print_function(self.df))


def tkeg_fromcluster_fixture(tkegs):
    # print tkegs
    tkegs.loc[tkegs.hwQual == 1, 'hwQual'] = 3
    return tkegs


def cl3d_fixtures(clusters, tcs):
    # print clusters.columns
    # for backward compatibility
    clusters.rename(columns={'clusters_id': 'clusters',
                             'clusters_n': 'nclu'},
                    inplace=True)
    clusters['hwQual'] = clusters['quality']
    # clusters['nclu'] = [len(x) for x in clusters.clusters]

    def compute_hoe(cluster):
        # print cluster
        components = tcs[tcs.id.isin(cluster.clusters)]
        e_energy = components[components.layer <= 28].energy.sum()
        h_enery = components[components.layer > 28].energy.sum()
        if e_energy != 0.:
            cluster.hoe = h_enery/e_energy
        return cluster

    if 'hoe' not in clusters.columns:
        clusters['hoe'] = 666
        clusters = clusters.apply(compute_hoe, axis=1)

    em_layers = range(1, 29, 2)

    def compute_layer_energy(cluster):
        components = tcs[tcs.id.isin(cluster.clusters)]
        cluster['layer_energy'] = [components[components.layer == layer].energy.sum() for layer in range(1, 29, 2)]
        return cluster

    def compute_layer_energy2(cluster):
        components = tcs[tcs.id.isin(cluster.clusters)]
        cluster['layer_energy'] = [np.sum(components[components.layer == layer].energy.values) for layer in em_layers]
        return cluster

    if 'layer_energy' not in clusters.columns:
        clusters = clusters.apply(compute_layer_energy2, axis=1)

    clusters['ptem'] = clusters.pt/(1+clusters.hoe)
    clusters['eem'] = clusters.energy/(1+clusters.hoe)
    if False:
        clusters['bdt_pu'] = rnptmva.evaluate_reader(
            classifiers.mva_pu_classifier_builder(), 'BDT', clusters[['pt', 'eta', 'maxlayer', 'hoe', 'emaxe', 'szz']])

        clusters['bdt_pi'] = rnptmva.evaluate_reader(
            classifiers.mva_pi_classifier_builder(), 'BDT', clusters[['pt', 'eta', 'maxlayer', 'hoe', 'emaxe', 'szz']])
    return clusters

def gen_fixtures(particles, mc_particles):
    # print particles.columns
    particles['pdgid'] = particles.pid
    particles['abseta'] = np.abs(particles.eta)

    def get_mother_pdgid(particle, mc_particles):
        if particle.gen == -1:
            return -1
        return mc_particles.df.loc[particle.gen-1].firstmother_pdgid
    particles['firstmother_pdgid'] = particles.apply(func=lambda x: get_mother_pdgid(x, mc_particles), axis=1)
    return particles


def mc_fixtures(particles):
    particles['firstmother'] = particles.index
    particles['firstmother_pdgid'] = particles.pdgid

    for particle in particles.itertuples():
        # print particle.Index
        particles.loc[particle.daughters, 'firstmother'] = particle.Index
        particles.loc[particle.daughters, 'firstmother_pdgid'] = particle.pdgid
        # print particles.loc[particle.daughters]['firstmother']
    return particles


def tc_fixtures(tcs):
    tcs['ncells'] = 1
    if not tcs.empty:
        tcs['cells'] = tcs.apply(func=lambda x: [int(x.id)], axis=1)
    # tcs['xproj'] = tcs.x/tcs.z
    # tcs['yproj'] = tcs.y/tcs.z
    return tcs


def cl2d_fixtures(clusters):
    clusters['ncells'] = 1
    if not clusters.empty:
        clusters['ncells'] = [len(x) for x in clusters.cells]
    return clusters


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
    return towers


def get_cylind_clusters_mp(cl3ds, tcs, cylind_size, pool):
    cluster_sides = [x for x in [cl3ds[cl3ds.eta > 0],
                                 cl3ds[cl3ds.eta < 0]]]
    tc_sides = [x for x in [tcs[tcs.eta > 0],
                            tcs[tcs.eta < 0]]]

    cylind_sizes = [cylind_size, cylind_size]

    cluster_and_tc_sides = zip(cluster_sides, tc_sides, cylind_sizes)

    result_3dcl = pool.map(clAlgo.get_cylind_clusters_unpack, cluster_and_tc_sides)
    merged_clusters = pd.DataFrame(columns=cl3ds.columns)
    for res3D in result_3dcl:
        merged_clusters = merged_clusters.append(res3D, ignore_index=True, sort=False)
    return merged_clusters


def get_dr_clusters_mp(cl3ds, tcs, dr_size, pool):
    cluster_sides = [x for x in [cl3ds[cl3ds.eta > 0],
                                 cl3ds[cl3ds.eta < 0]]]
    tc_sides = [x for x in [tcs[tcs.eta > 0],
                            tcs[tcs.eta < 0]]]

    dr_sizes = [dr_size, dr_size]

    cluster_and_tc_sides = zip(cluster_sides, tc_sides, dr_sizes)

    result_3dcl = pool.map(clAlgo.get_dr_clusters_unpack2, cluster_and_tc_sides)
    # result_3dcl = []
    # result_3dcl.append(clAlgo.get_dr_clusters_unpack2(cluster_and_tc_sides[0]))
    # result_3dcl.append(clAlgo.get_dr_clusters_unpack2(cluster_and_tc_sides[1]))

    merged_clusters = pd.DataFrame(columns=cl3ds.columns)
    for res3D in result_3dcl:
        merged_clusters = merged_clusters.append(res3D, ignore_index=True, sort=False)
    return merged_clusters


def get_emint_clusters(triggerClusters):
    clusters_emint = triggerClusters.copy(deep=True)
    def interpret(cluster):
        cluster.energy = cluster.ienergy[-1]
        cluster.pt = cluster.ipt[-1]
        return cluster

    clusters_emint = clusters_emint.apply(interpret, axis=1)
    return clusters_emint


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


def get_layer_calib_clusters(input_clusters, layer_calib_factors, eta_corr=(0., 0.)):
    calibrated_clusters = input_clusters.copy(deep=True)

    def apply_calibration(cluster):
        cluster['energy'] = np.sum(np.array(cluster['layer_energy'])*np.array(layer_calib_factors))+eta_corr[1]+np.abs(cluster['eta'])*eta_corr[0]
        cluster['pt'] = cluster.energy/np.cosh(cluster.eta)
        return cluster
    calibrated_clusters = calibrated_clusters.apply(apply_calibration, axis=1)
    return calibrated_clusters


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


def merge_collections(barrel, endcap):
    return barrel.append(endcap, ignore_index=True)


def barrel_quality(electrons):
    hwqual = pd.to_numeric(electrons['hwQual'], downcast='integer')
    electrons['looseTkID'] = ((hwqual.values >> 1) & 1) > 0
    electrons['photonID'] = ((hwqual.values >> 2) & 1) > 0

    return electrons

def fake_endcap_quality(electrons):
    # just added for compatibility with barrel
    electrons['looseTkID'] = True
    electrons['photonID'] = True
    return electrons

# v96bis
calib_table = {}

calib_table['HMvDRCalib'] = [1., 0.98, 1.03, 1.07, 0.94, 0.96, 1.09, 1.03, 0.81, 1.0, 0.9, 1.08, 1.5, 1.81]
calib_table['HMvDRcylind10Calib'] = [1., 0.99, 1.03, 1.07, 0.94, 0.96, 1.09, 1.03, 0.8, 1.02, 0.9, 1.08, 1.5, 1.83]
calib_table['HMvDRcylind5Calib'] = [1., 0.98, 1.05, 1.09, 0.96, 0.97, 1.11, 1.06, 0.83, 1.04, 0.92, 1.08, 1.5, 1.89]
calib_table['HMvDRcylind2p5Calib'] = [1., 0.87, 1.16, 1.21, 1.06, 1.04, 1.25, 1.15, 0.97, 1.11, 1.01, 1.15, 1.64, 2.15]
calib_table['HMvDRshapeCalib'] = [1., 1.38, 1.05, 1.07, 0.96, 0.97, 1.11, 1.04, 0.81, 1.02, 0.9, 1.07, 1.52, 1.84]
calib_table['HMvDRshapeDrCalib'] = [1., 0.98, 1.05, 1.09, 0.96, 0.97, 1.11, 1.05, 0.83, 1.03, 0.91, 1.07, 1.51, 1.89]



gen = DFCollection(name='MC', label='MC particles',
                   filler_function=lambda event: event.getDataFrame(prefix='gen'),
                   fixture_function=mc_fixtures, debug=0)


gen_parts = DFCollection(name='GEN', label='GEN particles',
                         filler_function=lambda event: event.getDataFrame(prefix='genpart'),
                         fixture_function=lambda gen_parts: gen_fixtures(gen_parts, gen),
                         depends_on=[gen],
                         debug=0,
                         print_function=lambda df: df[['eta', 'phi', 'pt', 'energy', 'mother', 'fbrem', 'ovz', 'pid', 'gen', 'reachedEE']]
                         )
# gen_parts.activate()

tcs = DFCollection(name='TC', label='Trigger Cells',
                   filler_function=lambda event: event.getDataFrame(prefix='tc'),
                   fixture_function=tc_fixtures, debug=0)
# tcs.activate()
tcs_truth = DFCollection(name='TCTrue', label='Trigger Cells True',
                         filler_function=lambda event: event.getDataFrame(prefix='tctruth'),
                         fixture_function=tc_fixtures)

cl2d_def = DFCollection(name='DEF2D', label='dRC2d',
                        filler_function=lambda event: event.getDataFrame(prefix='cl'),
                        fixture_function=cl2d_fixtures)

cl2d_truth = DFCollection(name='DEF2DTrue', label='dRC2d True',
                          filler_function=lambda event: event.getDataFrame(prefix='cltruth'),
                          fixture_function=cl2d_fixtures)

cl3d_truth = DFCollection(name='HMvDRTrue', label='HM+dR(layer) True Cl3d',
                          filler_function=lambda event: event.getDataFrame(prefix='cl3dtruth'),
                          fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
                          depends_on=[tcs], debug=0)


cl3d_def = DFCollection(name='DEF', label='dRC3d',
                        filler_function=lambda event: event.getDataFrame(prefix='cl3d'),
                        fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
                        depends_on=[tcs])

cl3d_def_nc = DFCollection(name='DEFNC', label='dRC3d NewTh',
                           filler_function=lambda event: event.getDataFrame(prefix='cl3dNC'),
                           fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
                           depends_on=[tcs])

cl3d_hm = DFCollection(name='HMvDR', label='HM+dR(layer) Cl3d',
                       filler_function=lambda event: event.getDataFrame(prefix='hmVRcl3d'),
                       fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
                       depends_on=[tcs],
                       debug=0,
                       print_function=lambda df: df[['id', 'energy', 'pt', 'eta', 'phi', 'quality', 'hwQual', 'ienergy', 'ipt']].sort_values(by='pt', ascending=False))
# cl3d_hm.activate()


cl3d_hm_emint = DFCollection(name='HMvDREmInt', label='HM+dR(layer) Cl3d EM Int',
                           filler_function=lambda event: get_emint_clusters(cl3d_hm.df[cl3d_hm.df.quality>0]),
                           # fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
                           depends_on=[cl3d_hm],
                           debug=0,
                           print_function=lambda df: df[['id', 'energy', 'pt', 'eta', 'quality', 'hwQual', 'ienergy', 'ipt']].sort_values(by='pt', ascending=False)[:10])


cl3d_hm_emint_merged = DFCollection(name='HMvDREmIntMerged', label='HM+dR(layer) Cl3d EM Int Merged',
                                    filler_function=lambda event: get_merged_cl3d(cl3d_hm_emint.df[cl3d_hm.df.quality >= 0], POOL),
                                    # fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
                                    depends_on=[cl3d_hm_emint],
                                    debug=0,
                                    print_function=lambda df: df[['id', 'energy', 'pt', 'eta', 'quality', 'hwQual', 'ienergy', 'ipt']])

# cl3d_hm_emint.activate()

cl3d_hm_rebin = DFCollection(name='HMvDRRebin', label='HM+dR(layer) rebin Cl3d ',
                             filler_function=lambda event: event.getDataFrame(prefix='hmVRcl3dRebin'),
                             fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
                             depends_on=[tcs])

cl3d_hm_stc = DFCollection(name='HMvDRsTC', label='HM+dR(layer) SuperTC Cl3d ',
                           filler_function=lambda event: event.getDataFrame(prefix='hmVRcl3dSTC'),
                           fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
                           depends_on=[tcs])

cl3d_hm_nc0 = DFCollection(name='HMvDRNC0', label='HM+dR(layer) Cl3d + NewTh0',
                           filler_function=lambda event: event.getDataFrame(prefix='hmVRcl3dNC0'),
                           fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
                           depends_on=[tcs])

cl3d_hm_nc1 = DFCollection(name='HMvDRNC1', label='HM+dR(layer) Cl3d + NewTh1',
                           filler_function=lambda event: event.getDataFrame(prefix='hmVRcl3dNC1'),
                           fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
                           depends_on=[tcs])

cl3d_def_merged = DFCollection(name='DEFMerged', label='dRC3d merged',
                               filler_function=lambda event: get_merged_cl3d(cl3d_def.df[cl3d_def.df.quality > 0], POOL),
                               depends_on=[cl3d_def])

cl3d_def_calib = DFCollection(name='DEFCalib', label='dRC3d calib.',
                              filler_function=lambda event: get_calibrated_clusters(calib.get_calib_factors(), cl3d_def.df),
                              depends_on=[cl3d_def])

cl3d_hm_merged = DFCollection(name='HMvDRMerged', label='HM+dR(layer) merged',
                              filler_function=lambda event: get_merged_cl3d(cl3d_hm.df[cl3d_hm.df.quality >= 0], POOL),
                              depends_on=[cl3d_hm])

cl3d_hm_fixed = DFCollection(name='HMvDRfixed', label='HM fixed',
                             filler_function=lambda event: get_cylind_clusters_mp(cl3d_hm.df, tcs.df, [999.]*52, POOL),
                             depends_on=[cl3d_hm, tcs], debug=0)

cl3d_hm_cylind10 = DFCollection(name='HMvDRcylind10', label='HM Cylinder 10cm',
                                filler_function=lambda event: get_cylind_clusters_mp(cl3d_hm.df, tcs.df, [10.]*52, POOL),
                                depends_on=[cl3d_hm, tcs], debug=0)

cl3d_hm_cylind5 = DFCollection(name='HMvDRcylind5', label='HM Cylinder 5cm',
                               filler_function=lambda event: get_cylind_clusters_mp(cl3d_hm.df, tcs.df, [5.]*52, POOL),
                               depends_on=[cl3d_hm, tcs], debug=0)

cl3d_hm_cylind2p5 = DFCollection(name='HMvDRcylind2p5', label='HM Cylinder 2.5cm',
                                 filler_function=lambda event: get_cylind_clusters_mp(cl3d_hm.df, tcs.df, [2.5]*52, POOL),
                                 depends_on=[cl3d_hm, tcs], debug=0)

# cl3d_hm_shape = DFCollection(name='HMvDRshape', label='HM shape ',
#                              filler_function=lambda event: get_cylind_clusters_mp(cl3d_hm.df, tcs.df, [1.]*2+[1.6]*2+[2.5]*2+[5]*2+[5]*2+[5]*2+[5]*2+[5.]*2+[6.]*2+[7.]*2+[7.2]*2+[7.4]*2+[7.2]*2+[7.]*2+[2.5]*25, POOL),
#                              depends_on=[cl3d_hm, tcs], debug=0)

cl3d_hm_shape = DFCollection(name='HMvDRshape', label='HM shape',
                             filler_function=lambda event: get_cylind_clusters_mp(cl3d_hm.df,
                                                                                  tcs.df,
                                                                                  [1.]*2+[1.6]*2+[2.5]*2+[5.0]*2+[5.0]*2+[5.0]*2+[5.0]*2+[5.]*2+[6.]*2+[7.]*2+[7.2]*2+[7.4]*2+[7.2]*2+[7.]*2+[2.5]*25,
                                                                                  # [1.]*2+[1.6]*2+[1.8]*2+[2.2]*2+[2.6]*2+[3.4]*2+[4.2]*2+[5.]*2+[6.]*2+[7.]*2+[7.2]*2+[7.4]*2+[7.2]*2+[7.]*2+[2.5]*25,
                                                                                  POOL),
                             depends_on=[cl3d_hm, tcs], debug=0)


cl3d_hm_shapeDr = DFCollection(name='HMvDRshapeDr', label='HM #Delta#rho < 0.015',
                               filler_function=lambda event: get_dr_clusters_mp(cl3d_hm.df[cl3d_hm.df.quality>0],
                                                                                tcs.df,
                                                                                [0.015]*53,
                                                                                # [1.]*2+[1.6]*2+[1.8]*2+[2.2]*2+[2.6]*2+[3.4]*2+[4.2]*2+[5.]*2+[6.]*2+[7.]*2+[7.2]*2+[7.4]*2+[7.2]*2+[7.]*2+[2.5]*25,
                                                                                POOL),
                               depends_on=[cl3d_hm, tcs],
                               debug=0,
                               print_function=lambda df: df[['id', 'energy', 'pt', 'eta', 'quality', 'hwQual']].sort_values(by='pt', ascending=False)[:10])


cl3d_hm_calib = DFCollection(name='HMvDRCalib', label='HM calib.',
                             filler_function=lambda event: get_layer_calib_clusters(cl3d_hm.df,
                                                                                    calib_table['HMvDRCalib']),
                             depends_on=[cl3d_hm, tcs], debug=0)

cl3d_hm_cylind10_calib = DFCollection(name='HMvDRcylind10Calib', label='HM Cylinder 10cm calib.',
                                      filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_cylind10.df,
                                                                                             calib_table['HMvDRcylind10Calib']),
                                      depends_on=[cl3d_hm_cylind10, tcs], debug=0)

cl3d_hm_cylind5_calib = DFCollection(name='HMvDRcylind5Calib', label='HM Cylinder 5cm calib.',
                                     filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_cylind5.df,
                                                                                            calib_table['HMvDRcylind5Calib']),
                                     depends_on=[cl3d_hm_cylind5, tcs])

cl3d_hm_cylind2p5_calib = DFCollection(name='HMvDRcylind2p5Calib', label='HM Cylinder 2.5cm calib.',
                                       filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_cylind2p5.df,
                                                                                              calib_table['HMvDRcylind2p5Calib']),
                                       depends_on=[cl3d_hm_cylind2p5, tcs], debug=0)

cl3d_hm_fixed_calib = DFCollection(name='HMvDRfixedCalib', label='HM fixed calib.',
                                   filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_fixed.df, calib_table['HMvDRfixedCalib']),
                                   depends_on=[cl3d_hm_fixed, tcs], debug=0)

cl3d_hm_shape_calib = DFCollection(name='HMvDRshapeCalib', label='HM shape calib.',
                                   filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_shape.df, calib_table['HMvDRshapeCalib']),
                                   depends_on=[cl3d_hm_shape, tcs], debug=0, print_function=lambda df: df[['id', 'pt', 'eta', 'quality', 'hwQual']])


cl3d_hm_shapeDr_calib = DFCollection(name='HMvDRshapeDrCalib', label='HM #Delta#rho < 0.015 calib.',
                                   filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_shapeDr.df, [0.0, 1.4, 0.84, 1.14, 1.0, 0.98, 1.03, 1.03, 1.03, 0.92, 0.99, 0.93, 1.45, 1.88], (-17.593281, 38.969376)),
                                   depends_on=[cl3d_hm_shapeDr, tcs], debug=0, print_function=lambda df: df[['id', 'pt', 'eta', 'quality', 'hwQual']])



cl3d_hm_calib_merged = DFCollection(name='HMvDRCalibMerged', label='HM calib. merged',
                                    filler_function=lambda event: get_merged_cl3d(cl3d_hm_calib.df[cl3d_hm_calib.df.quality > 0], POOL),
                                    depends_on=[cl3d_hm_calib])

cl3d_hm_shape_calib_merged = DFCollection(name='HMvDRshapeCalibMerged', label='HM shape calib. merged',
                                          filler_function=lambda event: get_merged_cl3d(cl3d_hm_shape_calib.df[cl3d_hm_shape_calib.df.quality > 0], POOL),
                                          depends_on=[cl3d_hm_shape_calib])

cl3d_hm_cylind2p5_calib_merged = DFCollection(name='HMvDRcylind2p5CalibMerged', label='HM cyl. 2.5cms calib. merged',
                                              filler_function=lambda event: get_merged_cl3d(cl3d_hm_cylind2p5_calib.df[cl3d_hm_cylind2p5_calib.df.quality > 0], POOL),
                                              depends_on=[cl3d_hm_cylind2p5_calib])

cl3d_hm_shape_calib1 = DFCollection(name='HMvDRshapeCalib1', label='HM shape calib. dedx',
                                    filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_shape.df, [1.527]+[1.]*12+[1.98]),
                                    depends_on=[cl3d_hm_shape, tcs], debug=0)

cl3d_hm_fixed_calib1 = DFCollection(name='HMvDRfixedCalib1', label='HM fixed calib. dedx',
                                    filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_fixed.df, [1.527]+[1.]*12+[1.98]),
                                    depends_on=[cl3d_hm_fixed, tcs], debug=0)

cl3d_hm_cylind10_calib1 = DFCollection(name='HMvDRcylind10Calib1', label='HM Cylinder 10cm calib. dedx',
                                       filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_cylind10.df, [1.527]+[1.]*12+[1.98]),
                                       depends_on=[cl3d_hm_cylind10, tcs], debug=0)

cl3d_hm_cylind5_calib1 = DFCollection(name='HMvDRcylind5Calib1', label='HM Cylinder 5cm calib. dedx',
                                      filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_cylind5.df, [1.527]+[1.]*12+[1.98]),
                                      depends_on=[cl3d_hm_cylind5, tcs])

cl3d_hm_cylind2p5_calib1 = DFCollection(name='HMvDRcylind2p5Calib1', label='HM Cylinder 2.5cm calib. dedx',
                                        filler_function=lambda event: get_layer_calib_clusters(cl3d_hm_cylind2p5.df, [1.527]+[1.]*12+[1.98]),
                                        depends_on=[cl3d_hm_cylind2p5, tcs], debug=0)

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
                   filler_function=lambda event: event.getDataFrame(prefix='egammaEE'),
                   # print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
                   fixture_function=fake_endcap_quality,
                   debug=0)

egs_brl = DFCollection(name='EGBRL', label='EG barrel',
                       filler_function=lambda event: event.getDataFrame(prefix='egammaEB'),
                       fixture_function=barrel_quality,
                       # print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
                       debug=0)

egs_all = DFCollection(name='EGALL', label='EG all',
                       filler_function=lambda event: merge_collections(barrel=egs_brl.df, endcap=egs.df[egs.df.hwQual == 5]),
                       print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
                       debug=0,
                       depends_on=[egs, egs_brl])

tracks = DFCollection(name='L1Trk', label='L1Track',
                      filler_function=lambda event: event.getDataFrame(prefix='l1track'), debug=0)

tracks_emu = DFCollection(name='L1TrkEmu', label='L1Track EMU',
                          filler_function=lambda event: event.getDataFrame(prefix='l1trackemu'), debug=0)


tkeles = DFCollection(name='TkEle', label='TkEle',
                      filler_function=lambda event: event.getDataFrame(prefix='tkEle'),
                      fixture_function=fake_endcap_quality,
                      debug=0)

tkelesEL = DFCollection(name='TkEleEL', label='TkEle Ell. match',
                        filler_function=lambda event: event.getDataFrame(prefix='tkEleEl'),
                        fixture_function=fake_endcap_quality,
                        debug=0)

tkisoeles = DFCollection(name='TkIsoEle', label='TkIsoEle',
                         filler_function=lambda event: event.getDataFrame(prefix='tkIsoEle'))

tkegs = DFCollection(name='TkEG', label='TkEG',
                     filler_function=lambda event: get_trackmatched_egs(egs=egs.df, tracks=tracks.df),
                     depends_on=[egs, tracks],
                     debug=0)

tkegs_shape_calib = DFCollection(name='TkEGshapeCalib', label='TkEGshapecalib',
                     filler_function=lambda event: get_trackmatched_egs(egs=cl3d_hm_shape_calib.df, tracks=tracks.df),
                     fixture_function=tkeg_fromcluster_fixture,
                     depends_on=[cl3d_hm_shape_calib, tracks],
                     debug=0)
# tkegs_shape_calib.activate()

tkegs_emu = DFCollection(name='TkEGEmu', label='TkEG Emu',
                         filler_function=lambda event: get_trackmatched_egs(egs=egs.df, tracks=tracks_emu.df),
                         depends_on=[egs, tracks_emu])

tkeles_brl = DFCollection(name='TkEleBRL', label='TkEle barrel',
                          filler_function=lambda event: event.getDataFrame(prefix='tkEleBARREL'),
                          fixture_function=barrel_quality,
                          debug=0)

tkelesEL_brl = DFCollection(name='TkEleELBRL', label='TkEle Ell. match barrel',
                            filler_function=lambda event: event.getDataFrame(prefix='tkEleElBARREL'),
                            fixture_function=barrel_quality,
                            debug=0)

tkelesEL_all = DFCollection(name='TkEleELALL', label='TkEle Ell. match all',
                            filler_function=lambda event: merge_collections(barrel=tkelesEL_brl.df,
                                                                            endcap=tkelesEL.df[tkelesEL.df.hwQual==5]),
                            debug=0,
                            depends_on=[tkelesEL, tkelesEL_brl])

tkeles_all = DFCollection(name='TkEleALL', label='TkEle all',
                         filler_function=lambda event: merge_collections(barrel=tkeles_brl.df, endcap=tkeles.df[tkeles.df.hwQual==5]),
                         debug=0,
                         depends_on=[tkeles, tkeles_brl])


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
tp_truth = TPSet(tcs, tcs, cl3d_truth)
tp_def_nc = TPSet(tcs, cl2d_def, cl3d_def_nc)
tp_def_merged = TPSet(tcs, cl2d_def, cl3d_def_merged)
tp_def_calib = TPSet(tcs, cl2d_def, cl3d_def_calib)
tp_hm_vdr = TPSet(tcs, tcs, cl3d_hm)
tp_hm_fixed = TPSet(tcs, tcs, cl3d_hm_fixed)
tp_hm_shape = TPSet(tcs, tcs, cl3d_hm_shape)
tp_hm_shapeDr = TPSet(tcs, tcs, cl3d_hm_shapeDr)
tp_hm_emint = TPSet(tcs, tcs, cl3d_hm_emint)
tp_hm_cylind10 = TPSet(tcs, tcs, cl3d_hm_cylind10)
tp_hm_cylind5 = TPSet(tcs, tcs, cl3d_hm_cylind5)
tp_hm_cylind2p5 = TPSet(tcs, tcs, cl3d_hm_cylind2p5)
tp_hm_fixed_calib = TPSet(tcs, tcs, cl3d_hm_fixed_calib)
tp_hm_shape_calib = TPSet(tcs, tcs, cl3d_hm_shape_calib)
tp_hm_shapeDr_calib = TPSet(tcs, tcs, cl3d_hm_shapeDr_calib)
tp_hm_calib = TPSet(tcs, tcs, cl3d_hm_calib)
tp_hm_cylind10_calib = TPSet(tcs, tcs, cl3d_hm_cylind10_calib)
tp_hm_cylind5_calib = TPSet(tcs, tcs, cl3d_hm_cylind5_calib)
tp_hm_cylind2p5_calib = TPSet(tcs, tcs, cl3d_hm_cylind2p5_calib)
tp_hm_shape_calib_merged = TPSet(tcs, tcs, cl3d_hm_shape_calib_merged)
tp_hm_calib_merged = TPSet(tcs, tcs, cl3d_hm_calib_merged)
# tp_hm_cylind10_calib = TPSet(tcs, tcs, cl3d_hm_cylind10_calib)
# tp_hm_cylind5_calib = TPSet(tcs, tcs, cl3d_hm_cylind5_calib)
tp_hm_cylind2p5_calib_merged = TPSet(tcs, tcs, cl3d_hm_cylind2p5_calib_merged)

tp_hm_fixed_calib1 = TPSet(tcs, tcs, cl3d_hm_fixed_calib1)
tp_hm_shape_calib1 = TPSet(tcs, tcs, cl3d_hm_shape_calib1)
tp_hm_cylind10_calib1 = TPSet(tcs, tcs, cl3d_hm_cylind10_calib1)
tp_hm_cylind5_calib1 = TPSet(tcs, tcs, cl3d_hm_cylind5_calib1)
tp_hm_cylind2p5_calib1 = TPSet(tcs, tcs, cl3d_hm_cylind2p5_calib1)
tp_hm_vdr_rebin = TPSet(tcs, tcs, cl3d_hm_rebin)
tp_hm_vdr_stc = TPSet(tcs, tcs, cl3d_hm_stc)
tp_hm_vdr_nc0 = TPSet(tcs, tcs, cl3d_hm_nc0)
tp_hm_vdr_nc1 = TPSet(tcs, tcs, cl3d_hm_nc1)
tp_hm_vdr_merged = TPSet(tcs, tcs, cl3d_hm_merged)
tp_hm_emint_merged = TPSet(tcs, tcs, cl3d_hm_emint_merged)
