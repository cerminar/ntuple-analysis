"""
Manager of the ntuple data collections.

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

from __future__ import print_function
from __future__ import absolute_import
import pandas as pd
import numpy as np
import ROOT
import math
import sys

import root_numpy.tmva as rnptmva

from .utils import debugPrintOut
import python.clusterTools as clAlgo
from python.mp_pool import POOL
import python.classifiers as classifiers
import python.calibrations as calib
import python.pf_regions as pf_regions


class WeightFile(object):
    def __init__(self, file_name):
        self.file_ = ROOT.TFile(file_name)
        self.cache_ = {}

    def get_weight_1d(self, histo_name, variable):
        histo = None
        if histo_name not in self.cache_.keys():
            self.cache_[histo_name] = self.file_.Get(histo_name)
        histo = self.cache_[histo_name]
        bin_n = histo.FindBin(variable)
        return histo.GetBinContent(bin_n)


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
            self.weight_file = None

        def read_weight_file(self, weight_file_name):
            self.weight_file = WeightFile(weight_file_name)

        def registerCollection(self, collection):
            # print '[EventManager] registering collection: {}'.format(collection.name)
            self.collections.append(collection)

        def registerActiveCollection(self, collection):
            print('[EventManager] registering collection as active: {}'.format(collection.name))
            self.active_collections.append(collection)

        def read(self, event, debug):
            for collection in self.active_collections:
                if debug >= 3:
                    print('[EventManager] filling collection: {}'.format(collection.name))
                collection.fill(event, self.weight_file, debug)

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
                 read_entry_block=1000,
                 depends_on=[],
                 debug=0,
                 print_function=lambda df: df,
                 weight_function=None):
        self.df = None
        self.name = name
        self.label = label
        self.is_active = False
        self.filler_function = filler_function
        self.fixture_function = fixture_function
        self.depends_on = depends_on
        self.debug = debug
        self.print_function = print_function
        self.weight_function = weight_function
        self.n_queries = 0
        self.cached_queries = dict()
        self.cached_entries = dict()
        self.entries = None
        self.empty_df = None
        self.next_entry_read = 0
        self.read_entry_block = read_entry_block
        # print (f'Create collection: {self.name} with read_entry_block: {read_entry_block}')

        self.new_read = False
        self.new_read_nentries = 0
        self.register()

    def register(self):
        event_manager = EventManager()
        event_manager.registerCollection(self)

    def activate(self):
        if len(self.depends_on) > 0:
            common_block_size = -1
            for coll in self.depends_on:
                if common_block_size == -1:
                    common_block_size = coll.read_entry_block
                else:
                    if coll.read_entry_block != common_block_size:
                        raise ValueError(f'Collection {self.name} depends on collections with different common_block_size!')
            if common_block_size != self.read_entry_block:
                print(f'Collection {self.name}: common_block_size set to dependent value: {common_block_size}')
                self.read_entry_block = common_block_size

        if not self.is_active:
            for dep in self.depends_on:
                dep.activate()
            self.is_active = True
            event_manager = EventManager()
            event_manager.registerActiveCollection(self)
        return self.is_active

    def fill(self, event, weight_file=None, debug=0):
        stride = self.read_entry_block
        # print (f'Coll: {self.name} fill for entry: {event.file_entry}')
        if event.file_entry == 0 or event.file_entry == self.next_entry_read or event.global_entry == event.entry_range[0]:
            # print ([self.read_entry_block, (event.entry_range[1]-event.global_entry), (event.tree.num_entries - event.file_entry)])
            stride = min([self.read_entry_block, (1+event.entry_range[1]-event.global_entry), (event.tree.num_entries - event.file_entry)])
            if stride == 0:
                print('ERROR Last event????')
                self.new_read = False
            else:
                self.new_read = True
                self.next_entry_read = event.file_entry + stride
                self.new_read_nentries = stride
                self.fill_real(event, stride, weight_file, debug)
        else:
            self.new_read = False

        if self.debug > 0:
            df_print = self.empty_df
            if event.file_entry in self.df.index.get_level_values('entry'):
                df_print = self.df.loc[event.file_entry]
            debugPrintOut(max(debug, self.debug), self.label,
                          toCount=df_print,
                          toPrint=self.print_function(df_print))

    def fill_real(self, event, stride, weight_file=None, debug=0):
        self.clear_query_cache(debug)
        self.df = self.filler_function(event, stride)
        if self.fixture_function is not None:
            # FIXME: wouldn't this be more efficient
            # self.fixture_function(self.df)
            self.df = self.fixture_function(self.df)
        if self.weight_function is not None:
            self.df = self.weight_function(self.df, weight_file)
        self.empty_df = pd.DataFrame(columns=self.df.columns)
        self.entries = self.df.index.get_level_values('entry').unique()
        if debug > 2:
            print(f'read coll. {self.name} from entry: {event.file_entry} to entry: {event.file_entry+stride} (stride: {stride}), # rows: {self.df.shape[0]}, # entries: {len(self.entries)}')

    def query(self, selection):
        self.n_queries += 1
        if selection.all or self.df.empty:
            return self.df
        if selection.selection not in self.cached_queries:
            ret = self.df.query(selection.selection)
            self.cached_queries[sys.intern(selection.selection)] = ret
            entries = ret.index.get_level_values('entry').unique()
            self.cached_entries[selection.hash] = entries
            return ret
        return self.cached_queries[selection.selection]

    def query_event(self, selection, idx):
        self.n_queries += 1
        # print (f'coll: {self.name}, query selection: {selection.selection} for entry: {idx}')
        if idx not in self.entries:
            # print ('  enrty not in frame!')
            return self.empty_df
        if selection.all or self.df.empty:
            # print ('  frame is empty')
            return self.df.loc[idx]
        if selection.selection not in self.cached_queries:
            # print ('   query already cached')
            ret = self.df.query(selection.selection)
            self.cached_queries[sys.intern(selection.selection)] = ret
            entries = ret.index.get_level_values('entry').unique()
            self.cached_entries[selection.hash] = entries
            if idx not in entries:
                return self.empty_df
            return ret.loc[idx]
        # print ('    query not cached')
        # print (f'     {self.cached_queries.keys()}')
        entries = self.cached_entries[selection.hash]
        if idx not in entries:
            return self.empty_df
        return self.cached_queries[selection.selection].loc[idx]

    def clear_query_cache(self, debug=0):
        if (debug > 5):
            print('Coll: {} # queries: {} # unique queries: {}'.format(
                self.name, self.n_queries, len(self.cached_queries.keys())))
        self.n_queries = 0
        self.cached_entries.clear()
        self.cached_queries.clear()


def tkeg_fromcluster_fixture(tkegs):
    # print tkegs
    tkegs.loc[tkegs.hwQual == 1, 'hwQual'] = 3
    return tkegs


# NOTE: scorporate the part wich computes the layer_weights
# (needed only by rthe calib plotters) from the rest (creating ad-hoc collections)
# this should also allow for removing the tc dependency -> huge speedup in filling
# FIXME: this needs to be ported to the new interface reading several entries at once
def cl3d_layerEnergy_hoe(clusters, tcs):
    """ """
    if clusters.empty:
        return clusters
    do_compute_hoe = False
    do_compute_layer_energy = False
    if 'hoe' not in clusters.columns:
        do_compute_hoe = True
    if 'layer_energy' not in clusters.columns:
        do_compute_layer_energy = True

    def compute_layer_energy(cluster, do_layer_energy=True, do_hoe=False):
        components = tcs[tcs.id.isin(cluster.clusters)]
        hist, bins = np.histogram(components.layer.values,
                                  bins=range(0, 29, 2),
                                  weights=components.energy.values)
        results = []
        if do_layer_energy:
            results.append(hist)
        if do_hoe:
            em_energy = np.sum(hist)
            hoe = -1
            if em_energy != 0:
                hoe = max(0, cluster.energy - em_energy)/em_energy
            results.append(hoe)
        return results

    if do_compute_hoe or do_compute_layer_energy:
        new_columns = []
        if do_compute_layer_energy:
            new_columns.append('layer_energy')
        if do_compute_hoe:
            new_columns.append('hoe')
        clusters[new_columns] = clusters.apply(
            lambda cl: compute_layer_energy(
                cl,
                do_compute_layer_energy,
                do_compute_hoe),
            result_type='expand',
            axis=1)
    return clusters


def cl3d_fixtures(clusters):
    # print(clusters.columns)
    # for backward compatibility
    if clusters.empty:
        return clusters

    clusters.rename(columns={'clusters_id': 'clusters',
                             'clusters_n': 'nclu'},
                    inplace=True)

    clusters['ptem'] = clusters.pt/(1+clusters.hoe)
    clusters['eem'] = clusters.energy/(1+clusters.hoe)
    if False:
        clusters['bdt_pu'] = rnptmva.evaluate_reader(
            classifiers.mva_pu_classifier_builder(), 'BDT', clusters[['pt', 'eta', 'maxlayer', 'hoe', 'emaxe', 'szz']])

        clusters['bdt_pi'] = rnptmva.evaluate_reader(
            classifiers.mva_pi_classifier_builder(), 'BDT', clusters[['pt', 'eta', 'maxlayer', 'hoe', 'emaxe', 'szz']])
    return clusters


def gen_fixtures(particles, mc_particles):
    if particles.empty:
        return particles
    # print particles.columns
    particles['pdgid'] = particles.pid
    particles['abseta'] = np.abs(particles.eta)

    def get_mother_pdgid(particle, mc_particles):
        if particle.gen == -1:
            return -1
        return mc_particles.df.loc[(particle.name[0], particle.gen-1)].firstmother_pdgid
    particles['firstmother_pdgid'] = particles.apply(func=lambda x: get_mother_pdgid(x, mc_particles), axis=1)
    return particles


def mc_fixtures(particles):
    particles['firstmother'] = particles.index.to_numpy()
    particles['firstmother_pdgid'] = particles.pdgid
    return particles
    # FIXME: this is broken
    # print(particles)

    for particle in particles.itertuples():
        print(particle.daughters)
        if particle.daughters == [[], []]:
            continue
        particles.loc[particle.daughters, 'firstmother'] = particle.Index
        particles.loc[particle.daughters, 'firstmother_pdgid'] = particle.pdgid
    return particles


def tc_fixtures(tcs):
    # print tcs.columns
    tcs['ncells'] = 1
    if not tcs.empty:
        tcs['cells'] = tcs.apply(func=lambda x: [int(x.id)], axis=1)
        tcs['abseta'] = np.abs(tcs.eta)
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


def recluster_mp(cl3ds, tcs, cluster_size, cluster_function, pool):
    # FIXME: need to be ported to uproot multiindexing
    # cluster_sides = [x for x in [cl3ds[cl3ds.eta > 0],
    #                              cl3ds[cl3ds.eta < 0]]]
    # tc_sides = [x for x in [tcs[tcs.eta > 0],
    #                         tcs[tcs.eta < 0]]]
    #
    # cluster_sizes = [cluster_size, cluster_size]
    #
    # cluster_and_tc_sides = zip(cluster_sides, tc_sides, cluster_sizes)
    #
    # result_3dcl = pool.map(cluster_function, cluster_and_tc_sides)
    #
    # # result_3dcl = []
    # # result_3dcl.append(cluster_function(cluster_and_tc_sides[0]))
    # # result_3dcl.append(cluster_function(cluster_and_tc_sides[1]))
    #
    # # print result_3dcl[0]
    # # print result_3dcl[1]
    #
    merged_clusters = pd.DataFrame(columns=cl3ds.columns)
    # for res3D in result_3dcl:
    #     merged_clusters = merged_clusters.append(res3D, ignore_index=True, sort=False)
    return merged_clusters


def get_cylind_clusters_mp(cl3ds, tcs, cylind_size, pool):
    return recluster_mp(cl3ds, tcs,
                        cluster_size=cylind_size,
                        cluster_function=clAlgo.get_cylind_clusters_unpack,
                        pool=pool)


def get_dr_clusters_mp(cl3ds, tcs, dr_size, pool):
    return recluster_mp(cl3ds, tcs,
                        cluster_size=dr_size,
                        cluster_function=clAlgo.get_dr_clusters_unpack,
                        pool=pool)


def get_dtdu_clusters_mp(cl3ds, tcs, dr_size, pool):
    return recluster_mp(cl3ds, tcs,
                        cluster_size=dr_size,
                        cluster_function=clAlgo.get_dtdu_clusters_unpack,
                        pool=pool)


def get_emint_clusters(triggerClusters):
    clusters_emint = triggerClusters.copy(deep=True)

    def interpret(cluster):
        cluster.energy = cluster.ienergy[-1]
        cluster.pt = cluster.ipt[-1]
        return cluster

    clusters_emint = clusters_emint.apply(interpret, axis=1)
    return clusters_emint


def get_merged_cl3d(triggerClusters, pool, debug=0):
    # FIXME: need to be ported to uproot multiindexing
    merged_clusters = pd.DataFrame(columns=triggerClusters.columns)
    # if triggerClusters.empty:
    #     return merged_clusters
    # # FIXME: filter only interesting clusters
    # clusterSides = [x for x in [triggerClusters[triggerClusters.eta > 0],
    #                             triggerClusters[triggerClusters.eta < 0]] if not x.empty]
    #
    # results3Dcl = pool.map(clAlgo.merge3DClustersEtaPhi, clusterSides)
    # for res3D in results3Dcl:
    #     merged_clusters = merged_clusters.append(res3D, ignore_index=True, sort=False)
    return merged_clusters


def get_trackmatched_egs(egs, tracks, debug=0):
    newcolumns = ['pt', 'energy', 'eta', 'phi', 'hwQual']
    newcolumns.extend(['tkpt', 'tketa', 'tkphi', 'tkz0', 'tkchi2', 'tkchi2Red', 'tknstubs', 'deta', 'dphi', 'dr'])
    matched_egs = pd.DataFrame(columns=newcolumns)
    # FIXME: need to be ported to uproot multiindexing
    #
    #
    # if egs.empty or tracks.empty:
    #     return matched_egs
    # best_match_indexes, allmatches = match_etaphi(egs[['eta', 'phi']],
    #                                               tracks[['caloeta', 'calophi']],
    #                                               tracks['pt'],
    #                                               deltaR=0.1)
    # for bestmatch_idxes in best_match_indexes.iteritems():
    #     bestmatch_eg = egs.loc[bestmatch_idxes[0]]
    #     bestmatch_tk = tracks.loc[bestmatch_idxes[1]]
    #     matched_egs = matched_egs.append({'pt': bestmatch_eg.pt,
    #                                       'energy': bestmatch_eg.energy,
    #                                       'eta': bestmatch_eg.eta,
    #                                       'phi': bestmatch_eg.phi,
    #                                       'hwQual': bestmatch_eg.hwQual,
    #                                       'tkpt': bestmatch_tk.pt,
    #                                       'tketa': bestmatch_tk.eta,
    #                                       'tkphi': bestmatch_tk.phi,
    #                                       'tkz0': bestmatch_tk.z0,
    #                                       'tkchi2': bestmatch_tk.chi2,
    #                                       'tkchi2Red': bestmatch_tk.chi2Red,
    #                                       'tknstubs': bestmatch_tk.nStubs,
    #                                       'deta': bestmatch_tk.eta - bestmatch_eg.eta,
    #                                       'dphi': bestmatch_tk.phi - bestmatch_eg.phi,
    #                                       'dr': math.sqrt((bestmatch_tk.phi-bestmatch_eg.phi)**2+(bestmatch_tk.eta-bestmatch_eg.eta)**2)},
    #                                      ignore_index=True, sort=False)
    return matched_egs


def get_layer_calib_clusters(input_clusters,
                             layer_calib_factors,
                             eta_corr=(0., 0.),
                             debug=False):
    if debug:
        print(layer_calib_factors)
        print(eta_corr)
    calibrated_clusters = input_clusters.copy(deep=True)

    # def apply_calibration(cluster):
    #     cluster['energy'] = np.sum(np.array(cluster['layer_energy'])*np.array(layer_calib_factors))+eta_corr[1]+np.abs(cluster['eta'])*eta_corr[0]
    #     cluster['pt'] = cluster.energy/np.cosh(cluster.eta)
    #     return cluster
    # calibrated_clusters = calibrated_clusters.apply(apply_calibration, axis=1)
    if not calibrated_clusters.empty:
        energies_lcalib = calibrated_clusters.layer_energy.apply(lambda x: np.dot(x, layer_calib_factors))
        calibrated_clusters['energy'] = energies_lcalib+eta_corr[1]+np.abs(calibrated_clusters['eta'])*eta_corr[0]
        calibrated_clusters['pt'] = calibrated_clusters.energy/np.cosh(calibrated_clusters.eta)

        # print calibrated_clusters[['energy', 'energy_1']]
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
    # FIXME: need to be ported to uproot multiindexing
    #
    # if triggerClusters.empty:
    #     return trigger3DClusters
    # clusterSides = [x for x in [triggerClusters[triggerClusters.eta > 0],
    #                             triggerClusters[triggerClusters.eta < 0]] if not x.empty]
    # results3Dcl = pool.map(algorithm, clusterSides)
    # for res3D in results3Dcl:
    #     trigger3DClusters = trigger3DClusters.append(res3D, ignore_index=True, sort=False)
    #
    # debugPrintOut(debug, name='{} 3D clusters'.format(name),
    #               toCount=trigger3DClusters,
    #               toPrint=trigger3DClusters.iloc[:3])
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
                print(cluster[['pt', 'eta']])

        cluster['pt2'] = cluster.pt*calib_factor
        return cluster
        # input_3Dclusters[(input_3Dclusters.eta_l > abs(cluster.eta)) & ()]
    calibrated_clusters = calibrated_clusters.apply(apply_calibration, axis=1)
    return calibrated_clusters


def select_and_merge_collections(barrel, endcap):
    barrel_final = barrel
    endcap_final = endcap
    if not endcap.empty:
        endcap_final = endcap[endcap.hwQual == 5]
    return merge_collections(
        barrel=barrel_final,
        endcap=endcap_final)


def merge_collections(barrel, endcap):
    if barrel.empty:
        return endcap
    return barrel.append(endcap, ignore_index=True)


def barrel_quality(electrons):
    if electrons.empty:
        # electrons.columns = ['pt', 'energy', 'eta', 'phi', 'hwQual']
        return electrons
    hwqual = pd.to_numeric(electrons['hwQual'], downcast='integer')
    electrons['looseTkID'] = ((hwqual.values >> 1) & 1) > 0
    electrons['photonID'] = ((hwqual.values >> 2) & 1) > 0
    return electrons


def fake_endcap_quality(electrons):
    # just added for compatibility with barrel
    electrons['looseTkID'] = True
    electrons['photonID'] = True
    return electrons


def tkele_fixture_ee(electrons):
    electrons['looseTkID'] = True
    electrons['photonID'] = True
    electrons['dpt'] = electrons.tkPt - electrons.pt
    return electrons


def tkele_fixture_eb(electrons):
    # if electrons.empty:
    #     # electrons.columns = ['pt', 'energy', 'eta', 'phi', 'hwQual']
    #     return electrons
    hwqual = pd.to_numeric(electrons['hwQual'], downcast='integer')
    electrons['looseTkID'] = ((hwqual.values >> 1) & 1) > 0
    electrons['photonID'] = ((hwqual.values >> 2) & 1) > 0
    electrons['dpt'] = electrons.tkPt - electrons.pt
    return electrons


def print_columns(df):
    print(df.columns)
    return df


def gen_part_pt_weights(gen_parts, weight_file):
    def compute_weight(gen_part):
        return weight_file.get_weight_1d('h_weights', gen_part.pt)

    if weight_file is None:
        gen_parts['weight'] = 1
    else:
        gen_parts['weight'] = gen_parts.apply(compute_weight, axis=1)
    return gen_parts


def map2pfregions(objects, eta_var, phi_var, fiducial=False):
    for ieta in range(0, pf_regions.regionizer.n_eta_regions()):
        objects['eta_reg_{}'.format(ieta)] = False
    for iphi in range(0, pf_regions.regionizer.n_phi_regions()):
        objects['phi_reg_{}'.format(iphi)] = False

    for ieta, eta_range in enumerate(pf_regions.regionizer.get_eta_boundaries(fiducial)):
        query = '({} > {}) & ({} <= {})'.format(
            eta_var,
            eta_range[0],
            eta_var,
            eta_range[1]
            )
        region_objects = objects.query(query).index
        objects.loc[region_objects, ['eta_reg_{}'.format(ieta)]] = True

    for iphi, phi_range in enumerate(pf_regions.regionizer.get_phi_boundaries(fiducial)):
        query = '({} > {}) & ({} <= {})'.format(
            phi_var,
            phi_range[0],
            phi_var,
            phi_range[1])
        region_objects = objects.query(query).index
        objects.loc[region_objects, ['phi_reg_{}'.format(iphi)]] = True

    return objects


def maptk2pfregions_in(objects):
    return map2pfregions(objects, 'caloeta', 'calophi', fiducial=False)


def mapcalo2pfregions_in(objects):
    return map2pfregions(objects, 'eta', 'phi', fiducial=False)


def mapcalo2pfregions_out(objects):
    return map2pfregions(objects, 'eta', 'phi', fiducial=True)


def decodedTk_fixtures(objects):
    objects['deltaZ0'] = objects.z0 - objects.simz0
    objects['deltaPt'] = objects.pt - objects.simpt
    objects['deltaEta'] = objects.eta - objects.simeta
    objects['deltaCaloEta'] = objects.caloeta - objects.simcaloeta
    # have dphi between -pi and pi
    comp_remainder = np.vectorize(math.remainder)
    objects['deltaCaloPhi'] = comp_remainder(objects.calophi - objects.simcalophi, 2*np.pi)

    objects['abseta'] = np.abs(objects.eta)
    objects['simabseta'] = np.abs(objects.simeta)
    return objects


calib_mgr = calib.CalibManager()

gen = DFCollection(
    name='MC', label='MC particles',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='gen', entry_block=entry_block),
    fixture_function=mc_fixtures,
    debug=0)

gen_parts = DFCollection(
    name='GEN', label='GEN particles',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='genpart', entry_block=entry_block),
    fixture_function=lambda gen_parts: gen_fixtures(gen_parts, gen),
    # read_entry_block=10,
    depends_on=[gen],
    debug=0,
    # print_function=lambda df: df[['eta', 'phi', 'pt', 'energy', 'mother', 'fbrem', 'ovz', 'pid', 'gen', 'reachedEE', 'firstmother_pdgid']],
    print_function=lambda df: df[['gen', 'pid', 'eta', 'phi', 'pt', 'mother', 'ovz', 'dvz', 'reachedEE']].sort_values(by='mother', ascending=False),
    # print_function=lambda df: df.columns,
    weight_function=gen_part_pt_weights)

tcs = DFCollection(
    name='TC', label='Trigger Cells',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tc', entry_block=entry_block),
    read_entry_block=200,
    fixture_function=tc_fixtures, debug=0)

tcs_truth = DFCollection(
    name='TCTrue', label='Trigger Cells True',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tctruth', entry_block=entry_block),
    fixture_function=tc_fixtures)

cl2d_def = DFCollection(
    name='DEF2D', label='dRC2d',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='cl', entry_block=entry_block),
    fixture_function=cl2d_fixtures)

cl2d_truth = DFCollection(
    name='DEF2DTrue', label='dRC2d True',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='cltruth', entry_block=entry_block),
    fixture_function=cl2d_fixtures)

cl3d_truth = DFCollection(
    name='HMvDRTrue', label='HM+dR(layer) True Cl3d',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='cl3dtruth', entry_block=entry_block),
    fixture_function=lambda clusters: cl3d_fixtures(clusters),
    debug=0)

cl3d_def = DFCollection(
    name='DEF', label='dRC3d',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='cl3d', entry_block=entry_block),
    fixture_function=lambda clusters: cl3d_fixtures(clusters))

cl3d_def_nc = DFCollection(
    name='DEFNC', label='dRC3d NewTh',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='cl3dNC', entry_block=entry_block),
    fixture_function=lambda clusters: cl3d_fixtures(clusters))

cl3d_hm = DFCollection(
    name='HMvDR', label='HM+dR(layer) Cl3d',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='HMvDR', entry_block=entry_block, fallback='hmVRcl3d'),
    fixture_function=lambda clusters: cl3d_fixtures(clusters),
    read_entry_block=200,
    debug=0,
    print_function=lambda df: df[['id', 'energy', 'pt', 'eta', 'phi', 'quality', 'ienergy', 'ipt']].sort_values(by='pt', ascending=False))
# cl3d_hm.activate()

cl3d_hm_emint = DFCollection(
    name='HMvDREmInt', label='HM+dR(layer) Cl3d EM Int',
    filler_function=lambda event, entry_block: get_emint_clusters(cl3d_hm.df[cl3d_hm.df.quality > 0]),
    # fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
    depends_on=[cl3d_hm],
    debug=0,
    print_function=lambda df: df[['id', 'energy', 'pt', 'eta', 'quality', 'ienergy', 'ipt']].sort_values(by='pt', ascending=False)[:10])


cl3d_hm_emint_merged = DFCollection(
    name='HMvDREmIntMerged', label='HM+dR(layer) Cl3d EM Int Merged',
    filler_function=lambda event, entry_block: get_merged_cl3d(cl3d_hm_emint.df[cl3d_hm.df.quality >= 0], POOL),
    # fixture_function=lambda clusters: cl3d_fixtures(clusters, tcs.df),
    depends_on=[cl3d_hm_emint],
    debug=0,
    print_function=lambda df: df[['id', 'energy', 'pt', 'eta', 'quality', 'ienergy', 'ipt']])
# cl3d_hm_emint.activate()

cl3d_hm_rebin = DFCollection(
    name='HMvDRRebin', label='HM+dR(layer) rebin Cl3d ',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='hmVRcl3dRebin', entry_block=entry_block),
    fixture_function=lambda clusters: cl3d_fixtures(clusters))

cl3d_hm_stc = DFCollection(
    name='HMvDRsTC', label='HM+dR(layer) SuperTC Cl3d ',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='hmVRcl3dSTC', entry_block=entry_block),
    fixture_function=lambda clusters: cl3d_fixtures(clusters))

cl3d_hm_nc0 = DFCollection(
    name='HMvDRNC0', label='HM+dR(layer) Cl3d + NewTh0',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='hmVRcl3dNC0', entry_block=entry_block),
    fixture_function=lambda clusters: cl3d_fixtures(clusters))

cl3d_hm_nc1 = DFCollection(
    name='HMvDRNC1', label='HM+dR(layer) Cl3d + NewTh1',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='hmVRcl3dNC1', entry_block=entry_block),
    fixture_function=lambda clusters: cl3d_fixtures(clusters))

cl3d_def_merged = DFCollection(
    name='DEFMerged', label='dRC3d merged',
    filler_function=lambda event, entry_block: get_merged_cl3d(cl3d_def.df[cl3d_def.df.quality > 0], POOL),
    depends_on=[cl3d_def])

cl3d_def_calib = DFCollection(
    name='DEFCalib', label='dRC3d calib.',
    filler_function=lambda event, entry_block: get_calibrated_clusters(calib.get_calib_factors(), cl3d_def.df),
    depends_on=[cl3d_def])

cl3d_hm_merged = DFCollection(
    name='HMvDRMerged', label='HM+dR(layer) merged',
    filler_function=lambda event, entry_block: get_merged_cl3d(cl3d_hm.df[cl3d_hm.df.quality >= 0], POOL),
    depends_on=[cl3d_hm])

cl3d_hm_fixed = DFCollection(
    name='HMvDRfixed', label='HM fixed',
    filler_function=lambda event, entry_block: get_cylind_clusters_mp(cl3d_hm.df, tcs.df, [999.]*52, POOL),
    depends_on=[cl3d_hm, tcs], debug=0)

cl3d_hm_cylind10 = DFCollection(
    name='HMvDRcylind10', label='HM Cylinder 10cm',
    filler_function=lambda event, entry_block: get_cylind_clusters_mp(cl3d_hm.df, tcs.df, [10.]*52, POOL),
    depends_on=[cl3d_hm, tcs], debug=0)

cl3d_hm_cylind5 = DFCollection(
    name='HMvDRcylind5', label='HM Cylinder 5cm',
    filler_function=lambda event, entry_block: get_cylind_clusters_mp(cl3d_hm.df, tcs.df, [5.]*52, POOL),
    depends_on=[cl3d_hm, tcs], debug=0)

cl3d_hm_cylind2p5 = DFCollection(
    name='HMvDRcylind2p5', label='HM Cylinder 2.5cm',
    filler_function=lambda event, entry_block: get_cylind_clusters_mp(cl3d_hm.df, tcs.df, [2.5]*52, POOL),
    depends_on=[cl3d_hm, tcs],
    debug=0)

cl3d_hm_shape = DFCollection(
    name='HMvDRshape', label='HM shape',
    filler_function=lambda event, entry_block: get_cylind_clusters_mp(
        cl3d_hm.df, tcs.df,
        [1.]*2+[1.6]*2+[2.5]*2+[5.0]*2+[5.0]*2+[5.0]*2+[5.0]*2+[5.]*2+[6.]*2+[7.]*2+[7.2]*2+[7.4]*2+[7.2]*2+[7.]*2+[2.5]*25,
        POOL),
    depends_on=[cl3d_hm, tcs], debug=0)


cl3d_hm_shapeDr = DFCollection(
    name='HMvDRshapeDr', label='HM #Delta#rho < 0.015',
    filler_function=lambda event, entry_block: get_dr_clusters_mp(
        cl3d_hm.df[cl3d_hm.df.quality > 0],
        tcs.df,
        0.015,
        POOL),
    depends_on=[cl3d_hm, tcs],
    debug=0,
    print_function=lambda df: df[['id', 'energy', 'pt', 'eta', 'quality']].sort_values(by='pt', ascending=False)[:10])


cl3d_hm_shapeDtDu = DFCollection(
    name='HMvDRshapeDtDu', label='HM #Deltau #Deltat',
    filler_function=lambda event, entry_block: get_dtdu_clusters_mp(
        cl3d_hm.df[cl3d_hm.df.quality > 0],
        tcs.df,
        (0.015, 0.007),
        POOL),
    depends_on=[cl3d_hm, tcs],
    debug=0,
    print_function=lambda df: df[['id', 'energy', 'pt', 'eta', 'quality']].sort_values(by='pt', ascending=False)[:10])


cl3d_hm_calib = DFCollection(
    name='HMvDRCalib', label='HM calib.',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(
        cl3d_hm.df,
        calib_mgr.get_calibration('HMvDRCalib', 'layer_calibs')),
    depends_on=[cl3d_hm, tcs],
    debug=0)

cl3d_hm_cylind10_calib = DFCollection(
    name='HMvDRcylind10Calib', label='HM Cylinder 10cm calib.',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(
        cl3d_hm_cylind10.df,
        calib_mgr.get_calibration('HMvDRcylind10Calib', 'layer_calibs')),
    depends_on=[cl3d_hm_cylind10, tcs], debug=0)

cl3d_hm_cylind5_calib = DFCollection(
    name='HMvDRcylind5Calib', label='HM Cylinder 5cm calib.',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(
        cl3d_hm_cylind5.df,
        calib_mgr.get_calibration('HMvDRcylind5Calib', 'layer_calibs')),
    depends_on=[cl3d_hm_cylind5, tcs])

cl3d_hm_cylind2p5_calib = DFCollection(
    name='HMvDRcylind2p5Calib', label='HM Cylinder 2.5cm calib.',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(
        cl3d_hm_cylind2p5.df,
        calib_mgr.get_calibration('HMvDRcylind2p5Calib', 'layer_calibs')),
    depends_on=[cl3d_hm_cylind2p5, tcs], debug=0)

cl3d_hm_fixed_calib = DFCollection(
    name='HMvDRfixedCalib', label='HM fixed calib.',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(
        cl3d_hm_fixed.df,
        calib_mgr.get_calibration('HMvDRfixedCalib', 'layer_calibs')),
    depends_on=[cl3d_hm_fixed, tcs], debug=0)

cl3d_hm_shape_calib = DFCollection(
    name='HMvDRshapeCalib', label='HM shape calib.',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(
        cl3d_hm_shape.df,
        calib_mgr.get_calibration('HMvDRshapeCalib', 'layer_calibs')),
    depends_on=[cl3d_hm_shape, tcs],
    debug=0,
    print_function=lambda df: df[['id', 'pt', 'eta', 'quality']])

cl3d_hm_shapeDr_calib = DFCollection(
    name='HMvDRshapeDrCalib', label='HM #Delta#rho < 0.015 calib.',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(
        cl3d_hm_shapeDr.df,
        calib_mgr.get_calibration('HMvDRshapeDrCalib', 'layer_calibs'),
        calib_mgr.get_calibration('HMvDRshapeDrCalib', 'eta_calibs'),
        debug=False),
    depends_on=[cl3d_hm_shapeDr, tcs],
    debug=0,
    print_function=lambda df: df[['id', 'pt', 'eta', 'quality']])

cl3d_hm_shapeDr_calib_new = DFCollection(
    name='HMvDRshapeDrCalibNew', label='HM #Delta#rho < 0.015 calib.',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(
        cl3d_hm_shapeDr.df,
        calib_mgr.get_calibration('HMvDRshapeDrCalibNew', 'layer_calibs'),
        calib_mgr.get_calibration('HMvDRshapeDrCalibNew', 'eta_calibs'),
        debug=False),
    depends_on=[cl3d_hm_shapeDr, tcs],
    debug=0,
    print_function=lambda df: df[['id', 'pt', 'eta', 'quality']])

cl3d_hm_shapeDtDu_calib = DFCollection(
    name='HMvDRshapeDtDuCalib', label='HM #Deltat#Deltau calib.',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(
        cl3d_hm_shapeDr.df,
        calib_mgr.get_calibration('HMvDRshapeDtDuCalib', 'layer_calibs'),
        calib_mgr.get_calibration('HMvDRshapeDtDuCalib', 'eta_calibs'),
        debug=False),
    depends_on=[cl3d_hm_shapeDtDu, tcs],
    debug=0,
    print_function=lambda df: df[['id', 'pt', 'eta', 'quality']])


cl3d_hm_calib_merged = DFCollection(
    name='HMvDRCalibMerged', label='HM calib. merged',
    filler_function=lambda event, entry_block: get_merged_cl3d(cl3d_hm_calib.df[cl3d_hm_calib.df.quality > 0], POOL),
    depends_on=[cl3d_hm_calib])

cl3d_hm_shape_calib_merged = DFCollection(
    name='HMvDRshapeCalibMerged', label='HM shape calib. merged',
    filler_function=lambda event, entry_block: get_merged_cl3d(cl3d_hm_shape_calib.df[cl3d_hm_shape_calib.df.quality > 0], POOL),
    depends_on=[cl3d_hm_shape_calib])

cl3d_hm_cylind2p5_calib_merged = DFCollection(
    name='HMvDRcylind2p5CalibMerged', label='HM cyl. 2.5cms calib. merged',
    filler_function=lambda event, entry_block: get_merged_cl3d(cl3d_hm_cylind2p5_calib.df[cl3d_hm_cylind2p5_calib.df.quality > 0], POOL),
    depends_on=[cl3d_hm_cylind2p5_calib])

cl3d_hm_shape_calib1 = DFCollection(
    name='HMvDRshapeCalib1', label='HM shape calib. dedx',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(cl3d_hm_shape.df, [1.527]+[1.]*12+[1.98]),
    depends_on=[cl3d_hm_shape, tcs], debug=0)

cl3d_hm_fixed_calib1 = DFCollection(
    name='HMvDRfixedCalib1', label='HM fixed calib. dedx',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(cl3d_hm_fixed.df, [1.527]+[1.]*12+[1.98]),
    depends_on=[cl3d_hm_fixed, tcs], debug=0)

cl3d_hm_cylind10_calib1 = DFCollection(
    name='HMvDRcylind10Calib1', label='HM Cylinder 10cm calib. dedx',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(cl3d_hm_cylind10.df, [1.527]+[1.]*12+[1.98]),
    depends_on=[cl3d_hm_cylind10, tcs], debug=0)

cl3d_hm_cylind5_calib1 = DFCollection(
    name='HMvDRcylind5Calib1', label='HM Cylinder 5cm calib. dedx',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(cl3d_hm_cylind5.df, [1.527]+[1.]*12+[1.98]),
    depends_on=[cl3d_hm_cylind5, tcs])

cl3d_hm_cylind2p5_calib1 = DFCollection(
    name='HMvDRcylind2p5Calib1', label='HM Cylinder 2.5cm calib. dedx',
    filler_function=lambda event, entry_block: get_layer_calib_clusters(cl3d_hm_cylind2p5.df, [1.527]+[1.]*12+[1.98]),
    depends_on=[cl3d_hm_cylind2p5, tcs], debug=0)

towers_tcs = DFCollection(
    name='TT', label='TT (TC)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tower', entry_block=entry_block),
    fixture_function=tower_fixtures)

towers_sim = DFCollection(
    name='SimTT', label='TT (sim)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='simTower', entry_block=entry_block),
    fixture_function=tower_fixtures)

towers_hgcroc = DFCollection(
    name='HgcrocTT', label='TT (HGCROC)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='hgcrocTower', entry_block=entry_block),
    fixture_function=tower_fixtures)

towers_wafer = DFCollection(
    name='WaferTT', label='TT (Wafer)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='waferTower', entry_block=entry_block),
    fixture_function=tower_fixtures)

egs = DFCollection(
    name='EG', label='EG',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='egammaEE', entry_block=entry_block),
    # print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
    fixture_function=fake_endcap_quality,
    debug=0)

egs_brl = DFCollection(
    name='EGBRL', label='EG barrel',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='egammaEB', entry_block=entry_block),
    fixture_function=barrel_quality,
    # print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
    debug=0)

egs_all = DFCollection(
    name='EGALL', label='EG all',
    filler_function=lambda event, entry_block: select_and_merge_collections(
        barrel=egs_brl.df,
        endcap=egs.df),
    print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(
        by='hwQual', ascending=False)[:10],
    debug=0,
    depends_on=[egs, egs_brl])

tracks = DFCollection(
    name='L1Trk', label='L1Track',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='l1Trk', entry_block=entry_block),
    print_function=lambda df: df.sort_values(by='pt', ascending=False)[:10],
    debug=0)

tracks_emu = DFCollection(
    name='L1TrkEmu', label='L1Track EMU',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='l1trackemu', entry_block=entry_block),
    debug=0)


tkeles = DFCollection(
    name='TkEle', label='TkEle',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tkEle', entry_block=entry_block),
    fixture_function=fake_endcap_quality,
    debug=0)

tkelesEL = DFCollection(
    name='tkEleEE', label='TkEle (Ell.) EE',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tkEleEE', entry_block=entry_block),
    fixture_function=fake_endcap_quality,
    debug=0)

tkisoeles = DFCollection(
    name='TkIsoEle', label='TkIsoEle',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tkIsoEle', entry_block=entry_block))

tkegs = DFCollection(
    name='TkEG', label='TkEG',
    filler_function=lambda event, entry_block: get_trackmatched_egs(egs=egs.df, tracks=tracks.df),
    depends_on=[egs, tracks],
    debug=0)

tkegs_shape_calib = DFCollection(
    name='TkEGshapeCalib', label='TkEGshapecalib',
    filler_function=lambda event, entry_block: get_trackmatched_egs(egs=cl3d_hm_shape_calib.df, tracks=tracks.df),
    fixture_function=tkeg_fromcluster_fixture,
    depends_on=[cl3d_hm_shape_calib, tracks],
    debug=0)
# tkegs_shape_calib.activate()

tkegs_emu = DFCollection(
    name='TkEGEmu', label='TkEG Emu',
    filler_function=lambda event, entry_block: get_trackmatched_egs(egs=egs.df, tracks=tracks_emu.df),
    depends_on=[egs, tracks_emu])

tkeles_brl = DFCollection(
    name='TkEleBRL', label='TkEle barrel',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tkEleBARREL', entry_block=entry_block),
    fixture_function=barrel_quality,
    debug=0)

tkelesEL_brl = DFCollection(
    name='tkEleEB', label='TkEle (Ell.) EB',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tkEleEB', entry_block=entry_block),
    fixture_function=tkele_fixture_eb,
    debug=0)

tkelesEL_all = DFCollection(
    name='tkEleEllAll', label='TkEle Ell. match all',
    filler_function=lambda event, entry_block: select_and_merge_collections(
        barrel=tkelesEL_brl.df,
        endcap=tkelesEL.df),
    debug=0,
    depends_on=[tkelesEL, tkelesEL_brl])

tkeles_all = DFCollection(
    name='TkEleALL', label='TkEle all',
    filler_function=lambda event, entry_block: merge_collections(
        barrel=tkeles_brl.df,
        endcap=tkeles.df[tkeles.df.hwQual == 5]),
    debug=0,
    depends_on=[tkeles, tkeles_brl])

# try to cleanup a bit

egs_EE = DFCollection(
    name='EgEE', label='EG EE',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='egammaEE', entry_block=entry_block),
    # print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
    fixture_function=fake_endcap_quality,
    read_entry_block=100,
    debug=0)

egs_EB = DFCollection(
    name='EgEB', label='EG EB',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='egammaEB', entry_block=entry_block),
    # print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
    fixture_function=barrel_quality,
    read_entry_block=200,
    debug=0)

egs_EE_pf = DFCollection(
    name='PFEgEE', label='EG EE Corr.',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='PFegammaEE', entry_block=entry_block),
    # print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
    # fixture_function=mapcalo2pfregions,
    fixture_function=fake_endcap_quality,
    debug=0)

egs_EE_pfnf = DFCollection(
    name='PFNFEgEE', label='EG EE Corr. New',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='PFNFegammaEE', entry_block=entry_block),
    # print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
    # fixture_function=mapcalo2pfregions,
    fixture_function=fake_endcap_quality,
    debug=0)

tkeles_EE = DFCollection(
    name='tkEleEE', label='TkEle EE',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tkEleEE', entry_block=entry_block),
    fixture_function=tkele_fixture_ee,
    debug=0)

tkeles_EB = DFCollection(
    name='tkEleEB', label='TkEle EB',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tkEleEB', entry_block=entry_block),
    fixture_function=tkele_fixture_eb,
    debug=0)

tkeles_EE_pf = DFCollection(
    name='PFtkEleEE', label='TkEle EE Corr.',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='PFtkEleEE', entry_block=entry_block),
    fixture_function=tkele_fixture_ee,
    debug=0)

tkeles_EB_pf = DFCollection(
    name='PFtkEleEB', label='TkEle EB Corr',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='PFtkEleEB', entry_block=entry_block),
    fixture_function=tkele_fixture_eb,
    debug=0)

tkeles_EE_pfnf = DFCollection(
    name='PFNFtkEleEE', label='TkEle EE Corr. New',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='PFNFtkEleEE', entry_block=entry_block),
    fixture_function=tkele_fixture_ee,
    debug=0)

tkeles_EB_pfnf = DFCollection(
    name='PFNFtkEleEB', label='TkEle EB Corr. New',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='PFNFtkEleEB', entry_block=entry_block),
    fixture_function=tkele_fixture_eb,
    debug=0)

# --------

tkem_EE = DFCollection(
    name='tkEmEE', label='TkEm EE',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tkEmEE', entry_block=entry_block),
    fixture_function=fake_endcap_quality,
    read_entry_block=100,
    debug=0)

tkem_EB = DFCollection(
    name='tkEmEB', label='TkEm EB',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tkEmEB', entry_block=entry_block),
    fixture_function=barrel_quality,
    read_entry_block=200,
    debug=0)

tkem_EE_pf = DFCollection(
    name='PFtkEmEE', label='TkEm EE Corr.',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='PFtkEmEE', entry_block=entry_block),
    fixture_function=fake_endcap_quality,
    debug=0)

tkem_EB_pf = DFCollection(
    name='PFtkEmEB', label='TkEm EB Corr',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='PFtkEmEB', entry_block=entry_block),
    fixture_function=barrel_quality,
    read_entry_block=200,
    debug=0)

tkem_EE_pfnf = DFCollection(
    name='PFNFtkEmEE', label='TkEm EE Corr. New',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='PFNFtkEmEE', entry_block=entry_block),
    fixture_function=fake_endcap_quality,
    debug=0)

tkem_EB_pfnf = DFCollection(
    name='PFNFtkEmEB', label='TkEm EB Corr. New',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='PFNFtkEmEB', entry_block=entry_block),
    fixture_function=barrel_quality,
    read_entry_block=200,
    debug=0)

egs_EE_pf_reg = DFCollection(
    name='PFOutEgEE', label='EG EE Corr.',
    filler_function=lambda event, entry_block: egs_EE_pf.df,
    print_function=lambda df: df[[
        'pt', 'eta', 'hwQual',
        'eta_reg_0', 'eta_reg_1', 'eta_reg_2', 'eta_reg_3',
        'eta_reg_4', 'eta_reg_5', 'eta_reg_6', 'eta_reg_7',
        'eta_reg_8', 'eta_reg_9', 'eta_reg_10']].sort_values(by='eta', ascending=False)[:10],
    fixture_function=mapcalo2pfregions_out,
    depends_on=[egs_EE_pf],
    debug=0)

tkeles_EE_pf_reg = DFCollection(
    name='PFOuttkEleEE', label='TkEle EE Corr.',
    filler_function=lambda event, entry_block: tkeles_EE_pf.df,
    fixture_function=mapcalo2pfregions_out,
    depends_on=[tkeles_EE_pf],
    debug=0)

tkeles_EB_pf_reg = DFCollection(
    name='PFOuttkEleEB', label='TkEle EB Corr',
    filler_function=lambda event, entry_block: tkeles_EB_pf.df,
    fixture_function=mapcalo2pfregions_out,
    depends_on=[tkeles_EB_pf],
    debug=0)

tkem_EE_pf_reg = DFCollection(
    name='PFOuttkEmEE', label='TkEm EE Corr.',
    filler_function=lambda event, entry_block: tkem_EE_pf.df,
    fixture_function=mapcalo2pfregions_out,
    depends_on=[tkem_EE_pf],
    debug=0)

tkem_EB_pf_reg = DFCollection(
    name='PFOuttkEmEB', label='TkEm EB Corr',
    filler_function=lambda event, entry_block: tkem_EB_pf.df,
    fixture_function=mapcalo2pfregions_out,
    depends_on=[tkem_EB_pf],
    debug=0)

tk_pfinputs = DFCollection(
    name='L1Trk', label='L1Track',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='l1Trk', entry_block=entry_block),
    fixture_function=maptk2pfregions_in,
    debug=0)

eg_EE_pfinputs = DFCollection(
    name='egEEPFin', label='EG EE Input',
    filler_function=lambda event, entry_block: egs_EE.df,
    fixture_function=mapcalo2pfregions_in,
    depends_on=[egs_EE],
    debug=0)

eg_EB_pfinputs = DFCollection(
    name='egEBPFin', label='EG EB Input',
    filler_function=lambda event, entry_block: egs_EB.df,
    fixture_function=mapcalo2pfregions_in,
    depends_on=[egs_EB],
    debug=0)

cl3d_hm_pfinputs = DFCollection(
    name='HMvDRPFin', label='HMvDR Input',
    filler_function=lambda event, entry_block: cl3d_hm.df,
    fixture_function=mapcalo2pfregions_in,
    depends_on=[cl3d_hm],
    debug=0)

decTk = DFCollection(
    name='PFDecTk', label='decoded Tk',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='pfdtk', entry_block=entry_block),
    fixture_function=decodedTk_fixtures,
    debug=0)


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
tp_hm_shapeDtDu = TPSet(tcs, tcs, cl3d_hm_shapeDtDu)
tp_hm_shapeDtDu_calib = TPSet(tcs, tcs, cl3d_hm_shapeDtDu_calib)
tp_hm_emint = TPSet(tcs, tcs, cl3d_hm_emint)
tp_hm_cylind10 = TPSet(tcs, tcs, cl3d_hm_cylind10)
tp_hm_cylind5 = TPSet(tcs, tcs, cl3d_hm_cylind5)
tp_hm_cylind2p5 = TPSet(tcs, tcs, cl3d_hm_cylind2p5)
tp_hm_fixed_calib = TPSet(tcs, tcs, cl3d_hm_fixed_calib)
tp_hm_shape_calib = TPSet(tcs, tcs, cl3d_hm_shape_calib)
tp_hm_shapeDr_calib = TPSet(tcs, tcs, cl3d_hm_shapeDr_calib)
tp_hm_shapeDr_calib_new = TPSet(tcs, tcs, cl3d_hm_shapeDr_calib_new)
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
