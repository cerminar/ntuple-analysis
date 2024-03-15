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
import awkward as ak
import ROOT
import math
import sys
import xgboost
# import root_numpy.tmva as rnptmva

from .utils import debugPrintOut
import python.clusterTools as clAlgo
from python.mp_pool import POOL
import python.classifiers as classifiers
import python.calibrations as calib
import python.pf_regions as pf_regions
from scipy.spatial import cKDTree    
import python.selections as selections


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

    This class represents the data for the objects which need to be plotted.
    The objects are registered with the EventManager at creation time but they
    are actually created/read only if one plotter object activates
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
                 read_entry_block=10000,
                 depends_on=[],
                 debug=0,
                 print_function=lambda df: df,
                 max_print_lines=-1,
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
        self.max_print_lines = max_print_lines
        self.weight_function = weight_function
        self.entries = None
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
            if event.entry_range[1] != -1:
                stride = min([self.read_entry_block, (1+event.entry_range[1]-event.global_entry), (event.tree.num_entries - event.file_entry)])
            else:
                stride = min([self.read_entry_block, (event.tree.num_entries - event.file_entry)])
            # print(f'[fill] stride: {stride}')
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
            df_print = ak.to_dataframe(self.df[event.file_entry])
            debugPrintOut(max(debug, self.debug), self.label,
                          toCount=df_print,
                          toPrint=self.print_function(df_print),
                          max_lines=self.max_print_lines)

    def fill_real(self, event, stride, weight_file=None, debug=0):
        self.df = self.filler_function(event, stride)
        if self.fixture_function is not None:
            self.df = self.fixture_function(self.df)
        if self.weight_function is not None:
            self.df = self.weight_function(self.df, weight_file)
        self.entries = range(0,1000)# FIXME: self.df.index.get_level_values('entry').unique()
        if debug > 2:
            print(f'read coll. {self.name} from entry: {event.file_entry} to entry: {event.file_entry+stride} (stride: {stride}), # rows: {len(self.df)}, # entries: {len(self.entries)}')




def cl3d_fixtures(clusters):
    # print(clusters.show())

    # print(clusters)
    # print(clusters.type.show())
    # print(clusters.energy)

    mask_loose = 0b0010
    mask_tight = 0b0001
    clusters['IDTightEm'] = np.bitwise_and(clusters.hwQual, mask_tight) > 0
    clusters['IDLooseEm'] = np.bitwise_and(clusters.hwQual, mask_loose) > 0
    clusters['eMax'] = clusters.emaxe*clusters.energy
    clusters['meanz_scaled'] = clusters.meanz-320.
    clusters['abseta'] =  np.abs(clusters.eta)

    if False:
        input_array = ak.flatten(
            clusters[[
                'coreshowerlength', 
                'showerlength', 
                'firstlayer', 
                'maxlayer', 
                'szz', 
                'srrmean', 
                'srrtot', 
                'seetot', 
                'spptot']], 
            axis=1)
        input_data = ak.concatenate(ak.unzip(input_array[:, np.newaxis]), axis=1)
        input_matrix = xgboost.DMatrix(np.asarray(input_data))
        score =  classifiers.eg_hgc_model_xgb.predict(input_matrix)

        pu_input_array = ak.flatten(
            clusters[[
                'eMax', 
                'emaxe', 
                'spptot', 
                'srrtot', 
                'ntc90']], 
            axis=1)
        pu_input_data = ak.concatenate(ak.unzip(pu_input_array[:, np.newaxis]), axis=1)
        pu_input_matrix = xgboost.DMatrix(np.asarray(pu_input_data))
        pu_score =  classifiers.pu_veto_model_xgb.predict(pu_input_matrix)

        counts = ak.num(clusters)
        clusters_flat = ak.flatten(clusters)
        clusters_flat['egbdtscore'] = score
        clusters_flat['pubdtscore'] = pu_score

        clusters_flat['egbdtscoreproba'] = -np.log(1.0/score - 1.0)
        clusters_flat['pubdtscoreproba'] = -np.log(1.0/pu_score - 1.0)


        clusters = ak.unflatten(clusters_flat, counts)
        # print(clusters.type.show())

    return clusters


def gen_fixtures(particles, mc_particles):
    # if particles.empty:
    #     return particles
    # print particles.columns
    particles['pdgid'] = particles.pid
    particles['abseta'] = np.abs(particles.eta)
    particles['firstmother_pdgid'] = mc_particles.df.pdgid[particles[particles.gen != -1].gen-1]
    return particles


def mc_fixtures(particles):
    particles['abseta'] = np.abs(particles.eta)
    return particles

def ele_mc_fixtures(particles):
    if 'pdgid' not in particles.fields:
        particles['pdgid'] = particles.charge*11
    return mc_fixtures(particles)

def pho_mc_fixtures(particles):
    if 'pdgid' not in particles.fields:
        particles['pdgid'] = 22
    return mc_fixtures(particles)

def pi_mc_fixtures(particles):
    if 'pdgid' not in particles.fields:
        particles['pdgid'] = particles.charge*211
    return mc_fixtures(particles)

def tc_fixtures(tcs):
    # print tcs.columns
    tcs['ncells'] = 1
    if not tcs.empty:
        tcs['cells'] = tcs.apply(func=lambda x: [int(x.id)], axis=1)
        tcs['abseta'] = np.abs(tcs.eta)
    # tcs['xproj'] = tcs.x/tcs.z
    # tcs['yproj'] = tcs.y/tcs.z
    return tcs


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
    entries = tracks.df.index.get_level_values('entry').union(egs.df.index.get_level_values('entry')).unique()
    
    data = []
    index = []
    for entry in entries:
        subentry = 0
        kd_tree1 = cKDTree(egs.df.loc[entry][['eta', 'phi']])
        kd_tree2 = cKDTree(tracks.df.loc[entry][['caloeta', 'calophi']])
        sdm = kd_tree1.sparse_distance_matrix(kd_tree2, 0.2)
        for key,dr in sdm.items():
            eg_match = egs.df.loc[(entry, key[0])]
            tk_match = tracks.df.loc[(entry, key[1])]
            index.append((entry, subentry))
            data_entry=[
                eg_match.pt_em, 
                eg_match.eta, 
                eg_match.phi, 
                eg_match.quality, 
                eg_match.bdteg, 
                eg_match.bdt_pu, 
                tk_match.pt,
                tk_match.eta,
                tk_match.phi,
                tk_match.caloeta,
                tk_match.calophi,
                tk_match.z0,
                tk_match.chi2,
                tk_match.chi2Red,
                dr,
                key[0],
                key[1]]

            data.append(data_entry)
            subentry += 1
    
    newcolumns = ['pt', 'eta', 'phi', 'quality', 'bdteg', 'bdt_pu']
    newcolumns.extend(['tkpt', 'tketa', 'tkphi', 'tkcaloeta', 'tkcalophi', 'tkz0', 'tkchi2', 'tkchi2red', 'dr', 'clidx', 'tkidx'])
    
    newindex = pd.MultiIndex.from_tuples(index, names=egs.df.index.names)
    matched_egs = pd.DataFrame(data=data, columns=newcolumns, index=newindex)

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


def quality_flags(objs):
    # print(objs.hwQual)
    objs['hwQual'] = ak.values_astype(objs.hwQual, np.int32)
    mask_tight_sta = 0b0001
    mask_tight_ele = 0b0010
    mask_tight_pho = 0b0100
    mask_no_brem = 0b1000
    objs['IDTightSTA'] = np.bitwise_and(objs.hwQual, mask_tight_sta) > 0
    objs['IDTightEle'] = np.bitwise_and(objs.hwQual, mask_tight_ele) > 0
    objs['IDTightPho'] = np.bitwise_and(objs.hwQual, mask_tight_pho) > 0
    objs['IDNoBrem'] = np.bitwise_and(objs.hwQual, mask_no_brem) > 0
    objs['IDBrem'] = np.bitwise_and(objs.hwQual, mask_no_brem) == 0
    return objs

def quality_ele_fixtures(objs):
    # print(objs)
    objs['dpt'] = objs.tkPt - objs.pt
    return quality_flags(objs)


def print_columns(df):
    print(df.fields)
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


def build_double_obj(obj):
    ret = ak.combinations(
        array=obj, 
        n=2, 
        axis=1,
        fields=['leg0', 'leg1'])
    # ret.show()
    return ret

def highest_pt(objs, num=2):
    sel_objs = objs[objs.prompt >= 2]
    index = ak.argsort(sel_objs.pt)
    array = sel_objs[index]
    # print (ak.local_index(array))
    return array[ak.local_index(array.pt, axis=1)<num]



calib_mgr = calib.CalibManager()

# --- FP collections

gen_ele = DFCollection(
    name='GEN', label='GEN particles (ele)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='GenEl', entry_block=entry_block),
    fixture_function=ele_mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    max_print_lines=None,
    debug=0)
# gen_ele.activate()


gen_highestpt_ele = DFCollection(
    name='GEN', label='GEN particles (ele highest-pT)',
    filler_function=lambda event, entry_block: highest_pt(gen_ele.df),
    # fixture_function=mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    max_print_lines=None,
    depends_on=[gen_ele],
    debug=0)
# gen_highestpt_ele.activate()


gen_pho = DFCollection(
    name='GEN', label='GEN particles (pho)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='GenPh', entry_block=entry_block),
    fixture_function=pho_mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    max_print_lines=None,
    debug=0)

gen_pi = DFCollection(
    name='GEN', label='GEN particles (pi)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='GenPi', entry_block=entry_block),
    fixture_function=pi_mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    max_print_lines=None,
    debug=0)


gen = DFCollection(
    name='GEN', label='GEN particles',
    filler_function=lambda event, entry_block: ak.concatenate([gen_ele.df, gen_pho.df], axis=1),
    # fixture_function=mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    depends_on=[gen_ele, gen_pho],
    max_print_lines=None,
    debug=0)
# gen.activate()


gen_jet = DFCollection(
    name='GEN', label='GEN jets',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='GenJets', entry_block=entry_block),
    fixture_function=mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    max_print_lines=None,
    debug=0)


hgc_cl3d = DFCollection(
    name='HGCCl3d', label='HGC Cl3d',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='HGCal3DCl', entry_block=entry_block, fallback='HMvDR'),
    fixture_function=lambda clusters: cl3d_fixtures(clusters),
    # read_entry_block=500,
    debug=0,
    # print_function=lambda df: df[['rho', 'eta', 'phi', 'hwQual', 'ptEm', 'egbdtscore', 'pubdtscore', 'egbdtscoreproba', 'pubdtscoreproba', 'pfPuIdScore', 'egEmIdScore']].sort_values(by='rho', ascending=False)
    print_function=lambda df: df.columns

    )


tracks = DFCollection(
    name='L1Trk', label='L1Track',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='l1Trk', entry_block=entry_block),
    print_function=lambda df: df.sort_values(by='pt', ascending=False)[:10],
    debug=0)


TkEleEE = DFCollection(
    name='TkEleEE', label='TkEle EE',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEleEE', entry_block=entry_block),
    fixture_function=quality_ele_fixtures,
    print_function=lambda df:df.columns,
    debug=0)


TkEleEB = DFCollection(
    name='TkEleEB', label='TkEle EB',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEleEB', entry_block=entry_block),
    fixture_function=quality_ele_fixtures,
    debug=0)

TkEleEllEE = DFCollection(
    name='TkEleEllEE', label='TkEle EE (Ell.)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEleEllEE', entry_block=entry_block),
    fixture_function=quality_ele_fixtures,
    debug=0)

TkEmEE = DFCollection(
    name='TkEmEE', label='TkEm EE',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEmEE', entry_block=entry_block),
    print_function=lambda df: df.loc[(abs(df.eta) > 2.4), ['energy', 'pt', 'eta', 'phi','hwQual']].sort_values(by='pt', ascending=False)[:10],
    fixture_function=quality_flags,
    debug=0)

TkEmEB = DFCollection(
    name='TkEmEB', label='TkEm EB',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEmEB', entry_block=entry_block),
    fixture_function=quality_flags,
    # read_entry_block=200,
    debug=0)

TkEmL2 = DFCollection(
    name='TkEmL2', label='TkEm L2',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEmL2', entry_block=entry_block),
    fixture_function=quality_flags,
    debug=0)

# -- FP
TkEleL2 = DFCollection(
    name='TkEleL2', label='TkEle L2',
    filler_function=lambda event, entry_block : event.getDataFrame(
        prefix='TkEleL2', entry_block=entry_block, fallback='L2TkEle'),
    fixture_function=quality_ele_fixtures,
    debug=0)

TkEmL2Ell = DFCollection(
    name='TkEmL2Ell', label='TkEm L2 (ell.)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='L2TkEmEll', entry_block=entry_block),
    fixture_function=quality_flags,
    debug=0)

TkEleL2Ell = DFCollection(
    name='TkEleL2Ell', label='TkEle L2 (ell.)',
    filler_function=lambda event, entry_block : event.getDataFrame(
        prefix='L2TkEleEll', entry_block=entry_block, fallback='TkEleL2Ell'),
    fixture_function=quality_ele_fixtures,
    debug=0)

DoubleTkEleL2 = DFCollection(
    name='DoubleTkEleL2', label='DoubleTkEle L2',
    filler_function=lambda event, entry_block: build_double_obj(obj=TkEleL2.df),
    # fixture_function=,
    depends_on=[TkEleL2],
    debug=0)

DoubleTkEmL2 = DFCollection(
    name='DoubleTkEmL2', label='DoubleTkEm L2',
    filler_function=lambda event, entry_block: build_double_obj(obj=TkEmL2.df),
    # fixture_function=,
    depends_on=[TkEmL2],
    debug=0)

EGStaEE = DFCollection(
    name='EGStaEE', label='EG EE',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='EGStaEE', entry_block=entry_block),
    print_function=lambda df: df.loc[(abs(df.eta) > 2.4), ['energy', 'pt', 'eta', 'phi','hwQual']].sort_values(by='pt', ascending=False)[:10],
    # fixture_function=mapcalo2pfregions,
    fixture_function=quality_flags,
    debug=0)


EGStaEB = DFCollection(
    name='EGStaEB', label='EG EB',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='EGStaEB', entry_block=entry_block),
    # print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
    fixture_function=quality_flags,
    # read_entry_block=200,
    debug=0)

decTk = DFCollection(
    name='PFDecTk', label='decoded Tk',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='pfdtk', entry_block=entry_block),
    fixture_function=decodedTk_fixtures,
    debug=0)

tkCl3DMatch = DFCollection(
    name='TkCl3DMatch', label='TkCl3DMatch',
    filler_function=lambda event, entry_block: get_trackmatched_egs(egs=hgc_cl3d, tracks=tracks),
    fixture_function=mapcalo2pfregions_in,    
    depends_on=[hgc_cl3d, tracks],
    debug=0)


hgc_cl3d_pfinputs = DFCollection(
    name='HGCCl3dPfIN', label='HGC Cl3d L1TC IN',
    filler_function=lambda event, entry_block: hgc_cl3d.df,
    fixture_function=mapcalo2pfregions_in,
    depends_on=[hgc_cl3d],
    debug=0)

EGStaEB_pfinputs = DFCollection(
    name='EGStaEBPFin', label='EG EB  L1TC IN',
    filler_function=lambda event, entry_block: EGStaEB.df,
    fixture_function=mapcalo2pfregions_in,
    depends_on=[EGStaEB],
    print_function=lambda df: df.loc[~(df.eta_reg_4 | df.eta_reg_5 | df.eta_reg_6 | df.eta_reg_7 | df.eta_reg_8 | df.eta_reg_9), ['eta', 'phi', 'eta_reg_0', 'eta_reg_1', 'eta_reg_2', 'eta_reg_3', 'eta_reg_4', 'eta_reg_5', 'eta_reg_6', 'eta_reg_7', 'eta_reg_8', 'eta_reg_9', 'eta_reg_10', 'eta_reg_11', 'eta_reg_12', 'eta_reg_13']].sort_values(by='eta', ascending=False),
    debug=0)

TkEleEB_pf_reg = DFCollection(
    name='PFOuttkEleEB', label='TkEle EB (old EMU)',
    filler_function=lambda event, entry_block: tkeles_EB_pf.df,
    fixture_function=mapcalo2pfregions_out,
    depends_on=[TkEleEB],
    debug=0)

tk_pfinputs = DFCollection(
    name='L1TrkPfIn', label='L1Track Input',
    filler_function=lambda event, entry_block: tracks.df,
    fixture_function=maptk2pfregions_in,
    depends_on=[tracks],
    debug=0)

pfjets = DFCollection(
    name='PFJets', label='PFJets',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='L1PFJets', entry_block=entry_block),
    print_function=lambda df: df.sort_values(by='pt', ascending=False)[:10],
    debug=0)

# --------------


# -- FP


sim_parts = DFCollection(
    name='SIM', label='SIM particles',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='simpart', entry_block=entry_block),
    fixture_function=lambda gen_parts: gen_fixtures(gen_parts, gen),
    # read_entry_block=10,
    depends_on=[gen],
    debug=0,
    # print_function=lambda df: df[['eta', 'phi', 'pt', 'energy', 'mother', 'fbrem', 'ovz', 'pid', 'gen', 'reachedEE', 'firstmother_pdgid']],
    print_function=lambda df: df[['gen', 'pid', 'pt', 'eta', 'phi', 'mother', 'reachedEE', 'ovz', 'dvz', ]],
    max_print_lines=None,
    # print_function=lambda df: df.columns,
    weight_function=gen_part_pt_weights)

gen_parts = DFCollection(
    name='GEN', label='GEN particles',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='genpart', entry_block=entry_block),
    fixture_function=lambda gen_parts: gen_fixtures(gen_parts, gen),
    # read_entry_block=10,
    depends_on=[gen],
    debug=0,
    # print_function=lambda df: df[['eta', 'phi', 'pt', 'energy', 'mother', 'fbrem', 'ovz', 'pid', 'gen', 'reachedEE', 'firstmother_pdgid']],
    print_function=lambda df: df[['gen', 'pid',  'pt', 'eta', 'phi', 'mother', 'ovz', 'dvz', 'reachedEE']].sort_values(by='mother', ascending=False),
    max_print_lines=None,
    # print_function=lambda df: df.columns,
    weight_function=gen_part_pt_weights)


tcs = DFCollection(
    name='TC', label='Trigger Cells',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='tc', entry_block=entry_block),
    read_entry_block=200,
    fixture_function=tc_fixtures, debug=0)

# try to cleanup a bit



# --------


def dy_gen_selection(gen):
    vec_bos = gen[gen.pdgid == 23]
    print(vec_bos.status)
    print(gen)


selected_gen_parts = DFCollection(
    name='SelectedSimParts', label='Double Sim e/g',
    filler_function=lambda event, entry_block: dy_gen_selection(gen.df),
    # fixture_function=,
    depends_on=[gen],
    debug=0)


SelectedSimParts = DFCollection(
    name='SelectedSimParts', label='Double Sim e/g',
    filler_function=lambda event, entry_block: sim_parts.df[selections.Selector('^GEN$').one().selection(sim_parts.df)],
    # fixture_function=,
    depends_on=[sim_parts],
    debug=0)


DoubleSimEle = DFCollection(
    name='DoubleSimEle', label='Double Sim e/g',
    filler_function=lambda event, entry_block: build_double_obj(obj=gen_highestpt_ele.df),
    # fixture_function=,
    depends_on=[gen_highestpt_ele],
    debug=0)
# DoubleSimEle.activate()





# tkCl3DMatch.activate()



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


# tp_hm_vdr = TPSet(tcs, tcs, cl3d_hm)
