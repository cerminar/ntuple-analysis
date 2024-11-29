"""
Definines and instantiate the plotter classes.

These are the classes where the analysis logic is implemented.
The plotter classes need to implement a standard interface:
init:
    accept one or more DFCollection (or TPSet) and a corresponding list of selections.

book_histos:
    activates the relevant DFCollection instances and books the HistoClasses defined in the
    `l1THistos` module for each combination of the object and selections.
     The naming convention of the histograms uses the namse of the objects,
     the name of the selection and the name of the gen-matched object (if any).

fill_histos:
    actually implements the analysis logic, running the selections on the input
    collections and filling the histograms.

Several collections of plotters are also instantiated. Which one will actually be run
is steered via the configuration file.
"""
import awkward as ak
import numpy as np
import pandas as pd
import ROOT
import math 

from . import calibrations as calib
from . import clusterTools as clAlgo
from . import histos as histos
from . import selections as selections
from . import utils as utils

from rich import print as pprint
from rich.console import Console
from rich.table import Table
# import collections as collections
# ROOT.gROOT.ProcessLine('#include "src/fastfilling.h"')


class Test:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'<TEST class, {self.name}>'


class BasePlotter:
    def __init__(self, data_set, data_selections, gen_set=None, gen_selections=None):
        self.data_set = data_set
        self.data_selections = data_selections
        self.gen_set = gen_set
        self.gen_selections = gen_selections

    @property
    def tp_set(self):
        return self.data_set

    @property
    def tp_selections(self):
        return self.data_selections

    def get_histo_primitives(self):
        histo_primitives = pd.DataFrame(columns=['data', 'data_sel', 'gen_sel', 'data_label', 'data_sel_label', 'gen_sel_label'])
        gen_sel_names = ['nomatch']
        gen_sel_labels = ['']
        if self.gen_selections is not None:
            gen_sel_names = [sel.name for sel in self.gen_selections]
            gen_sel_labels = [sel.label for sel in self.gen_selections]

        for data_sel in self.data_selections:
            for idx, gen_sel_name in enumerate(gen_sel_names):
                histo_primitives = histo_primitives.append({'data': self.data_set.name,
                                                            'data_sel': data_sel.name,
                                                            'gen_sel': gen_sel_name,
                                                            'data_label': self.data_set.label,
                                                            'data_sel_label': data_sel.label,
                                                            'gen_sel_label': gen_sel_labels[idx]},
                                                           ignore_index=True)
        return histo_primitives

    def fill_histos(self, debug=0):
        pass

    def fill_histos_event(self, idx, debug=0):
        if self.data_set.new_read:
            self.fill_histos(debug)

    def __repr__(self):
        return f'<{self.__class__.__name__}, ds: {self.data_set}, ds_sel: {self.data_selections}, g: {self.gen_set}, g_sel: {self.gen_selections} >'
    # def change_genpart_selection(self, newselection):
    #     """Allow customization of gen selection per sample."""
    #     if self.gen_selections is not None:
    #         self.gen_selections = newselection
    #
    # def __repr__(self):
    #     if  self.gen_selections is not None:
    #         return '<{}, tp: {}, gen_sel: {}>'.format(self.__class__.__name__,
    #                                                   self.data_set.name,
    #                                                   self.gen_selections[0].selection)
    #     else:
    #         return '<{}, tp: {}>'.format(self.__class__.__name__,
    #                                      self.data_set.name)
    def print(self):
        table = Table(title=f'* Plotter: {self.__class__.__name__}', title_justify='left',title_style='bold on blue')
        table.add_column('Data coll.', justify='right', style='cyan', no_wrap=True)
        table.add_column('Data sel.', justify='left', style='cyan', no_wrap=True)
        # table.add_column('Histo class', style='blue')

        table.add_row(self.data_set.name, 
                      ('\n').join(['- '+s.name for s in self.data_selections]),
                    #   self.HistoClass.__name__
                      )
        console = Console()
        console.print(table)

        

class GenericDataFramePlotter(BasePlotter):
    def __init__(self, HistoClass, data_set, selections=[selections.Selection('all')], pt_bins=None):
        self.HistoClass = HistoClass
        self.h_set = {}
        self.pt_bins = pt_bins
        super(GenericDataFramePlotter, self).__init__(data_set, selections)

    def book_histos(self):
        self.data_set.activate()
        data_name = self.data_set.name
        for selection in self.data_selections:
            if self.pt_bins:
                self.h_set[selection.name] = self.HistoClass(
                    name=f'{data_name}_{selection.name}_nomatch',
                    pt_bins=self.pt_bins)
            else:
                self.h_set[selection.name] = self.HistoClass(
                    name=f'{data_name}_{selection.name}_nomatch')


    def fill_histos(self, debug=0):
        for data_sel in self.data_selections:
            data = self.data_set.df
            if not data_sel.all:
                data = data[data_sel.selection(data)]
            self.h_set[data_sel.name].fill(data)

    def print(self):
        table = Table(title=f'* Plotter: {self.__class__.__name__}', title_justify='left',title_style='bold on blue')
        table.add_column('Data coll.', justify='right', style='cyan', no_wrap=True)
        table.add_column('Data sel.', justify='left', style='cyan', no_wrap=True)
        table.add_column('Histo class', style='blue')

        table.add_row(self.data_set.name, 
                      ('\n').join(['- '+s.name for s in self.data_selections]),
                      self.HistoClass.__name__
                      )
        console = Console()
        console.print(table)


class GenPlotter(GenericDataFramePlotter):
    def __init__(self, gen_set, gen_selections=[selections.Selection('all')], pt_bins=None):
        super(GenPlotter, self).__init__(
            histos.GenParticleExtraHistos,
            gen_set,
            gen_selections,
            pt_bins)


class TkElePlotter(GenericDataFramePlotter):
    def __init__(self, tkeg_set, tkeg_selections=[selections.Selection('all')]):
        super(TkElePlotter, self).__init__(histos.TkEleHistos, tkeg_set, tkeg_selections)


class TkEmPlotter(GenericDataFramePlotter):
    def __init__(self, tkeg_set, tkeg_selections=[selections.Selection('all')]):
        super(TkEmPlotter, self).__init__(histos.TkEmHistos, tkeg_set, tkeg_selections)


class TkEGPlotter(GenericDataFramePlotter):
    def __init__(self, tkeg_set, tkeg_selections=[selections.Selection('all')]):
        super(TkEGPlotter, self).__init__(histos.TkEGHistos, tkeg_set, tkeg_selections)


class TrackPlotter(GenericDataFramePlotter):
    def __init__(self, trk_set, track_selections=[selections.Selection('all')]):
        super(TrackPlotter, self).__init__(histos.TrackHistos, trk_set, track_selections)


class TTPlotter(GenericDataFramePlotter):
    def __init__(self, tt_set, tt_selections=[selections.Selection('all')]):
        super(TTPlotter, self).__init__(histos.TriggerTowerHistos, tt_set, tt_selections)


class DecTkPlotter(GenericDataFramePlotter):
    def __init__(self, tk_set, tk_selections=[selections.Selection('all')]):
        super(DecTkPlotter, self).__init__(histos.DecTkHistos, tk_set, tk_selections)


class TPPlotter(BasePlotter):
    def __init__(self, tp_set, tp_selections=[selections.Selection('all')]):
        # self.tp_set = tp_set
        # self.tp_selections = tp_selections
        self.h_tpset = {}
        super(TPPlotter, self).__init__(tp_set, tp_selections)

    def book_histos(self):
        self.tp_set.activate()
        tp_name = self.tp_set.name
        for selection in self.tp_selections:
            self.h_tpset[selection.name] = histos.HistoSetClusters(name=f'{tp_name}_{selection.name}_nomatch')

    def fill_histos(self, debug=0):
        # FIXME: migrate to the new query caching system
        for tp_sel in self.tp_selections:
            tcs = self.tp_set.tc_df
            cl2Ds = self.tp_set.cl2d_df
            cl3Ds = self.tp_set.cl3d_df
            if not tp_sel.all:
                cl3Ds = self.tp_set.cl3d_df.query(tp_sel.selection)
            # debug = 4
            # utils.debugPrintOut(debug, '{}_{}'.format(self.tp_set.name, 'TCs'), tcs, tcs[:3])
            # utils.debugPrintOut(debug, '{}_{}'.format(self.tp_set.name, 'CL2D'), cl2Ds, cl2Ds[:3])
            # utils.debugPrintOut(debug, '{}_{}'.format(self.tp_set.name, 'CL3D'), cl3Ds, cl3Ds[:3])
            if not cl3Ds.empty and not cl2Ds.empty and not tcs.empty:
                self.h_tpset[tp_sel.name].fill(tcs, cl2Ds, cl3Ds)


class TPGenMatchPlotter(BasePlotter):
    def __init__(self, tp_set, gen_set,
                 tp_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')],
                 extended_range=False):
        # self.tp_set = tp_set
        # self.tp_selections = tp_selections
        # self.gen_set = gen_set
        # self.gen_selections = gen_selections
        self.h_tpset = {}
        self.h_resoset = {}
        self.h_effset = {}
        self.h_conecluster = {}
        self.extended_range = extended_range
        super(TPGenMatchPlotter, self).__init__(
            tp_set,
            tp_selections,
            gen_set,
            selections.multiply_selections(
                gen_selections,
                [selections.Selection('', '', 'gen > 0')]))

    def plot3DClusterMatch(self,
                           genParticles,
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
                           histoConeClusters,
                           algoname,
                           debug):
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

        def sumClustersInCone(all3DClusters, idx_incone, debug=0):
            components = all3DClusters[all3DClusters.index.isin(idx_incone)]
            ret = clAlgo.sum3DClusters(components)
            if debug > 0:
                print('-------- in cone:')
                print(components.sort_values(by='pt', ascending=False))
                print('   - Cone sum:')
                print(ret)
            return ret

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

                if False:
                    if 'energyCentral' not in matched3DCluster.columns:
                        calib_factor = 1.084
                        matched3DCluster['energyCentral'] = [matchedClusters[(matchedClusters.layer > 9) & (matchedClusters.layer < 21)].energy.sum()*calib_factor]

                if False:
                    iso_df = computeIsolation(trigger3DClusters,
                                              idx_best_match=best_match_indexes[idx],
                                              idx_incone=allmatches[idx], dr=0.2)
                    matched3DCluster['iso0p2'] = iso_df.energy
                    matched3DCluster['isoRel0p2'] = iso_df.pt/matched3DCluster.pt

                # fill the plots
                histoTCMatch.fill(matchedTriggerCells)
                histoClMatch.fill(matchedClusters)
                histo3DClMatch.fill(matched3DCluster)

                # print matchedClusters
                # print matchedClusters.layer.unique()
                for layer in matchedClusters.layer.unique():
                    if histoReso2D is not None:
                        histoReso2D.fill(reference=genParticle,
                                         target=clAlgo.build2D(matchedClusters[matchedClusters.layer == layer]))

                if False:
                    histoReso2D.fill(reference=genParticle, target=matchedClusters)
                histoReso.fill(reference=genParticle, target=matched3DCluster)

                if False:
                    # now we fill the reso plot for all the clusters in the cone
                    clustersInCone = sumClustersInCone(trigger3DClusters, allmatches[idx])

                    def fill_cluster_incone_histos(cl3ds,
                                                   idx_allmatches,
                                                   idx_bestmatch,
                                                   charge,
                                                   h_clustersInCone,
                                                   debug=0):
                        if debug > 4:
                            print(f'- best match: {idx_bestmatch}, all matches: {idx_allmatches}')
                        bestcl = cl3ds.loc[idx_bestmatch]
                        h_clustersInCone.fill_n(len(idx_allmatches)-1)
                        for idx in idx_allmatches:
                            if idx == idx_bestmatch:
                                continue
                            clincone = cl3ds.loc[idx]
                            h_clustersInCone.fill(reference=bestcl,
                                                  target=clincone,
                                                  charge=charge)
                    # print genParticle
                    # print genParticle.pid/abs(genParticle.pid)
                    fill_cluster_incone_histos(trigger3DClusters,
                                               allmatches[idx],
                                               best_match_indexes[idx],
                                               genParticle.pid/abs(genParticle.pid),
                                               histoConeClusters)

                    # print ('----- in cone sum:')
                    # print (clustersInCone)
                    # histoResoCone.fill(reference=genParticle, target=clustersInCone.iloc[0])

                if histoGenMatched is not None:
                    histoGenMatched.fill(genParticles.loc[[idx]])

                if debug >= 6:
                    print(f'--- Dump match for algo {algoname} ---------------')
                    print(f'GEN particle: idx: {idx}')
                    print(genParticle)
                    print('Matched to 3D cluster:')
                    print(matched3DCluster)
                    print('Matched 2D clusters:')
                    print(matchedClusters)
                    print('matched cells:')
                    print(matchedTriggerCells)

                    print(f'3D cluster energy: {matched3DCluster.energy.sum()}')
                    print(f'3D cluster pt: {matched3DCluster.pt.sum()}')
                    calib_factor = 1.084
                    print(f'sum 2D cluster energy: {matchedClusters.energy.sum() * calib_factor}')
                    # print ('sum 2D cluster pt: {}'.format(matchedClusters.pt.sum()*calib_factor))
                    print(f'sum TC energy: {matchedTriggerCells.energy.sum()}')
                    print('Sum of matched clusters in cone:')
                    print(clustersInCone)
            elif debug >= 5:
                print(f'==== Warning no match found for algo {algoname}, idx {idx} ======================')
                if debug >= 2:
                    print(genParticle)
                    print(trigger3DClusters)

        # if len(allmatched2Dclusters) != 0:
        #     matchedClustersAll = pd.concat(allmatched2Dclusters)
        # return matchedClustersAll

    def book_histos(self):
        self.gen_set.activate()
        self.tp_set.activate()
        for tp_sel in self.tp_selections:
            for gen_sel in self.gen_selections:
                histo_name = f'{self.tp_set.name}_{tp_sel.name}_{gen_sel.name}'
                self.h_tpset[histo_name] = histos.HistoSetClusters(histo_name)
                self.h_resoset[histo_name] = histos.HistoSetReso(histo_name)
                self.h_effset[histo_name] = histos.HistoSetEff(histo_name, extended_range=self.extended_range)
                self.h_conecluster[histo_name] = histos.ClusterConeHistos(histo_name)

    def fill_histos(self, debug=0):
        for tp_sel in self.data_selections:
            tcs = self.tp_set.tc_df
            cl2Ds = self.tp_set.cl2d_df
            cl3Ds = self.tp_set.cl3ds.query(tp_sel)
            for gen_sel in self.gen_selections:
                genReference = self.gen_set.query(gen_sel)
                histo_name = f'{self.tp_set.name}_{tp_sel.name}_{gen_sel.name}'

                h_tpset_match = self.h_tpset[histo_name]
                h_resoset = self.h_resoset[histo_name]
                h_genseleff = self.h_effset[histo_name]
                h_conecl = self.h_conecluster[histo_name]
                # print 'TPsel: {}, GENsel: {}'.format(tp_sel.name, gen_sel.name)
                # print cl3Ds
                # print cl2Ds
                # print tcs
                self.plot3DClusterMatch(genReference,
                                        cl3Ds,
                                        cl2Ds,
                                        tcs,
                                        h_genseleff.h_den,
                                        h_genseleff.h_num,
                                        h_tpset_match.htc,
                                        h_tpset_match.hcl2d,
                                        h_tpset_match.hcl3d,
                                        h_resoset.hreso,
                                        h_resoset.hresoCone,
                                        h_resoset.hreso2D,
                                        h_conecl,
                                        self.tp_set.name,
                                        debug)

    def __repr__(self):
        return f'<{self.__class__.__name__} tps: {self.tp_set.name}, tps_s: {[sel.name for sel in self.tp_selections]}, gen:{self.gen_set.name}, gen_s:{[sel.name for sel in self.gen_selections]}> '


class GenericGenMatchPlotter(BasePlotter):
    def __init__(self, ObjectHistoClass, ResoHistoClass,
                 data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')],
                 gen_eta_phi_columns=('caloeta', 'calophi'),
                 drcut=0.1,
                 pt_bins=None):
        self.ObjectHistoClass = ObjectHistoClass
        self.ResoHistoClass = ResoHistoClass
        # self.data_set = data_set
        # self.data_selections = data_selections
        # self.gen_set = gen_set
        # self.gen_selections = gen_selections
        self.h_dataset = {}
        self.h_resoset = {}
        self.h_effset = {}
        self.gen_eta_phi_columns = gen_eta_phi_columns
        self.dr2 = drcut*drcut
        self.pt_bins = pt_bins

        super(GenericGenMatchPlotter, self).__init__(
            data_set,
            data_selections,
            gen_set,
            gen_selections)

        # print self
        # print gen_selections

    def plotObjectMatch(self,
                        gen,
                        objects,
                        h_gen,
                        h_gen_matched,
                        h_object_matched,
                        h_reso,
                        algoname,
                        debug):


        # fill histo with all selected GEN particles before any match
        h_gen.fill(gen)

        #FIXME: use utils.gen_match 

        # perform the matching
        match_eta = ak.cartesian([objects.eta, gen[self.gen_eta_phi_columns[0]]])
        match_phi = ak.cartesian([objects.phi, gen[self.gen_eta_phi_columns[1]]])
        match_pt = ak.cartesian([objects.pt, gen.pt])
        match_idx = ak.argcartesian([objects.eta, gen.eta])

        obj_eta, gen_eta = ak.unzip(match_eta)

        obj_phi, gen_phi = ak.unzip(match_phi)
        obj_pt, gen_pt = ak.unzip(match_pt)
        obj_idx, gen_idx = ak.unzip(match_idx)
        dpt = np.abs(obj_pt - gen_pt)

        # dphi = obj_phi-gen_phi
        dphi = utils.angle_range(obj_phi-gen_phi)
        # dphi = ak.where(dphi>np.pi, dphi-2*np.pi, dphi)
        # dphi = np.where(dphi<-np.pi, dphi+2*np.pi, dphi)
        dr2 = (obj_eta-gen_eta)**2+(dphi)**2
        match = ak.Array(data={'ele_idx': obj_idx, 'gen_idx': gen_idx, 'dpt': dpt, 'dr2': dr2})
        dr_match=match[match.dr2<self.dr2]
        for genid in np.unique(ak.flatten(dr_match.gen_idx)):
            gen_match_id = dr_match[dr_match.gen_idx == genid]
            dpt_min_index = ak.argmin(gen_match_id.dpt, axis=1, keepdims=True)
            best_match_id = gen_match_id[dpt_min_index]
            matched_obj = objects[best_match_id.ele_idx]
            matched_gen = gen[best_match_id.gen_idx]
            h_object_matched.fill(matched_obj)
            if h_gen_matched is not None:
                h_gen_matched.fill(matched_gen)
            h_reso.fill(reference=matched_gen,
                        target=matched_obj)
            # FIXME: [AWKWARD]
            # if hasattr(h_reso, 'fill_nMatch'):
            #     h_reso.fill_nMatch(len(allmatches[idx]))


    def book_histos(self):
        self.gen_set.activate()
        self.data_set.activate()
        # print(f'# data sel: {len(self.data_selections)} x # gen sel: {len(self.gen_selections)} = {len(self.data_selections)*len(self.gen_selections)}')
        for tp_sel in self.data_selections:
            for gen_sel in self.gen_selections:
                histo_name = f'{self.data_set.name}_{tp_sel.name}_{gen_sel.name}'
                self.h_dataset[histo_name] = self.ObjectHistoClass(histo_name)
                self.h_resoset[histo_name] = self.ResoHistoClass(histo_name)
                self.h_effset[histo_name] = histos.HistoSetEff(histo_name, pt_bins=self.pt_bins)

    def fill_histos(self, debug=0):
        # FIXME: we need to reduce the # of jugged dimensions for the selection slicing to work in AWKWARD....
        # print(self.gen_set.df.fields)
        # gen = self.gen_set.df[['eta', 'abseta', 'phi', 'pt', 'energy', 'exeta', 'exphi', 'fbrem', 'gen', 'pid', 'reachedEE', 'pdgid', 'ovx', 'ovy', 'ovz']]
        gen = self.gen_set.df

        for tp_sel in self.data_selections:
            # print(tp_sel)
            if tp_sel.all:
                # FIXME: workaround for now
                objects = self.data_set.df
            else:
                objects = self.data_set.df[tp_sel.selection(self.data_set.df)]
            for gen_sel in self.gen_selections:
                # print(gen_sel)
                # print(gen_sel.selection(gen))
                if gen_sel.all:
                    genReference = gen
                else:
                    genReference = gen[gen_sel.selection(gen)]
                histo_name = f'{self.data_set.name}_{tp_sel.name}_{gen_sel.name}'
                # print (histo_name)
                # print (f'# data: {objects.shape[0]}')
                # print (f'# gen: {genReference.shape[0]}')
                # print (genReference)
                h_obj_match = self.h_dataset[histo_name]
                h_resoset = self.h_resoset[histo_name]
                h_genseleff = self.h_effset[histo_name]
                self.plotObjectMatch(genReference,
                                     objects,
                                     h_genseleff.h_den,
                                     h_genseleff.h_num,
                                     h_obj_match,
                                     h_resoset,
                                     self.data_set.name,
                                     debug)

    def print(self):
        table = Table(title=f'* Plotter: {self.__class__.__name__}', title_justify='left',title_style='bold on blue')
        table.add_column('Data coll.', justify='right', style='cyan', no_wrap=True)
        table.add_column('Data sel.', justify='left', style='cyan', no_wrap=True)
        table.add_column('GEN coll.', style='blue')
        table.add_column('GEN sel.', style='blue', justify='left')
        table.add_column('DR cut', style='blue', justify='left')
        table.add_column('GEN match', style='blue', justify='left')

        table.add_row(self.data_set.name, 
                      ('\n').join(['- '+s.name for s in self.data_selections]),
                      self.gen_set.name,
                      ('\n').join(['- '+s.name for s in self.gen_selections]),
                      str(math.sqrt(self.dr2)),
                      f'{self.gen_eta_phi_columns[0]},{self.gen_eta_phi_columns[1]}'
                      )
        console = Console()
        console.print(table)

class TrackGenMatchPlotter(GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')]):
        super(TrackGenMatchPlotter, self).__init__(histos.TrackHistos, histos.TrackResoHistos,
                                                   data_set, gen_set,
                                                   data_selections, gen_selections,
                                                   gen_eta_phi_columns=['eta', 'phi'])


class TrackGenMatchPlotter(GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')]):
        super(TrackGenMatchPlotter, self).__init__(histos.TrackHistos, histos.TrackResoHistos,
                                                   data_set, gen_set,
                                                   data_selections, gen_selections,
                                                   gen_eta_phi_columns=['eta', 'phi'])


class TkEGGenMatchPlotter(GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')]):
        super(TkEGGenMatchPlotter, self).__init__(histos.TkEGHistos, histos.EGResoHistos,
                                                  data_set, gen_set,
                                                  data_selections, gen_selections)



# class ResoNtupleMatchPlotter(BasePlotter):
#     def __init__(self, data_set, gen_set,
#                  data_selections=[selections.Selection('all')], gen_selections=[selections.Selection('all')]):

#         self.h_calibration = {}
#         super(ResoNtupleMatchPlotter, self).__init__(
#             data_set,
#             data_selections,
#             gen_set,
#             selections.multiply_selections(
#                 gen_selections,
#                 [selections.Selection('', '', 'gen > 0')]))

#         # print self
#         # print gen_selections

#     def plotObjectMatch(self,
#                         genParticles,
#                         objects,
#                         h_calibration,
#                         algoname,
#                         debug):
#         best_match_indexes = {}
#         if not objects.empty:
#             best_match_indexes, allmatches = utils.match_etaphi(genParticles[['exeta', 'exphi']],
#                                                                 objects[['eta', 'phi']],
#                                                                 objects['pt'],
#                                                                 deltaR=0.1)

#         for idx, genParticle in genParticles.iterrows():
#             if idx in best_match_indexes.keys():
#                 # print ('-----------------------')
#                 #  print(genParticle)
#                 obj_matched = objects.loc[[best_match_indexes[idx]]]
#                 # print obj_matched
#                 # print obj_matched.clusters
#                 # print obj_matched.clusters[0]
#                 # print algoname
#                 # print obj_matched[['energy', 'layer_energy']]
#                 h_calibration.fill(reference=genParticle, target=obj_matched)

#                 if debug >= 4:
#                     print(('--- Dump match for algo {} ---------------'.format(algoname)))
#                     print(('GEN particle: idx: {}'.format(idx)))
#                     print(genParticle)
#                     print('Matched to track object:')
#                     print(obj_matched)
#             else:
#                 if debug >= 5:
#                     print(('==== Warning no match found for algo {}, idx {} ======================'.format(algoname, idx)))
#                     print(genParticle)
#                     print(objects)

#     def book_histos(self):
#         self.gen_set.activate()
#         self.data_set.activate()
#         for tp_sel in self.data_selections:
#             for gen_sel in self.gen_selections:
#                 histo_name = '{}_{}_{}'.format(self.data_set.name, tp_sel.name, gen_sel.name)
#                 self.h_calibration[histo_name] = histos.ResoTuples(histo_name)

#     def fill_histos(self, debug=0):
#         for tp_sel in self.data_selections:
#             objects = self.data_set.query(tp_sel)
#             for gen_sel in self.gen_selections:
#                 genReference = self.gen_set.query(gen_sel)
#                 histo_name = '{}_{}_{}'.format(self.data_set.name, tp_sel.name, gen_sel.name)

#                 h_calib = self.h_calibration[histo_name]
#                 # print 'TPsel: {}, GENsel: {}'.format(tp_sel.name, gen_sel.name)
#                 self.plotObjectMatch(genReference,
#                                      objects,
#                                      h_calib,
#                                      self.data_set.name,
#                                      debug)


class CalibrationPlotter(BasePlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')], gen_selections=[selections.Selection('all')]):

        self.h_calibration = {}
        super(CalibrationPlotter, self).__init__(
            data_set,
            data_selections,
            gen_set,
            selections.multiply_selections(
                gen_selections,
                [selections.Selection('', '', 'gen > 0')]))

        # print self
        # print gen_selections

    def plotObjectMatch(self,
                        genParticles,
                        objects,
                        tcs,
                        h_calibration,
                        algoname,
                        debug):
        best_match_indexes = {}
        if not objects.empty:
            best_match_indexes, allmatches = utils.match_etaphi(genParticles[['exeta', 'exphi']],
                                                                objects[['eta', 'phi']],
                                                                objects['pt'],
                                                                deltaR=0.1)

        for idx, genParticle in genParticles.iterrows():
            if idx in best_match_indexes.keys():
                # print ('-----------------------')
                #  print(genParticle)
                obj_matched = objects.loc[[best_match_indexes[idx]]]
                # print obj_matched
                # print obj_matched.clusters
                # print obj_matched.clusters[0]
                components = tcs[tcs.id.isin(obj_matched.iloc[0].clusters)].copy()
                if 'layer_energy' not in obj_matched.columns:
                    layer_energy = []
                    for layer in range(1, 29, 2):
                        # components[components.layer == layer].energy.sum()
                        layer_energy.append(components[components.layer == layer].energy.sum())
                    obj_matched['layer_energy'] = [layer_energy]

                # print algoname
                # print obj_matched[['energy', 'layer_energy']]
                h_calibration.fill(reference=genParticle, target=obj_matched)

                if debug >= 4:
                    print(f'--- Dump match for algo {algoname} ---------------')
                    print(f'GEN particle: idx: {idx}')
                    print(genParticle)
                    print('Matched to track object:')
                    print(obj_matched)
            elif debug >= 5:
                print(f'==== Warning no match found for algo {algoname}, idx {idx} ======================')
                print(genParticle)
                print(objects)

    def book_histos(self):
        self.gen_set.activate()
        self.data_set.activate()
        for tp_sel in self.data_selections:
            for gen_sel in self.gen_selections:
                histo_name = f'{self.data_set.name}_{tp_sel.name}_{gen_sel.name}'
                self.h_calibration[histo_name] = histos.CalibrationHistos(histo_name)

    def fill_histos(self, debug=0):
        for tp_sel in self.data_selections:
            objects = self.data_set.query(tp_sel)
            for gen_sel in self.gen_selections:
                genReference = self.gen_set.query(gen_sel)
                histo_name = f'{self.data_set.name}_{tp_sel.name}_{gen_sel.name}'

                h_calib = self.h_calibration[histo_name]
                # print 'TPsel: {}, GENsel: {}'.format(tp_sel.name, gen_sel.name)
                self.plotObjectMatch(genReference,
                                     objects,
                                     self.data_set.tcs.df,
                                     h_calib,
                                     self.data_set.name,
                                     debug)


class TTGenMatchPlotter:
    def __init__(self, tt_set, gen_set,
                 tt_selections=[selections.Selection('all')], gen_selections=[selections.Selection('all')]):
        self.tt_set = tt_set
        self.tt_selections = tt_selections
        self.gen_set = gen_set
        self.gen_selections = gen_selections
        self.h_tt = {}
        self.h_reso_tt = {}
        self.h_reso_ttcl = {}

    def book_histos(self):
        self.tt_set.activate()
        self.gen_set.activate()
        for tp_sel in self.tt_selections:
            for gen_sel in self.gen_selections:
                histo_name = f'{tp_sel.name}_{gen_sel.name}'
                self.h_tt[histo_name] = histos.TriggerTowerHistos(f'{self.tt_set.name}_{histo_name}')
                self.h_reso_tt[histo_name] = histos.TriggerTowerResoHistos(f'{self.tt_set.name}_{histo_name}')
                self.h_reso_ttcl[histo_name] = histos.TriggerTowerResoHistos(f'{self.tt_set.name}Cl_{histo_name}')

    def fill_histos(self, debug=0):
        triggerTowers_all = self.tt_set.df
        genParts_all = self.gen_set.df[(self.gen_set.df.gen > 0)]
        for tp_sel in self.tt_selections:
            triggerTowers = triggerTowers_all
            if not tp_sel.all:
                triggerTowers = triggerTowers_all.query(tp_sel.selection)
            for gen_sel in self.gen_selections:
                histo_name = f'{tp_sel.name}_{gen_sel.name}'
                genReference = genParts_all
                if not gen_sel.all:
                    genReference = genParts_all.query(gen_sel.selection)
                self.plotTriggerTowerMatch(genReference,
                                           None,
                                           triggerTowers,
                                           self.h_tt[histo_name],
                                           self.h_reso_tt[histo_name],
                                           self.h_reso_ttcl[histo_name],
                                           'TThighestPt',
                                           debug)

    def plotTriggerTowerMatch(self,
                              genParticles,
                              histoGen,
                              triggerTowers,
                              histoTowersMatch,
                              histoTowersReso,
                              histoTowersResoCl,
                              algoname,
                              debug):

        best_match_indexes = {}
        if triggerTowers.shape[0] != 0:
            best_match_indexes, allmatches = utils.match_etaphi(genParticles[['exeta', 'exphi']],
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

                ttCluster = clAlgo.buildTriggerTowerCluster(triggerTowers, matchedTower, debug)
                histoTowersResoCl.fill(reference=genParticle, target=ttCluster)

                # clustersInCone = sumClustersInCone(trigger3DClusters, allmatches[idx])
                # print ('----- in cone sum:')
                # print (clustersInCone)
                # histoResoCone.fill(reference=genParticle, target=clustersInCone.iloc[0])

                if debug >= 4:
                    print(f'--- Dump match for algo {algoname} ---------------')
                    print(f'GEN particle: idx: {idx}')
                    print(genParticle)
                    print('Matched Trigger Tower:')
                    print(matchedTower)
            elif debug >= 0:
                print(f'==== Warning no match found for algo {algoname}, idx {idx} ======================')
                if debug >= 2:
                    print(genParticle)



class ClusterTCGenMatchPlotter(BasePlotter):
    def __init__(self,  # ResoHistoClass,
                 data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')]):
        # self.ResoHistoClass = ResoHistoClass
        self.h_tcmatching = {}
        super(ClusterTCGenMatchPlotter, self).__init__(
            data_set,
            data_selections,
            gen_set,
            selections.multiply_selections(
                gen_selections,
                [selections.Selection('', '', 'gen > 0')]))

    def plotObjectMatch(self,
                        genParticles,
                        objects,
                        tcs,
                        h_tc_matched,
                        algoname,
                        debug):
        # fill histo with all selected GEN particles before any match

        best_match_indexes = {}
        if not objects.empty:
            best_match_indexes, allmatches = utils.match_etaphi(genParticles[['exeta', 'exphi']],
                                                                objects[['eta', 'phi']],
                                                                objects['pt'],
                                                                deltaR=0.1)
        if tcs.empty:
            return

        for idx, genParticle in genParticles.iterrows():
            if idx in best_match_indexes.keys():
                obj_matched = objects.loc[[best_match_indexes[idx]]].iloc[0]
                sel_tcs = tcs[(tcs.z * obj_matched.eta > 0)]
                if sel_tcs.empty:
                    continue

                sel_tcs.loc[sel_tcs.index, 'delta_eta'] = sel_tcs.eta - obj_matched.eta
                sel_tcs.loc[sel_tcs.index, 'delta_phi'] = sel_tcs.apply(lambda tc: ROOT.TVector2.Phi_mpi_pi(tc.phi - obj_matched.phi),
                                                                        axis=1)
                clAlgo.compute_tcs_to_cluster_deltaro(cluster=obj_matched,
                                                      tcs=sel_tcs)

                h_tc_matched.fill(sel_tcs, obj_matched)

                if debug >= 4:
                    print(f'--- Dump match for algo {algoname} ---------------')
                    print(f'GEN particle: idx: {idx}')
                    print(genParticle)

    def book_histos(self):
        self.gen_set.activate()
        self.data_set.activate()
        for tp_sel in self.data_selections:
            for gen_sel in self.gen_selections:
                histo_name = f'{self.data_set.name}_{tp_sel.name}_{gen_sel.name}'
                self.h_tcmatching[histo_name] = histos.TCClusterMatchHistos(histo_name)

    def fill_histos(self, debug=0):
        for tp_sel in self.data_selections:
            objects = self.data_set.query(tp_sel)
            for gen_sel in self.gen_selections:
                genReference = self.gen_set.query(gen_sel)
                histo_name = f'{self.data_set.name}_{tp_sel.name}_{gen_sel.name}'

                h_tc_match = self.h_tcmatching[histo_name]
                # h_resoset = self.h_resoset[histo_name]
                # h_genseleff = self.h_effset[histo_name]
                # print 'TPsel: {}, GENsel: {}'.format(tp_sel.name, gen_sel.name)
                self.plotObjectMatch(genReference,
                                     objects,
                                     self.data_set.tcs.df,
                                     h_tc_match,
                                     self.data_set.name,
                                     debug)


class IsoTuplePlotter(BasePlotter):
    def __init__(self,
                 data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')]):
        self.h_resoset = {}
        super(IsoTuplePlotter, self).__init__(
            data_set,
            data_selections,
            gen_set,
            selections.multiply_selections(
                gen_selections,
                [selections.Selection('', '', 'gen > 0')]))

        # print self
        # print gen_selections

    def plotObjectMatch(self,
                        genParticles,
                        objects,
                        h_gen,
                        h_gen_matched,
                        h_object_matched,
                        h_reso,
                        algoname,
                        debug):
        # fill histo with all selected GEN particles before any match
        if h_gen:
            h_gen.fill(genParticles)

        best_match_indexes = {}
        if not objects.empty:
            best_match_indexes, allmatches = utils.match_etaphi(genParticles[['exeta', 'exphi']],
                                                                objects[['eta', 'phi']],
                                                                objects['pt'],
                                                                deltaR=0.1)
        # print best_match_indexes

        for idx, genParticle in genParticles.iterrows():
            if idx in best_match_indexes.keys():
                # print ('-----------------------')
                # print (genParticle)
                obj_matched = objects.loc[[best_match_indexes[idx]]]
                # h_object_matched.fill(obj_matched)
                h_reso.fill(reference=genParticle, target=obj_matched)

                if h_gen_matched is not None:
                    h_gen_matched.fill(genParticles.loc[[idx]])

        for idx, obj in objects.loc[~objects.index.isin(best_match_indexes.values())].iterrows():
            h_reso.fill(reference=None, target=obj)

    def book_histos(self):
        self.gen_set.activate()
        self.data_set.activate()
        for tp_sel in self.data_selections:
            for gen_sel in self.gen_selections:
                histo_name = f'{self.data_set.name}_{tp_sel.name}_{gen_sel.name}'
                self.h_resoset[histo_name] = histos.IsoTuples(histo_name)

    def fill_histos(self, debug=0):
        pass

    def fill_histos_event(self, idx, debug=0):
        for tp_sel in self.data_selections:
            objects = self.data_set.query_event(tp_sel, idx)
            for gen_sel in self.gen_selections:
                genReference = self.gen_set.query_event(gen_sel, idx)
                histo_name = f'{self.data_set.name}_{tp_sel.name}_{gen_sel.name}'
                # print (histo_name)
                # print (f'# data: {objects.shape[0]}')
                # print (f'# gen: {genReference.shape[0]}')
                # print (genReference)
                h_resoset = self.h_resoset[histo_name]
                # print ('TPsel: {}, GENsel: {}'.format(tp_sel.name, gen_sel.name))
                self.plotObjectMatch(genReference,
                                     objects,
                                     None,
                                     None,
                                     None,
                                     h_resoset,
                                     self.data_set.name,
                                     debug)


class QuantizationPlotter(GenericDataFramePlotter):
# class QuantizationPlotter(GenericDataFrameLazyPlotter):
    def __init__(self, data_set, data_selections, features):
        self.features = features
        super(QuantizationPlotter, self).__init__(histos.QuantizationHistos, data_set, data_selections)

    def book_histos(self):
        self.data_set.activate()
        data_name = self.data_set.name
        for selection in self.data_selections:
            self.h_set[selection.name] = self.HistoClass(
                name=f'{data_name}_{selection.name}_nomatch',
                features=self.features)


class CompTuplesPlotter(GenericDataFramePlotter):
    def __init__(self, obj_set, obj_selections=[selections.Selection('all')]):
        super(CompTuplesPlotter, self).__init__(histos.CompTuples, obj_set, obj_selections)


class CompCatTuplePlotter(GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')]):
        super(CompCatTuplePlotter, self).__init__(histos.EGHistos, histos.CompCatTuples,
                                                data_set, gen_set,
                                                data_selections, gen_selections, drcut=0.2)







class DiObjMassPlotter(GenericDataFramePlotter):
    def __init__(self, obj_set, obj_selections=[selections.Selection('all')]):
        super(DiObjMassPlotter, self).__init__(histos.DiObjMassHistos, obj_set, obj_selections)


if __name__ == '__main__':
    for sel in selections.multiply_selections(
            selections.tp_id_selections,
            selections.tp_eta_selections):
        print(sel)

    print(selections.multiply_selections(
            selections.tp_id_selections,
            selections.tp_pt_selections))

    # print(selections.gen_selection)
