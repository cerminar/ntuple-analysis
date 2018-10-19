import l1THistos as histos
import utils as utils
import pandas as pd
import numpy as np
import clusterTools as clAlgo


class PID:
    electron = 11
    photon = 22
    pizero = 111
    pion = 211
    kzero = 130


class Selection:
    def __init__(self, name, label='', selection=''):
        self.name = name
        self.label = label
        self.selection = selection

    def __add__(self, sel_obj):
        """ & operation """
        if sel_obj.all:
            return self
        if self.all:
            return sel_obj
        new_label = '{} & {}'.format(self.label, sel_obj.label)
        if self.label == '':
            new_label = sel_obj.label
        if sel_obj.label == '':
            new_label = self.label
        return Selection(name='{}{}'.format(self.name, sel_obj.name),
                         label=new_label,
                         selection='({}) & ({})'.format(self.selection, sel_obj.selection))

    def __str__(self):
        return 'n: {}, s: {}, l:{}'.format(self.name, self.selection, self.label)

    def __repr__(self):
        return '<{} n: {}, s: {}, l:{}> '.format(self.__class__.__name__,
                                                 self.name,
                                                 self.selection,
                                                 self.label)

    @property
    def all(self):
        if self.name == 'all':
            return True
        return False


def add_selections(list1, list2):
    ret = []
    for sel1 in list1:
        for sel2 in list2:
            ret.append(sel1+sel2)
    return ret


class RatePlotter:
    def __init__(self, tp_set, tp_selections=[Selection('all')]):
        self.tp_set = tp_set
        self.tp_selections = tp_selections
        self.h_rate = {}

    def book_histos(self):
        tp_name = self.tp_set.name
        for selection in self.tp_selections:
            self.h_rate[selection.name] = histos.RateHistos(name='{}_{}'.format(tp_name, selection.name))

    def fill_histos(self, debug=False):
        for selection in self.tp_selections:
            if not selection.all:
                sel_clusters = self.tp_set.cl3d_df.query(selection.selection)
            else:
                sel_clusters = self.tp_set.cl3d_df
            trigger_clusters = sel_clusters[['pt', 'eta']].sort_values(by='pt', ascending=False)
            if not trigger_clusters.empty:
                self.h_rate[selection.name].fill(trigger_clusters.iloc[0].pt, trigger_clusters.iloc[0].eta)
            self.h_rate[selection.name].fill_norm()


class TPPlotter:
    def __init__(self, tp_set, tp_selections=[Selection('all')]):
        self.tp_set = tp_set
        self.tp_selections = tp_selections
        self.h_tpset = {}

    def book_histos(self):
        tp_name = self.tp_set.name
        for selection in self.tp_selections:
            self.h_tpset[selection.name] = histos.HistoSetClusters(name='{}_{}_nomatch'.format(tp_name, selection.name))

    def fill_histos(self, debug=False):
        for tp_sel in self.tp_selections:
            tcs = self.tp_set.tc_df
            cl2Ds = self.tp_set.cl2d_df
            cl3Ds = self.tp_set.cl3d_df
            if not tp_sel.all:
                cl3Ds = self.tp_set.cl3d_df.query(tp_sel.selection)
            # debugPrintOut(debug, '{}_{}'.format(self.name, 'TCs'), tcs, tcs[:3])
            # debugPrintOut(debug, '{}_{}'.format(self.name, 'CL2D'), cl2Ds, cl2Ds[:3])
            # debugPrintOut(debug, '{}_{}'.format(self.name, 'CL3D'), cl3Ds, cl3Ds[:3])
            self.h_tpset[tp_sel.name].fill(tcs, cl2Ds, cl3Ds)


class GenPlotter:
    def __init__(self, gen_set, gen_selections=[Selection('all')]):
        self.gen_set = gen_set
        self.gen_selections = gen_selections
        self.h_gen = {}

    def book_histos(self):
        for selection in self.gen_selections:
            self.h_gen[selection.name] = histos.GenParticleHistos(name='h_genParts_{}'.format(selection.name))

    def fill_histos(self, debug=False):
        gen_parts_all = self.gen_set.gen_df[self.gen_set.gen_df.gen > 0]
        for gen_sel in self.gen_selections:
            gen_parts = gen_parts_all
            if not gen_sel.all:
                gen_parts = gen_parts_all.query(gen_sel.selection)
            self.h_gen[gen_sel.name].fill(gen_parts)


class TPGenMatchPlotter:
    def __init__(self, tp_set, gen_set,
                 tp_selections=[Selection('all')], gen_selections=[Selection('all')]):
        self.tp_set = tp_set
        self.tp_selections = tp_selections
        self.gen_set = gen_set
        self.gen_selections = gen_selections
        self.h_tpset = {}
        self.h_resoset = {}
        self.h_effset = {}

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
                if histoGenMatched is None:
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

    def book_histos(self):
        for tp_sel in self.tp_selections:
            for gen_sel in self.gen_selections:
                histo_name = '{}_{}_{}'.format(self.tp_set.name, tp_sel.name, gen_sel.name)
                self.h_tpset[histo_name] = histos.HistoSetClusters(histo_name)
                self.h_resoset[histo_name] = histos.HistoSetReso(histo_name)
                self.h_effset[histo_name] = histos.HistoSetEff(histo_name)

    def fill_histos(self, debug=False):
        for tp_sel in self.tp_selections:
            tcs = self.tp_set.tc_df
            cl2Ds = self.tp_set.cl2d_df
            cl3Ds = self.tp_set.cl3d_df
            if not tp_sel.all:
                cl3Ds = self.tp_set.cl3d_df.query(tp_sel.selection)
            for gen_sel in self.gen_selections:
                histo_name = '{}_{}_{}'.format(self.tp_set.name, tp_sel.name, gen_sel.name)
                genReference = self.gen_set.gen_df[(self.gen_set.gen_df.gen > 0)]
                if not gen_sel.all:
                    genReference = self.gen_set.gen_df[(self.gen_set.gen_df.gen > 0)].query(gen_sel.selection)
                    # FIXME: this doesn't work for pizeros since they are never listed in the genParticles...we need a working solution
                    # elif  particle.pdgid == PID.pizero:
                    #     genReference = genParts[(genParts.pid == particle.pdgid)]
                h_tpset_match = self.h_tpset[histo_name]
                h_resoset = self.h_resoset[histo_name]
                h_genseleff = self.h_effset[histo_name]
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
                                        self.tp_set.name,
                                        debug)


class TTPlotter:
    def __init__(self, tt_set, tt_selections=[Selection('all')]):
        self.tt_set = tt_set
        self.tt_selections = tt_selections
        self.h_tt = {}

    def book_histos(self):
        for sel in self.tt_selections:
            self.h_tt[sel.name] = histos.TriggerTowerHistos('h_{}_{}'.format(self.tt_set.name, sel.name))

    def fill_histos(self, debug=False):
        triggerTowers_all = self.tt_set.tt_df
        for sel in self.tt_selections:
            triggerTowers = triggerTowers_all
            if not sel.all:
                triggerTowers = triggerTowers_all.query(sel.selection)
            self.h_tt[sel.name].fill(triggerTowers)


class TTGenMatchPlotter:
    def __init__(self, tt_set, gen_set,
                 tt_selections=[Selection('all')], gen_selections=[Selection('all')]):
        self.tt_set = tt_set
        self.tt_selections = tt_selections
        self.gen_set = gen_set
        self.gen_selections = gen_selections
        self.h_tt = {}
        self.h_reso_tt = {}
        self.h_reso_ttcl = {}

    def book_histos(self):
        for tp_sel in self.tt_selections:
            for gen_sel in self.gen_selections:
                histo_name = '{}_{}'.format(tp_sel.name, gen_sel.name)
            self.h_tt[histo_name] = histos.TriggerTowerHistos('h_{}_{}'.format(self.tt_set.name, histo_name))
            self.h_reso_tt[histo_name] = histos.TriggerTowerResoHistos('h_reso_{}_{}'.format(self.tt_set.name, histo_name))
            self.h_reso_ttcl[histo_name] = histos.TriggerTowerResoHistos('h_reso_{}Cl_{}'.format(self.tt_set.name, histo_name))

    def fill_histos(self, debug=False):
        triggerTowers_all = self.tt_set.tt_df
        genParts_all = self.gen_set.gen_df[(self.gen_set.gen_df.gen > 0)]
        for tp_sel in self.tt_selections:
            triggerTowers = triggerTowers_all
            if not tp_sel.all:
                triggerTowers = triggerTowers_all.query(tp_sel.selection)
            for gen_sel in self.gen_selections:
                histo_name = '{}_{}'.format(tp_sel.name, gen_sel.name)
                genReference = genParts_all
                if not gen_sel.all:
                    genReference = genParts_all.query(gen_sel.selection)
            self.plotTriggerTowerMatch(genReference,
                                       None,
                                       triggerTowers,
                                       self.h_tt[histo_name],
                                       self.h_reso_tt[histo_name],
                                       self.h_reso_ttcl[histo_name],
                                       "TThighestPt",
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

                ttCluster = clAlgo.buildTriggerTowerCluster(triggerTowers, matchedTower, debug)
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







if __name__ == "__main__":
    for sel in add_selections(tp_id_selections, tp_eta_selections):
        print sel

    print add_selections(tp_id_selections, tp_pt_selections)

    print gen_selection
