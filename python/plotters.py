import l1THistos as histos
import utils as utils
import pandas as pd
import numpy as np
import clusterTools as clAlgo
import selections as selections


class RatePlotter:
    def __init__(self, tp_set, tp_selections=[selections.Selection('all')]):
        self.tp_set = tp_set
        self.tp_selections = tp_selections
        self.h_rate = {}

    def book_histos(self):
        tp_name = self.tp_set.name
        for selection in self.tp_selections:
            self.h_rate[selection.name] = histos.RateHistos(name='{}_{}'.format(tp_name,
                                                                                selection.name))

    def fill_histos(self, debug=False):
        for selection in self.tp_selections:
            if not selection.all:
                sel_clusters = self.tp_set.cl3d_df.query(selection.selection)
            else:
                sel_clusters = self.tp_set.cl3d_df
            trigger_clusters = sel_clusters[['pt', 'eta']].sort_values(by='pt',
                                                                       ascending=False)
            if not trigger_clusters.empty:
                self.h_rate[selection.name].fill(trigger_clusters.iloc[0].pt,
                                                 trigger_clusters.iloc[0].eta)
            self.h_rate[selection.name].fill_norm()


class TkEGPlotter:
    def __init__(self, tkeg_set, tkeg_selections=[selections.Selection('all')]):
        self.data_set = tkeg_set
        self.data_selections = tkeg_selections
        self.h_set = {}

    def book_histos(self):
        data_name = self.data_set.name
        for selection in self.data_selections:
            self.h_set[selection.name] = histos.TkEGHistos(name='{}_{}_nomatch'.format(data_name,
                                                                                        selection.name))

    def fill_histos(self, debug=False):
        for data_sel in self.data_selections:
            data = self.data_set.tkeg_df
            if not data_sel.all:
                data = self.data_set.tkeg_df.query(data_sel.selection)
            self.h_set[data_sel.name].fill(data)


class TrackPlotter:
    def __init__(self, trk_set, track_selections=[selections.Selection('all')]):
        self.data_set = trk_set
        self.data_selections = track_selections
        self.h_set = {}

    def book_histos(self):
        data_name = self.data_set.name
        for selection in self.data_selections:
            self.h_set[selection.name] = histos.TrackHistos(name='{}_{}_nomatch'.format(data_name,
                                                                                        selection.name))

    def fill_histos(self, debug=False):
        for data_sel in self.data_selections:
            data = self.data_set.trk_df
            if not data_sel.all:
                data = self.data_set.trk_df.query(data_sel.selection)
            self.h_set[data_sel.name].fill(data)


class EGPlotter:
    def __init__(self, eg_set, eg_selections=[selections.Selection('all')]):
        self.data_set = eg_set
        self.data_selections = eg_selections
        self.h_set = {}

    def book_histos(self):
        data_name = self.data_set.name
        for selection in self.data_selections:
            self.h_set[selection.name] = histos.EGHistos(name='{}_{}_nomatch'.format(data_name,
                                                                                     selection.name))

    def fill_histos(self, debug=False):
        for data_sel in self.data_selections:
            data = self.data_set.cl3d_df
            if not data_sel.all:
                data = self.data_set.cl3d_df.query(data_sel.selection)
            self.h_set[data_sel.name].fill(data)


class TPPlotter:
    def __init__(self, tp_set, tp_selections=[selections.Selection('all')]):
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
    def __init__(self, gen_set, gen_selections=[selections.Selection('all')]):
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
                 tp_selections=[selections.Selection('all')], gen_selections=[selections.Selection('all')]):
        self.tp_set = tp_set
        self.tp_selections = tp_selections
        self.gen_set = gen_set
        self.gen_selections = gen_selections
        self.h_tpset = {}
        self.h_resoset = {}
        self.h_effset = {}
        self.h_conecluster = {}

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
                print '-------- in cone:'
                print components.sort_values(by='pt', ascending=False)
                print '   - Cone sum:'
                print ret
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

                def fill_cluster_incone_histos(cl3ds,
                                               idx_allmatches,
                                               idx_bestmatch,
                                               charge,
                                               h_clustersInCone,
                                               debug=0):
                    if debug > 4:
                        print '- best match: {}, all matches: {}'.format(idx_bestmatch,
                                                                         idx_allmatches)
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
                histoResoCone.fill(reference=genParticle, target=clustersInCone.iloc[0])
                if histoGenMatched is not None:
                    histoGenMatched.fill(genParticles.loc[[idx]])

                if debug >= 6:
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
                    print ('==== Warning no match found for algo {}, idx {} ======================'.format(algoname,
                                                                                                           idx))
                    if debug >= 2:
                        print (genParticle)
                        print (trigger3DClusters)

        # if len(allmatched2Dclusters) != 0:
        #     matchedClustersAll = pd.concat(allmatched2Dclusters)
        # return matchedClustersAll

    def book_histos(self):
        for tp_sel in self.tp_selections:
            for gen_sel in self.gen_selections:
                histo_name = '{}_{}_{}'.format(self.tp_set.name,
                                               tp_sel.name,
                                               gen_sel.name)
                self.h_tpset[histo_name] = histos.HistoSetClusters(histo_name)
                self.h_resoset[histo_name] = histos.HistoSetReso(histo_name)
                self.h_effset[histo_name] = histos.HistoSetEff(histo_name)
                self.h_conecluster[histo_name] = histos.ClusterConeHistos(histo_name)

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
                h_conecl = self.h_conecluster[histo_name]
                # print 'TPsel: {}, GENsel: {}'.format(tp_sel.name, gen_sel.name)
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
        return '<{} tps: {}, tps_s: {}, gen:{}, gen_s:{}> '.format(self.__class__.__name__,
                                                                   self.tp_set.name,
                                                                   [sel.name for sel in self.tp_selections],
                                                                   self.gen_set.name,
                                                                   [sel.name for sel in self.gen_selections])


class TrackGenMatchPlotter:
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')], gen_selections=[selections.Selection('all')]):
        self.data_set = data_set
        self.data_selections = data_selections
        self.gen_set = gen_set
        self.gen_selections = gen_selections
        self.h_dataset = {}
        self.h_resoset = {}
        self.h_effset = {}

    def plotTrackMatch(self,
                       genParticles,
                       tracks,
                       h_gen,
                       h_gen_matched,
                       h_track_matched,
                       h_reso,
                       algoname,
                       debug):
        # fill histo with all selected GEN particles before any match
        if h_gen:
            h_gen.fill(genParticles)

        best_match_indexes = {}
        if not tracks.empty:
            best_match_indexes, allmatches = utils.match_etaphi(genParticles[['eta', 'phi']],
                                                                tracks[['eta', 'phi']],
                                                                tracks['pt'],
                                                                deltaR=0.1)

        for idx, genParticle in genParticles.iterrows():
            if idx in best_match_indexes.keys():
                # print ('-----------------------')
                #  print(genParticle)
                track_matched = tracks.loc[[best_match_indexes[idx]]]
                h_track_matched.fill(track_matched)
                h_reso.fill(reference=genParticle, target=track_matched)

                if h_gen_matched is not None:
                    h_gen_matched.fill(genParticles.loc[[idx]])

                if debug >= 4:
                    print ('--- Dump match for algo {} ---------------'.format(algoname))
                    print ('GEN particle: idx: {}'.format(idx))
                    print (genParticle)
                    print ('Matched to track object:')
                    print (track_matched)
            else:
                if debug >= 5:
                    print ('==== Warning no match found for algo {}, idx {} ======================'.format(algoname, idx))
                    print (genParticle)
                    print (tracks)

    def book_histos(self):
        for tp_sel in self.data_selections:
            for gen_sel in self.gen_selections:
                histo_name = '{}_{}_{}'.format(self.data_set.name, tp_sel.name, gen_sel.name)
                self.h_dataset[histo_name] = histos.TrackHistos(histo_name)
                self.h_resoset[histo_name] = histos.TrackResoHistos(histo_name)
                self.h_effset[histo_name] = histos.HistoSetEff(histo_name)

    def fill_histos(self, debug=False):
        for tp_sel in self.data_selections:
            tracks = self.data_set.trk_df
            if not tp_sel.all:
                tracks = self.data_set.trk_df.query(tp_sel.selection)
            for gen_sel in self.gen_selections:
                histo_name = '{}_{}_{}'.format(self.data_set.name, tp_sel.name, gen_sel.name)
                genReference = self.gen_set.gen_df[(self.gen_set.gen_df.gen > 0)]
                if not gen_sel.all:
                    genReference = self.gen_set.gen_df[(self.gen_set.gen_df.gen > 0)].query(gen_sel.selection)
                    # FIXME: this doesn't work for pizeros since they are never listed in the genParticles...we need a working solution
                    # elif  particle.pdgid == PID.pizero:
                    #     genReference = genParts[(genParts.pid == particle.pdgid)]

                h_tpset_match = self.h_dataset[histo_name]
                h_resoset = self.h_resoset[histo_name]
                h_genseleff = self.h_effset[histo_name]
                # print 'TPsel: {}, GENsel: {}'.format(tp_sel.name, gen_sel.name)
                self.plotTrackMatch(genReference,
                                    tracks,
                                    h_genseleff.h_den,
                                    h_genseleff.h_num,
                                    h_tpset_match,
                                    h_resoset,
                                    self.data_set.name,
                                    debug)


class EGGenMatchPlotter:
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')], gen_selections=[selections.Selection('all')]):
        self.data_set = data_set
        self.data_selections = data_selections
        self.gen_set = gen_set
        self.gen_selections = gen_selections
        self.h_dataset = {}
        self.h_resoset = {}
        self.h_effset = {}

    def plotEGMatch(self,
                    genParticles,
                    egs,
                    h_gen,
                    h_gen_matched,
                    h_eg_matched,
                    h_reso,
                    algoname,
                    debug):
        # fill histo with all selected GEN particles before any match
        if h_gen:
            h_gen.fill(genParticles)

        best_match_indexes = {}
        if not egs.empty:
            best_match_indexes, allmatches = utils.match_etaphi(genParticles[['eta', 'phi']],
                                                                egs[['eta', 'phi']],
                                                                egs['pt'],
                                                                deltaR=0.1)

        for idx, genParticle in genParticles.iterrows():
            if idx in best_match_indexes.keys():
                # print ('-----------------------')
                #  print(genParticle)
                eg_matched = egs.loc[[best_match_indexes[idx]]]
                h_eg_matched.fill(eg_matched)
                h_reso.fill(reference=genParticle, target=eg_matched)

                if h_gen_matched is not None:
                    h_gen_matched.fill(genParticles.loc[[idx]])

                if debug >= 4:
                    print ('--- Dump match for algo {} ---------------'.format(algoname))
                    print ('GEN particle: idx: {}'.format(idx))
                    print (genParticle)
                    print ('Matched to EG object:')
                    print (eg_matched)
            else:
                if debug >= 5:
                    print ('==== Warning no match found for algo {}, idx {} ======================'.format(algoname, idx))
                    print (genParticle)
                    print (egs)

    def book_histos(self):
        for tp_sel in self.data_selections:
            for gen_sel in self.gen_selections:
                histo_name = '{}_{}_{}'.format(self.data_set.name, tp_sel.name, gen_sel.name)
                self.h_dataset[histo_name] = histos.EGHistos(histo_name)
                self.h_resoset[histo_name] = histos.EGResoHistos(histo_name)
                self.h_effset[histo_name] = histos.HistoSetEff(histo_name)

    def fill_histos(self, debug=False):
        for tp_sel in self.data_selections:
            cl3Ds = self.data_set.cl3d_df
            if not tp_sel.all and not self.data_set.cl3d_df.empty:
                cl3Ds = self.data_set.cl3d_df.query(tp_sel.selection)
            for gen_sel in self.gen_selections:
                histo_name = '{}_{}_{}'.format(self.data_set.name, tp_sel.name, gen_sel.name)
                genReference = self.gen_set.gen_df[(self.gen_set.gen_df.gen > 0)]
                if not gen_sel.all:
                    genReference = self.gen_set.gen_df[(self.gen_set.gen_df.gen > 0)].query(gen_sel.selection)
                    # FIXME: this doesn't work for pizeros since they are never listed in the genParticles...we need a working solution
                    # elif  particle.pdgid == PID.pizero:
                    #     genReference = genParts[(genParts.pid == particle.pdgid)]

                h_tpset_match = self.h_dataset[histo_name]
                h_resoset = self.h_resoset[histo_name]
                h_genseleff = self.h_effset[histo_name]
                # print 'TPsel: {}, GENsel: {}'.format(tp_sel.name, gen_sel.name)
                self.plotEGMatch(genReference,
                                 cl3Ds,
                                 h_genseleff.h_den,
                                 h_genseleff.h_num,
                                 h_tpset_match,
                                 h_resoset,
                                 self.data_set.name,
                                 debug)


class TTPlotter:
    def __init__(self, tt_set, tt_selections=[selections.Selection('all')]):
        self.tt_set = tt_set
        self.tt_selections = tt_selections
        self.h_tt = {}

    def book_histos(self):
        for sel in self.tt_selections:
            self.h_tt[sel.name] = histos.TriggerTowerHistos('{}_{}_nomatch'.format(self.tt_set.name, sel.name))

    def fill_histos(self, debug=False):
        triggerTowers_all = self.tt_set.tt_df
        for sel in self.tt_selections:
            triggerTowers = triggerTowers_all
            if not sel.all:
                triggerTowers = triggerTowers_all.query(sel.selection)
            self.h_tt[sel.name].fill(triggerTowers)


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
        for tp_sel in self.tt_selections:
            for gen_sel in self.gen_selections:
                histo_name = '{}_{}'.format(tp_sel.name, gen_sel.name)
                self.h_tt[histo_name] = histos.TriggerTowerHistos('{}_{}'.format(self.tt_set.name, histo_name))
                self.h_reso_tt[histo_name] = histos.TriggerTowerResoHistos('{}_{}'.format(self.tt_set.name, histo_name))
                self.h_reso_ttcl[histo_name] = histos.TriggerTowerResoHistos('{}Cl_{}'.format(self.tt_set.name, histo_name))

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


tp_plotters = [TPPlotter(selections.tp_def, selections.tp_id_selections),
               # TPPlotter(selections.tp_def_calib, selections.tp_id_selections)
               ]
eg_plotters = [EGPlotter(selections.eg_set, selections.eg_qual_selections)]
track_plotters = [TrackPlotter(selections.track_set, selections.tracks_selections)]
tkeg_plotters = [TkEGPlotter(selections.tkeg_set, selections.eg_qual_selections)]
rate_plotters = [RatePlotter(selections.tp_def, selections.tp_rate_selections),
                 # RatePlotter(selections.tp_def_calib, selections.tp_rate_selections),
                 RatePlotter(selections.tp_def_merged, selections.tp_rate_selections)]
eg_rate_plotters = [RatePlotter(selections.eg_set, selections.eg_rate_selections),
                    RatePlotter(selections.tkeg_set, selections.tkeg_rate_selections)]
tp_genmatched_plotters = [TPGenMatchPlotter(selections.tp_def, selections.gen_set,
                                            selections.tp_match_selections,
                                            selections.gen_part_selections),
                          # TPGenMatchPlotter(selections.tp_def_calib, selections.gen_set,
                          #                   selections.tp_match_selections,
                          #                   selections.gen_part_selections),
                          TPGenMatchPlotter(selections.tp_def_merged, selections.gen_set,
                                            selections.tp_match_selections,
                                            selections.gen_part_selections)
                                            ]
eg_genmatched_plotters = [EGGenMatchPlotter(selections.eg_set, selections.gen_set,
                                            selections.eg_qual_selections,
                                            selections.gen_part_selections),
                          EGGenMatchPlotter(selections.tkeg_set, selections.gen_set,
                                            selections.tkeg_qual_selections,
                                            selections.gen_part_selections)]
track_genmatched_plotters = [TrackGenMatchPlotter(selections.track_set, selections.gen_set,
                                                  selections.tracks_selections,
                                                  selections.gen_part_selections)]
genpart_plotters = [GenPlotter(selections.gen_set, selections.genpart_ele_genplotting)]
ttower_plotters = [TTPlotter(selections.tt_set),
                   TTPlotter(selections.simtt_set),
                   TTPlotter(selections.hgcroc_tt),
                   TTPlotter(selections.wafer_tt)
                   ]
ttower_genmatched_plotters = [TTGenMatchPlotter(selections.tt_set, selections.gen_set,
                              [selections.Selection('all')], selections.gen_part_selections),
                              TTGenMatchPlotter(selections.simtt_set, selections.gen_set,
                              [selections.Selection('all')], selections.gen_part_selections),
                              TTGenMatchPlotter(selections.hgcroc_tt, selections.gen_set,
                              [selections.Selection('all')], selections.gen_part_selections),
                              TTGenMatchPlotter(selections.wafer_tt, selections.gen_set,
                              [selections.Selection('all')], selections.gen_part_selections)
                              ]


if __name__ == "__main__":
    for sel in selections.add_selections(selections.tp_id_selections,
                                         selections.tp_eta_selections):
        print sel

    print selections.add_selections(selections.tp_id_selections,
                                    selections.tp_pt_selections)

    print selections.gen_selection
