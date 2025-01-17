from python import plotters, selections, calibrations, histos
import python.boost_hist as bh
import cfg.datasets.fastpuppi_collections as coll
import awkward as ak
import math
import numpy as np

# ------ Histogram classes ----------------------------------------------

class GenDiEleHistos(histos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptPair = bh.TH1F(f'{name}_ptPair', 'Pt (GeV); p_{T} [GeV]', 50, 0, 50)

            self.h_ptLead = bh.TH1F(f'{name}_ptLead', 'Pt (GeV); p_{T} [GeV]', 50, 0, 50)
            self.h_etaLead = bh.TH1F(f'{name}_etaLead', 'Pt (GeV); p_{T} [GeV]', 50, 0, 4)
            self.h_ptSubLead = bh.TH1F(f'{name}_ptSubLead', 'Pt (GeV); p_{T} [GeV]', 50, 0, 50)
            self.h_etaSubLead = bh.TH1F(f'{name}_etaSubLead', 'Pt (GeV); p_{T} [GeV]', 50, 0, 4)
            self.h_mass = bh.TH1F(f'{name}_mass', 'mass (GeV); mass [GeV]', 150, 0, 300)
            self.h_dR = bh.TH1F(f'{name}_dR', '#DeltaR; #DeltaR', 50, 0, 7)
            self.h_dPhi = bh.TH1F(f'{name}_dPhi', '#Delta#phi; |#Delta#phi|', 50, 0, 4)
            self.h_dEta = bh.TH1F(f'{name}_dEta', '#Delta#eta; |#Delta#eta|', 50, 0, 10)
            self.h_etaSign = bh.TH1F(f'{name}_etaSign', '#Delta#eta; #eta_{0}*#eta_{1}>0', 2, 0, 2)

            # self.h_eta = bh.TH1F(f'{name}_eta', 'EG eta; #eta;', 100, -4, 4)
            # self.h_energy = bh.TH1F(f'{name}_energy', 'EG energy (GeV); E [GeV]', 1000, 0, 1000)
            # self.h_hwQual = bh.TH1F(f'{name}_hwQual', 'EG energy (GeV); hwQual', 5, 0, 5)
            # self.h_tkIso = bh.TH1F(f'{name}_tkIso', 'Iso; rel-iso_{tk}', 100, 0, 2)
            # self.h_pfIso = bh.TH1F(f'{name}_pfIso', 'Iso; rel-iso_{pf}', 100, 0, 2)
            # self.h_tkIsoPV = bh.TH1F(f'{name}_tkIsoPV', 'Iso; rel-iso^{PV}_{tk}', 100, 0, 2)
            # self.h_pfIsoPV = bh.TH1F(f'{name}_pfIsoPV', 'Iso; rel-iso^{PV}_{pf}', 100, 0, 2)
            # self.h_n = bh.TH1F(f'{name}_n', '# objects per event', 100, 0, 100)
            # self.h_compBdt = bh.TH1F(f'{name}_compBdt', 'BDT Score Comp ID', 50, 0, 1)

        histos.BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, egs):
        weight = None
        if 'weight' in egs.fields:
            weight = egs.weight
        bh.fill_1Dhist(hist=self.h_ptPair,     array=egs.ptPair,     weights=weight)
        bh.fill_1Dhist(hist=self.h_ptLead,     array=egs.leg0.pt,     weights=weight)
        bh.fill_1Dhist(hist=self.h_etaLead,     array=np.abs(egs.leg0.eta),     weights=weight)
        bh.fill_1Dhist(hist=self.h_ptSubLead,     array=egs.leg1.pt,     weights=weight)
        bh.fill_1Dhist(hist=self.h_etaSubLead,     array=np.abs(egs.leg1.eta),     weights=weight)
        bh.fill_1Dhist(hist=self.h_mass,     array=egs.mass,     weights=weight)
        bh.fill_1Dhist(hist=self.h_dR,     array=egs.dr,     weights=weight)
        # print(egs.leg0.deltaphi(egs.leg1))
        bh.fill_1Dhist(hist=self.h_dPhi,     array=np.abs(egs.leg0.deltaphi(egs.leg1)),     weights=weight)
        bh.fill_1Dhist(hist=self.h_dEta,     array=np.abs(egs.leg0.deltaeta(egs.leg1)),     weights=weight)

        # print(egs.leg0.eta)
        # print(egs.leg1.eta)

        # print(egs.leg0.eta*egs.leg1.eta)
        # print((egs.leg0.eta*egs.leg1.eta)>0)
        bh.fill_1Dhist(hist=self.h_etaSign,     array=((egs.leg0.eta*egs.leg1.eta)>0),     weights=weight)

        # bh.fill_1Dhist(hist=self.h_eta,    array=egs.eta,    weights=weight)
        # # bh.fill_1Dhist(hist=self.h_energy, array=egs.energy, weights=weight)
        # bh.fill_1Dhist(hist=self.h_hwQual, array=egs.hwQual, weights=weight)


class DiEleHistos(histos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptPair = bh.TH1F(f'{name}_ptPair', 'Pt (GeV); p_{T} [GeV]', 50, 0, 50)

            self.h_ptLead = bh.TH1F(f'{name}_ptLead', 'Pt (GeV); p_{T} [GeV]', 50, 0, 50)
            self.h_etaLead = bh.TH1F(f'{name}_etaLead', 'Pt (GeV); p_{T} [GeV]', 50, 0, 4)
            self.h_ptSubLead = bh.TH1F(f'{name}_ptSubLead', 'Pt (GeV); p_{T} [GeV]', 50, 0, 50)
            self.h_etaSubLead = bh.TH1F(f'{name}_etaSubLead', 'Pt (GeV); p_{T} [GeV]', 50, 0, 4)
            self.h_mass = bh.TH1F(f'{name}_mass', 'mass (GeV); mass [GeV]', 150, 0, 300)
            self.h_dR = bh.TH1F(f'{name}_dR', '#DeltaR; #DeltaR', 50, 0, 7)
            self.h_dPhi = bh.TH1F(f'{name}_dPhi', '#Delta#phi; #Delta#phi', 50, 0, 4)

            # self.h_eta = bh.TH1F(f'{name}_eta', 'EG eta; #eta;', 100, -4, 4)
            # self.h_energy = bh.TH1F(f'{name}_energy', 'EG energy (GeV); E [GeV]', 1000, 0, 1000)
            # self.h_hwQual = bh.TH1F(f'{name}_hwQual', 'EG energy (GeV); hwQual', 5, 0, 5)
            self.h_tkIsoLead = bh.TH1F(f'{name}_tkIsoLead', 'Iso; rel-iso_{tk}', 100, 0, 2)
            self.h_tkIsoSubLead = bh.TH1F(f'{name}_tkIsoSubLead', 'Iso; rel-iso_{tk}', 100, 0, 2)
            self.h_pfIsoLead = bh.TH1F(f'{name}_pfIsoLead', 'Iso; rel-iso_{tk}', 100, 0, 2)
            self.h_pfIsoSubLead = bh.TH1F(f'{name}_pfIsoSubLead', 'Iso; rel-iso_{tk}', 100, 0, 2)

            # self.h_pfIsoPV = bh.TH1F(f'{name}_pfIsoPV', 'Iso; rel-iso^{PV}_{pf}', 100, 0, 2)
            self.h_n = bh.TH1F(f'{name}_n', '# objects per event', 100, 0, 100)

            self.h_idScoreLead = bh.TH1F(f'{name}_idScoreLead', 'ID BDT Score Lead', 50, -1, 1)
            self.h_idScoreSubLead = bh.TH1F(f'{name}_idScoreSubLead', 'ID BDT Score Lead', 50, -1, 1)

            # self.h_compBdt = bh.TH1F(f'{name}_compBdt', 'BDT Score Comp ID', 50, 0, 1)

        histos.BaseHistos.__init__(self, name, root_file, debug)


    def fill(self, egs):
        weight = None
        if 'weight' in egs.fields:
            weight = egs.weight
        
        bh.fill_1Dhist(hist=self.h_ptPair,     array=egs.ptPair,     weights=weight)
        bh.fill_1Dhist(hist=self.h_ptLead,     array=egs.leg0.pt,     weights=weight)
        bh.fill_1Dhist(hist=self.h_etaLead,     array=np.abs(egs.leg0.eta),     weights=weight)
        bh.fill_1Dhist(hist=self.h_ptSubLead,     array=egs.leg1.pt,     weights=weight)
        bh.fill_1Dhist(hist=self.h_etaSubLead,     array=np.abs(egs.leg1.eta),     weights=weight)
        bh.fill_1Dhist(hist=self.h_mass,     array=egs.mass,     weights=weight)
        bh.fill_1Dhist(hist=self.h_dR,     array=egs.dr,     weights=weight)
        bh.fill_1Dhist(hist=self.h_dPhi,     array=egs.dphi,     weights=weight)
        # print(self.h_mass.counts())
        # bh.fill_1Dhist(hist=self.h_eta,    array=egs.eta,    weights=weight)
        # # bh.fill_1Dhist(hist=self.h_energy, array=egs.energy, weights=weight)
        # bh.fill_1Dhist(hist=self.h_hwQual, array=egs.hwQual, weights=weight)
        # print(egs.leg0.fields)
        if 'tkIso' in egs.leg0.fields:
            bh.fill_1Dhist(hist=self.h_tkIsoLead, array=egs.leg0.tkIso, weights=weight)
            bh.fill_1Dhist(hist=self.h_tkIsoSubLead, array=egs.leg1.tkIso, weights=weight)
        if 'pfIso' in egs.leg0.fields:
            bh.fill_1Dhist(hist=self.h_pfIsoLead, array=egs.leg0.pfIso, weights=weight)
            bh.fill_1Dhist(hist=self.h_pfIsoSubLead, array=egs.leg1.pfIso, weights=weight)
        if 'idScore' in egs.leg0.fields:
            bh.fill_1Dhist(hist=self.h_idScoreLead, 
                           array=egs.leg0.idScore, 
                           weights=weight)
            bh.fill_1Dhist(hist=self.h_idScoreSubLead, 
                           array=egs.leg1.idScore, 
                           weights=weight)

        # if 'pfIso' in egs.fields:
        #     bh.fill_1Dhist(hist=self.h_pfIso, array=egs.pfIso, weights=weight)
        # if 'tkIsoPV' in egs.fields:
        #     bh.fill_1Dhist(hist=self.h_tkIsoPV, array=egs.tkIsoPV, weights=weight)
        #     bh.fill_1Dhist(hist=self.h_pfIsoPV, array=egs.pfIsoPV, weights=weight)
        # if 'compBDTScore' in egs.fields:
        #     bh.fill_1Dhist(hist=self.h_compBdt, array=egs.compBDTScore, weights=weight)
        # if 'idScore' in egs.fields:
        #     bh.fill_1Dhist(hist=self.h_compBdt, array=expit(egs.idScore), weights=weight)
        # # print(ak.count(egs.pt, axis=1))
        # # print(egs.pt.type.show())
        # # print(ak.count(egs.pt, axis=1).type.show())

        self.h_n.fill(ak.count(egs.mass, axis=1))
        
        # # bh.fill_1Dhist(hist=self.h_n, array=ak.count(egs.pt, axis=1), weights=weight)
        # # self.h_n.Fill()


# ------ Plotter classes ------------------------------------------------

class GenDiElePlotter(plotters.GenericDataFramePlotter):
    def __init__(self, data_set,
                 data_selections=[selections.Selection('all')],):
        super(GenDiElePlotter, self).__init__(GenDiEleHistos,
                                                data_set,
                                                data_selections)


class DiElePlotter(plotters.GenericDataFramePlotter):
    def __init__(self, data_set,
                 data_selections=[selections.Selection('all')],
                 data_selections_best=[selections.Selection('all')]):
        self.best_sels = data_selections_best
        self.d_sels = data_selections
        all_data_sels = selections.multiply_selections(data_selections, data_selections_best)
        super(DiElePlotter, self).__init__(DiEleHistos,
                                                data_set,
                                                all_data_sels)

    def fill_histos(self, debug=0):
        for data_sel in self.d_sels:
            data = self.data_set.df
            if not data_sel.all:
                data = data[data_sel.selection(data)]
            hname = data_sel.name
            for bs in self.best_sels:
                if not bs.all:
                    # print('BEFORE')
                    # print(data)
                    # print(bs.selection(data))
                    data = ak.drop_none(data[bs.selection(data)], axis=1)
                    
                    # print(data)
                    if not data_sel.all:
                        hname = data_sel.name + bs.name
                    else:
                        hname = bs.name
                    # print(hname)
                self.h_set[hname].fill(data)
                    # print(self.h_set[hname])

# ------ Plotter instances
sm = selections.SelectionManager()
# Selector.selection_primitives = sm.selections.copy()

diobjsel = [
    selections.Selection('OS', 'O.S.', lambda ar: ar.sign < 1),
    selections.Selection('Dz1', '|#Delta z|<1cm', lambda ar: ar.dz < 1),
    selections.Selection('IDScore0p1', 'score_{ID}>0.1', lambda ar: ar.idScore > 0.1),
]

best_pair_sel = [
    selections.Selection('OS', 'O.S.', lambda ar: ar.sign < 1),
]

selections.Selector.selection_primitives = sm.selections.copy()

diele_sel = [
    selections.build_DiObj_selection('DiElePt5', 'p_{T}^{leg}>5GeV',
                                     (selections.Selector('^Pt5$')).one(),
                                     (selections.Selector('^Pt5$')).one()),
    selections.build_DiObj_selection('DiElePt5EtaEB', 'p_{T}^{leg}>5GeV EB',
                                     (selections.Selector('^Pt5$')*('^EtaEB$')).one(),
                                     (selections.Selector('^Pt5$')*('^EtaEB$')).one()),

    # selections.build_DiObj_selection('DiEle', '',
    #                                  (selections.Selector('^all$')).one(),
    #                                  (selections.Selector('^all$')).one()),
    # selections.build_DiObj_selection('DiEleID', '',
    #                                  (selections.Selector('^IDScore0p1$')).one(),
    #                                  (selections.Selector('^IDScore0p1$')).one()),
    # selections.build_DiObj_selection('DiEleIDPt5', '',
    #                                  (selections.Selector('^IDScore0p1$')*('^Pt5$')).one(),
    #                                  (selections.Selector('^IDScore0p1$')*('^Pt5$')).one()),
    



    # selections.Selection('DiElePt5', 'p_{t}^{leg}>5',  lambda ar: (ar.leg0.pt > 5) & (ar.leg1.pt > 5)),
    # selections.Selection('DiElePt5OS', 'p_{t}^{leg}>5 & OS',  lambda ar: (ar.leg0.pt > 5) & (ar.leg1.pt > 5) & (ar.sign < 1)),
    # selections.Selection('DiElePt5OSDz1', 'p_{t}^{leg}>5 & OS & #DeltaZ<1cm',  lambda ar: (ar.leg0.pt > 5) & (ar.leg1.pt > 5) & (ar.sign < 1) &  (ar.dz < 1)),


    # selections.Selection('DiElePt5B2B', 'p_{t}^{leg}>5',  lambda ar: (ar.leg0.pt > 5) & (ar.leg1.pt > 5)),

    # selections.Selection('DiEleOS', 'OS',  lambda ar: ar.sign < 1),
    # selections.Selection('DiEleOSDZ', 'OS + DZ<1',  lambda ar: ((ar.sign < 1) & (ar.dz < 1))),
    # selections.Selection('DiEleOSDZId', 'OS + DZ<1 + ID',  lambda ar: ((ar.sign < 1) & (ar.dz < 1) & (ar.leg0.idScore > 0.1) & (ar.leg1.idScore > 0.1))),
    # selections.Selection('DiEleOSDZIdPt5', 'OS + DZ<1 + ID + Pt>5',  lambda ar: ((ar.sign < 1) & (ar.dz < 1) & (ar.leg0.idScore > 0.1) & (ar.leg1.idScore > 0.1) & (ar.leg0.pt > 5) & (ar.leg1.pt > 5))),

    # selections.Selection('DiEleOSDZPt5', 'OS + DZ<1 + Pt>5',  lambda ar: ((ar.sign < 1) & (ar.dz < 1) & (ar.leg0.pt > 5) & (ar.leg1.pt > 5))),

]

bestsel = [
    selections.Selection('BestID', 'best-pair(IDScore)', lambda ar: ak.argmax(ar.idScore, axis=1, keepdims=True)),
    selections.Selection('BestB2B', 'best-pair(B2B)', lambda ar: ak.argmax(ar.dphi, axis=1, keepdims=True)),
    selections.Selection('BestPt', 'best-pair(p_{T}^{pair})', lambda ar: ak.argmax(ar.ptPair, axis=1, keepdims=True)),

]

selections.Selector.selection_primitives = sm.selections.copy()

gen_diele_selections = selections.Selector('^DiGEN$')()
diele_selections = (selections.Selector('^DiEle|all')*('^OS$|all')*('^Dz1$|all')*('^IDScore|all'))()
best_selections = (selections.Selector('^Best|all'))()

diele_plots = [
    GenDiElePlotter(
        coll.gen_diele,
        gen_diele_selections),
    DiElePlotter(
        coll.diTkEle,
        diele_selections,
        best_selections),
    DiElePlotter(
        coll.diTkEle_GENMatched,
        diele_selections,
        best_selections),

]

diele_plots_mb = [
    GenDiElePlotter(
        coll.gen_diele,
        gen_diele_selections),
    DiElePlotter(
        coll.diTkEle,
        diele_selections,
        best_selections),
    # DiElePlotter(
    #     coll.diTkEle_GENMatched,
    #     diele_selections),

]
