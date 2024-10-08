from python import collections, plotters, selections, histos
from python import boost_hist as bh
import cfg.datasets.fastpuppi_collections as coll
import awkward as ak


from cfg.eg_genmatch import EGGenMatchPlotter

class DecTkHistos(histos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(
                f'{name}_pt',
                'Pt (GeV); p_{T} [GeV]',
                100, 0, 100)
            # self.h_deltaPt = bh.TH1F(
            #     f'{name}_deltaPt',
            #     'Pt (GeV); p_{T}^{decoded}-p_{T}^{float}  [GeV]',
            #     100, -10, 10)
            # self.h_deltaPtVeta = bh.TH2F(
            #     f'{name}_deltaPtVeta',
            #     'Pt (GeV); #eta^{float}; p_{T}^{decoded}-p_{T}^{float}  [GeV]',
            #     50, -2.5, 2.5,
            #     50, -0.25, 0.25)
            # self.h_deltaPtVabseta = bh.TH2F(
            #     f'{name}_deltaPtVabseta',
            #     'Pt (GeV); |#eta^{float}|; p_{T}^{decoded}-p_{T}^{float}  [GeV]',
            #     50, 0, 2.5,
            #     50, -0.25, 0.25)
            self.h_eta = bh.TH1F(
                f'{name}_eta',
                '#eta; #eta;',
                100, -4, 4)
            self.h_z0 = bh.TH1F(
                f'{name}_z0',
                'z0; z_{0} [cm];',
                100, -10, 10)
            # FIXME: plots to check extrapolation...should be restored
            # self.h_deltaZ0 = bh.TH1F(
            #     f'{name}_deltaZ0',
            #     '#Delta z0; z0^{decoded}-z0^{float};',
            #     50, -0.2, 0.2)
            # self.h_deltaZ0Veta = bh.TH2F(
            #     f'{name}_deltaZ0Veta',
            #     '#Delta z0; #eta^{float}; z0^{decoded}-z0^{float};',
            #     100, -2.5, 2.5,
            #     50, -0.05, 0.05)
            # self.h_deltaEta = bh.TH1F(
            #     f'{name}_deltaEta',
            #     '#Delta #eta_{@vtx}; #eta_{@vtx}^{decoded}-#eta_{@vtx}^{float};',
            #     100, -1, 1)
            # self.h_deltaEtaVabseta = bh.TH2F(
            #     f'{name}_deltaEtaVabseta',
            #     '#Delta #eta_{@vtx} vs |#eta^{float}|; |#eta^{float}|; #eta_{@vtx}^{decoded}-#eta_{@vtx}^{float};',
            #     25, 0, 2.5,
            #     100, -0.004, 0.004)
            # self.h_deltaEtaVeta = bh.TH2F(
            #     f'{name}_deltaEtaVeta',
            #     '#Delta #eta_{@vtx} vs #eta^{float}; #eta^{float}; #eta_{@vtx}^{decoded}-#eta_{@vtx}^{float};',
            #     50, -2.5, 2.5,
            #     50, -0.004, 0.004)
            # self.h_deltaCaloEta = bh.TH1F(
            #     f'{name}_deltaCaloEta',
            #     '#Delta #eta_{@calo}; #eta_{@calo}^{decoded}-#eta_{@calo}^{float};',
            #     100, -1, 1)
            # self.h_deltaCaloEtaVabseta = bh.TH2F(
            #     f'{name}_deltaCaloEtaVabseta',
            #     '#Delta #eta_{@calo} vs |#eta^{float}|; |#eta^{float}|; #eta_{@calo}^{decoded}-#eta_{@calo}^{float};',
            #     50, 0, 2.5,
            #     100, -0.04, 0.04)
            # self.h_deltaCaloEtaVeta = bh.TH2F(
            #     f'{name}_deltaCaloEtaVeta',
            #     '#Delta #eta_{@calo} vs #eta^{float}; #eta^{float}; #eta_{@calo}^{decoded}-#eta_{@calo}^{float};',
            #     100, -2.5, 2.5,
            #     100, -0.04, 0.04)
            # self.h_deltaCaloPhi = bh.TH1F(
            #     f'{name}_deltaCaloPhi',
            #     '#Delta #phi_{@calo}; #phi_{@calo}^{decoded}-#phi_{@calo}^{float};',
            #     100, -1, 1)
            # self.h_deltaCaloPhiVabseta = bh.TH2F(
            #     f'{name}_deltaCaloPhiVabseta',
            #     '#Delta #phi_{@calo} vs |#eta^{float}|; |#phi^{float}|; #phi_{@calo}^{decoded}-#phi_{@calo}^{float};',
            #     100, 0, 2.5,
            #     100, -0.1, 0.1)

        histos.BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, egs):
        bh.fill_1Dhist(self.h_pt, egs.pt)
        # bh.fill_1Dhist(self.h_deltaPt, egs.deltaPt)
        # bh.fill_2Dhist(self.h_deltaPtVeta, egs.simeta, egs.deltaPt)
        # bh.fill_2Dhist(self.h_deltaPtVabseta, egs.simabseta, egs.deltaPt)
        bh.fill_1Dhist(self.h_eta, egs.eta)
        bh.fill_1Dhist(self.h_z0, egs.vz)
        # bh.fill_1Dhist(self.h_deltaZ0, egs.deltaZ0)
        # bh.fill_2Dhist(self.h_deltaZ0Veta, egs.simeta, egs.deltaZ0)
        # bh.fill_1Dhist(self.h_deltaEta, egs.deltaEta)
        # bh.fill_2Dhist(self.h_deltaEtaVabseta, egs.simabseta, egs.deltaEta)
        # bh.fill_2Dhist(self.h_deltaEtaVeta, egs.simeta, egs.deltaEta)
        # bh.fill_1Dhist(self.h_deltaCaloEta, egs.deltaCaloEta)
        # bh.fill_2Dhist(self.h_deltaCaloEtaVabseta, egs.simabseta, egs.deltaCaloEta)
        # bh.fill_2Dhist(self.h_deltaCaloEtaVeta, egs.simeta, egs.deltaCaloEta)
        # bh.fill_1Dhist(self.h_deltaCaloPhi, egs.deltaCaloPhi)
        # bh.fill_2Dhist(self.h_deltaCaloPhiVabseta, egs.simabseta, egs.deltaCaloPhi)


class DecTkResoHistos(histos.BaseResoHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptResVpt = bh.TH2F(
                f'{name}_ptResVpt',
                'Track Pt reso. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}-p_{T}^{GEN} [GeV];',
                50, 0, 100, 100, -20, 20)
            self.h_ptResp = bh.TH1F(
                f'{name}_ptResp',
                'Track Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 3)
            self.h_ptRespVpt = bh.TH2F(
                f'{name}_ptRespVpt',
                'Track Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 3)
            self.h_ptRespVeta = bh.TH2F(
                f'{name}_ptRespVeta',
                'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                50, -4, 4, 100, 0, 3)
            self.h_etaRes = bh.TH1F(
                f'{name}_etaRes',
                'Track eta reso',
                100, -0.15, 0.15)
            self.h_etaResVabseta = bh.TH2F(
                f'{name}_etaResVabseta',
                '#eta_{@vtx} reso; |#eta^{GEN}|; #eta_{@vtx}^{L1} vs #eta_{@vtx}^{GEN}',
                50, 0, 2.5,
                100, -0.1, 0.1)
            self.h_etaResVeta = bh.TH2F(
                f'{name}_etaResVeta',
                '#eta_{@vtx} reso; #eta^{GEN}; #eta_{@vtx}^{L1} vs #eta_{@vtx}^{GEN}',
                200, -2.5, 2.5,
                100, -0.1, 0.1)
            self.h_phiRes = bh.TH1F(
                f'{name}_phiRes',
                'Track phi reso',
                100, -0.4, 0.4)
            self.h_caloEtaRes = bh.TH1F(
                f'{name}_caloEtaRes',
                '#eta_{@calo} reso; #eta_{@calo}^{L1} vs #eta_{@calo}^{GEN}',
                100, -0.15, 0.15)
            self.h_caloEtaResVabseta = bh.TH2F(
                f'{name}_caloEtaResVabseta',
                '#eta_{@calo} reso; |#eta^{GEN}|; #eta_{@calo}^{L1} vs #eta_{@calo}^{GEN}',
                50, 0, 2.5,
                100, -0.1, 0.1)
            self.h_caloEtaResVeta = bh.TH2F(
                f'{name}_caloEtaResVeta',
                '#eta_{@calo} reso; #eta^{GEN}; #eta_{@calo}^{L1} vs #eta_{@calo}^{GEN}',
                200, -2.5, 2.5,
                100, -0.1, 0.1)
            self.h_caloPhiRes = bh.TH1F(
                f'{name}_caloPhiRes',
                '#phi_{@calo} reso; #phi_{@calo}^{L1} vs #phi_{@calo}^{GEN}',
                100, -0.4, 0.4)
            self.h_caloPhiResVabseta = bh.TH2F(
                f'{name}_caloPhiResVabseta',
                '#phi_{@calo} reso; |#eta^{GEN}|; #phi_{@calo}^{L1} vs #phi_{@calo}^{GEN}',
                50, 0, 3,
                100, -0.4, 0.4)
            self.h_dzRes = bh.TH1F(
                f'{name}_dzRes',
                '#DeltaZ_{0} res; #DeltaZ_{0}^{L1}-#DeltaZ_{0}^{GEN}',
                100, -10, 10)

            # self.h_caloPhiResVeta = bh.TH2F(
            #     name+'_caloPhiResVabseta',
            #     '#phi_{@calo} reso; #eta^{GEN}; #phi_{@calo}^{L1} vs #phi_{@calo}^{GEN}',
            #     50, 0, 3,
            #     100, -0.4, 0.4)
            self.h_nMatch = bh.TH1F(
                f'{name}_nMatch',
                '# matches',
                100, 0, 100)

            # self.h_pt2stResVpt = bh.TH2F(name+'_pt2stResVpt', 'EG Pt 2stubs reso. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}-p_{T}^{GEN} [GeV];',
            #                                50, 0, 100, 100, -20, 20)
            #
            # self.h_pt2stResp = bh.TH1F(name+'_pt2stResp', 'Track Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
            #                              100, 0, 3)
            # self.h_pt2stRespVpt = bh.TH2F(name+'_pt2stRespVpt', 'Track Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
            #                                 50, 0, 100, 100, 0, 3)
            # self.h_pt2stRespVeta = bh.TH2F(name+'_pt2stRespVeta', 'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
            #                                  50, -4, 4, 100, 0, 3)

        histos.BaseResoHistos.__init__(self, name, root_file, debug)

    def fill(self, reference, target):
        bh.fill_1Dhist(self.h_ptResp, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptResVpt, reference.pt, target.pt-reference.pt)
        bh.fill_2Dhist(self.h_ptRespVeta, reference.eta, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVpt, reference.pt, target.pt/reference.pt)
        bh.fill_1Dhist(self.h_etaRes, target.eta - reference.eta)
        bh.fill_2Dhist(self.h_etaResVabseta, reference.abseta, target.eta - reference.eta)
        bh.fill_2Dhist(self.h_etaResVeta, reference.eta, target.eta - reference.eta)
        bh.fill_1Dhist(self.h_phiRes, target.phi - reference.phi)
        bh.fill_1Dhist(self.h_caloEtaRes, target.caloEta - reference.caloeta)
        bh.fill_1Dhist(self.h_caloPhiRes, target.caloPhi - reference.calophi)
        bh.fill_2Dhist(self.h_caloEtaResVabseta, reference.abseta, target.caloEta - reference.caloeta)
        bh.fill_2Dhist(self.h_caloPhiResVabseta, reference.abseta, target.caloPhi - reference.calophi)
        bh.fill_2Dhist(self.h_caloEtaResVeta, reference.eta, target.caloEta - reference.caloeta)
        bh.fill_1Dhist(self.h_dzRes, target.vz - reference.vz)
        # bh.fill_1Dhist(self.h_nMatch, ak.count(reference.pt))



class DecTkPlotter(plotters.GenericDataFramePlotter):
    def __init__(self, tk_set, tk_selections=[selections.Selection('all')]):
        super(DecTkPlotter, self).__init__(DecTkHistos, tk_set, tk_selections)


class DecTrackGenMatchPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')]):
        super(DecTrackGenMatchPlotter, self).__init__(
            DecTkHistos, DecTkResoHistos,
            data_set, gen_set,
            data_selections, gen_selections,
            gen_eta_phi_columns=['eta', 'phi'])



dectk_selections = (selections.Selector('^EtaE[BE]$|all')*('^Pt[1,2,5][0]$|all'))()
dectk_match_selections = (selections.Selector('^Pt5$|^Pt[1,2,5][0]$|all'))()
# track_selections = (selections.Selector('^TkCTL1|all')&('^Pt5$|^Pt[1,2,5][0]$|all'))()
gen_tk_selections = (selections.Selector('GEN$')*('EtaE[BE]$|all')+selections.Selector('GEN$')*('Pt15|Pt30'))()

decTk_plotters = [
    DecTkPlotter(
        coll.decTkBarrel,
        dectk_selections
    ),
    DecTrackGenMatchPlotter(
        coll.decTkBarrel,
        coll.gen,
        dectk_match_selections,
        gen_tk_selections
    ),
    DecTrackGenMatchPlotter(
        coll.tkClMatchBarrel,
        coll.gen,
        dectk_match_selections,
        gen_tk_selections
    ),
    EGGenMatchPlotter(
        coll.TkEleL2,
        coll.gen,
        dectk_match_selections,
        gen_tk_selections,
        gen_eta_phi_columns=('eta', 'phi')
    )
]

# tk_plotters = [
#     plotters.TrackPlotter(
#         collections.tracks,
#         track_selections
#     ),
#     plotters.TrackGenMatchPlotter(
#         collections.tracks,
#         collections.sim_parts,
#         track_selections,
#         gen_tk_selections
#     )
# ]

# for sel in gen_tk_selections:
#     print (sel)
