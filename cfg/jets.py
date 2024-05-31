from python import collections, plotters, selections, histos
import python.boost_hist as bh
import cfg.datasets.fastpuppi_collections as coll
import numpy as np

class JetResoHistos(histos.BaseResoHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
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
                100, -0.4, 0.4)
            self.h_phiRes = bh.TH1F(
                f'{name}_phiRes',
                'Track phi reso',
                100, -0.4, 0.4)
            self.h_drRes = bh.TH1F(
                f'{name}_drRes',
                'Track DR reso',
                100, 0, 0.4)
            self.h_nMatch = bh.TH1F(
                f'{name}_nMatch',
                '# matches',
                100, 0, 100)

        histos.BaseResoHistos.__init__(self, name, root_file, debug)

    def fill(self, reference, target):
        bh.fill_1Dhist(self.h_ptResp, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVeta, reference.eta, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVpt, reference.pt, target.pt/reference.pt)
        bh.fill_1Dhist(self.h_etaRes, (target.eta - reference.eta))
        bh.fill_1Dhist(self.h_phiRes, (target.phi - reference.phi))
        bh.fill_1Dhist(self.h_drRes, np.sqrt((reference.phi-target.phi)**2+(reference.eta-target.eta)**2))

    def fill_nMatch(self, n_matches):
        self.h_nMatch.Fill(n_matches)


class JetHistos(histos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(f'{name}_pt',
                                'Track Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(f'{name}_eta',
                                 'Track eta; #eta;', 100, -4, 4)

        histos.BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tracks):
        bh.fill_1Dhist(self.h_pt, tracks.pt)
        bh.fill_1Dhist(self.h_eta, tracks.eta)




class JetGenMatchPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')]):
        super(JetGenMatchPlotter, self).__init__(JetHistos, JetResoHistos,
                                                   data_set, gen_set,
                                                   data_selections, gen_selections,
                                                   gen_eta_phi_columns=['eta', 'phi'],
                                                   drcut=0.3)


pfjet_selections = (selections.Selector('^Pt[34]0$|all'))()

genjet_selections = (selections.Selector('^GENJ$')*('^EtaE[EB]$|all')+selections.Selector('GENJ$')*('^Pt30'))()


jets_genmatched = [
    JetGenMatchPlotter(
        coll.pfjets, coll.gen_jet,
        pfjet_selections, genjet_selections),
]

if False:
    print('---- pfjet_selections ----------------')
    for sel in pfjet_selections:
        print(sel)

    print('---- genjet_selections ----------------')
    for sel in genjet_selections:
        print(sel)
