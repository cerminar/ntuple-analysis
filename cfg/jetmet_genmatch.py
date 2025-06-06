from python import plotters, selections, calibrations, histos
import python.boost_hist as bh
import cfg.datasets.fastpuppi_collections as coll
import awkward as ak
import numpy as np
import math

# ------ Histogram classes ----------------------------------------------

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
                100, 0, 500, 100, 0, 3)
            self.h_ptRespVeta = bh.TH2F(
                f'{name}_ptRespVeta',
                'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 4, 100, 0, 3)
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
                                'Track Pt (GeV); p_{T} [GeV]', 100, 0, 500)
            self.h_eta = bh.TH1F(f'{name}_eta',
                                 'Track eta; #eta;', 100, -4, 4)

        histos.BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tracks):
        bh.fill_1Dhist(self.h_pt, tracks.pt)
        bh.fill_1Dhist(self.h_eta, tracks.eta)

# ------ Plotter classes ------------------------------------------------

class JetGenMatchPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')],
                 gen_eta_phi_columns=('eta', 'phi'),
                 pt_bins=range(0, 500, 5)):
        super(JetGenMatchPlotter, self).__init__(JetHistos, JetResoHistos,
                                                data_set, gen_set,
                                                data_selections, gen_selections,
                                                gen_eta_phi_columns=gen_eta_phi_columns,
                                                pt_bins=pt_bins,
                                                drcut=0.3)


class EGGenMatchPtWPSPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set, gen_selections):
        super(EGGenMatchPtWPSPlotter, self).__init__(
            JetHistos, EGResoHistos,
            data_set, gen_set,
            [], gen_selections)

    def book_histos(self):
        calib_mgr = calibrations.CalibManager()
        rate_pt_wps = calib_mgr.get_calib('rate_pt_wps')
        self.data_selections = selections.rate_pt_wps_selections(
            rate_pt_wps, self.data_set.name)
        plotters.GenericGenMatchPlotter.book_histos(self)



# ------ Plotter instances



gen_selections = (selections.Selector('GENJ$')*('^EtaE[EB]$|^Eta(V)?Fwd$|all')+selections.Selector('GENJ$')*('^Pt(30|100)'))()
jet_selections = (selections.Selector('all$'))()


jets = [
    JetGenMatchPlotter(
        coll.calo_jets, coll.gen_jet,
        jet_selections, gen_selections),
    JetGenMatchPlotter(
        coll.pf_jets, coll.gen_jet,
        jet_selections, gen_selections),
    JetGenMatchPlotter(
        coll.puppi_jets, coll.gen_jet,
        jet_selections, gen_selections),
    JetGenMatchPlotter(
        coll.tk_jets, coll.gen_jet,
        jet_selections, gen_selections),
    JetGenMatchPlotter(
        coll.sc_corr_jets, coll.gen_jet,
        jet_selections, gen_selections),
    JetGenMatchPlotter(
        coll.sc_jets, coll.gen_jet,
        jet_selections, gen_selections),

]

