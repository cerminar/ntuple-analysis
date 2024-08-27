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
            self.h_mass = bh.TH1F(f'{name}_mass', 'mass (GeV); mass [GeV]', 50, 0, 25)
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
        # bh.fill_1Dhist(hist=self.h_eta,    array=egs.eta,    weights=weight)
        # # bh.fill_1Dhist(hist=self.h_energy, array=egs.energy, weights=weight)
        # bh.fill_1Dhist(hist=self.h_hwQual, array=egs.hwQual, weights=weight)
        # if 'tkIso' in egs.fields:
        #     bh.fill_1Dhist(hist=self.h_tkIso, array=egs.tkIso, weights=weight)
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
        # self.h_n.fill(ak.count(egs.pt, axis=1))
        # # bh.fill_1Dhist(hist=self.h_n, array=ak.count(egs.pt, axis=1), weights=weight)
        # # self.h_n.Fill()


# ------ Plotter classes ------------------------------------------------

class GenDiElePlotter(plotters.GenericDataFramePlotter):
    def __init__(self, data_set,
                 data_selections=[selections.Selection('all')],):
        super(GenDiElePlotter, self).__init__(GenDiEleHistos,
                                                data_set,
                                                data_selections)


# ------ Plotter instances

gen_diele_selections = [selections.Selection('all')]



diele_plots = [
    GenDiElePlotter(
        coll.gen_diele,
        gen_diele_selections),
]
