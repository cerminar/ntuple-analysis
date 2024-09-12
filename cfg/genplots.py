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


# ------ Plotter classes ------------------------------------------------

class GenDiElePlotter(plotters.GenericDataFramePlotter):
    def __init__(self, data_set,
                 data_selections=[selections.Selection('all')],):
        super(GenDiElePlotter, self).__init__(GenDiEleHistos,
                                                data_set,
                                                data_selections)


# ------ Plotter instances

gen_diele_selections = selections.Selector('^DiGEN$')()
diele_selections = selections.Selector('all|^DiEle')()


diele_plots = [
    GenDiElePlotter(
        coll.gen_diele,
        gen_diele_selections),
    GenDiElePlotter(
        coll.diTkEle,
        diele_selections),

]
