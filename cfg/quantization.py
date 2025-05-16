from python import plotters, selections, calibrations, histos
import python.boost_hist as bh
import cfg.datasets.fastpuppi_collections as coll
import awkward as ak
import math
import numpy as np


# ------ Histogram classes ----------------------------------------------
class QuantizationHistos(histos.BaseHistos):
    def __init__(self, name, features=None, root_file=None, debug=False):
        if not root_file:
            self.features = features
            self.h_features = bh.TH2F_category(
                f'{name}_features',
                'features; feature; value',
                 self.features,
                 1000, -1000, 1000)
            self.h_featuresLog2 = bh.TH2F_category(
                f'{name}_featuresLog2',
                'featuresLog2; features; log_{2}(value)',
                 self.features,
                 64, -32, 32)
            # for bin,ft in enumerate(features):
            #     self.h_features.GetXaxis().SetBinLabel(bin+1, ft)
            #     self.h_featuresLog2.GetXaxis().SetBinLabel(bin+1, ft)

        histos.BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, df):
        fill = df
        # print(df.fields)
        for bin,ft in enumerate(self.features):
            fill[f'{ft}_bin'] = [ft]
            fill[f'{ft}_log2'] = np.log2(fill[ft])

            bh.fill_2Dhist(self.h_features, fill[f'{ft}_bin'], fill[ft])
            bh.fill_2Dhist(self.h_featuresLog2, fill[f'{ft}_bin'], fill[f'{ft}_log2'])


# ------ Plotter classes ------------------------------------------------
class QuantizationPlotter(plotters.GenericDataFramePlotter):
# class QuantizationPlotter(GenericDataFrameLazyPlotter):
    def __init__(self, data_set, data_selections, features):
        self.features = features
        super(QuantizationPlotter, self).__init__(QuantizationHistos, data_set, data_selections)

    def book_histos(self):
        self.data_set.activate()
        data_name = self.data_set.name
        for selection in self.data_selections:
            self.h_set[selection.name] = self.HistoClass(
                name=f'{data_name}_{selection.name}_nomatch',
                features=self.features)



simple_selections = (selections.Selector('^Pt[1-5]$|all')*('^EtaE[EB]$|all'))()

hgcid_plotters = [
    QuantizationPlotter(coll.decHadCaloEndcap,  simple_selections, ['pt',  'emf', 'abseta', 'meanz', 'sigmaetaeta', 'sigmaphiphi', 'sigmazz', 'showerlength', 'coreshowerlength', 'hwMeanZ', 'hwEmf', 'hwAbseta', 'hwAbsetaOffset256','hwAbsetaOffset320', 'hwSigmaetaeta', 'hwSigmaphiphi', 'hwSigmazz', 'hwFPMeanz']),
]

# for sel in simple_selections:
#     print(sel)

# l1tc_pho_plotters = [

# ]
