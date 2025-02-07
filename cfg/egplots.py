from python import plotters, selections, calibrations, histos
import python.boost_hist as bh
import cfg.datasets.fastpuppi_collections as coll
import awkward as ak
import math
import numpy as np


# ------ Histogram classes ----------------------------------------------
from cfg.eg_genmatch import EGHistos


# ------ Plotter classes ------------------------------------------------

class EGPlotter(plotters.GenericDataFramePlotter):
    def __init__(self, eg_set, eg_selections=[selections.Selection('all')]):
        super(EGPlotter, self).__init__(EGHistos, eg_set, eg_selections)


# simple_selections = (selections.Selector('^EGq[4-5]$')*('^Pt[1-3][0]$|all'))()

simple_selections = (selections.Selector('^Pt[1-5]$|all')*('^EtaE[EB]$|all')*('^IDTight[EPS]$|all'))()
sta_selection = (selections.Selector('^IDTight[EPS]|all')*selections.Selector('^Pt5|all')*selections.Selector('^EtaABC$|all'))()
# print(f"simple_selections: {simple_selections}")

egid_iso_etatk_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Iso|all')*selections.Selector('^Eta[F]$|^Eta[AF][ABCD]*[C]$'))()

l1tc_simple_plotters = [
    # EE Tk-electrons
    # plotters.TkElePlotter(collections.TkEleEE, egid_iso_etatk_selections),
    # plotters.TkElePlotter(collections.TkEleEB, egid_iso_etatk_selections),
    # plotters.TkElePlotter(collections.TkEleL2, egid_iso_etatk_selections),

    # plotters.EGPlotter(collections.TkEleEE, sta_selection),

    # plotters.EGPlotter(collections.TkEmEE, sta_selection),
    # plotters.EGPlotter(collections.TkEmEB, sta_selection),
    # plotters.EGPlotter(collections.EGStaEE, sta_selection),
    # plotters.EGPlotter(collections.EGStaEB, sta_selection),
    ]

l1ct_tkegl2_plotters = [
    EGPlotter(coll.TkEleL2, simple_selections),
    EGPlotter(coll.TkEmL2, simple_selections),
]


quantization_plotters = [
    plotters.QuantizationPlotter(coll.hgc_cl3d,  simple_selections, ['pt', 'hoe', 'srrtot', 'meanz_scaled']),
    plotters.QuantizationPlotter(coll.tracks,  simple_selections, ['nStubs', 'chi2'])

]

# for sel in simple_selections:
#     print(sel)

# l1tc_pho_plotters = [

# ]
