from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections


# simple_selections = (selections.Selector('^EGq[4-5]$')*('^Pt[1-3][0]$|all'))()

comp_selections = (selections.Selector('^Pt15|all')&('^EtaABC$|^EtaBC$|all'))()
sim_selections = (selections.Selector('^GEN$')&('^Ee$|all')&('^Pt15|all')&('^EtaABC$|^EtaBC$|all'))()

compid_plotters = [
    plotters.CompTuplesPlotter(collections.TkEleEE, comp_selections),
    plotters.CompCatTuplePlotter(collections.TkEleEE, collections.sim_parts, comp_selections, sim_selections)
]

# for sel in sim_selections:
#     print(sel)