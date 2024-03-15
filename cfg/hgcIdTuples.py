from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections


# simple_selections = (selections.Selector('^EGq[4-5]$')*('^Pt[1-3][0]$|all'))()




comp_selections = (selections.Selector('^Pt15|all')&('^EtaABC$|^EtaBC$|all'))()
sim_eg_selections = (selections.Selector('^GEN$'))()
sim_pi_selections = (selections.Selector('^GENPi$'))()


egid_plotters = [
    # plotters.HGCIdTuplesPlotter(collections.hgc_cl3d, comp_selections),
    plotters.HGCIdMatchTuplesPlotter(collections.hgc_cl3d, collections.gen, comp_selections, sim_eg_selections)
]

piid_plotters = [
    # plotters.HGCIdTuplesPlotter(collections.hgc_cl3d, comp_selections),
    plotters.HGCIdMatchTuplesPlotter(collections.hgc_cl3d, collections.gen_pi, comp_selections, sim_pi_selections)
]

pu_plotters = [
    plotters.HGCIdTuplesPlotter(collections.hgc_cl3d, comp_selections),
    # plotters.CompCatTuplePlotter(collections.hgc_cl3d, collections.sim_parts, comp_selections, sim_selections)
]


# for sel in sim_selections:
#     print(sel)