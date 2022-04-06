from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections

gen_ee_tk_selections = (selections.Selector('GEN$')*('Ee$')*('^Eta[A-C]$|EtaBC$|all')+selections.Selector('GEN$')*('Ee$')*('Pt15|Pt30'))()
gen_ee_selections = (selections.Selector('GEN$')*('Ee')*('^Eta[BC]+[CD]$|^Eta[A-D]$|all')+selections.Selector('GEN$')*('Ee')*('^Pt15|^Pt30'))()





hgc_tp_selections = (selections.Selector('^EgBdt*|^Em|all'))()
# hgc_tp_selections = (selections.Selector('^Eta[BC]+[CD]$|^Eta[A]$|all'))()
hgc_tp_rate_selections = (selections.Selector('^EgBdt*|^Em|all')*('^Eta[ABC]+[CD]$|all'))()

# print('\n'.join([str(sel) for sel in hgc_tp_rate_selections]))
hgc_tp_unmatched = [
    plotters.Cl3DPlotter(collections.hgc_cl3d, hgc_tp_selections)
]


hgc_tp_genmatched = [
    plotters.Cl3DGenMatchPlotter(
        collections.hgc_cl3d, collections.sim_parts,
        hgc_tp_selections, gen_ee_selections)                                 
]


hgc_tp_rate = [
    plotters.HGCCl3DRatePlotter(
        collections.hgc_cl3d, hgc_tp_rate_selections),
]

hgc_tp_rate_pt_wps = [
    plotters.HGCCl3DGenMatchPtWPSPlotter(
        collections.hgc_cl3d, collections.sim_parts, 
        gen_ee_selections)
]
