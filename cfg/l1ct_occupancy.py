from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections


pfin_hgc_tp_selections = (selections.Selector('^EgBdtLE|^Em|all')*('PUId|all')*('^Pt[1,2,5]$|all'))()
pfin_eb_selections = (selections.Selector('^Pt[1,2,5]$'))()
pfin_tk_selections = (selections.Selector('^TkPt'))()


# pfeg_tp_input_selections = (Selector('^PFinH')*('^Pt[1,2,5]$|all')*('^Em$|all'))()
# pfeg_ee_input_selections = (Selector('^PFinH')*('^Pt[1,2,5]$|all')*('^EGq[1]$|all'))()
# pftkinput_selections = (Selector('^PFinBRL|^PFinHGC$')*('^TkPt'))()


l1tcorr_input_occ = [
    plotters.CorrOccupancyPlotter(
        collections.hgc_cl3d_pfinputs, 
        pfin_hgc_tp_selections),
    plotters.CorrOccupancyPlotter(
        collections.EGStaEB_pfinputs, 
        pfin_eb_selections),
    plotters.CorrOccupancyPlotter(
        collections.tk_pfinputs, 
        pfin_tk_selections),
]





# # print('\n'.join([str(sel) for sel in hgc_tp_rate_selections]))
# hgc_tp_unmatched = [
#     plotters.Cl3DPlotter(collections.hgc_cl3d, hgc_tp_selections)
# ]


# hgc_tp_genmatched = [
#     plotters.Cl3DGenMatchPlotter(
#         collections.hgc_cl3d, collections.sim_parts,
#         hgc_tp_selections, gen_ee_selections)                                 
# ]


# hgc_tp_rate = [
#     plotters.HGCCl3DRatePlotter(
#         collections.hgc_cl3d, hgc_tp_rate_selections),
# ]

# hgc_tp_rate_pt_wps = [
#     plotters.HGCCl3DGenMatchPtWPSPlotter(
#         collections.hgc_cl3d, collections.sim_parts, 
#         gen_ee_selections)
# ]


# correlator_occupancy_plotters = [
#     plotters.CorrOccupancyPlotter(
#         collections.tk_pfinputs, selections.pftkinput_selections),
#     plotters.CorrOccupancyPlotter(
#         collections.egs_EE_pf_reg, selections.pfinput_regions),
#     plotters.CorrOccupancyPlotter(
#         collections.tkeles_EE_pf_reg, selections.pfinput_regions),
#     plotters.CorrOccupancyPlotter(
#         collections.tkeles_EB_pf_reg, selections.pfinput_regions),
#     plotters.CorrOccupancyPlotter(
#         collections.tkem_EE_pf_reg, selections.pfinput_regions),
#     plotters.CorrOccupancyPlotter(
#         collections.tkem_EB_pf_reg, selections.pfinput_regions),
#     plotters.CorrOccupancyPlotter(
#         collections.eg_EE_pfinputs, selections.pfeg_ee_input_selections),
#     plotters.CorrOccupancyPlotter(
#         collections.eg_EB_pfinputs, selections.pfeg_eb_input_selections),
#     plotters.CorrOccupancyPlotter(
#         collections.cl3d_hm_pfinputs, selections.pfeg_tp_input_selections),
# ]
