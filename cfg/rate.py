from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections


simeg_rate_ee_selections = (selections.Selector('^EGq[4-5]$')*('^Eta[^DA][BC]*[BCD]$|all'))()
emueg_rate_ee_selections = (selections.Selector('^^EGq[1,3]$')*('^Eta[^DA][BC]*[BCD]$|all'))()
emueg_fw_rate_ee_selections = (selections.Selector('^^EGq[1,3]$')*('^Eta[^DA][BC]*[BCD]$|all'))()

# sim_eg_match_ee_selections = (selections.Selector('^EGq[4-5]$')*('^Pt[1-3][0]$|all'))()
# gen_ee_tk_selections = (selections.Selector('GEN$')*('Ee$')*('^Eta[A-C]$|EtaBC$|all')+selections.Selector('GEN$')*('Ee$')*('Pt15|Pt30'))()
# gen_ee_selections = (selections.Selector('GEN$')*('Ee')*('^Eta[BC]+[CD]$|^Eta[A-D]$|all')+selections.Selector('GEN$')*('Ee')*('^Pt15|^Pt30'))()
tp_plotters = [
    plotters.RatePlotter(collections.cl3d_hm, selections.tp_rate_selections),
    # plotters.RatePlotter(collections.cl3d_hm_calib, selections.tp_rate_selections),
    # plotters.RatePlotter(collections.cl3d_hm_shapeDr, selections.tp_rate_selections),
    # plotters.RatePlotter(collections.cl3d_hm_shapeDr_calib, selections.tp_rate_selections),
    # plotters.RatePlotter(collections.cl3d_hm_shapeDr_calib_new, selections.tp_rate_selections),
    # plotters.RatePlotter(collections.cl3d_hm_shapeDtDu_calib, selections.tp_rate_selections),
    # plotters.RatePlotter(collections.cl3d_hm_emint, selections.tp_rate_selections),
]

eg_tdrsim_plotters = [
    # plotters.RatePlotter(
    #     collections.egs_EE, selections.eg_id_eta_ee_selections),
    # plotters.RatePlotter(
    #     collections.egs_EB, selections.eg_barrel_rate_selections),
    plotters.RatePlotter(
        collections.tkeles_EE, simeg_rate_ee_selections),
    # plotters.RatePlotter(
    #     collections.tkeles_EB, selections.barrel_rate_selections),
    plotters.RatePlotter(
        collections.tkem_EE, simeg_rate_ee_selections),
]

eg_emu_plotters = [
    plotters.RatePlotter(
        collections.TkEmEE, emueg_rate_ee_selections),
    plotters.RatePlotter(
        collections.TkEmEB, selections.barrel_rate_selections),
    plotters.RatePlotter(
        collections.TkEleEE, emueg_rate_ee_selections),
    plotters.RatePlotter(
        collections.TkEleEB, selections.barrel_rate_selections),
    # plotters.RatePlotter(
    #     collections.tkem_EE_pfnf, selections.eg_id_iso_eta_ee_selections),
    # plotters.RatePlotter(
    #     collections.tkem_EB_pfnf, selections.barrel_rate_selections),
]

eg_emufw_plotters = [
    plotters.RatePlotter(
        collections.tkem_EE_pfnf, emueg_fw_rate_ee_selections),
    # plotters.RatePlotter(
    #     collections.tkem_EB_pfnf, selections.barrel_rate_selections),
    plotters.RatePlotter(
        collections.tkeles_EE_pfnf, emueg_fw_rate_ee_selections),
    # plotters.RatePlotter(
    #     collections.tkeles_EB_pfnf, selections.barrel_rate_selections),
    # plotters.RatePlotter(
    #     collections.tkem_EE_pfnf, selections.eg_id_iso_eta_ee_selections),
    # plotters.RatePlotter(
    #     collections.tkem_EB_pfnf, selections.barrel_rate_selections),

]
