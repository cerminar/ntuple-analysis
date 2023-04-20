from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections


simeg_rate_ee_selections = (selections.Selector('^EGq[4-5]$')*('^Eta[^DA][BC]*[BCD]$|all'))()
emueg_rate_ee_selections = (selections.Selector('^EGq[1,3]$')*('^Eta[A][BC]*[C]$')*('^Iso|all'))()

emueg_fw_rate_ee_selections = (selections.Selector('^EGq[1,3]$')*('^Eta[A][BC]*[BCD]$|all')*('^Iso|all'))()
emueg_rate_eb_selections = (selections.Selector('^LooseTkID$|all')*('^Eta[F]$')*('^Iso|all'))()


# print(egid_eta_selections)

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

eg_emu_oldID_plotters = [
    plotters.RatePlotter(
        collections.TkEmEEOldID, emueg_rate_ee_selections),
    plotters.RatePlotter(
        collections.TkEmEBOldID, emueg_rate_eb_selections),
    plotters.RatePlotter(
        collections.TkEleEEOldID, emueg_rate_ee_selections),
    plotters.RatePlotter(
        collections.TkEleEBOldID, emueg_rate_eb_selections),
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

egid_eta_selections = (selections.Selector('^IDTightS|all')*selections.Selector('^Eta[F]$|^Eta[AF][ABCD]*[CD]$'))()
egid_etatk_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Eta[F]$|^Eta[AF][ABCD]*[C]$'))()
egid_iso_etatk_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Iso|all')*selections.Selector('^Eta[F]$|^Eta[AF][ABCD]*[C]$'))()

egid_menu_ele_selections = (selections.Selector('^MenuEle')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_pho_selections = (selections.Selector('^MenuPho')*selections.Selector('^EtaE[BE]$|all'))()

egid_menu_ele_rate_selections = (selections.Selector('^SingleIsoTkEle|^SingleTkEle')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_pho_rate_selections = (selections.Selector('^SingleIsoTkPho')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_diele_rate_selections = (selections.Selector('^DoubleTkEle'))()
egid_menu_dipho_rate_selections = (selections.Selector('^DoubleIsoTkPho'))()


egid_eta_ee_selections = (selections.Selector('^IDTightS|all')*selections.Selector('^Eta[A][BCD]*[CD]$'))()
egid_eta_eetk_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Eta[A][BCD]*[C]$'))()
egid_iso_eta_eetk_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Iso|all')*selections.Selector('^Eta[A][BCD]*[C]$'))()

egid_iso_eta_eetk_selections_comp = (selections.Selector('^IDTight[E]|^IDCompWP|all')*selections.Selector('^Eta[AB][BCD]*[C]$'))()

egid_eta_eb_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Eta[F]$|all'))()
egid_iso_eta_eb_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Iso|all')*selections.Selector('^Eta[F]$|all'))()


# for sel in egid_iso_etatk_selections:
#     print(sel)

eg_emuCTl1_plotters = [
    plotters.RatePlotter(
        collections.EGStaEE, egid_eta_ee_selections),
    plotters.RatePlotter(
        collections.EGStaEB, egid_eta_eb_selections),
    plotters.RatePlotter(
        collections.TkEmEE, egid_iso_eta_eetk_selections),
    plotters.RatePlotter(
        collections.TkEmEB, egid_iso_eta_eb_selections),
    plotters.RatePlotter(
        collections.TkEleEE, egid_iso_eta_eetk_selections_comp),
    plotters.RatePlotter(
        collections.TkEleEB, egid_iso_eta_eb_selections),
]

eg_emuCTl2_plotters = [
    plotters.RatePlotter(
        collections.TkEmL2, egid_iso_etatk_selections),
    plotters.RatePlotter(
        collections.TkEleL2, egid_iso_etatk_selections),
]

eg_emuCTl1_ell_plotters = [
    plotters.RatePlotter(
        collections.TkEleEllEE, egid_iso_eta_eetk_selections_comp),
]


eg_emuCTl2_ell_plotters = [
    plotters.RatePlotter(
        collections.TkEmL2Ell, egid_iso_etatk_selections),
    plotters.RatePlotter(
        collections.TkEleL2Ell, egid_iso_etatk_selections),
]


eg_menuCTl2_plotters = [
    plotters.RatePlotter(
        collections.TkEmL2, egid_menu_pho_selections),
    plotters.RatePlotter(
        collections.TkEleL2, egid_menu_ele_selections),
]

eg_menuCTl2_ell_plotters = [
    plotters.RatePlotter(
        collections.TkEmL2Ell, egid_menu_pho_selections),
    plotters.RatePlotter(
        collections.TkEleL2Ell, egid_menu_ele_selections),
]



eg_menuCTl2_rate = [
    plotters.RateCounter(
        collections.TkEmL2, egid_menu_pho_rate_selections),
    plotters.RateCounter(
        collections.TkEleL2, egid_menu_ele_rate_selections),
    plotters.RateCounter(
        collections.TkEleL2Ell, egid_menu_ele_rate_selections),
    plotters.DoubleObjRateCounter(
        collections.DoubleTkEleL2, egid_menu_diele_rate_selections),
    plotters.DoubleObjRateCounter(
        collections.DoubleTkEmL2, egid_menu_dipho_rate_selections)
]
