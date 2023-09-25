from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections

sim_eg_match_ee_selections = (selections.Selector('^EGq[4-5]$')*('^Pt[1-3][0]$|all'))()
gen_ee_tk_selections = (selections.Selector('GEN$')*('Ee$')*('^Eta[A-C]$|EtaBC$|all')+selections.Selector('GEN$')*('Ee$')*('Pt15|Pt30'))()
gen_ee_selections = (selections.Selector('GEN$')*('Ee')*('^Eta[BC]+[CD]$|^Eta[A-D]$|all')+selections.Selector('GEN$')*('Ee')*('^Pt15|^Pt30'))()

tdrsim_eg_genmatched = [
    plotters.EGGenMatchPlotter(
        collections.egs_EE, collections.gen_parts,
        sim_eg_match_ee_selections, gen_ee_selections),
    # plotters.EGGenMatchPlotter(
    #     collections.egs_EB, collections.gen_parts,
    #     selections.eg_id_pt_eb_selections, selections.gen_eb_selections),
    plotters.EGGenMatchPlotter(
        collections.tkeles_EE, collections.gen_parts,
        sim_eg_match_ee_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.tkeles_EB, collections.gen_parts,
        selections.eg_id_pt_eb_selections, selections.gen_eb_selections),
    plotters.EGGenMatchPlotter(
        collections.tkem_EE, collections.gen_parts,
        sim_eg_match_ee_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.tkem_EB, collections.gen_parts,
        selections.eg_id_pt_eb_selections, selections.gen_eb_selections),
]

# FIXME: should become in newer versions
# l1tc_match_ee_selections = (selections.Selector('^EGq[1,3]$')*('^Pt[1-2][0]$|all'))()
l1tc_match_ee_selections = (selections.Selector('^EGq[1,3]$')*('^Pt[1-2][0]$|all'))()

l1tc_eg_genmatched = [
    plotters.EGGenMatchPlotter(
        collections.EGStaEE, collections.sim_parts,
        l1tc_match_ee_selections, gen_ee_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEleEE, collections.sim_parts,
        l1tc_match_ee_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEleEB, collections.sim_parts,
        selections.eg_id_pt_eb_selections, selections.gen_eb_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEmEE, collections.sim_parts,
        l1tc_match_ee_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEmEB, collections.sim_parts,
        selections.eg_id_pt_eb_selections, selections.gen_eb_selections),
]

# FIXME: this one can be dropped in newer versions
l1tc_fw_match_ee_selections = (selections.Selector('^EGq[2,4]or[3,5]$')*('^Pt[1-2][0]$|all'))()

l1tc_fw_eg_genmatched = [
    # plotters.EGGenMatchPlotter(
    #     collections.egs_EE_pfnf, collections.gen_parts,
    #     l1tc_fw_match_ee_selections, gen_ee_selections),
    # plotters.EGGenMatchPlotter(
    #     collections.tkeles_EE_pfnf, collections.gen_parts,
    #     l1tc_fw_match_ee_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.tkeles_EB_pfnf, collections.gen_parts,
        selections.eg_id_pt_eb_selections, selections.gen_eb_selections),
    # plotters.EGGenMatchPlotter(
    #     collections.tkem_EE_pfnf, collections.gen_parts,
    #     l1tc_fw_match_ee_selections, gen_ee_tk_selections),
    # plotters.EGGenMatchPlotter(
    #     collections.tkem_EB_pfnf, collections.gen_parts,
    #     selections.eg_id_pt_eb_selections, selections.gen_eb_selections),

]

l1tc_rate_pt_wps = [
    plotters.EGGenMatchPtWPSPlotter(
        collections.EGStaEE, collections.sim_parts,
        gen_ee_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.TkEleEE, collections.sim_parts,
        gen_ee_tk_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.TkEleEB, collections.sim_parts,
        selections.gen_eb_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.TkEmEE, collections.sim_parts,
        gen_ee_tk_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.TkEmEB, collections.sim_parts,
        selections.gen_eb_selections),
]


egid_sta_selections = (selections.Selector('^IDTightS|all')*('^Pt[1-3][0]$|all'))()
# egid_iso_tkele_selections = (selections.Selector('^IDTight[E]|all')*('^Pt[1-3][0]$|all')*('^Iso0p[1-2]|all'))()
# egid_iso_tkpho_selections = (selections.Selector('^IDTight[P]|all')*('^Pt[1-3][0]$|all')*('^Iso0p[1-2]|all'))()
egid_iso_tkele_selections = (selections.Selector('^IDTight[E]|all')*('^Pt[1-3][0]$|all'))()
egid_iso_tkpho_selections = (selections.Selector('^IDTight[P]|all')*('^Pt[1-3][0]$|all'))()
egid_iso_tkele_comp_selections = (selections.Selector('^IDTight[E]|^IDComp|all')*('^Pt[1-3][0]$|all'))()


gen_selections = (selections.Selector('GEN$')*('^Eta[F]$|^Eta[AF][ABCD]*[C]$|all')+selections.Selector('GEN$')*('^Ee|all')*('^Pt15|^Pt30'))()

# for sels in [gen_selections, selections.gen_selections]:
#     print('--------------------')
#     print(f'# of sels: {len(sels)}')
#     for sel in sels:
#         print(sel)

l1tc_emu_genmatched = [
    # plotters.EGGenMatchPlotter(
    #     collections.EGStaEE, collections.sim_parts,
    #     egid_sta_selections, gen_ee_selections),
    # plotters.EGGenMatchPlotter(
    #     collections.EGStaEB, collections.sim_parts,
    #     egid_sta_selections, selections.gen_eb_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEleEE, collections.sim_parts,
        egid_iso_tkele_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEleEB, collections.sim_parts,
        egid_iso_tkele_selections, selections.gen_eb_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEmEE, collections.sim_parts,
        egid_iso_tkpho_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEmEB, collections.sim_parts,
        egid_iso_tkpho_selections, selections.gen_eb_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEmL2, collections.sim_parts,
        egid_iso_tkpho_selections, selections.gen_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEleL2, collections.sim_parts,
        egid_iso_tkele_selections, selections.gen_selections),
    
]


l1tc_l1emu_eb_genmatched = [
    plotters.EGGenMatchPlotter(
        collections.EGStaEB, collections.sim_parts,
        egid_sta_selections, selections.gen_eb_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEleEB, collections.sim_parts,
        egid_iso_tkele_selections, selections.gen_eb_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEmEB, collections.sim_parts,
        egid_iso_tkpho_selections, selections.gen_eb_selections),
]

l1tc_l1emu_ee_genmatched = [
    # plotters.EGGenMatchPlotter(
    #     collections.EGStaEE, collections.sim_parts,
    #     egid_sta_selections, gen_ee_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEleEE, collections.sim_parts,
        egid_iso_tkele_comp_selections, gen_ee_tk_selections),
    # plotters.EGGenMatchPlotter(
    #     collections.TkEmEE, collections.sim_parts,
    #     egid_iso_tkpho_selections, gen_ee_tk_selections),
]

l1tc_l1emu_ee_ell_genmatched = [
    plotters.EGGenMatchPlotter(
        collections.TkEleEllEE, collections.sim_parts,
        egid_iso_tkele_comp_selections, gen_ee_tk_selections),
]


l1tc_l2emu_genmatched = [
    plotters.EGGenMatchPlotter(
        collections.TkEmL2, collections.sim_parts,
        egid_iso_tkpho_selections, selections.gen_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEleL2, collections.sim_parts,
        egid_iso_tkele_selections, selections.gen_selections),   
]

l1tc_l2emu_ell_genmatched = [
    # plotters.EGGenMatchPlotter(
    #     collections.TkEmL2Ell, collections.sim_parts,
    #     egid_iso_tkpho_selections, selections.gen_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEleL2Ell, collections.sim_parts,
        egid_iso_tkele_selections, selections.gen_selections),   
]


egid_menu_ele_rate_selections = (selections.Selector('^SingleIsoTkEle|^SingleTkEle|^MenuEle')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_pho_rate_selections = (selections.Selector('^SingleIsoTkPho|^SingleEGEle|^MenuSta|^MenuPho')*selections.Selector('^EtaE[BE]$|all'))()


l1tc_l2emu_singlelepton_genmatched = [
    plotters.EGGenMatchPlotter(
        collections.TkEmL2, collections.sim_parts,
        egid_menu_pho_rate_selections, gen_selections),
    plotters.EGGenMatchPlotter(
        collections.TkEleL2, collections.sim_parts,
        egid_menu_ele_rate_selections, gen_selections),   
]


l1tc_l2emu_singlelepton_rate_pt_wps = [
    plotters.EGGenMatchPtWPSPlotter(
        collections.TkEmL2, collections.sim_parts,
        gen_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.TkEleL2, collections.sim_parts,
        gen_selections),
]

l1tc_emu_rate_pt_wps = [
    # plotters.EGGenMatchPtWPSPlotter(
    #     collections.EGStaEE, collections.sim_parts,
    #     gen_ee_selections),
    # plotters.EGGenMatchPtWPSPlotter(
    #     collections.TkEleEE, collections.sim_parts,
    #     gen_ee_tk_selections),
    # # plotters.EGGenMatchPtWPSPlotter(
    #     collections.TkEleEB, collections.sim_parts,
    #     selections.gen_eb_selections),
    # plotters.EGGenMatchPtWPSPlotter(
    #     collections.TkEmEE, collections.sim_parts,
    #     gen_ee_tk_selections),
    # plotters.EGGenMatchPtWPSPlotter(
    #     collections.TkEmEB, collections.sim_parts,
    #     selections.gen_eb_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.TkEleEE, collections.sim_parts,
        gen_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.TkEleEllEE, collections.sim_parts,
        gen_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.TkEmL2, collections.sim_parts,
        gen_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.TkEleL2, collections.sim_parts,
        gen_selections),
]

# for sel in egid_menu_pho_rate_selections:
#     print(sel)