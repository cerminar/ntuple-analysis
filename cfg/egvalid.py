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
l1tc_match_ee_selections = (selections.Selector('^EGq[1,2]$')*('^Pt[1-2][0]$|all'))()

l1tc_eg_genmatched = [
    plotters.EGGenMatchPlotter(
        collections.egs_EE_pfnf, collections.gen_parts,
        l1tc_match_ee_selections, gen_ee_selections),
    plotters.EGGenMatchPlotter(
        collections.tkeles_EE_pfnf, collections.gen_parts,
        l1tc_match_ee_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.tkeles_EB_pfnf, collections.gen_parts,
        selections.eg_id_pt_eb_selections, selections.gen_eb_selections),
    plotters.EGGenMatchPlotter(
        collections.tkem_EE_pfnf, collections.gen_parts,
        l1tc_match_ee_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.tkem_EB_pfnf, collections.gen_parts,
        selections.eg_id_pt_eb_selections, selections.gen_eb_selections),

]

# FIXME: this one can be dropped in newer versions
l1tc_fw_match_ee_selections = (selections.Selector('^EGq[2,4]or[3,5]$')*('^Pt[1-2][0]$|all'))()

l1tc_fw_eg_genmatched = [
    plotters.EGGenMatchPlotter(
        collections.egs_EE_pfnf, collections.gen_parts,
        l1tc_fw_match_ee_selections, gen_ee_selections),
    plotters.EGGenMatchPlotter(
        collections.tkeles_EE_pfnf, collections.gen_parts,
        l1tc_fw_match_ee_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.tkeles_EB_pfnf, collections.gen_parts,
        selections.eg_id_pt_eb_selections, selections.gen_eb_selections),
    plotters.EGGenMatchPlotter(
        collections.tkem_EE_pfnf, collections.gen_parts,
        l1tc_fw_match_ee_selections, gen_ee_tk_selections),
    plotters.EGGenMatchPlotter(
        collections.tkem_EB_pfnf, collections.gen_parts,
        selections.eg_id_pt_eb_selections, selections.gen_eb_selections),

]

l1tc_rate_pt_wps = [
    plotters.EGGenMatchPtWPSPlotter(
        collections.egs_EE_pfnf, collections.gen_parts,
        gen_ee_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.tkeles_EE_pfnf, collections.gen_parts,
        gen_ee_tk_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.tkeles_EB_pfnf, collections.gen_parts,
        selections.gen_eb_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.tkem_EE_pfnf, collections.gen_parts,
        gen_ee_tk_selections),
    plotters.EGGenMatchPtWPSPlotter(
        collections.tkem_EB_pfnf, collections.gen_parts,
        selections.gen_eb_selections),
]
