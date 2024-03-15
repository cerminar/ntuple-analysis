from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections

gen_ee_tk_selections = (selections.Selector('GEN$')*('Ee$')*('^Eta[A-C]$|EtaBC$|all')+selections.Selector('GEN$')*('Ee$')*('Pt15|Pt30'))()
gen_ee_selections = (selections.Selector('GEN$')*('^Eta[ABC]+[CD]$|^Eta[A-D]$|all')+selections.Selector('GEN$')*('^Pt15|^Pt30'))()
gen_selections = (selections.Selector('GEN$')*('^Eta[ABC]+[CD]$|^Eta[A-D,F]$|all')+selections.Selector('GEN$')*('^Pt15|^Pt30'))()

ctl2_eg_selections = (selections.Selector('^IDTightE$|all')*('^EtaE[EB]$|all')+selections.Selector('^Pt15|^Pt30'))()



hgc_tp_selections = (selections.Selector('^EgBdt*|^Em|all')*('PUId|all'))()
# hgc_tp_selections = (selections.Selector('^Eta[BC]+[CD]$|^Eta[A]$|all'))()
hgc_tp_rate_selections = (selections.Selector('^EgBdt*|^Em|all')*('PUId|all')*('^Eta[ABC]+[CD]$|all'))()

tkcl3dmatch_selections = (selections.Selector('PUId')*('^EgBdtLE|all')*('^Pt[1,2,5]$|all')*('^MTkPt[2-5]|all'))()

hgc_tp_id_selections = (selections.Selector('^IDTightEm$|^IDLooseEm$|all')+selections.Selector('^EgBdt|all'))()


double_gen_selections = [
    selections.build_DiObj_selection('DoubleGENEtaEB', 'GENEtaEB',
                          (selections.Selector('GEN$')*('^EtaEB$')).one(),
                          (selections.Selector('GEN$')*('^EtaEB$')).one()),
    selections.build_DiObj_selection('DoubleGENEtaEE', 'GENEtaEE',
                          (selections.Selector('GEN$')*('^EtaEE$')).one(),
                          (selections.Selector('GEN$')*('^EtaEE$')).one()),                          
    selections.build_DiObj_selection('DoubleGEN', 'GEN',
                          (selections.Selector('GEN$')).one(),
                          (selections.Selector('GEN$')).one()),                          

]


# *('PUId|all')

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

# for sel in tkcl3dmatch_selections:
#     print(sel)

hgc_tp_tkmatch_genmatched = [
    plotters.Cl3DGenMatchPlotter(
        collections.tkCl3DMatch, collections.sim_parts,
        tkcl3dmatch_selections, gen_ee_selections)                                 
]


zprime_eff_pt_bins = list(range(0,100, 10))+list(range(100,500, 100))+list(range(500, 1000, 250))+list(range(1000, 2000, 500))


hgc_tp_highpt_genmatched = [
    plotters.GenPlotter(
        collections.gen_ele,
        gen_ee_selections,
        pt_bins=range(0,4000, 5)),
    plotters.DiObjMassPlotter(
        collections.DoubleSimEle,
        double_gen_selections
    ),
    plotters.Cl3DGenMatchPlotter(
        collections.hgc_cl3d, collections.gen_ele,
        hgc_tp_id_selections, gen_ee_selections,
        pt_bins=zprime_eff_pt_bins),
    plotters.EGGenMatchPlotter(
        collections.TkEleL2, collections.gen_ele,
        ctl2_eg_selections, gen_selections,
        pt_bins=zprime_eff_pt_bins),

                               
]

for sel in gen_selections:
    print (sel)