from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections

# sim_eg_match_ee_selections = (selections.Selector('^EGq[4-5]$')*('^Pt[1-3][0]$|all'))()
# gen_ee_tk_selections = (selections.Selector('GEN$')*('Ee$')*('^Eta[A-C]$|EtaBC$|all')+selections.Selector('GEN$')*('Ee$')*('Pt15|Pt30'))()
gen_selections = (selections.Selector('GEN$')*('Ee|all')*('^EtaE[EB]$|all'))()
egid_tkele_selections = (selections.Selector('^IDTight[E]|all'))()


diobj_eta_selections = [
    selections.build_DiObj_selection('DoubleEtaEB', 'EtaEB',
                          (selections.Selector('^EtaEB$')).one(),
                          (selections.Selector('^EtaEB$')).one()),
    selections.build_DiObj_selection('DoubleEtaEE', 'EE',
                          (selections.Selector('^EtaEE$')).one(),
                          (selections.Selector('^EtaEE$')).one()),                          
]

double_gen_selections = [
    selections.build_DiObj_selection('DoubleGENEtaEB', 'GENEtaEB',
                          (selections.Selector('GEN$')*('^EtaEB$')).one(),
                          (selections.Selector('GEN$')*('^EtaEB$')).one()),
    selections.build_DiObj_selection('DoubleGENEtaEE', 'GENEtaEE',
                          (selections.Selector('GEN$')*('^EtaEE$')).one(),
                          (selections.Selector('GEN$')*('^EtaEE$')).one()),                          
]

diobj_mass = [
    plotters.DiObjMassPlotter(
        collections.DoubleTkEleL2,
        diobj_eta_selections
    ),
    plotters.DiObjMassPlotter(
        collections.DoubleSimEle,
        double_gen_selections
    ),

]



eg_resotuples_plotters = [
    plotters.ResoNtupleMatchPlotter(
        collections.TkEleL2, collections.sim_parts,
        egid_tkele_selections,
        gen_selections),
    # plotters.ResoNtupleMatchPlotter(
    #     collections.egs_brl, collections.gen_parts,
    #     selections.barrel_quality_selections,
    #     selections.gen_eb_selections),
    # plotters.ResoNtupleMatchPlotter(
    #     collections.tkelesEL, collections.gen_parts,
    #     selections.tkisoeg_selections,
    #     selections.gen_ee_tk_selections),
    # plotters.ResoNtupleMatchPlotter(
    #     collections.tkelesEL_brl, collections.gen_parts,
    #     selections.barrel_quality_selections,
    #     selections.gen_eb_selections),
    ]
