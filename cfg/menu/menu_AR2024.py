from python.selections import *
from cfg.menu.wps import *


menu_obj_sel = [
    # ---------- Standalone egamma - 
    ((Selector('^EtaEB$')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightS$'))).one('MenuSta', 'TightID'),
    ((Selector('^EtaEB$')&('^IDTightE$'))|(Selector('^EtaEEFwd$')&('^IDTightS$'))).one('MenuStaFwd', 'TightID'),
    # ---------- Isolated electrons -
    ((Selector('^EtaEB$')&('^IsoEleEB$'))|(Selector('^EtaEE$')&('^IsoEleEE'))).one('MenuEleIso', 'Iso Menu WP'),
    ((Selector('^EtaEB$')&('^IsoEleEB$'))|(Selector('^EtaEE$')&('^IsoEleEE'))).one('MenuEleIsoLoose', 'Iso LooseID'),
    ((Selector('^EtaEB$')&('^IDTightE$')&('^IsoEleEB$'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^IDTightE$'))).one('MenuEleIsoTight', 'Iso TightID'),
    # ---------- Electrons -
    ((Selector('^EtaEB$')&('^IDTightE$$'))|(Selector('^EtaEE$'))).one('MenuEle', 'Menu WP'),
    ((Selector('^EtaEB$')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightE$'))).one('MenuEleTight', 'TightID'),
    ((Selector('^EtaEB$')&('^IDTightE$$'))|(Selector('^EtaEE$'))).one('MenuEleLoose', 'LooseID'),
    # ---------- Photons -
    ((Selector('^EtaEB$')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightP'))).one('MenuPho', 'no-iso'),
    ((Selector('^EtaEB$')&('^IsoPhoEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP'))).one('MenuPhoIso', 'Iso'),

]

Selector.selection_primitives = sm.selections.copy()

trigger_sel = [
    (Selector('^MenuEleIso$')&('^PtIsoEle28')).one('SingleIsoTkEle28', 'SingleIsoTkEle28'),
    (Selector('^MenuEleTight$')&('^PtEle36')).one('SingleTkEle36', 'SingleTkEle36'),
    (Selector('^MenuPhoIso$')&('^PtIsoPho36')).one('SingleIsoTkPho36', 'SingleIsoTkPho36'),
    (Selector('^MenuPhoIso$')&('^PtIsoPho22')).one('SingleIsoTkPho22', 'SingleIsoTkPho22'),
    (Selector('^MenuPhoIso$')&('^PtIsoPho12')).one('SingleIsoTkPho12', 'SingleIsoTkPho12'),
    (Selector('^MenuSta$')&('^PtSta51')).one('SingleEGEle51', 'SingleEGEle51'),
    (Selector('^MenuStaFwd$')&('^PtSta51')).one('SingleEGEle51Fwd', 'SingleEGEle51 |#eta| < 3'),

]

diobj_sel = [
    build_DiObj_selection('DoubleIsoTkPho22-12', 'DoubleIsoTkPho22-12',
                    ((Selector('^MenuPhoIso$')&('^PtIsoPho22'))).one(),
                    ((Selector('^MenuPhoIso$')&('^PtIsoPho12'))).one()),
    build_DiObj_selection('MenuDoubleIsoTkPho22-X', 'DoubleIsoTkPho22-X',
                        ((Selector('^MenuPhoIso$')&('^PtIsoPho22'))).one(),
                        (Selector('^MenuPhoIso$')).one()),
    build_DiObj_selection('DoubleTkEle25-12', 'DoubleTkEle25-12',
                        (Selector('^MenuEleLoose$')&('^PtEle25')).one(),
                        (Selector('^MenuEleLoose$')&('^PtEle12')).one(),
                        Selector('^Dz1$').one()),
    build_DiObj_selection('DoubleStaEG37-24', 'DoubleStaEG37-24',
                        (Selector('^MenuSta$')&('^PtSta37')).one(),
                        (Selector('^MenuSta$')&('^PtSta24')).one()),
    build_DiObj_selection('DoubleStaEG37-24Fwd', 'DoubleStaEG37-24 |#eta| < 3',
                        (Selector('^MenuStaFwd$')&('^PtSta37')).one(),
                        (Selector('^MenuStaFwd$')&('^PtSta24')).one()),
    build_DiObj_selection('DoubleIsoTkEleStaEG22-12', 'DoubleIsoTkEleStaEG22-12',
                        (Selector('^MenuEleIsoLoose$')&('^PtIsoEle22')).one(),
                        (Selector('^MenuSta$')&('^PtSta12')).one(),
                        Selector('^DRg0p1').one()),
    build_DiObj_selection('DoubleIsoTkEleStaEG22-12Fwd', 'DoubleIsoTkEleStaEG22-12  |#eta| < 3',
                        (Selector('^MenuEleIsoLoose$')&('^PtIsoEle22')).one(),
                        (Selector('^MenuStaFwd$')&('^PtSta12')).one(),
                        Selector('^DRg0p1').one()),

]


diobj_legs_sel = [
    (Selector('^MenuPhoIso$')&('^PtIsoPho22')).one('SingleIsoTkPho22', 'SingleIsoTkPho22'),
    (Selector('^MenuPhoIso$')&('^PtIsoPho12')).one('SingleIsoTkPho12', 'SingleIsoTkPho12'),
    (Selector('^MenuEleLoose$')&('^PtEle25')).one('SingleTkEle25', 'SingleTkEle25'),
    (Selector('^MenuEleLoose$')&('^PtEle12')).one('SingleTkEle12', 'SingleTkEle12'),
    (Selector('^MenuEleIso$')&('^PtIsoEle22')).one('SingleIsoTkEle22', 'SingleIsoTkEle22'),
]


# menu_sel = [
#     ((Selector('^EtaEB')&('^IDTightP$'))|(Selector('^EtaEE$')&('^IDTightS$'))).one('MenuStaNew', 'TightID-new'),

#     # ((Selector('^EtaEB')&('^IsoEleEB$'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^IDTightE$'))).one('MenuEleIsoTight', 'Iso TightID'),
#     # ((Selector('^EtaEB')&('^IsoEleEB$'))|(Selector('^EtaEE$')&('^IsoEleEE'))).one('MenuEleIsoLoose', 'Iso LooseID'),
#     ((Selector('^EtaEB')&('^IDTightE$')&('^IsoEleEB$'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^IDTightE$'))).one('MenuEleIsoTight', 'Iso TightID'),

#     ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightE$'))).one('MenuEleTight', 'TightID'),
#     ((Selector('^EtaEB')&('^IDTightE$$'))|(Selector('^EtaEE$'))).one('MenuEleLoose', 'LooseID'),
#     ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightP'))).one('MenuPho', 'no-iso'),
#     ((Selector('^EtaEB')&('^IDTightP$'))|(Selector('^EtaEE$')&('^IDTightP'))).one('MenuPhoNew', 'no-iso-new'),

#     ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP'))).one('MenuPhoIso', 'Iso'),
#     ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightP$'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP'))).one('MenuPhoIsoNew', 'Iso-new'),

#     # Rate selections
#     ((Selector('^EtaEB')&('^IDTightE$')&('^IsoEleEB')&('^PtIsoEleEB28'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^IDTightE$')&('^PtIsoEleEE28'))).one('SingleIsoTkEle28New', 'SingleIsoTkEle28-new'),
#     # ((Selector('^EtaEB')&('^IsoEleEB')&('^PtIsoEleEB28'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^PtIsoEleEE28'))).one('SingleIsoTkEle28', 'SingleIsoTkEle28'),
#     # ((Selector('^EtaEB')&('^IsoEleEB')&('^PtIsoEleEB28'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^IDTightE$')&('^PtIsoEleEE28'))).one('SingleIsoTkEle28Tight', 'SingleIsoTkEle28Tight'),
#     ((Selector('^EtaEB')&('^IDTightE$')&('^PtEleEB36'))|(Selector('^EtaEE$')&('^IDTightE$')&('^PtEleEE36'))).one('SingleTkEle36', 'SingleTkEle36'),

#     ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB36'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE36'))).one('SingleIsoTkPho36', 'SingleIsoTkPho36'),
#     ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one('SingleIsoTkPho22', 'SingleIsoTkPho22'),
#     ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB12'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE12'))).one('SingleIsoTkPho12', 'SingleIsoTkPho12'),

#     ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightP$')&('^PtIsoPhoEB36'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE36'))).one('SingleIsoTkPho36New', 'SingleIsoTkPho36-new'),
#     ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightP$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one('SingleIsoTkPho22New', 'SingleIsoTkPho22-new'),
#     ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightP$')&('^PtIsoPhoEB12'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE12'))).one('SingleIsoTkPho12New', 'SingleIsoTkPho12-new'),

#     ((Selector('^EtaEB')&('^IDTightE$')&('^PtStaEB51'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE51'))).one('SingleEGEle51', 'SingleEGEle51'),
#     ((Selector('^EtaEB')&('^IDTightP$')&('^PtStaEB51'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE51'))).one('SingleEGEle51New', 'SingleEGEle51-new'),

#     build_DiObj_selection('DoubleIsoTkPho22-12', 'DoubleIsoTkPho22-12',
#                         ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
#                         ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB12'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE12'))).one()),
#     build_DiObj_selection('DoubleIsoTkPho22-12New', 'DoubleIsoTkPho22-12-new',
#                         ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightP$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
#                         ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightP$')&('^PtIsoPhoEB12'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE12'))).one()),

#     build_DiObj_selection('MenuDoubleIsoTkPho22-X', 'DoubleIsoTkPho22-X',
#                         ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
#                         ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP'))).one()),
#     build_DiObj_selection('MenuDoubleIsoTkPho22-XNew', 'DoubleIsoTkPho22-X-new',
#                         ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightP$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
#                         ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightP$'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP'))).one()),


#     build_DiObj_selection('DoubleTkEle25-12', 'DoubleTkEle25-12',
#                         ((Selector('^EtaEB')&('^IDTightE$')&('^PtEleEB25'))|(Selector('^EtaEE$')&('^PtEleEE25'))).one(),
#                         ((Selector('^EtaEB')&('^IDTightE$')&('^PtEleEB12'))|(Selector('^EtaEE$')&('^PtEleEE12'))).one(),
#                         Selector('^Dz1$').one()),

#     build_DiObj_selection('DoubleTkEle25-12Tight', 'DoubleTkEle25-12 tight',
#                         ((Selector('^EtaEB')&('^IDTightP$')&('^PtEleEB25'))|(Selector('^EtaEE$')&('^IDTightE$')&('^PtEleEE25'))).one(),
#                         ((Selector('^EtaEB')&('^IDTightP$')&('^PtEleEB12'))|(Selector('^EtaEE$')&('^IDTightE$')&('^PtEleEE12'))).one(),
#                         Selector('^Dz1$').one()),

#     build_DiObj_selection('MenuDoubleTkPho22-X', 'DoubleTkPho22-X',
#                         ((Selector('^EtaEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
#                         ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightP'))).one()),
#     build_DiObj_selection('MenuDoubleTkPho22-XNew', 'DoubleTkPho22-X-new',
#                         ((Selector('^EtaEB')&('^IDTightP$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
#                         ((Selector('^EtaEB')&('^IDTightP$'))|(Selector('^EtaEE$')&('^IDTightP'))).one()),

#     build_DiObj_selection('MenuDoubleIsoOneTkPho22-X', 'DoubleIsoOneTkPho22-X',
#                         ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
#                         ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightP'))).one()),

#     build_DiObj_selection('MenuDoubleIsoOneTkPho22-XNew', 'DoubleIsoOneTkPho22-X-new',
#                         ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightP$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
#                         ((Selector('^EtaEB')&('^IDTightP$'))|(Selector('^EtaEE$')&('^IDTightP'))).one()),

#     build_DiObj_selection('DoubleStaEG37-24', 'DoubleStaEG37-24',
#                         ((Selector('^EtaEB')&('^IDTightE$')&('^PtStaEB37'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE37'))).one(),
#                         ((Selector('^EtaEB')&('^IDTightE$')&('^PtStaEB24'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE24'))).one()),
#     build_DiObj_selection('DoubleStaEG37-24New', 'DoubleStaEG37-24-new',
#                         ((Selector('^EtaEB')&('^IDTightP$')&('^PtStaEB37'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE37'))).one(),
#                         ((Selector('^EtaEB')&('^IDTightP$')&('^PtStaEB24'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE24'))).one()),

#     build_DiObj_selection('DoubleIsoTkEleStaEG22-12', 'DoubleIsoTkEleStaEG22-12',
#                         ((Selector('^EtaEB')&('^IsoEleEB$')&('^PtIsoEleEB22'))|(Selector('^EtaEE$')&('^IsoEleEE$')&('^PtIsoEleEE22'))).one(),
#                         ((Selector('^EtaEB')&('^IDTightE$')&('^PtStaEB12'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE12'))).one(),
#                         Selector('^DRg0p1').one()),
#     build_DiObj_selection('DoubleIsoTkEleStaEG22-12New', 'DoubleIsoTkEleStaEG22-12-new',
#                         ((Selector('^EtaEB')&('^IDTightE$')&('^IsoEleEB$')&('^PtIsoEleEB22'))|(Selector('^EtaEE$')&('^IDTightE$')&('^IsoEleEE$')&('^PtIsoEleEE22'))).one(),
#                         ((Selector('^EtaEB')&('^IDTightP$')&('^PtStaEB12'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE12'))).one(),
#                         Selector('^DRg0p1').one()),
    
    # build_DiObj_selection('DoubleIsoTkEleStaEG22-12', 'DoubleIsoTkEleStaEG22-12',
    #                     ((Selector('^EtaEB')&('^IsoEleEB$')&('^PtIsoEleEB22'))|(Selector('^EtaEE$')&('^IsoEleEE$')&('^PtIsoEleEE22'))).one(),
    #                     ((Selector('^EtaEB')&('^IDTightE$')&('^PtStaEB12'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE12'))).one(),
    #                     Selector('^DRg0p1').one()),
                        

# ]


# for eff in [90, 92, 94, 96, 98]:
#     menu_sel.append(build_DiObj_selection(f'MenuDoubleIso{eff}TkPho22-X', f'DoubleIso{eff}TkPho22-X',
#                         ((Selector('^EtaEB')&(f'^IsoPho{eff}')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&(f'^IsoPho{eff}')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
#                         ((Selector('^EtaEB')&(f'^IsoPho{eff}')&('^IDTightE$'))|(Selector('^EtaEE$')&(f'^IsoPho{eff}')&('^IDTightP'))).one()))
    
    
    
#     menu_sel.append(((Selector('^EtaEB')&(f'^IsoPho{eff}')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&(f'^IsoPho{eff}')&('^IDTightP')&('^PtIsoPhoEE22'))).one(f'Iso@{eff}TkPho22', f'p_{{T}}>22, iso@{eff}'))
#     menu_sel.append(((Selector('^EtaEB')&(f'^IsoPho{eff}')&('^IDTightE$')&('^PtIsoPhoEB12'))|(Selector('^EtaEE$')&(f'^IsoPho{eff}')&('^IDTightP')&('^PtIsoPhoEE12'))).one(f'Iso@{eff}TkPho12', f'p_{{T}}>12, iso@{eff}'))
#     menu_sel.append(((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB12'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE12'))).one(f'IsoTkPho12', 'p_{T}>12, iso@Menu'))
#     menu_sel.append(((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(f'IsoTkPho22', 'p_{T}>22, iso@Menu'))






# repeat the call: we want the menu selections to be avaialble via the selectors
Selector.selection_primitives = sm.selections.copy()

# selm = SelectionManager()
# pprint(selm.get_labels())

