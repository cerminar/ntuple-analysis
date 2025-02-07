import cfg.eg_genmatch
import python.histos as histos
from python.draw.drawingTools import *
from cfg.eg_genmatch import EGResoHistos




def what(what):
    match what:
        case 'egmenu_ele':
            return [histos.HistoSetEff], 'eff', egmenu_ele_draw
        case 'egmenu_pho':
            return [histos.HistoSetEff], 'eff', egmenu_pho_draw
        case 'ctl2_tkem_iso_pho_draw':
            return [histos.HistoSetEff], 'eff', ctl2_tkem_iso_pho_draw
        case 'ctl2_tkeg_eff', 'ctl2_tkeg':
            return [histos.HistoSetEff], 'eff_ctl2', ctl2_tkem_draw
        case 'ctl2_tkeg_reso':
            return [EGResoHistos], 'reso_ctl2', ctl2_tkeg_reso_draw




draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #it{#scale[1.]{Phase-2 Simulation}}'),
                    (0.69, 0.91, '#it{14TeV}, 200 PU')]



def egmenu_ele_draw(hplot, smps, wc):

    egmenu_configs = [    
        (['TkEmL2',], ['MenuSta'],      ['GENPt15'],  'TkEmL2_MenuSta_GENPt15_ele', {}),
        (['TkEmL2',], ['MenuSta'],      ['GENPt30'],  'TkEmL2_MenuSta_GENPt30_ele', {}),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENPt10to25'],  'TkEmL2_MenuPhoIso_GENPt10to25_ele', {}),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENPt15'],  'TkEmL2_MenuPhoIso_GENPt15_ele', {}),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENPt30'],  'TkEmL2_MenuPhoIso_GENPt30_ele', {}),

        # (smps_pho, ['TkEmL2',], ['MenuSta'],      ['GENPt15'],  'TkEmL2_MenuSta_GENPt15'),
        # (smps_pho, ['TkEmL2',], ['MenuSta'],      ['GENPt30'],  'TkEmL2_MenuSta_GENPt30'),
        # (smps_pho, ['TkEmL2',], ['MenuPhoIso'],      ['GENPt15'],  'TkEmL2_MenuPhoIso_GENPt15'),
        # (smps_pho, ['TkEmL2',], ['MenuPhoIso'],      ['GENPt30'],  'TkEmL2_MenuPhoIso_GENPt30'),

        (['TkEleL2',], ['MenuEleLoose'],      ['GENPt10to25'],  'TkEleL2_MenuEleLoose_GENPt10to25', {}),
        (['TkEleL2',], ['MenuEleLoose'],      ['GENPt15'],  'TkEleL2_MenuEleLoose_GENPt15', {}),
        (['TkEleL2',], ['MenuEleLoose'],      ['GENPt30'],  'TkEleL2_MenuEleLoose_GENPt30', {}),
        (['TkEleL2',], ['MenuEleTight'],      ['GENPt10to25'],  'TkEleL2_MenuEleTight_GENPt10to25', {}),
        (['TkEleL2',], ['MenuEleTight'],      ['GENPt15'],  'TkEleL2_MenuEleTight_GENPt15', {}),
        (['TkEleL2',], ['MenuEleTight'],      ['GENPt30'],  'TkEleL2_MenuEleTight_GENPt30', {}),

        (['TkEleL2',], ['MenuEleIsoLoose'],      ['GENPt10to25'],  'TkEleL2_MenuEleIsoLoose_GENPt10to25', {}),
        (['TkEleL2',], ['MenuEleIsoLoose'],      ['GENPt15'],  'TkEleL2_MenuEleIsoLoose_GENPt15', {}),
        (['TkEleL2',], ['GENPt30'],  'TkEleL2_MenuEleIsoLoose_GENPt30', {}),
        (['TkEleL2',], ['MenuEleIsoTight'],      ['GENPt10to25'],  'TkEleL2_MenuEleIsoTight_GENPt10to25', {}),
        (['TkEleL2',], ['MenuEleIsoTight'],      ['GENPt15'],  'TkEleL2_MenuEleIsoTight_GENPt15', {}),
        (['TkEleL2',], ['MenuEleIsoTight'],      ['GENPt30'],  'TkEleL2_MenuEleIsoTight_GENPt30', {}),

        (['TkEleL2',], ['SingleTkEle36'],      ['GENPt30'],  'TkEleL2_SingleTkEle36_GENPt30', {}),
        (['TkEleL2',], ['SingleIsoTkEle28'],      ['GENPt30'],  'TkEleL2_SingleIsoTkEle28_GENPt30', {})
    ]
    draw_effvseta(hplot, smps, wc, draw_style=draw_config, configs=egmenu_configs)

    menu_configs = [
        (['EGStaEB',], ['MenuSta'],      ['GENEtaEB'],    'EGStaEB_MenuSta_GENEtaEB_ele', {}),
        (['EGStaEE',], ['MenuSta'],      ['GENEtaEE'],  'EGStaEE_MenuSta_GENEtaEE_ele', {}),
        (['EGStaEE',], ['MenuSta'],      ['GENEtaEEb'],  'EGStaEE_MenuSta_GENEtaEEb_ele', {}),
        (['TkEmL2',], ['MenuSta'],      ['GENEtaEB'],    'TkEmL2_MenuSta_GENEtaEB_ele', {}),
        (['TkEmL2',], ['MenuSta'],      ['GENEtaEE'],  'TkEmL2_MenuSta_GENEtaEE_ele', {}),
        (['TkEmL2',], ['MenuSta'],      ['GENEtaEEb'],  'TkEmL2_MenuSta_GENEtaEEb_ele', {}),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENEtaEB'],    'TkEmL2_MenuPhoIso_GENEtaEB_ele', {}),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENEtaEE'],  'TkEmL2_MenuPhoIso_GENEtaEE_ele', {}),

        (['TkEleL2',], ['MenuEleLoose'],      ['GENEtaEB'],    'TkEleL2_MenuEleLoose_GENEtaEB', {}),
        (['TkEleL2',], ['MenuEleLoose'],      ['GENEtaEE'],  'TkEleL2_MenuEleLoose_GENEtaEE', {}),
        (['TkEleL2',], ['MenuEleLoose'],      ['GENEtaEEb'],  'TkEleL2_MenuEleLoose_GENEtaEEb', {}),
        (['TkEleL2',], ['MenuEleTight'],      ['GENEtaEB'],    'TkEleL2_MenuEleTight_GENEtaEB', {}),
        (['TkEleL2',], ['MenuEleTight'],      ['GENEtaEE'],  'TkEleL2_MenuEleTight_GENEtaEE', {}),
        (['TkEleL2',], ['MenuEleTight'],      ['GENEtaEEb'],  'TkEleL2_MenuEleTight_GENEtaEEb', {}),

        (['TkEleL2',], ['MenuEleIsoLoose'],      ['GENEtaEB'],    'TkEleL2_MenuEleIsoLoose_GENEtaEB', {}),
        (['TkEleL2',], ['MenuEleIsoLoose'],      ['GENEtaEE'],  'TkEleL2_MenuEleIsoLoose_GENEtaEE', {}),
        (['TkEleL2',], ['MenuEleIsoLoose'],      ['GENEtaEEb'],  'TkEleL2_MenuEleIsoLoose_GENEtaEEb', {}),

        (['TkEleL2',], ['MenuEleIsoTight'],      ['GENEtaEB'],    'TkEleL2_MenuEleIsoTight_GENEtaEB', {}),
        (['TkEleL2',], ['MenuEleIsoTight'],      ['GENEtaEE'],  'TkEleL2_MenuEleIsoTight_GENEtaEE', {}),
        (['TkEleL2',], ['MenuEleIsoTight'],      ['GENEtaEEb'],  'TkEleL2_MenuEleIsoTight_GENEtaEEb', {}),

        (['TkEleL2',], ['SingleTkEle36'],      ['GENEtaEB'],    'TkEleL2_SingleTkEle36_GENEtaEB', {}),
        (['TkEleL2',], ['SingleTkEle36'],      ['GENEtaEE'],  'TkEleL2_SingleTkEle36_GENEtaEE', {}),
        (['TkEleL2',], ['SingleTkEle36'],      ['GENEtaEEb'],  'TkEleL2_SingleTkEle36_GENEtaEEb', {}),

        (['TkEleL2',], ['SingleIsoTkEle28'],      ['GENEtaEB'],    'TkEleL2_SingleIsoTkEle28_GENEtaEB', {}),
        (['TkEleL2',], ['SingleIsoTkEle28'],      ['GENEtaEE'],  'TkEleL2_SingleIsoTkEle28_GENEtaEE', {}),
        (['TkEleL2',], ['SingleIsoTkEle28'],      ['GENEtaEEb'],  'TkEleL2_SingleIsoTkEle28_GENEtaEEb', {}),
    ]
    draw_effvspt(hplot, smps, wc, draw_style=draw_config, configs=menu_configs)
    menu_configs = [
        (['EGStaEB',], ['MenuSta'],      ['GENEtaEB'],    'ele'),
        (['EGStaEE',], ['MenuSta'],      ['GENEtaEE'],    'ele'),
        # (smps_ele, ['TkEmL2',], ['MenuSta'],      ['GENEtaEB'],     'ele'),
        # (smps_ele, ['TkEmL2',], ['MenuSta'],      ['GENEtaEE'],     'ele'),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENEtaEB'],  'ele'),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENEtaEE'],  'ele'),

        (['TkEleL2',], ['MenuEleLoose'],      ['GENEtaEB'],  ''),
        (['TkEleL2',], ['MenuEleLoose'],      ['GENEtaEE'],  ''),
        (['TkEleL2',], ['MenuEleTight'],      ['GENEtaEB'],  ''),
        (['TkEleL2',], ['MenuEleTight'],      ['GENEtaEE'],  ''),

        (['TkEleL2',], ['MenuEleIsoLoose'],      ['GENEtaEB'],  ''),
        (['TkEleL2',], ['MenuEleIsoLoose'],      ['GENEtaEE'],  ''),
        (['TkEleL2',], ['MenuEleIsoTight'],      ['GENEtaEB'],  ''),
        (['TkEleL2',], ['MenuEleIsoTight'],      ['GENEtaEE'],  ''),
    ]
    draw_ton(hplot, smps, wc, draw_style=draw_config, configs=menu_configs)


def egmenu_pho_draw(hplot, smps, wc):

    egmenu_configs = [    
        (['TkEmL2',], ['MenuSta'],      ['GENPt15'],  'TkEmL2_MenuSta_GENPt15', {}),
        (['TkEmL2',], ['MenuSta'],      ['GENPt30'],  'TkEmL2_MenuSta_GENPt30', {}),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENPt10to25'],  'TkEmL2_MenuPhoIso_GENPt10to25', {}),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENPt15'],  'TkEmL2_MenuPhoIso_GENPt15', {}),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENPt30'],  'TkEmL2_MenuPhoIso_GENPt30', {}),
    ]
    draw_effvseta(hplot, smps, wc, draw_style=draw_config, configs=egmenu_configs)

    menu_configs = [
        (['EGStaEB',], ['MenuSta'],      ['GENEtaEB'],    'EGStaEB_MenuSta_GENEtaEB', {}),
        (['EGStaEE',], ['MenuSta'],      ['GENEtaEE'],    'EGStaEE_MenuSta_GENEtaEE', {}),
        (['TkEmL2',], ['MenuSta'],      ['GENEtaEB'],    'TkEmL2_MenuSta_GENEtaEB', {}),
        (['TkEmL2',], ['MenuSta'],      ['GENEtaEE'],    'TkEmL2_MenuSta_GENEtaEE', {}),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENEtaEB'],    'TkEmL2_MenuPhoIso_GENEtaEB', {}),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENEtaEE'],    'TkEmL2_MenuPhoIso_GENEtaEE', {}),
        (['TkEmL2',], ['MenuPho'],      ['GENEtaEB'],    'TkEmL2_MenuPho_GENEtaEB', {}),
        (['TkEmL2',], ['MenuPho'],      ['GENEtaEE'],    'TkEmL2_MenuPho_GENEtaEE', {}),
    ]
    draw_effvspt(hplot, smps, wc, draw_style=draw_config, configs=menu_configs)
    menu_configs = [
        (['EGStaEB',], ['MenuSta'],      ['GENEtaEB'],    ''),
        (['EGStaEE',], ['MenuSta'],      ['GENEtaEE'],    ''),
        # (smps_ele, ['TkEmL2',], ['MenuSta'],      ['GENEtaEB'],     'ele'),
        # (smps_ele, ['TkEmL2',], ['MenuSta'],      ['GENEtaEE'],     'ele'),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENEtaEB'],  ''),
        (['TkEmL2',], ['MenuPhoIso'],      ['GENEtaEE'],  ''),

    ]
    draw_ton(hplot, smps, wc, draw_style=draw_config, configs=menu_configs)


def ctl2_tkem_iso_pho_draw(hplot, smps, wc):

    egmenu_configs = [    
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoT'],      ['GENPt10to25'],  'TkEmL2_Pho_GENPt110to25', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoT'],      ['GENPt15'],  'TkEmL2_Pho_GENPt15', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoT'],      ['GENPt30'],  'TkEmL2_Pho_GENPt30', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoLL2Iso', 'L2IDPhoLIsoPho90', 'L2IDPhoLIsoPho92', 'L2IDPhoLIsoPho94', 'L2IDPhoLIsoPho96', 'L2IDPhoLIsoPho98'],      ['GENPt10to25'],  'TkEmL2_LoosePhoIso_GENPt10to25', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoLL2Iso', 'L2IDPhoLIsoPho90', 'L2IDPhoLIsoPho92', 'L2IDPhoLIsoPho94', 'L2IDPhoLIsoPho96', 'L2IDPhoLIsoPho98'],      ['GENPt15'],      'TkEmL2_LoosePhoIso_GENPt15', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoLL2Iso', 'L2IDPhoLIsoPho90', 'L2IDPhoLIsoPho92', 'L2IDPhoLIsoPho94', 'L2IDPhoLIsoPho96', 'L2IDPhoLIsoPho98'],      ['GENPt30'],      'TkEmL2_LoosePhoIso_GENPt30', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoT', 'L2IDPhoTL2Iso', 'L2IDPhoTIsoPho90', 'L2IDPhoTIsoPho92', 'L2IDPhoTIsoPho94', 'L2IDPhoTIsoPho96', 'L2IDPhoTIsoPho98'],      ['GENPt10to25'],  'TkEmL2_TightPhoIso_GENPt10to25', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoT', 'L2IDPhoTL2Iso', 'L2IDPhoTIsoPho90', 'L2IDPhoTIsoPho92', 'L2IDPhoTIsoPho94', 'L2IDPhoTIsoPho96', 'L2IDPhoTIsoPho98'],      ['GENPt15'],      'TkEmL2_TightPhoIso_GENPt15', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoT', 'L2IDPhoTL2Iso', 'L2IDPhoTIsoPho90', 'L2IDPhoTIsoPho92', 'L2IDPhoTIsoPho94', 'L2IDPhoTIsoPho96', 'L2IDPhoTIsoPho98'],      ['GENPt30'],      'TkEmL2_TightPhoIso_GENPt30', {}),
    ]
    draw_effvseta(hplot, smps, wc, draw_style=draw_config, configs=egmenu_configs)

    menu_configs = [
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoLL2Iso', 'L2IDPhoLIsoPho90', 'L2IDPhoLIsoPho92', 'L2IDPhoLIsoPho94', 'L2IDPhoLIsoPho96', 'L2IDPhoLIsoPho98'], ['GENEtaEB'], 'TkEmL2_PhoIso_GENEtaEB', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoT'], ['GENEtaEE'], 'TkEmL2_Pho_GENEtaEE', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoLL2Iso', 'L2IDPhoLIsoPho90', 'L2IDPhoLIsoPho92', 'L2IDPhoLIsoPho94', 'L2IDPhoLIsoPho96', 'L2IDPhoLIsoPho98'], ['GENEtaEE'], 'TkEmL2_LoosePhoIso_GENEtaEE', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoT', 'L2IDPhoTL2Iso'], ['GENEtaEE'], 'TkEmL2_TightPhoIso_GENEtaEE', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoT'], ['GENEtaEEb'], 'TkEmL2_Pho_GENEtaEEb', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoL', 'L2IDPhoLL2Iso', 'L2IDPhoLIsoPho90', 'L2IDPhoLIsoPho92', 'L2IDPhoLIsoPho94', 'L2IDPhoLIsoPho96', 'L2IDPhoLIsoPho98'], ['GENEtaEEb'], 'TkEmL2_LoosePhoIso_GENEtaEEb', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoT', 'L2IDPhoTL2Iso'], ['GENEtaEEb'], 'TkEmL2_TightPhoIso_GENEtaEEb', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoLL2Iso', 'L2IDPhoTL2Iso'], ['GENEtaEE'], 'TkEmL2_PhoIso_GENEtaEE', {}),
        (['TkEmL2IsoWP',], ['L2IDPhoLL2Iso', 'L2IDPhoTL2Iso'], ['GENEtaEEb'], 'TkEmL2_PhoIso_GENEtaEEb', {}),
        (['TkEmL2IsoWP',], ['IsoTkPho22', 'Iso@90TkPho22', 'Iso@92TkPho22', 'Iso@94TkPho22', 'Iso@96TkPho22'], ['GENEtaEEb'], 'TkEmL2_PhoIso22_GENEtaEEb', {}),
        (['TkEmL2IsoWP',], ['IsoTkPho22', 'Iso@90TkPho22', 'Iso@92TkPho22', 'Iso@94TkPho22', 'Iso@96TkPho22'], ['GENEtaEB'], 'TkEmL2_PhoIso22_GENEtaEB', {}),
        (['TkEmL2IsoWP',], ['IsoTkPho12', 'Iso@90TkPho12', 'Iso@92TkPho12', 'Iso@94TkPho12', 'Iso@96TkPho12'], ['GENEtaEEb'], 'TkEmL2_PhoIso12_GENEtaEEb', {}),
        (['TkEmL2IsoWP',], ['IsoTkPho12', 'Iso@90TkPho12', 'Iso@92TkPho12', 'Iso@94TkPho12', 'Iso@96TkPho12'], ['GENEtaEB'], 'TkEmL2_PhoIso12_GENEtaEB', {}),
    ]
    draw_effvspt(hplot, smps, wc, draw_style=draw_config, configs=menu_configs)



def ctl2_tkem_draw(hplot, smps, wc):

    egmenu_configs = [    
        (['TkEleL2',],            ['all'],              ['GENPt5to15'],       'TkEleL2_all_GENPt10to15', {}),
        (['TkEleL2',],            ['IDTightE'],         ['GENPt5to15'],  'TkEleL2_IDTightE_GENPt10to15', {}),
        (['TkEleL2',],            ['all'],              ['GENPt30'],  'TkEleL2_all_GENPt30', {}),
        (['TkEleL2',],            ['IDTightE'],         ['GENPt30'],  'TkEleL2_IDTightE_GENPt30', {}),
        (['TkEmL2',],             ['all',],  ['GENPt5to15'],  'TkEmL2_all_GENPt10to15', {}),
        (['TkEmL2',],             ['all',],  ['GENPt30'],     'TkEmL2_all_GENPt30', {}),
        (['TkEmL2',],             ['IDTightP'],         ['GENPt5to15'],  'TkEmL2_IDTightP_GENPt10to15', {}),
        (['TkEmL2',],             ['IDTightP'],         ['GENPt30'],     'TkEmL2_IDTightP_GENPt30', {}),
        (['EGStaEB', 'EGStaEE',], ['all'],      ['GENPt5to15'],   'EGSta_all_GENPt10to15', {}),
        (['EGStaEB', 'EGStaEE'],  ['all'],      ['GENPt30'],      'EGSta_all_GENPt30', {}),
        (['EGStaEB', 'EGStaEE',], ['IDTightS'],      ['GENPt5to15'],  'EGSta_IDTightS_GENPt10to15', {}),
        (['EGStaEB', 'EGStaEE'],  ['IDTightS'],      ['GENPt30'],     'EGSta_IDTightS_GENPt30', {}),

    ]
    draw_effvseta(hplot, smps, wc, draw_style=draw_config, configs=egmenu_configs)

    menu_configs = [
        (['TkEleL2',],  ['all'],       ['GENEtaEB'],  'TkEleL2_all_GENEtaEB', {}),
        (['TkEleL2',],  ['IDTightE'],  ['GENEtaEB'],  'TkEleL2_IDTightE_GENEtaEB', {}),
        (['TkEleL2',],  ['all'],       ['GENEtaEE'],  'TkEleL2_all_GENEtaEE', {}),
        (['TkEleL2',],  ['IDTightE'],  ['GENEtaEE'],  'TkEleL2_IDTightE_GENEtaEE', {}),
        (['TkEmL2',],   ['all'],       ['GENEtaEB'],  'TkEmL2_all_GENEtaEB', {}),
        (['TkEmL2',],   ['all'],       ['GENEtaEE'],  'TkEmL2_all_GENEtaEE', {}),
        (['TkEmL2',],   ['IDTightP'],  ['GENEtaEB'],  'TkEmL2_IDTightP_GENEtaEB', {}),
        (['TkEmL2',],   ['IDTightP'],  ['GENEtaEE'],  'TkEmL2_IDTightP_GENEtaEE', {}),
        (['EGStaEB',],  ['all'],       ['GENEtaEB'],  'EGSta_all_GENEtaEB', {}),
        (['EGStaEE'],   ['all'],       ['GENEtaEEa'],  'EGSta_all_GENEtaEEa', {}),
        (['EGStaEE'],   ['all'],       ['GENEtaEEb'],  'EGSta_all_GENEtaEEb', {}),
        (['EGStaEE'],   ['all'],       ['GENEtaEEc'],  'EGSta_all_GENEtaEEc', {}),
        (['EGStaEB'],   ['IDTightS'],  ['GENEtaEB'],  'EGSta_IDTightS_GENEtaEB', {}),
        (['EGStaEE'],   ['IDTightS'],  ['GENEtaEEa'],  'EGSta_IDTightS_GENEtaEEa', {}),
        (['EGStaEE'],   ['IDTightS'],  ['GENEtaEEb'],  'EGSta_IDTightS_GENEtaEEb', {}),
        (['EGStaEE'],   ['IDTightS'],  ['GENEtaEEc'],  'EGSta_IDTightS_GENEtaEEc', {}),

    ]
    draw_effvspt(hplot, smps, wc, draw_style=draw_config, configs=menu_configs)





def draw_effvseta(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)

        dm.config.legend_position = (0.15,0.05)

        hsets, labels, text = hplot.get_histo(
            histos.HistoSetEff, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)

        dm.addHistos([his.h_eff.h_abseta.CreateGraph() for his in hsets], labels=labels)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                x_min=opts.get('x_min', 0.), 
                x_max=opts.get('x_max', 3.2), 
                y_min=opts.get('y_min', 0.5),
                y_max=opts.get('y_max', 1.1), 
                v_lines=opts.get('v_lines', [1.52, 1.7, 2.4]),
                h_lines=opts.get('h_lines', [1., 0.9]),
                do_ratio=opts.get('do_ratio', True),                
                y_min_ratio=opts.get('y_min_ratio', 0.8),
                y_max_ratio=opts.get('y_max_ratio', 1.2),
                h_lines_ratio=opts.get('h_lines_ratio', [0.9, 1., 1.1]),
                y_axis_label=opts.get('y_axis_label', 'efficiency')
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hEffVsEta_{h_name}', page_creator=wc)


def draw_ton(hplot, smps, wc_eff, draw_style, configs):
    print('Computing Turn-on curves...')
    for smp in hplot.data['sample'].unique():
        # print ('Sample: {}'.format(smp))
        for pu in hplot.data[(hplot.data['sample'] == smp)].pu.unique():
            for tp in ['TkEmL2', 'TkEleL2', 'TkEmL2Ell', 'TkEleL2Ell', 'EGStaEE', 'EGStaEB']:
    #         for tp in ['EG', 'TkEleEL']:
                # print ('PU: {}, TP: {}'.format(pu, tp))
                for tp_sel in hplot.data[(hplot.data['sample'] == smp) & (hplot.data.pu == pu) & (hplot.data.tp == tp)].tp_sel.unique():
                    if 'Pt' not in tp_sel:
                        continue
                    if 'EGq2' in tp_sel or \
                    'EGq3' in tp_sel or \
                    'EGq4' in tp_sel:
                        continue
                        #                 tp_sel_den = 'all'
                    # print tp_sel
                    tp_sel_den = tp_sel.split('Pt')[0]
                    if tp_sel_den == '':
                        tp_sel_den = 'all'
                    # print ('  {} den -> {}'.format(tp_sel, tp_sel_den)) 
                    for gen_sel in hplot.data[(hplot.data['sample'] == smp) & (hplot.data.pu == pu) & (hplot.data.tp == tp) & (hplot.data.tp_sel == tp_sel)].gen_sel.unique():
                        if gen_sel == 'nomatch' or 'Pt' in gen_sel:
                            continue
                        # print(gen_sel)
                        hsetden = hplot.get_histo(histos.HistoSetEff, smp, pu, tp, tp_sel_den, gen_sel)
                        hset = hplot.get_histo(histos.HistoSetEff, smp, pu, tp, tp_sel, gen_sel)
                        hset[0][0].computeTurnOn(hsetden[0][0].h_num)

    pt_points = ['Pt20', 'Pt30', 'Pt40']
    for objs, objs_sel, gen_sel, h_name_sfx in configs:
        if len(smps) == 0:
            continue
        # print (smps)
        # print(f'obj: {objs}, sel: {objs_sel}, histo: {h_name}')
        objs_sel_base = objs_sel[0]
        for pt in pt_points:
            objs_sel = f'{objs_sel_base}{pt}'

            dm = DrawMachine(draw_style)
            dm.config.legend_position = (0.6,0.05)

            hsets, labels, text = hplot.get_histo(
                histos.HistoSetEff, 
                smps, 
                ['PU200'], 
                objs, 
                objs_sel, 
                gen_sel, debug=False)
            
            # print(f"# of hsets: {len(hsets)}")
            # for hset in hsets:
            #     hset.computeEff(rebin=2)
            if not hsets:
                continue
            dm.addHistos([his.h_ton.h_pt.CreateGraph() for his in hsets], labels=labels)

            for i in range(1,len(hsets)):
                # print(f'add ratio: {i} to 0')
                dm.addRatioHisto(i,0)
                # dm.addRatioHisto(2,0)
                # dm.addRatioHisto(3,0)
                # dm.addRatioHisto(4,0)

            dm.draw(
                text=text, 
                x_min=0, x_max=100, 
                y_min=0.0, y_max=1.1, 
                h_lines=[1.0, 0.9],
                do_ratio=True,
                y_min_ratio=0.9,
                y_max_ratio=1.1,
                h_lines_ratio=[0.95, 1., 1.05],
                y_axis_label='Turn-on efficiency (w.r.t matching)'
        )
            # dm.write(name='eg_TDRvsSummer20_matchig_eff')
            h_name = f'hTonVsPt_{objs[0]}_{objs_sel}_{gen_sel[0]}'
            if h_name_sfx != '':
                h_name = f'{h_name}_{h_name_sfx}'
            dm.toWeb(name=h_name, page_creator=wc_eff)


def draw_effvspt(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.05)
        hsets, labels, text = hplot.get_histo(
            histos.HistoSetEff, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_eff.h_pt.CreateGraph() for his in hsets], labels=labels)

        for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm.draw(
            text=text, 
            x_min=opts.get('x_min', 0), 
            x_max=opts.get('x_max', 100), 
            y_min=opts.get('y_min', 0.0), 
            y_max=opts.get('y_max', 1.1), 
            h_lines=opts.get('h_lines', [1.0, 0.9]),
            do_ratio=opts.get('do_ratio', True),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_axis_label=opts.get('y_axis_label', 'efficiency')
        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hEffVsPt_{h_name}', page_creator=wc_eff)

def ctl2_tkeg_reso_draw(hplot, smps, wc):
    ctl2_etaphi_reso_configs = [    
        (['TkEleL2',],            ['all'],              ['GEN'],            'TkEleL2_all_GEN', {}),
        (['TkEleL2',],            ['all'],              ['GENEtaEB'],       'TkEleL2_all_GENEtaEB', {}),
        (['TkEleL2',],            ['all'],              ['GENEtaEE'],       'TkEleL2_all_GENEtaEE', {}),

        # (['TkEleL2',],            ['IDTightE'],         ['GENPt5to15'],  'TkEleL2_IDTightE_GENPt10to15', {}),
        # (['TkEleL2',],            ['all'],              ['GENPt30'],  'TkEleL2_all_GENPt30', {}),
        # (['TkEleL2',],            ['IDTightE'],         ['GENPt30'],  'TkEleL2_IDTightE_GENPt30', {}),
        # (['TkEmL2',],             ['all',],  ['GENPt5to15'],  'TkEmL2_all_GENPt10to15', {}),
        # (['TkEmL2',],             ['all',],  ['GENPt30'],     'TkEmL2_all_GENPt30', {}),
        # (['TkEmL2',],             ['IDTightP'],         ['GENPt5to15'],  'TkEmL2_IDTightP_GENPt10to15', {}),
        # (['TkEmL2',],             ['IDTightP'],         ['GENPt30'],     'TkEmL2_IDTightP_GENPt30', {}),
        # (['EGStaEB', 'EGStaEE',], ['all'],      ['GENPt5to15'],   'EGSta_all_GENPt10to15', {}),
        # (['EGStaEB', 'EGStaEE'],  ['all'],      ['GENPt30'],      'EGSta_all_GENPt30', {}),
        # (['EGStaEB', 'EGStaEE',], ['IDTightS'],      ['GENPt5to15'],  'EGSta_IDTightS_GENPt10to15', {}),
        # (['EGStaEB', 'EGStaEE'],  ['IDTightS'],      ['GENPt30'],     'EGSta_IDTightS_GENPt30', {}),

    ]
    draw_reso_eta(hplot, smps, wc, draw_style=draw_config, configs=ctl2_etaphi_reso_configs)
    draw_reso_phi(hplot, smps, wc, draw_style=draw_config, configs=ctl2_etaphi_reso_configs)

    ctl2_caloetaphi_reso_configs = [    

        (['TkEmL2',],            ['all'],              ['GEN'],            'TkEmL2_all_GEN', {}),
        (['TkEmL2',],            ['all'],              ['GENEtaEB'],       'TkEmL2_all_GENEtaEB', {}),
        (['TkEmL2',],            ['all'],              ['GENEtaEE'],       'TkEmL2_all_GENEtaEE', {}),

    ]
    draw_reso_caloeta(hplot, smps, wc, draw_style=draw_config, configs=ctl2_caloetaphi_reso_configs)
    draw_reso_calophi(hplot, smps, wc, draw_style=draw_config, configs=ctl2_caloetaphi_reso_configs)

    ctl2_ptresp_configs = [    
        (['TkEleL2',],            ['all'],              ['GEN'],            'TkEleL2_all_GEN',      {'y_min': 1E-5}),
        (['TkEleL2',],            ['all'],              ['GENEtaEB'],       'TkEleL2_all_GENEtaEB', {'y_min': 1E-5}),
        (['TkEleL2',],            ['all'],              ['GENEtaEE'],       'TkEleL2_all_GENEtaEE', {'y_min': 1E-5}),
        (['TkEmL2',],            ['all'],               ['GEN'],            'TkEmL2_all_GEN',        {'y_min': 1E-5}),
        (['TkEmL2',],            ['all'],               ['GENEtaEB'],       'TkEmL2_all_GENEtaEB',   {'y_min': 1E-5}),
        (['TkEmL2',],            ['all'],               ['GENEtaEE'],       'TkEmL2_all_GENEtaEE',   {'y_min': 1E-5}),
        (['TkEmL2',],            ['all'],               ['GENEtaEE'],       'TkEmL2_all_GENEtaEE',   {'y_min': 1E-5}),

        # (['TkEleL2',],            ['IDTightE'],         ['GENPt5to15'],  'TkEleL2_IDTightE_GENPt10to15', {}),
        # (['TkEleL2',],            ['all'],              ['GENPt30'],  'TkEleL2_all_GENPt30', {}),
        # (['TkEleL2',],            ['IDTightE'],         ['GENPt30'],  'TkEleL2_IDTightE_GENPt30', {}),
        # (['TkEmL2',],             ['all',],  ['GENPt5to15'],  'TkEmL2_all_GENPt10to15', {}),
        # (['TkEmL2',],             ['all',],  ['GENPt30'],     'TkEmL2_all_GENPt30', {}),
        # (['TkEmL2',],             ['IDTightP'],         ['GENPt5to15'],  'TkEmL2_IDTightP_GENPt10to15', {}),
        # (['TkEmL2',],             ['IDTightP'],         ['GENPt30'],     'TkEmL2_IDTightP_GENPt30', {}),
        # (['EGStaEB', 'EGStaEE',], ['all'],      ['GENPt5to15'],   'EGSta_all_GENPt10to15', {}),
        # (['EGStaEB', 'EGStaEE'],  ['all'],      ['GENPt30'],      'EGSta_all_GENPt30', {}),
        # (['EGStaEB', 'EGStaEE',], ['IDTightS'],      ['GENPt5to15'],  'EGSta_IDTightS_GENPt10to15', {}),
        # (['EGStaEB', 'EGStaEE'],  ['IDTightS'],      ['GENPt30'],     'EGSta_IDTightS_GENPt30', {}),

    ]

    draw_resp_pt(hplot, smps, wc, draw_style=draw_config, configs=ctl2_ptresp_configs)


def draw_resp_pt(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            EGResoHistos, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_ptResp for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm.draw(
            text=text, 
            x_min=opts.get('x_min'), 
            x_max=opts.get('x_max'), 
            y_min=opts.get('y_min'), 
            y_max=opts.get('y_max'), 
            h_lines=opts.get('h_lines', []),
            norm=opts.get('norm', True),
            options=opts.get('options', 'hist'),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', True),
            y_axis_label=opts.get('y_axis_label', 'a.u.')
        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hPtResp_{h_name}', page_creator=wc_eff)




def draw_reso_eta(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            EGResoHistos, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_etaRes for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm.draw(
            text=text, 
            x_min=opts.get('x_min'), 
            x_max=opts.get('x_max'), 
            y_min=opts.get('y_min'), 
            y_max=opts.get('y_max'), 
            h_lines=opts.get('h_lines', []),
            norm=opts.get('norm', True),
            options=opts.get('options', 'hist'),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', 'a.u.')
        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hEtaRes_{h_name}', page_creator=wc_eff)


def draw_reso_eta(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            EGResoHistos, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_etaRes for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm.draw(
            text=text, 
            x_min=opts.get('x_min'), 
            x_max=opts.get('x_max'), 
            y_min=opts.get('y_min'), 
            y_max=opts.get('y_max'), 
            h_lines=opts.get('h_lines', []),
            norm=opts.get('norm', True),
            options=opts.get('options', 'hist'),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', 'a.u.')
        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hEtaRes_{h_name}', page_creator=wc_eff)


def draw_reso_phi(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            EGResoHistos, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_phiRes for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm.draw(
            text=text, 
            x_min=opts.get('x_min'), 
            x_max=opts.get('x_max'), 
            y_min=opts.get('y_min'), 
            y_max=opts.get('y_max'), 
            h_lines=opts.get('h_lines', []),
            norm=opts.get('norm', True),
            options=opts.get('options', 'hist'),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', 'a.u.')
        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hPhiRes_{h_name}', page_creator=wc_eff)



def draw_reso_caloeta(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            EGResoHistos, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_exetaRes for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm.draw(
            text=text, 
            x_min=opts.get('x_min'), 
            x_max=opts.get('x_max'), 
            y_min=opts.get('y_min'), 
            y_max=opts.get('y_max'), 
            h_lines=opts.get('h_lines', []),
            norm=opts.get('norm', True),
            options=opts.get('options', 'hist'),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', 'a.u.')
        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hCaloEtaRes_{h_name}', page_creator=wc_eff)


def draw_reso_calophi(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            EGResoHistos, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_exphiRes for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm.draw(
            text=text, 
            x_min=opts.get('x_min'), 
            x_max=opts.get('x_max'), 
            y_min=opts.get('y_min'), 
            y_max=opts.get('y_max'), 
            h_lines=opts.get('h_lines', []),
            norm=opts.get('norm', True),
            options=opts.get('options', 'hist'),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', 'a.u.')
        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hCaloPhiRes_{h_name}', page_creator=wc_eff)
