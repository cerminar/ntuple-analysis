import python.histos as histos
from python.draw.drawingTools import *
from cfg.eg_rate import SingleObjRateHistoCounter, DoubleObjRateHistoCounter
import tabulate


def what(what):
    match what:
        case 'menu_rate':
            return [histos.RateHistos], 'menu_rate', menu_rate_draw
        case 'menu_ratecounter':
            return [SingleObjRateHistoCounter, DoubleObjRateHistoCounter], 'menu_ratecounter', menu_ratecounter_draw
        case 'rate_pho_iso':
            return [histos.RateHistos], 'rate', rate_pho_iso_draw



draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #scale[1.]{Phase-2 Simulation}'),
                    (0.69, 0.91, '14TeV, 200 PU')]



def menu_ratecounter_draw(hplot, smps, wc):
    smp_table = {}
    for smp in smps:
        smp_table[smp] = []
    
    menu_single = [
        (['TkEleL2'], ['SingleTkEle36', 'SingleTkEle36EtaEB', 'SingleTkEle36EtaEE'], {}),
        (['TkEleL2'], ['SingleIsoTkEle28', 'SingleIsoTkEle28EtaEB', 'SingleIsoTkEle28EtaEE'], {}),
        (['TkEleL2'], ['SingleIsoTkEle28Tight', 'SingleIsoTkEle28TightEtaEB', 'SingleIsoTkEle28TightEtaEE'], {}),
        (['TkEmL2'], ['SingleIsoTkPho36', 'SingleIsoTkPho36EtaEB', 'SingleIsoTkPho36EtaEE'], {}),
        (['EGSta'], ['SingleEGEle51', 'SingleEGEle51EtaEB', 'SingleEGEle51EtaEE'], {}),

    ]
    menu_double = [
        (['DoubleTkEleL2'], ['DoubleTkEle25-12'], {}),
        (['DoubleTkEmL2'], ['DoubleIsoTkPho22-12'], {}),
        (['DoubleEGSta'], ['DoubleStaEG37-24'], {}),
        (['DoubleTkEleEGSta'], ['DoubleIsoTkEleStaEG22-12'], {}),

    ]


    for smp in smps:
        for obj, sels, opts in menu_single:
            singleobjcounter(hplot, smp_table, smp, obj, sels)

    for smp in smps:
        for obj, sels, opts in menu_double:
            doubleobjcounter(hplot, smp_table, smp, obj, sels)

    for smp in smps:
        print(f'--- {smp} ----------------------------------------')
        print(tabulate.tabulate(smp_table[smp], headers=[
            'seed', 
            'rate [kHz]', 
            f'rate EB [kHz]', 
            f'rate EE [kHz]']))
        print()
    # FXIME: dump to file


def doubleobjcounter(hplot, smp_table, smp, obj, sels):
    hsets, labels, text = hplot.get_histo(DoubleObjRateHistoCounter, smp, 'PU200', obj, sels, None)
    row = []
    row.append(sels[0])

    for hs in hsets:
        row.append(f'{round(hs.h_rate.GetBinContent(1), 2)} \u00B1 {round(hs.h_rate.GetBinError(1), 2)}')
    row.append('-')
    row.append('-')

    smp_table[smp].append(row)


def singleobjcounter(hplot, smp_table, smp, obj, sels):
    hsets, labels, text = hplot.get_histo(SingleObjRateHistoCounter, smp, 'PU200', obj, sels, None)
    row = []
    row.append(labels[0])
    for hs in hsets:
        row.append(f'{round(hs.h_rate.GetBinContent(1), 2)} \u00B1 {round(hs.h_rate.GetBinError(1), 2)}')

    smp_table[smp].append(row)



def menu_rate_draw(hplot, smps, wc):
    menu = [   
        (['TkEleL2'], ['MenuEleLoose'], 'hRate_TkEleL2_MenuEleLoose', {}),
        (['TkEleL2'], ['MenuEleTight'], 'hRate_TkEleL2_MenuEleTight', {}),
        (['TkEleL2'], ['MenuEleLooseEtaEE'], 'hRate_TkEleL2_MenuEleLooseEtaEE', {}),
        (['TkEleL2'], ['MenuEleTightEtaEE'], 'hRate_TkEleL2_MenuEleTightEtaEE', {}),
        (['TkEleL2'], ['MenuEleLooseEtaEB'], 'hRate_TkEleL2_MenuEleLooseEtaEB', {}),
        (['TkEleL2'], ['MenuEleTightEtaEB'], 'hRate_TkEleL2_MenuEleTightEtaEB', {}),
        (['TkEleL2'], ['MenuEleIsoLoose'], 'hRate_TkEleL2_MenuEleIsoLoose', {}),
        (['TkEleL2'], ['MenuEleIsoTight'], 'hRate_TkEleL2_MenuEleIsoTight', {}),
        (['TkEleL2'], ['MenuEleIsoTightEtaEB'], 'hRate_TkEleL2_MenuEleIsoTightEtaEB', {}),
        (['TkEleL2'], ['MenuEleIsoTight', 'MenuEleIsoLoose'], 'hRate_TkEleL2_MenuEleIso', {}),
        (['TkEmL2'], ['MenuPhoIso'], 'hRate_TkEmL2_MenuPhoIso', {}),
        (['TkEmL2'], ['MenuPhoIsoEtaEE'], 'hRate_TkEmL2_MenuPhoIsoEtaEE', {}),
        (['TkEmL2'], ['MenuPhoIsoEtaEB'], 'hRate_TkEmL2_MenuPhoIsoEtaEB', {}),
        (['EGStaEE'], ['MenuSta'], 'hRate_EGStaEE_MenuSta', {}),
        (['EGStaEB'], ['MenuSta'], 'hRate_EGStaEB_MenuSta', {}),
        (['DoubleTkEmL2'], ['MenuDoubleIsoTkPho22-X'], 'hRate_DoubleTkEmL2_DoubleIsoTkPho22-X', {})

    ]
    draw_rate(hplot, smps, wc, draw_style=draw_config, configs=menu)


def rate_pho_iso_draw(hplot, smps, wc):
    menu = [   
        (['TkEmL2IsoWP'], ['MenuPhoIso'], 'hRate_TkEmL2_MenuPhoIso', {}),
        (['TkEmL2IsoWP'], ['MenuPhoIsoEtaEE'], 'hRate_TkEmL2_MenuPhoIsoEtaEE', {}),
        (['TkEmL2IsoWP'], ['MenuPhoIsoEtaEB'], 'hRate_TkEmL2_MenuPhoIsoEtaEB', {}),
        (['TkEmL2IsoWP'], ['L2IDPhoL', 'L2IDPhoT', 'L2IDPhoLL2Iso', 'L2IDPhoTL2Iso'], 'hRate_TkEmL2_Pho', {}),
        (['TkEmL2IsoWP'], ['L2IDPhoL', 'L2IDPhoLL2Iso', 'L2IDPhoLIsoPho90','L2IDPhoLIsoPho92','L2IDPhoLIsoPho94','L2IDPhoLIsoPho96', 'L2IDPhoLIsoPho98'], 'hRate_TkEmL2_PhoIsoFlatEff', {'y_min_diff': -200, 'y_max_diff': 0}),

        (['DoubleTkEmL2IsoWP'], 
         ['MenuDoubleIsoTkPho22-X', 'MenuDoubleIso94TkPho22-X','MenuDoubleIso90TkPho22-X','MenuDoubleIso92TkPho22-X','MenuDoubleIso94TkPho22-X','MenuDoubleIso96TkPho22-X', 'MenuDoubleIso98TkPho22-X', 'MenuDoubleIsoOneTkPho22-X', 'MenuDoubleTkPho22-X'], 
         'hRate_DoubleTkEmL2_DoubleTkPho22-X', {'y_min': 0.5, 'y_max': 1000, 'x_min': 0, 'x_max': 40, 'v_lines': [12]})

    ]
    draw_rate(hplot, smps, wc, draw_style=draw_config, configs=menu)


def draw_rate(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.4, 0.45)

        hsets, labels, text = hplot.get_histo(histos.RateHistos, smps, 'PU200', objs, objs_sel, None)
        if not hsets:
            print(' -> skip draw')
            continue
        dm.addHistos([his.h_pt for his in hsets], labels=labels)
        # print(hsets[0].h_pt.GetName())

        for id in range(1,len(hsets)):
            dm.addDiffHisto(id,0)

        for id in range(1,len(hsets)):
            dm.addRatioHisto(id,0)

        # dm.addRatioHisto(1,0)
    #     dm.addRatioHisto(2,0)

        dm.draw(
            text=text,
            y_min=opts.get('y_min', 0.5), 
            y_max=opts.get('y_max', 40000),
            x_min=opts.get('x_min', 0.), 
            x_max=opts.get('x_max', 60.),
            y_min_ratio=opts.get('y_min_ratio', 0.8), 
            y_max_ratio=opts.get('y_max_ratio', 1.2),
            y_log=opts.get('y_log', True), 
            x_axis_label=opts.get('x_axis_label', 'online p_{T} thresh. [GeV]'),
            v_lines=opts.get('v_lines', []),
            h_lines=opts.get('h_lines', [20,100,1000]),
            h_lines_ratio=opts.get('h_lines_ratio', [0.9, 1, 1.1]),
            do_ratio=opts.get('do_ratio', False),
            y_min_diff=opts.get('y_min_diff', 0.), 
            y_max_diff=opts.get('y_max_diff', 100.),
            do_diff=opts.get('do_diff', True))
        dm.toWeb(name=h_name, page_creator=wc)