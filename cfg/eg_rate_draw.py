import python.histos as histos
from python.draw.drawingTools import *
from cfg.eg_rate import SingleObjRateHistoCounter, DoubleObjRateHistoCounter
import tabulate


def what(what):
    match what:
        case 'ctl2_rate':
            return [histos.RateHistos], 'eg_rate_ctl2', ctl2_rate_draw
        case 'menu_rate':
            return [histos.RateHistos], 'eg_rate_menu', menu_rate_draw
        case 'menu_ratecounter':
            return [SingleObjRateHistoCounter, DoubleObjRateHistoCounter], 'eg_menu_ratecounter', menu_ratecounter_draw
        case 'rate_pho_iso':
            return [histos.RateHistos], 'rate', rate_pho_iso_draw



draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #it{#scale[1.]{Phase-2 Simulation}}'),
                    (0.69, 0.91, '#it{14TeV}, 200 PU')]



def menu_ratecounter_draw(hplot, smps, wc):
    smp_table = {}
    for smp in smps:
        smp_table[smp.type] = []
    
    menu_single = [
        (['TkEleL2'], ['SingleTkEle36', 'SingleTkEle36EtaEB', 'SingleTkEle36EtaEE'], {}),
        (['TkEleL2'], ['SingleIsoTkEle28', 'SingleIsoTkEle28EtaEB', 'SingleIsoTkEle28EtaEE'], {}),
        (['TkEmL2'], ['SingleIsoTkPho36', 'SingleIsoTkPho36EtaEB', 'SingleIsoTkPho36EtaEE'], {}),
        (['EGSta'], ['SingleEGEle51', 'SingleEGEle51EtaEB', 'SingleEGEle51EtaEE'], {}),
        (['EGSta'], ['SingleEGEle51Fwd', 'SingleEGEle51FwdEtaEB', 'SingleEGEle51FwdEtaEE'], {}),
        # (['TkEleL2'], ['SingleIsoTkEle28', 'SingleIsoTkEle28EtaEB', 'SingleIsoTkEle28EtaEE'], {}),

    ]
    menu_double = [
        (['DoubleTkEleL2'], ['DoubleTkEle25-12'], {}),
        (['DoubleTkEmL2'], ['DoubleIsoTkPho22-12'], {}),
        (['DoubleEGSta'], ['DoubleStaEG37-24'], {}),
        (['DoubleEGSta'], ['DoubleStaEG37-24Fwd'], {}),
        (['DoubleTkEleEGSta'], ['DoubleIsoTkEleStaEG22-12'], {}),
        (['DoubleTkEleEGSta'], ['DoubleIsoTkEleStaEG22-12Fwd'], {}),

        # (['DoubleTkEleL2'], ['DoubleTkEle25-12Tight'], {}),
        # (['DoubleTkEmL2'], ['DoubleIsoTkPho22-12New'], {}),
        # (['DoubleEGSta'], ['DoubleStaEG37-24New'], {}),
        # (['DoubleTkEleEGSta'], ['DoubleIsoTkEleStaEG22-12New'], {}),

    ]


    for smp in smps:
        for obj, sels, opts in menu_single:
            singleobjcounter(hplot, smp_table, smp.type, obj, sels)

    for smp in smps:
        for obj, sels, opts in menu_double:
            doubleobjcounter(hplot, smp_table, smp.type, obj, sels)

    for smp in smps:
        print(f'--- {smp.label} ----------------------------------------')
        print(tabulate.tabulate(smp_table[smp.type], headers=[
            'seed', 
            'rate [kHz]', 
            f'rate EB [kHz]', 
            f'rate EE [kHz]']))
        print()

    table_by_seed = {}
    headers = ['seed']
    for smp in smps:
        headers.append(smp.label)
        for row in smp_table[smp.type]:
            seed = row[0]
            if seed not in table_by_seed:
                table_by_seed[seed] = [seed]
            # print(f'smp: {smp.label}, seed: {seed}, value: {row[1]}')
            table_by_seed[seed].append(row[1].split('\u00B1')[0].strip())

    values = []
    for seed, row in table_by_seed.items():
        values.append(row)

    # pprint(table_by_seed)

    print(tabulate.tabulate(values, headers=headers))
    print()
    
    # FXIME: dump to file


def doubleobjcounter(hplot, smp_table, smp, obj, sels):
    hsets, labels, text = hplot.get_histo(DoubleObjRateHistoCounter, smp, 'PU200', obj, sels, None)
    row = []
    # print(sels)
    # print(labels)
    row.append(sels[0]) #FIXME: why not labels[0]?

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
        (['TkEleL2'], ['MenuEle'], 'TkEleL2_MenuEle', {}),
        (['TkEleL2'], ['MenuEleEtaEB'], 'TkEleL2_MenuEleEtaEB', {}),
        (['TkEleL2'], ['MenuEleEtaEE'], 'TkEleL2_MenuEleEtaEE', {}),

        (['TkEleL2'], ['MenuEleLoose'], 'TkEleL2_MenuEleLoose', {}),
        (['TkEleL2'], ['MenuEleTight'], 'TkEleL2_MenuEleTight', {}),
        (['TkEleL2'], ['MenuEleLooseEtaEE'], 'TkEleL2_MenuEleLooseEtaEE', {}),
        (['TkEleL2'], ['MenuEleTightEtaEE'], 'TkEleL2_MenuEleTightEtaEE', {}),
        (['TkEleL2'], ['MenuEleLooseEtaEB'], 'TkEleL2_MenuEleLooseEtaEB', {}),
        (['TkEleL2'], ['MenuEleTightEtaEB'], 'TkEleL2_MenuEleTightEtaEB', {'y_min_diff': -5, 'y_max_diff': 5, 'v_lines': [29.0]}),
        (['TkEleL2'], ['MenuEleIsoLoose'], 'TkEleL2_MenuEleIsoLoose', {'is_iso': True}),
        (['TkEleL2'], ['MenuEleIsoTight'], 'TkEleL2_MenuEleIsoTight', {'is_iso': True}),
        (['TkEleL2'], ['MenuEleIsoTightEtaEB'], 'TkEleL2_MenuEleIsoTightEtaEB', {'is_iso': True, 'y_min_diff': 0, 'y_max_diff': 10, 'v_lines': [29.0]}),
        (['TkEleL2'], ['MenuEleIso'], 'TkEleL2_MenuEleIso', {'is_iso': True}),
        (['TkEleL2'], ['MenuEleIsoEtaEB'], 'TkEleL2_MenuEleIsoEtaEB', {'is_iso': True}),
        (['TkEleL2'], ['MenuEleIsoEtaEE'], 'TkEleL2_MenuEleIsoEtaEE', {'is_iso': True}),
        (['TkEleL2'], ['MenuEleIsoLooseEtaEE', 'MenuEleIsoTightEtaEE'], 'TkEleL2_MenuEleIsoAllEtaEE', {'is_iso': True}),
        (['TkEmL2'], ['MenuPhoIso'], 'TkEmL2_MenuPhoIso', {'is_iso': True}),
        (['TkEmL2'], ['MenuPhoIsoEtaEE'], 'TkEmL2_MenuPhoIsoEtaEE', {'is_iso': True}),
        (['TkEmL2'], ['MenuPhoIsoEtaEB'], 'TkEmL2_MenuPhoIsoEtaEB', {'is_iso': True}),
        (['EGStaEE'], ['MenuSta'], 'EGStaEE_MenuSta', {}),
        (['EGStaEB'], ['MenuSta'], 'EGStaEB_MenuSta', {}),

        (['DoubleTkEmL2'], ['MenuDoubleIsoTkPho22-X'], 'DoubleTkEmL2_DoubleIsoTkPho22-X', {'is_iso': True}),


    ]
    draw_rate(hplot, smps, wc, draw_style=draw_config, configs=menu)
    draw_rate(hplot, smps, wc, draw_style=draw_config, configs=menu, online=False)


def ctl2_rate_draw(hplot, smps, wc):
    menu = [
        (['TkEleL2'],       ['all'],                    'TkEleL2_all',          {}),
        (['TkEleL2'],       ['EtaEB'],                  'TkEleL2_EtaEB',        {}),
        (['TkEleL2'],       ['EtaEE'],                  'TkEleL2_EtaEE',        {}),
        (['TkEleL2'],       ['IDTightE'],               'TkEleL2_IDTightE',          {}),
        (['TkEleL2'],       ['IDTightEEtaEB'],          'TkEleL2_IDTightEEtaEB',        {}),
        (['TkEleL2'],       ['IDTightEEtaEE'],          'TkEleL2_IDTightEEtaEE',        {}),

        (['TkEmL2'],       ['all'],                    'TkEmL2_all',          {}),
        (['TkEmL2'],       ['EtaEB'],                  'TkEmL2_EtaEB',        {}),
        (['TkEmL2'],       ['EtaEE'],                  'TkEmL2_EtaEE',        {}),
        (['TkEmL2'],       ['IDTightP'],               'TkEmL2_IDTightP',          {}),
        (['TkEmL2'],       ['IDTightPEtaEB'],          'TkEmL2_IDTightPEtaEB',        {}),
        (['TkEmL2'],       ['IDTightPEtaEE'],          'TkEmL2_IDTightPEtaEE',        {}),

    ]
    draw_rate(hplot, smps, wc, draw_style=draw_config, configs=menu)


def rate_pho_iso_draw(hplot, smps, wc):
    menu = [   
        (['TkEmL2IsoWP'],       ['MenuPhoIso'],                    'TkEmL2_MenuPhoIso',                    {}),
        (['TkEmL2IsoWP'],       ['MenuPhoIsoEtaEE'],               'TkEmL2_MenuPhoIsoEtaEE',               {}),
        (['TkEmL2IsoWP'],       ['MenuPhoIsoEtaEB'],               'TkEmL2_MenuPhoIsoEtaEB',               {}),
        (['TkEmL2IsoWP'],       ['L2IDPhoL', 'L2IDPhoT', 'L2IDPhoLL2Iso', 'L2IDPhoTL2Iso'], 'TkEmL2_Pho', {}),
        (['TkEmL2IsoWP'],       ['L2IDPhoL', 'L2IDPhoLL2Iso', 'L2IDPhoLIsoPho90','L2IDPhoLIsoPho92','L2IDPhoLIsoPho94','L2IDPhoLIsoPho96', 'L2IDPhoLIsoPho98'], 'TkEmL2_PhoIsoFlatEff', {'y_min_diff': -200, 'y_max_diff': 0}),
        (['DoubleTkEmL2IsoWP'], ['MenuDoubleIsoTkPho22-X', 'MenuDoubleIso94TkPho22-X','MenuDoubleIso90TkPho22-X','MenuDoubleIso92TkPho22-X','MenuDoubleIso94TkPho22-X','MenuDoubleIso96TkPho22-X', 'MenuDoubleIso98TkPho22-X', 'MenuDoubleIsoOneTkPho22-X', 'MenuDoubleTkPho22-X'], 'DoubleTkEmL2_DoubleTkPho22-X', {'y_min': 0.5, 'y_max': 1000, 'x_min': 0, 'x_max': 40, 'v_lines': [12]})
    ]
    draw_rate(hplot, smps, wc, draw_style=draw_config, configs=menu)


def draw_rate(hplot, smps, wc, draw_style, configs, online=True):
    for objs, objs_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.4, 0.45)

        hsets, labels, text = hplot.get_histo(histos.RateHistos, [s.type for s in smps], 'PU200', objs, objs_sel, None)
        if not hsets:
            print(' -> skip draw')
            continue
        basename = 'hRate'
        if online:
            dm.addHistos([his.h_pt for his in hsets], labels=labels)
        else:
            basename = 'hRateOffline'
            is_iso = opts.get('is_iso', False)
            if is_iso:
                dm.addHistos([his.h_ptIsoOff for his in hsets], labels=labels)
            else:
                dm.addHistos([his.h_ptOff for his in hsets], labels=labels)
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
            x_axis_label=opts.get('x_axis_label'),
            v_lines=opts.get('v_lines', []),
            h_lines=opts.get('h_lines', [20,100,1000]),
            h_lines_ratio=opts.get('h_lines_ratio', [0.9, 1, 1.1]),
            do_ratio=opts.get('do_ratio', False),
            y_min_diff=opts.get('y_min_diff', 0.), 
            y_max_diff=opts.get('y_max_diff', 100.),
            do_diff=opts.get('do_diff', True))
        
        dm.toWeb(name=f'{basename}_{h_name}', page_creator=wc)