import python.histos as histos
from python.draw.drawingTools import *
from cfg.jetmet_rate import MetRateHistos, JetRateHistos
# from cfg.eg_rate import SingleObjRateHistoCounter, DoubleObjRateHistoCounter
import tabulate


def what(what):
    match what:
        case 'met':
            return [MetRateHistos], 'jetmet_rate', met_rate_draw
        case 'jets':
            return [JetRateHistos], 'jetmet_rate', jet_rate_draw
        case _:
            raise ValueError(f'Unknown draw function: {what}. Available: met, jets')



draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #it{#scale[1.]{Phase-2 Simulation}}'),
                    (0.69, 0.91, '#it{14TeV}, 200 PU')]




def jet_rate_draw(hplot, smps, wc):
    menu = [   
        (['CaloJets'], ['all'], 'CaloJets_all', {}),
        (['CaloJets'], ['EtaEB'], 'CaloJets_EtaEB', {}),
        (['CaloJets'], ['EtaEE'], 'CaloJets_EtaEE', {}),
        (['CaloJets'], ['EtaFwd'], 'CaloJets_EtaFwd', {}),
        (['CaloJets'], ['EtaVFwd'], 'CaloJets_EtaVFwd', {}),

        (['PFJets'],   ['all'], 'PFJets_all', {}),
        (['PFJets'],   ['EtaEB'], 'PFJets_EtaEB', {}),
        (['PFJets'],   ['EtaEE'], 'PFJets_EtaEE', {}),
        (['PFJets'],   ['EtaFwd'], 'PFJets_EtaFwd', {}),
        (['PFJets'],   ['EtaVFwd'], 'PFJets_EtaVFwd', {}),

        (['PuppiJets'],['all'], 'PuppiJets_all', {}),
        (['PuppiJets'],['EtaEB'], 'PuppiJets_EtaEB', {}),
        (['PuppiJets'],['EtaEE'], 'PuppiJets_EtaEE', {}),
        (['PuppiJets'],['EtaFwd'], 'PuppiJets_EtaFwd', {}),
        (['PuppiJets'],['EtaVFwd'], 'PuppiJets_EtaVFwd', {}),
        
        (['TkJets'],   ['all'], 'TkJets_all', {}),
        (['TkJets'],   ['EtaEB'], 'TkJets_EtaEB', {}),
        (['TkJets'],   ['EtaEE'], 'TkJets_EtaEE', {}),
        (['TkJets'],   ['EtaFwd'], 'TkJets_EtaFwd', {}),
        (['TkJets'],   ['EtaVFwd'], 'TkJets_EtaVFwd', {}),

        (['scPuppiJets'],   ['all'],     'scPuppiJets_all', {}),
        (['scPuppiJets'],   ['EtaEB'],   'scPuppiJets_EtaEB', {}),
        (['scPuppiJets'],   ['EtaEE'],   'scPuppiJets_EtaEE', {}),
        (['scPuppiJets'],   ['EtaFwd'],  'scPuppiJets_EtaFwd', {}),
        (['scPuppiJets'],   ['EtaVFwd'], 'scPuppiJets_EtaVFwd', {}),

        (['scPuppiCorrJets'],   ['all'],     'scPuppiCorrJets_all', {}),
        (['scPuppiCorrJets'],   ['EtaEB'],   'scPuppiCorrJets_EtaEB', {}),
        (['scPuppiCorrJets'],   ['EtaEE'],   'scPuppiCorrJets_EtaEE', {}),
        (['scPuppiCorrJets'],   ['EtaFwd'],  'scPuppiCorrJets_EtaFwd', {}),
        (['scPuppiCorrJets'],   ['EtaVFwd'], 'scPuppiCorrJets_EtaVFwd', {}),


    ]

    draw_rate(JetRateHistos, hplot, smps, wc, draw_style=draw_config, configs=menu)
    menu = [
        (['scPuppiCorrJets'],   ['all'],     'scPuppiCorrJets_all', {}),
        (['scPuppiCorrJets'],   ['EtaEB'],   'scPuppiCorrJets_EtaEB', {}),
        (['scPuppiCorrJets'],   ['EtaEE'],   'scPuppiCorrJets_EtaEE', {}),
        (['scPuppiCorrJets'],   ['EtaFwd'],  'scPuppiCorrJets_EtaFwd', {}),
        (['scPuppiCorrJets'],   ['EtaVFwd'], 'scPuppiCorrJets_EtaVFwd', {}),
    ]

    draw_rate(JetRateHistos, hplot, smps, wc, draw_style=draw_config, configs=menu, online=False)


def met_rate_draw(hplot, smps, wc):
    menu = [   
        (['CaloMet'], ['all'], 'CaloMet', {}),
        (['PFMet'],   ['all'], 'PFMet', {}),
        (['PuppiMet'],['all'], 'PuppiMet', {'x_max':200}),
        (['TkMet'],   ['all'], 'TkMet', {}),
        (['CaloMetCentral'], ['all'], 'CaloMet', {}),
        (['PFMetCentral'],   ['all'], 'PFMetCentral', {}),
        (['PuppiMetCentral'],['all'], 'PuppiMetCentral', {'x_max':200}),
        (['TkMetCentral'],   ['all'], 'TkMetCentral', {}),

    ]
    draw_rate(MetRateHistos, hplot, smps, wc, draw_style=draw_config, configs=menu)
    menu = [
        (['PuppiMet'],['all'], 'PuppiMet', {'x_max':200}),
    ]
    draw_rate(MetRateHistos, hplot, smps, wc, draw_style=draw_config, configs=menu, online=False)


def draw_rate(hclass, hplot, smps, wc, draw_style, configs, online=True):
    for objs, objs_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.4, 0.45)

        hsets, labels, text = hplot.get_histo(hclass, [s.type for s in smps], 'PU200', objs, objs_sel, None)
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
            x_max=opts.get('x_max'),
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