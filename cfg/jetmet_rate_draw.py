import python.histos as histos
from python.draw.drawingTools import *
from cfg.jetmet_rate import METRateHistos
# from cfg.eg_rate import SingleObjRateHistoCounter, DoubleObjRateHistoCounter
import tabulate


def what(what):
    match what:
        case 'met':
            return [METRateHistos], 'jetmet_rate', met_rate_draw



draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #it{#scale[1.]{Phase-2 Simulation}}'),
                    (0.69, 0.91, '#it{14TeV}, 200 PU')]





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
    draw_rate(hplot, smps, wc, draw_style=draw_config, configs=menu)

def draw_rate(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.4, 0.45)

        hsets, labels, text = hplot.get_histo(METRateHistos, smps, 'PU200', objs, objs_sel, None)
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
            x_max=opts.get('x_max', 500.),
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
        dm.toWeb(name=f'hRate_{h_name}', page_creator=wc)