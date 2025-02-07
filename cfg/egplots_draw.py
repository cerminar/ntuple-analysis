import cfg.egplots
import python.histos as histos
from cfg.eg_genmatch import EGHistos

from python.draw.drawingTools import *




def what(what):
    match what:
        case 'tkeg_plots':
            return [EGHistos], 'unmatched', tkeg_plots_draw




draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #it{#scale[1.]{Phase-2 Simulation}}'),
                    (0.69, 0.91, '#it{14TeV}, 200 PU')]



def tkeg_plots_draw(hplot, smps, wc):
    tkem_configs = [
        (['TkEmL2',], ['all'], ['nomatch'],  'TkEmL2_all_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['EtaEB'], ['nomatch'],  'TkEmL2_EtaEB_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['EtaEE'], ['nomatch'],  'TkEmL2_EtaEE_nomatch', {'y_min': 1E-3}),

        (['TkEmL2',], ['Pt2'], ['nomatch'],       'TkEmL2_Pt2_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt2EtaEB'], ['nomatch'],  'TkEmL2_Pt2EtaEB_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt2EtaEE'], ['nomatch'],  'TkEmL2_Pt2EtaEE_nomatch', {'y_min': 1E-3}),

        (['TkEmL2',], ['Pt5'], ['nomatch'],       'TkEmL2_Pt5_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt5EtaEB'], ['nomatch'],  'TkEmL2_Pt5EtaEB_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt5EtaEE'], ['nomatch'],  'TkEmL2_Pt5EtaEE_nomatch', {'y_min': 1E-3}),


        (['TkEmL2',], ['IDTightP'], ['nomatch'],  'TkEmL2_IDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['EtaEBIDTightP'], ['nomatch'],  'TkEmL2_EtaEBIDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['EtaEEIDTightP'], ['nomatch'],  'TkEmL2_EtaEEIDTightP_nomatch', {'y_min': 1E-3}),

        (['TkEmL2',], ['Pt1IDTightP'],      ['nomatch'],  'TkEmL2_Pt1IDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt1EtaEBIDTightP'], ['nomatch'],  'TkEmL2_Pt1EtaEBIDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt1EtaEEIDTightP'], ['nomatch'],  'TkEmL2_Pt1EtaEEIDTightP_nomatch', {'y_min': 1E-3}),

        (['TkEmL2',], ['Pt2IDTightP'],      ['nomatch'],  'TkEmL2_Pt2IDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt2EtaEBIDTightP'], ['nomatch'],  'TkEmL2_Pt2EtaEBIDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt2EtaEEIDTightP'], ['nomatch'],  'TkEmL2_Pt2EtaEEIDTightP_nomatch', {'y_min': 1E-3}),

        (['TkEmL2',], ['Pt5IDTightP'],      ['nomatch'],  'TkEmL2_Pt5IDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt5EtaEBIDTightP'], ['nomatch'],  'TkEmL2_Pt5EtaEBIDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt5EtaEEIDTightP'], ['nomatch'],  'TkEmL2_Pt5EtaEEIDTightP_nomatch', {'y_min': 1E-3}),

    ]

    draw_nobj(hplot, smps, wc, draw_style=draw_config, configs=tkem_configs)

    tkele_configs = [
        (['TkEleL2',], ['all'], ['nomatch'],    'TkEleL2_all_nomatch', {'y_min': 1E-3}),
        (['TkEleL2',], ['EtaEB'], ['nomatch'],  'TkEleL2_EtaEB_nomatch', {'y_min': 1E-3}),
        (['TkEleL2',], ['EtaEE'], ['nomatch'],  'TkEleL2_EtaEE_nomatch', {'y_min': 1E-3}),

        (['TkEleL2',], ['IDTightE'], ['nomatch'],       'TkEleL2_IDTightE_nomatch', {'y_min': 1E-3}),
        (['TkEleL2',], ['EtaEBIDTightE'], ['nomatch'],  'TkEleL2_IDTightEEtaEB_nomatch', {'y_min': 1E-3}),
        (['TkEleL2',], ['EtaEEIDTightE'], ['nomatch'],  'TkEleL2_IDTightEEtaEE_nomatch', {'y_min': 1E-3}),

    ]
    draw_nobj(hplot, smps, wc, draw_style=draw_config, configs=tkele_configs)
    draw_idscore(hplot, smps, wc, draw_style=draw_config, configs=tkele_configs)


    tkem_eta_configs = [
        (['TkEmL2',], ['all'], ['nomatch'],  'TkEmL2_all_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt2'], ['nomatch'],       'TkEmL2_Pt2_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt5'], ['nomatch'],       'TkEmL2_Pt5_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['IDTightP'], ['nomatch'],  'TkEmL2_IDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt1IDTightP'],      ['nomatch'],  'TkEmL2_Pt1IDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt2IDTightP'],      ['nomatch'],  'TkEmL2_Pt2IDTightP_nomatch', {'y_min': 1E-3}),
        (['TkEmL2',], ['Pt5IDTightP'],      ['nomatch'],  'TkEmL2_Pt5IDTightP_nomatch', {'y_min': 1E-3}),

        (['TkEleL2',], ['all'],              ['nomatch'],  'TkEleL2_all_nomatch', {'y_min': 1E-3}),
        (['TkEleL2',], ['Pt2'],              ['nomatch'],  'TkEleL2_Pt2_nomatch', {'y_min': 1E-3}),
        (['TkEleL2',], ['Pt5'],              ['nomatch'],  'TkEleL2_Pt5_nomatch', {'y_min': 1E-3}),
        (['TkEleL2',], ['IDTightE'],         ['nomatch'],  'TkEleL2_IDTightE_nomatch', {'y_min': 1E-3}),
        (['TkEleL2',], ['Pt1IDTightE'],      ['nomatch'],  'TkEleL2_Pt1IDTightE_nomatch', {'y_min': 1E-3}),
        (['TkEleL2',], ['Pt2IDTightE'],      ['nomatch'],  'TkEleL2_Pt2IDTightE_nomatch', {'y_min': 1E-3}),
        (['TkEleL2',], ['Pt5IDTightE'],      ['nomatch'],  'TkEleL2_Pt5IDTightE_nomatch', {'y_min': 1E-3}),

    ]

    draw_eta(hplot, smps, wc, draw_style=draw_config, configs=tkem_eta_configs)

    tkem_pt_configs = [
        (['TkEmL2',], ['all'], ['nomatch'],    'TkEmL2_all_nomatch',     {}),
        (['TkEmL2',], ['EtaEB'], ['nomatch'],  'TkEmL2_EtaEB_nomatch', {}),
        (['TkEmL2',], ['EtaEE'], ['nomatch'],  'TkEmL2_EtaEE_nomatch', {}),

        (['TkEmL2',], ['IDTightP'], ['nomatch'],  'TkEmL2_IDTightP_nomatch',           {}),
        (['TkEmL2',], ['EtaEBIDTightP'], ['nomatch'],  'TkEmL2_EtaEBIDTightP_nomatch', {}),
        (['TkEmL2',], ['EtaEEIDTightP'], ['nomatch'],  'TkEmL2_EtaEEIDTightP_nomatch', {}),

        (['TkEleL2',], ['all'], ['nomatch'],            'TkEleL2_all_nomatch',     {}),
        (['TkEleL2',], ['EtaEB'], ['nomatch'],          'TkEleL2_EtaEB_nomatch', {}),
        (['TkEleL2',], ['EtaEE'], ['nomatch'],          'TkEleL2_EtaEE_nomatch', {}),
        (['TkEleL2',], ['IDTightE'], ['nomatch'],       'TkEleL2_IDTightE_nomatch',           {}),
        (['TkEleL2',], ['EtaEBIDTightE'], ['nomatch'],  'TkEleL2_EtaEBIDTightE_nomatch', {}),
        (['TkEleL2',], ['EtaEEIDTightE'], ['nomatch'],  'TkEleL2_EtaEEIDTightE_nomatch', {}),

    ]

    draw_pt(hplot, smps, wc, draw_style=draw_config, configs=tkem_pt_configs)




def draw_nobj(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)

        dm.config.legend_position = (0.2,0.2)

        hsets, labels, text = hplot.get_histo(
            EGHistos, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)

        dm.addHistos([his.h_n for his in hsets], labels=labels)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                x_min=opts.get('x_min', 0.), 
                x_max=opts.get('x_max', 13.), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', False),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                x_axis_label=opts.get('x_axis_label', '# objs.'),
                y_log=opts.get('y_log', True),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hNObj_{h_name}', page_creator=wc)

def draw_idscore(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)

        dm.config.legend_position = (0.2,0.2)

        hsets, labels, text = hplot.get_histo(
            EGHistos, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)

        dm.addHistos([his.h_idScore for his in hsets], labels=labels)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                x_min=opts.get('x_min', -1.), 
                x_max=opts.get('x_max', 1.), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', False),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', True),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hIdScore_{h_name}', page_creator=wc)



def draw_eta(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)

        dm.config.legend_position = (0.2,0.2)

        hsets, labels, text = hplot.get_histo(
            EGHistos, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)

        dm.addHistos([his.h_eta for his in hsets], labels=labels)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                x_min=opts.get('x_min', -3), 
                x_max=opts.get('x_max', 3), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', False),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                y_log=opts.get('y_log', True),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hEta_{h_name}', page_creator=wc)


def draw_pt(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)

        dm.config.legend_position = (0.2,0.2)

        hsets, labels, text = hplot.get_histo(
            EGHistos, 
            smps, 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)

        dm.addHistos([his.h_pt for his in hsets], labels=labels)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                x_min=opts.get('x_min', 0), 
                x_max=opts.get('x_max', 100), 
                y_min=opts.get('y_min', 1E-5),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', False),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                y_log=opts.get('y_log', True),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hPt_{h_name}', page_creator=wc)

