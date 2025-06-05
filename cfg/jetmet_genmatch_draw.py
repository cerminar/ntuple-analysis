# import cfg.eg_genmatch
import python.histos as histos
from python.draw.drawingTools import *
import python.draw.utilities as draw_utils
from cfg.jetmet_genmatch import JetResoHistos
from cfg.eg_genmatch_draw import draw_effvseta,draw_effvspt



def what(what):
    match what:
        case 'jet_reso':
            return [JetResoHistos], 'jet_reso', jet_reso_draw
        case 'jet_eff':
            return [histos.HistoSetEff], 'jet_eff', jet_eff_draw
        case _:
            raise ValueError(f'Unknown draw function: {what}. Available: jet_reso, jet_eff')




draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #it{#scale[1.]{Phase-2 Simulation}}'),
                    (0.69, 0.91, '#it{14TeV}, 200 PU')]


def jet_eff_draw(hplot, smps, wc):
    effvseta_configs = [    
        (['CaloJets',],        ['all'],   ['GENJPt30'],  'CaloJets_all_GENJPt30',        {'x_max': 5., 'y_min': 0.0}),
        (['TkJets',],          ['all'],   ['GENJPt30'],  'TkJets_all_GENJPt30',          {'x_max': 5., 'y_min': 0.0}),
        (['PFJets',],          ['all'],   ['GENJPt30'],  'PFJets_all_GENJPt30',          {'x_max': 5., 'y_min': 0.0}),
        (['PuppiJets',],       ['all'],   ['GENJPt30'],  'PuppiJets_all_GENJPt30',       {'x_max': 5., 'y_min': 0.0}),
        (['scPuppiJets',],     ['all'],   ['GENJPt30'],  'scPuppiJets_all_GENJPt30',     {'x_max': 5., 'y_min': 0.0}),
        (['scPuppiCorrJets',], ['all'],   ['GENJPt30'],  'scPuppiCorrJets_all_GENJPt30', {'x_max': 5., 'y_min': 0.0}),

        (['CaloJets',],        ['all'],   ['GENJPt100'],  'CaloJets_all_GENJPt100',        {'x_max': 5., 'y_min': 0.0}),
        (['TkJets',],          ['all'],   ['GENJPt100'],  'TkJets_all_GENJPt100',          {'x_max': 5., 'y_min': 0.0}),
        (['PFJets',],          ['all'],   ['GENJPt100'],  'PFJets_all_GENJPt100',          {'x_max': 5., 'y_min': 0.0}),
        (['PuppiJets',],       ['all'],   ['GENJPt100'],  'PuppiJets_all_GENJPt100',       {'x_max': 5., 'y_min': 0.0}),
        (['scPuppiJets',],     ['all'],   ['GENJPt100'],  'scPuppiJets_all_GENJPt100',     {'x_max': 5., 'y_min': 0.0}),
        (['scPuppiCorrJets',], ['all'],   ['GENJPt100'],  'scPuppiCorrJets_all_GENJPt100', {'x_max': 5., 'y_min': 0.0}),

    ]
    draw_effvseta(hplot, smps, wc, draw_style=draw_config, configs=effvseta_configs)

    effvspt_configs = [
        (['CaloJets',],  ['all'],       ['GENJEtaEB'],  'CaloJets_all_GENEtaEB', {}),
        (['CaloJets',],  ['all'],       ['GENJEtaEE'],  'CaloJets_all_GENEtaEE', {}),
        (['TkJets',],    ['all'],       ['GENJEtaEB'],  'TkJets_all_GENEtaEB',   {}),
        (['TkJets',],    ['all'],       ['GENJEtaEE'],  'TkJets_all_GENEtaEE',   {}),
        (['PFJets',],    ['all'],       ['GENJEtaEB'],  'PFJets_all_GENEtaEB',   {}),
        (['PFJets',],    ['all'],       ['GENJEtaEE'],  'PFJets_all_GENEtaEE',   {}),
        (['PuppiJets',], ['all'],       ['GENJEtaEB'],  'PuppiJets_all_GENEtaEB', {}),
        (['PuppiJets',], ['all'],       ['GENJEtaEE'],  'PuppiJets_all_GENEtaEE', {}),
        (['scPuppiJets',], ['all'],     ['GENJEtaEB'],  'scPuppiJets_all_GENEtaEB', {}),
        (['scPuppiJets',], ['all'],     ['GENJEtaEE'],  'scPuppiJets_all_GENEtaEE', {}),
        (['scPuppiCorrJets',], ['all'], ['GENJEtaEB'],  'scPuppiCorrJets_all_GENEtaEB', {}),
        (['scPuppiCorrJets',], ['all'], ['GENJEtaEE'],  'scPuppiCorrJets_all_GENEtaEE', {}),

        (['CaloJets',],  ['all'],       ['GENJEtaFwd'],  'CaloJets_all_GENEtaFwd', {}),
        (['PFJets',],    ['all'],       ['GENJEtaFwd'],  'PFJets_all_GENEtaFwd',   {}),
        (['PuppiJets',], ['all'],       ['GENJEtaFwd'],  'PuppiJets_all_GENEtaFwd', {}),
        (['scPuppiJets',], ['all'],     ['GENJEtaFwd'],  'scPuppiJets_all_GENEtaFwd', {}),
        (['scPuppiCorrJets',], ['all'], ['GENJEtaFwd'],  'scPuppiCorrJets_all_GENEtaFwd', {}),

        (['CaloJets',],  ['all'],       ['GENJEtaVFwd'],  'CaloJets_all_GENEtaVFwd', {}),
        (['PFJets',],    ['all'],       ['GENJEtaVFwd'],  'PFJets_all_GENEtaVFwd',   {}),
        (['PuppiJets',], ['all'],       ['GENJEtaVFwd'],  'PuppiJets_all_GENEtaVFwd', {}),
        (['scPuppiJets',], ['all'],     ['GENJEtaVFwd'],  'scPuppiJets_all_GENEtaVFwd', {}),
        (['scPuppiCorrJets',], ['all'], ['GENJEtaVFwd'],  'scPuppiCorrJets_all_GENEtaVFwd', {}),

    ]
    draw_effvspt(hplot, smps, wc, draw_style=draw_config, configs=effvspt_configs)





def jet_reso_draw(hplot, smps, wc):
    jets_etaphi_reso_configs = [    
        (['CaloJets',],            ['all'],              ['GENJEtaEB'],       'CaloJets_all_GENEtaEB', {}),
        (['CaloJets',],            ['all'],              ['GENJEtaEE'],       'CaloJets_all_GENEtaEE', {}),
        (['TkJets',],            ['all'],              ['GENJEtaEB'],       'TkJets_all_GENEtaEB',   {}),
        (['TkJets',],            ['all'],              ['GENJEtaEE'],       'TkJets_all_GENEtaEE',   {}),
        (['PFJets',],            ['all'],              ['GENJEtaEB'],       'PFJets_all_GENEtaEB',   {}),
        (['PFJets',],            ['all'],              ['GENJEtaEE'],       'PFJets_all_GENEtaEE',   {}),
        (['PuppiJets',],         ['all'],              ['GENJEtaEB'],       'PuppiJets_all_GENEtaEB', {}),
        (['PuppiJets',],         ['all'],              ['GENJEtaEE'],       'PuppiJets_all_GENEtaEE', {}),
        (['scPuppiJets',],    ['all'],              ['GENJEtaEB'],       'scPuppiJets_all_GENEtaEB', {}),
        (['scPuppiJets',],    ['all'],              ['GENJEtaEE'],       'scPuppiJets_all_GENEtaEE', {}),
        (['scPuppiCorrJets',],    ['all'],              ['GENJEtaEB'],       'scPuppiCorrJets_all_GENEtaEB', {}),
        (['scPuppiCorrJets',],    ['all'],              ['GENJEtaEE'],       'scPuppiCorrJets_all_GENEtaEE', {}),
    ]
    draw_reso_eta(hplot, smps, wc, draw_style=draw_config, configs=jets_etaphi_reso_configs)
    draw_reso_phi(hplot, smps, wc, draw_style=draw_config, configs=jets_etaphi_reso_configs)

    # ctl2_caloetaphi_reso_configs = [    

    #     (['CaloJets',],            ['all'],              ['GEN'],            'CaloJets_all_GEN', {}),
    #     (['CaloJets',],            ['all'],              ['GENJEtaEB'],       'CaloJets_all_GENEtaEB', {}),
    #     (['CaloJets',],            ['all'],              ['GENJEtaEE'],       'CaloJets_all_GENEtaEE', {}),

    # ]
    # draw_reso_caloeta(hplot, smps, wc, draw_style=draw_config, configs=ctl2_caloetaphi_reso_configs)
    # draw_reso_calophi(hplot, smps, wc, draw_style=draw_config, configs=ctl2_caloetaphi_reso_configs)

    jet_ptresp_configs = [    
        (['CaloJets',],            ['all'],              ['GENJEtaEB'],       'CaloJets_all_GENEtaEB', {'y_min': 1E-5}),
        (['CaloJets',],            ['all'],              ['GENJEtaEE'],       'CaloJets_all_GENEtaEE', {'y_min': 1E-5}),
        (['TkJets',],            ['all'],              ['GENJEtaEB'],       'TkJets_all_GENEtaEB',   {'y_min': 1E-5}),
        (['TkJets',],            ['all'],              ['GENJEtaEE'],       'TkJets_all_GENEtaEE',   {'y_min': 1E-5}),
        (['PFJets',],            ['all'],              ['GENJEtaEB'],       'PFJets_all_GENEtaEB',   {'y_min': 1E-5}),
        (['PFJets',],            ['all'],              ['GENJEtaEE'],       'PFJets_all_GENEtaEE',   {'y_min': 1E-5}),
        (['PuppiJets',],         ['all'],              ['GENJEtaEB'],       'PuppiJets_all_GENEtaEB', {'y_min': 1E-5}),
        (['PuppiJets',],         ['all'],              ['GENJEtaEE'],       'PuppiJets_all_GENEtaEE', {'y_min': 1E-5}),
    ]

    draw_resp_pt(hplot, smps, wc, draw_style=draw_config, configs=jet_ptresp_configs)


    jet_ptrespvspt_configs = [    
        (['CaloJets',],            ['all'],              ['GENJEtaEB'],       'CaloJets_all_GENEtaEB',  {'y_min': 0}),
        (['CaloJets',],            ['all'],              ['GENJEtaEE'],       'CaloJets_all_GENEtaEE',  {'y_min': 0}),
        (['CaloJets',],            ['all'],              ['GENJEtaFwd'],       'CaloJets_all_GENEtaFwd',  {'y_min': 0}),
        (['CaloJets',],            ['all'],              ['GENJEtaVFwd'],       'CaloJets_all_GENEtaVFwd',  {'y_min': 0}),

        (['TkJets',],            ['all'],              ['GENJEtaEB'],       'TkJets_all_GENEtaEB',      {'y_min': 0}),
        (['TkJets',],            ['all'],              ['GENJEtaEE'],       'TkJets_all_GENEtaEE',      {'y_min': 0}),
        (['PFJets',],            ['all'],              ['GENJEtaEB'],       'PFJets_all_GENEtaEB',      {'y_min': 0}),
        (['PFJets',],            ['all'],              ['GENJEtaEE'],       'PFJets_all_GENEtaEE',      {'y_min': 0}),
        (['PFJets',],            ['all'],              ['GENJEtaFwd'],       'PFJets_all_GENEtaFwd',      {'y_min': 0}),
        (['PFJets',],            ['all'],              ['GENJEtaVFwd'],       'PFJets_all_GENEtaVFwd',      {'y_min': 0}),

        (['PuppiJets',],         ['all'],              ['GENJEtaEB'],       'PuppiJets_all_GENEtaEB',   {'y_min': 0}),
        (['PuppiJets',],         ['all'],              ['GENJEtaEE'],       'PuppiJets_all_GENEtaEE',   {'y_min': 0}),
        (['PuppiJets',],         ['all'],              ['GENJEtaFwd'],       'PuppiJets_all_GENEtaFwd',   {'y_min': 0}),
        (['PuppiJets',],         ['all'],              ['GENJEtaVFwd'],       'PuppiJets_all_GENEtaVFwd',   {'y_min': 0}),

        (['scPuppiJets',],    ['all'],              ['GENJEtaEB'],       'scPuppiJets_all_GENEtaEB', {'y_min': 0}),
        (['scPuppiJets',],    ['all'],              ['GENJEtaEE'],       'scPuppiJets_all_GENEtaEE', {'y_min': 0}),
        (['scPuppiJets',],    ['all'],              ['GENJEtaFwd'],       'scPuppiJets_all_GENEtaFwd', {'y_min': 0}),
        (['scPuppiJets',],    ['all'],              ['GENJEtaVFwd'],       'scPuppiJets_all_GENEtaVFwd', {'y_min': 0}),

        (['scPuppiCorrJets',],    ['all'],              ['GENJEtaEB'],       'scPuppiCorrJets_all_GENEtaEB', {'y_min': 0}),
        (['scPuppiCorrJets',],    ['all'],              ['GENJEtaEE'],       'scPuppiCorrJets_all_GENEtaEE', {'y_min': 0}),
        (['scPuppiCorrJets',],    ['all'],              ['GENJEtaFwd'],       'scPuppiCorrJets_all_GENEtaFwd', {'y_min': 0}),
        (['scPuppiCorrJets',],    ['all'],              ['GENJEtaVFwd'],       'scPuppiCorrJets_all_GENEtaVFwd', {'y_min': 0}),

    ]
    draw_resp_ptVpt_median(hplot, smps, wc, draw_style=draw_config, configs=jet_ptrespvspt_configs)


    jet_ptrespvspt_configs = [    
        (['CaloJets',],            ['all'],              ['GENJEtaEB'],       'CaloJets_all_GENEtaEB',  {'y_min': 0.1, 'y_max': 0.8}),
        (['CaloJets',],            ['all'],              ['GENJEtaEE'],       'CaloJets_all_GENEtaEE',  {'y_min': 0.1, 'y_max': 0.8}),
        (['TkJets',],            ['all'],              ['GENJEtaEB'],       'TkJets_all_GENEtaEB',      {'y_min': 0.1, 'y_max': 0.4}),
        (['TkJets',],            ['all'],              ['GENJEtaEE'],       'TkJets_all_GENEtaEE',      {'y_min': 0.1, 'y_max': 0.4}),
        (['PFJets',],            ['all'],              ['GENJEtaEB'],       'PFJets_all_GENEtaEB',      {'y_min': 0.1, 'y_max': 0.4}),
        (['PFJets',],            ['all'],              ['GENJEtaEE'],       'PFJets_all_GENEtaEE',      {'y_min': 0.1, 'y_max': 0.4}),
        (['PuppiJets',],         ['all'],              ['GENJEtaEB'],       'PuppiJets_all_GENEtaEB',   {'y_min': 0.1, 'y_max': 0.4}),
        (['PuppiJets',],         ['all'],              ['GENJEtaEE'],       'PuppiJets_all_GENEtaEE',   {'y_min': 0.1, 'y_max': 0.4}),
        (['scPuppiJets',],    ['all'],              ['GENJEtaEB'],       'scPuppiJets_all_GENEtaEB', {'y_min': 0.1, 'y_max': 0.4}),
        (['scPuppiJets',],    ['all'],              ['GENJEtaEE'],       'scPuppiJets_all_GENEtaEE', {'y_min': 0.1, 'y_max': 0.4}),
        (['scPuppiCorrJets',],    ['all'],              ['GENJEtaEB'],       'scPuppiCorrJets_all_GENEtaEB', {'y_min': 0.1, 'y_max': 0.4}),
        (['scPuppiCorrJets',],    ['all'],              ['GENJEtaEE'],       'scPuppiCorrJets_all_GENEtaEE', {'y_min': 0.1, 'y_max': 0.4}),
    ]
    draw_resp_ptVpt_sigma(hplot, smps, wc, draw_style=draw_config, configs=jet_ptrespvspt_configs)

    jet_ptrespvspt_configs = [    
        (['CaloJets',],            ['all'],              ['GENJEtaEB'],       'CaloJets_all_GENEtaEB',  {}),
        (['CaloJets',],            ['all'],              ['GENJEtaEE'],       'CaloJets_all_GENEtaEE',  {}),
        (['TkJets',],            ['all'],              ['GENJEtaEB'],       'TkJets_all_GENEtaEB',    {}),
        (['TkJets',],            ['all'],              ['GENJEtaEE'],       'TkJets_all_GENEtaEE',    {}),
        (['PFJets',],            ['all'],              ['GENJEtaEB'],       'PFJets_all_GENEtaEB',    {}),
        (['PFJets',],            ['all'],              ['GENJEtaEE'],       'PFJets_all_GENEtaEE',    {}),
        (['PuppiJets',],         ['all'],              ['GENJEtaEB'],       'PuppiJets_all_GENEtaEB', {}),
        (['PuppiJets',],         ['all'],              ['GENJEtaEE'],       'PuppiJets_all_GENEtaEE', {}),
        (['scPuppiJets',],    ['all'],              ['GENJEtaEB'],       'scPuppiJets_all_GENEtaEB', {}),
        (['scPuppiJets',],    ['all'],              ['GENJEtaEE'],       'scPuppiJets_all_GENEtaEE', {}),
        (['scPuppiCorrJets',],    ['all'],              ['GENJEtaEB'],       'scPuppiCorrJets_all_GENEtaEB', {}),
        (['scPuppiCorrJets',],    ['all'],              ['GENJEtaEE'],       'scPuppiCorrJets_all_GENEtaEE', {}),
    ]
    draw_resp_ptVpt(hplot, smps, wc, draw_style=draw_config, configs=jet_ptrespvspt_configs)



    jet_ptrespvseta_configs = [    
        (['CaloJets',],    ['all'],    ['GENJPt30'],      'CaloJets_all_GENPt30',     {'y_min': 0}),
        (['TkJets',],      ['all'],    ['GENJPt30'],      'TkJets_all_GENPt30',       {'y_min': 0}),
        (['PFJets',],      ['all'],    ['GENJPt30'],      'PFJets_all_GENPt30',       {'y_min': 0}),
        (['PuppiJets',],   ['all'],    ['GENJPt30'],      'PuppiJets_all_GENPt30',    {'y_min': 0}),
        (['scPuppiJets',], ['all'],    ['GENJPt30'],      'scPuppiJets_all_GENPt30',  {'y_min': 0}),
        (['scPuppiCorrJets',], ['all'],    ['GENJPt30'],      'scPuppiCorrJets_all_GENPt30',  {'y_min': 0}),
    ]
    draw_resp_ptVeta(hplot, smps, wc, draw_style=draw_config, configs=jet_ptrespvseta_configs)

    jet_ptrespvseta_configs = [    
        (['CaloJets',],    ['all'],    ['GENJPt30'],      'CaloJets_all_GENPt30',    {'y_min': 0}),
        (['TkJets',],      ['all'],    ['GENJPt30'],      'TkJets_all_GENPt30',      {'y_min': 0}),
        (['PFJets',],      ['all'],    ['GENJPt30'],      'PFJets_all_GENPt30',      {'y_min': 0}),
        (['PuppiJets',],   ['all'],    ['GENJPt30'],      'PuppiJets_all_GENPt30',   {'y_min': 0}),
        (['scPuppiJets',], ['all'],    ['GENJPt30'],      'scPuppiJets_all_GENPt30', {'y_min': 0}),
        (['scPuppiCorrJets',], ['all'],    ['GENJPt30'],      'scPuppiCorrJets_all_GENPt30', {'y_min': 0}),
    ]

    draw_resp_ptVeta_median(hplot, smps, wc, draw_style=draw_config, configs=jet_ptrespvseta_configs)

    jet_ptrespvseta_configs = [    
        (['CaloJets',],        ['all'],    ['GENJPt30'],      'CaloJets_all_GENPt30',         {'y_min': 0.15}),
        (['TkJets',],          ['all'],    ['GENJPt30'],      'TkJets_all_GENPt30',           {'y_min': 0.15}),
        (['PFJets',],          ['all'],    ['GENJPt30'],      'PFJets_all_GENPt30',           {'y_min': 0.15}),
        (['PuppiJets',],       ['all'],    ['GENJPt30'],      'PuppiJets_all_GENPt30',        {'y_min': 0.15}),
        (['scPuppiJets',],     ['all'],    ['GENJPt30'],      'scPuppiJets_all_GENPt30',      {'y_min': 0.15}),
        (['scPuppiCorrJets',], ['all'],    ['GENJPt30'],      'scPuppiCorrJets_all_GENPt30',  {'y_min': 0.15}),
    ]
    draw_resp_ptVeta_sigma(hplot, smps, wc, draw_style=draw_config, configs=jet_ptrespvseta_configs)



def draw_resp_pt(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            JetResoHistos, 
            [s.type for s in smps], 
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


def draw_resp_ptVpt_median(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm1 = DrawMachine(draw_style)
        dm1.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            JetResoHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue



        width = 1
        # FIXME: # of bins is hardcoded for now
        bin_limits_low=[(i, i+width-1) for i in range(1, 50, width)]
        width = 3
        bin_limits_high=[(i, i+width-1) for i in range(50, 100, width)]
        bin_limits = bin_limits_low + bin_limits_high
        for his in hsets:
            his.h_ptRespVpt_graph('sigma', '#sigma_{eff} [p_{T}^{L1}/p_{T}^{GEN}]', lambda histo: draw_utils.computeResolution_effSigma(histo, bin_limits=bin_limits, draw_bins=False))
            his.h_ptRespVpt_graph('median', 'median [p_{T}^{L1}/p_{T}^{GEN}]', lambda histo: draw_utils.computeResolution_mean(histo, bin_limits=bin_limits, draw_bins=False))

        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm1.addHistos([his.g_ptRespVpt_median for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm1.draw(
            text=text, 
            x_min=opts.get('x_min'), 
            x_max=opts.get('x_max'), 
            y_min=opts.get('y_min', 0.75), 
            y_max=opts.get('y_max', 1.5), 
            h_lines=opts.get('h_lines', [1]),
            # norm=opts.get('norm', False),
            options=opts.get('options', ''),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', None),
            y_max_ratio=opts.get('y_max_ratio', None),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            # y_axis_label=opts.get('y_axis_label', 'a.u.')
            x_axis_label=opts.get('x_axis_label', 'p_{T}^{GEN} [GeV]')

        )

        dm1.toWeb(name=f'hMedianPtRespVpt_{h_name}', page_creator=wc_eff)


def draw_resp_ptVpt_sigma(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm2 = DrawMachine(draw_style)
        dm2.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            JetResoHistos, 
            [s.type for s in smps], 
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
        dm2.addHistos([his.g_ptRespVpt_sigma for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm2.draw(
            text=text, 
            x_min=opts.get('x_min'), 
            x_max=opts.get('x_max'), 
            y_min=opts.get('y_min', 0), 
            y_max=opts.get('y_max', 0.4), 
            h_lines=opts.get('h_lines', [0]),
            # norm=opts.get('norm', False),
            options=opts.get('options', ''),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', ''),
            x_axis_label=opts.get('x_axis_label', 'p_{T}^{GEN} [GeV]')

        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')
        dm2.toWeb(name=f'hSigmaPtRespVpt_{h_name}', page_creator=wc_eff)



def draw_resp_ptVpt(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm3 = DrawMachine(draw_style)
        dm3.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            JetResoHistos, 
            [s.type for s in smps], 
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
        dm3.addHistos([his.h_ptRespVpt for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm3.draw(
            text=text, 
            x_min=opts.get('x_min'), 
            x_max=opts.get('x_max'), 
            y_min=opts.get('y_min'), 
            y_max=opts.get('y_max'), 
            h_lines=opts.get('h_lines', [1]),
            # norm=opts.get('norm', False),
            options=opts.get('options', 'COLZ'),
            do_profile=True,
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', None),
            y_max_ratio=opts.get('y_max_ratio', None),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            # y_axis_label=opts.get('y_axis_label', 'a.u.')
            x_axis_label=opts.get('x_axis_label', 'p_{T}^{GEN} [GeV]')

        )

        dm3.toWeb(name=f'hPtRespVpt_{h_name}', page_creator=wc_eff)


def draw_resp_ptVeta_median(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm1 = DrawMachine(draw_style)
        dm1.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            JetResoHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue



        width = 1
        # FIXME: # of bins is hardcoded for now
        bin_limits=[(i, i+width-1) for i in range(0, 40, width)]
        for his in hsets:
            his.h_ptRespVeta_graph('sigma', '#sigma_{eff} [p_{T}^{L1}/p_{T}^{GEN}]', lambda histo: draw_utils.computeResolution_effSigma(histo, bin_limits=bin_limits, draw_bins=False))
            his.h_ptRespVeta_graph('median', 'median [p_{T}^{L1}/p_{T}^{GEN}]', lambda histo: draw_utils.computeResolution_mean(histo, bin_limits=bin_limits, draw_bins=False))

        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm1.addHistos([his.g_ptRespVeta_median for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm1.draw(
            text=text, 
            x_min=opts.get('x_min',0), 
            x_max=opts.get('x_max', 3.5), 
            y_min=opts.get('y_min', 0.75), 
            y_max=opts.get('y_max', 1.5), 
            v_lines=opts.get('v_lines', [1.5, 2.4, 3]),
            h_lines=opts.get('h_lines', [1]),
            # norm=opts.get('norm', False),
            options=opts.get('options', ''),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', None),
            y_max_ratio=opts.get('y_max_ratio', None),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            # y_axis_label=opts.get('y_axis_label', 'a.u.')
            x_axis_label=opts.get('x_axis_label', '#eta^{GEN}')

        )

        dm1.toWeb(name=f'hMedianPtRespVeta_{h_name}', page_creator=wc_eff)

def draw_resp_ptVeta_sigma(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm2 = DrawMachine(draw_style)
        dm2.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            JetResoHistos, 
            [s.type for s in smps], 
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
        dm2.addHistos([his.g_ptRespVeta_sigma for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm2.draw(
            text=text, 
            x_min=opts.get('x_min', 0), 
            x_max=opts.get('x_max', 3.5), 
            y_min=opts.get('y_min', 0), 
            y_max=opts.get('y_max', 0.4), 
            v_lines=opts.get('v_lines', [1.5, 2.4, 3]),
            h_lines=opts.get('h_lines', [0]),
            # norm=opts.get('norm', False),
            options=opts.get('options', ''),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', ''),
            x_axis_label=opts.get('x_axis_label', '#eta^{GEN}')

        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm2.toWeb(name=f'hSigmaPtRespVeta_{h_name}', page_creator=wc_eff)


def draw_resp_ptVeta(hplot, smps, wc_eff, draw_style, configs):

    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm3 = DrawMachine(draw_style)
        dm3.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            JetResoHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue

        hsets, labels, text = hplot.get_histo(
            JetResoHistos, 
            [s.type for s in smps], 
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
        dm3.addHistos([his.h_ptRespVeta for his in hsets], labels=labels)

        # for i in range(1,len(hsets)):
            # print(f'add ratio: {i} to 0')
            # dm.addRatioHisto(i,0)
            # dm.addRatioHisto(2,0)
            # dm.addRatioHisto(3,0)
            # dm.addRatioHisto(4,0)


        dm3.draw(
            text=text, 
            x_min=opts.get('x_min', 0), 
            x_max=opts.get('x_max', 3.5), 
            y_min=opts.get('y_min'), 
            y_max=opts.get('y_max'), 
            v_lines=opts.get('v_lines', [ 1.5, 2.4, 3]),
            h_lines=opts.get('h_lines', [1]),
            # norm=opts.get('norm', False),
            options=opts.get('options', 'COLZ'),
            do_profile=True,
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', None),
            y_max_ratio=opts.get('y_max_ratio', None),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            # y_axis_label=opts.get('y_axis_label', 'a.u.')
            x_axis_label=opts.get('x_axis_label')

        )

        dm3.toWeb(name=f'hPtRespVeta_{h_name}', page_creator=wc_eff)





def draw_reso_eta(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            JetResoHistos, 
            [s.type for s in smps], 
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
            y_axis_label=opts.get('y_axis_label', 'a.u.'),
            x_axis_label=opts.get('x_axis_label', '#eta^{L1}-#eta^{GEN}')
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
            JetResoHistos, 
            [s.type for s in smps], 
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
            y_axis_label=opts.get('y_axis_label', 'a.u.'),
            x_axis_label=opts.get('x_axis_label', '#phi^{L1}-#phi^{GEN}')

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
            JetResoHistos, 
            [s.type for s in smps], 
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
            JetResoHistos, 
            [s.type for s in smps], 
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
