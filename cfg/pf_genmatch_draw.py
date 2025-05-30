import python.histos as histos
from python.draw.drawingTools import *
import python.draw.utilities as draw_utils
from cfg.pf_genmatch import PfResoHistos
from cfg.eg_genmatch_draw import draw_effvseta,draw_effvspt


def what(what):
    match what:
        case 'pf_reso':
            return [PfResoHistos], 'pf_reso', pf_reso_draw
        case 'pf_eff':
            return [histos.HistoSetEff], 'pf_eff', pf_eff_draw
        case _:
            raise ValueError(f'Unknown draw function: {what}. Available: pf_reso, pf_eff')




draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #it{#scale[1.]{Phase-2 Simulation}}'),
                    (0.69, 0.91, '#it{14TeV}, 200 PU')]


def pf_eff_draw(hplot, smps, wc):
    pf_effvseta_configs = [    
        (['PfCands',],     ['all'],              ['GENPi'],           'PfCands_all_GENPi', {'y_min': 0.0}),
        (['PfCands',],     ['PFTypeC'],              ['GENPi'],       'PfCands_Pi_GENPi',  {'y_min': 0.0}),
        (['PfCands',],     ['PFTypeN'],              ['GENPi'],       'PfCands_NH_GENPi',  {'y_min': 0.0}),
        (['PfCands',],     ['PFTypeE'],              ['GENPi'],       'PfCands_Ele_GENPi', {'y_min': 0.0}),

        (['DecHadCaloEndcap',],  ['IDHgcPFpi', 'IDHgcAgmPi'],       ['GENPi'],   'DecHadCaloEndcap_IDPi_GENPi', {'y_min': 0.0}),
        (['DecHadCaloEndcap',],  ['IDHgcPFem', 'IDHgcAgmEm'],       ['GENPi'],   'DecHadCaloEndcap_IDEm_GENPi', {'y_min': 0.0}),
        (['DecHadCaloEndcap',],  ['IDHgcAgmPu'],                    ['GENPi'],   'DecHadCaloEndcap_IDPu_GENPi', {'y_min': 0.0}),

    ]
    draw_effvseta(hplot, smps, wc, draw_style=draw_config, configs=pf_effvseta_configs)

    pf_effvspt_configs = [
        (['PfCands',],  ['PFTypeC'],       ['GENPiEtaEB'],  'PfCands_Pi_GENPiEtaEB', {}),
        (['PfCands',],  ['PFTypeC'],       ['GENPiEtaEE'],  'PfCands_Pi_GENPiEtaEE', {}),
        (['PfCands',],  ['PFTypeC'],       ['GENPiEtaFwd'],  'PfCands_Pi_GENPiEtaFwd', {}),
        (['PfCands',],  ['PFTypeN'],       ['GENPiEtaFwd'],  'PfCands_NH_GENPiEtaFwd', {}),

        (['DecHadCaloEndcap',],  ['IDHgcPFpi', 'IDHgcAgmPi'],       ['GENPiEtaEE'],  'DecHadCaloEndcap_IDPi_GENPiEtaEE', {}),
        (['DecHadCaloEndcap',],  ['IDHgcPFpi', 'IDHgcAgmPi'],       ['GENPiEtaFwd'],  'DecHadCaloEndcap_IDPi_GENPiEtaFwd', {}),
        (['DecHadCaloEndcap',],  ['IDHgcPFem', 'IDHgcAgmEm'],       ['GENPiEtaEE'],   'DecHadCaloEndcap_IDEm_GENPiEtaEE', {}),
        (['DecHadCaloEndcap',],  ['IDHgcPFem', 'IDHgcAgmEm'],       ['GENPiEtaFwd'],  'DecHadCaloEndcap_IDEm_GENPiEtaFwd', {}),
        (['DecHadCaloEndcap',],  ['IDHgcAgmPu'],       ['GENPiEtaEE'],   'DecHadCaloEndcap_IDPu_GENPiEtaEE', {}),
        (['DecHadCaloEndcap',],  ['IDHgcAgmPu'],       ['GENPiEtaFwd'],  'DecHadCaloEndcap_IDPu_GENPiEtaFwd', {}),


    ]
    draw_effvspt(hplot, smps, wc, draw_style=draw_config, configs=pf_effvspt_configs)



def pf_reso_draw(hplot, smps, wc):
    etaphi_reso_configs = [    
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaEB'],     'PfCands_Pi_GENEtaEB', {}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaEE'],    'PfCands_Pi_GENEtaEE', {}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaFwd'],    'PfCands_Pi_GENEtaFwd', {}),

    ]
    draw_reso_eta(hplot, smps, wc, draw_style=draw_config, configs=etaphi_reso_configs)
    draw_reso_phi(hplot, smps, wc, draw_style=draw_config, configs=etaphi_reso_configs)

    # ctl2_caloetaphi_reso_configs = [    

    #     (['CaloJets',],            ['all'],              ['GEN'],            'CaloJets_all_GEN', {}),
    #     (['CaloJets',],            ['all'],              ['GENJEtaEB'],       'CaloJets_all_GENEtaEB', {}),
    #     (['CaloJets',],            ['all'],              ['GENJEtaEE'],       'CaloJets_all_GENEtaEE', {}),

    # ]
    # draw_reso_caloeta(hplot, smps, wc, draw_style=draw_config, configs=ctl2_caloetaphi_reso_configs)
    # draw_reso_calophi(hplot, smps, wc, draw_style=draw_config, configs=ctl2_caloetaphi_reso_configs)

    ptresp_configs = [    
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaEB'],       'PfCands_Pi_GENPiEtaEB', {}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaEE'],       'PfCands_Pi_GENPiEtaEE', {}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaFwd'],      'PfCands_Pi_GENPiEtaFwd', {}),
    ]

    draw_resp_pt(hplot, smps, wc, draw_style=draw_config, configs=ptresp_configs)


    ptrespvspt_configs = [    
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaEB'],       'PfCands_Pi_GENPiEtaEB',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaEE'],       'PfCands_Pi_GENPiEtaEE',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaFwd'],      'PfCands_Pi_GENPiEtaFwd',  {'y_min': 0}),
    ]
    draw_resp_ptVpt_median(hplot, smps, wc, draw_style=draw_config, configs=ptrespvspt_configs)


    ptrespvspt_configs = [   
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaEB'],       'PfCands_Pi_GENPiEtaEB',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaEE'],       'PfCands_Pi_GENPiEtaEE',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaFwd'],      'PfCands_Pi_GENPiEtaFwd',  {'y_min': 0}),
    ]
    draw_resp_ptVpt_sigma(hplot, smps, wc, draw_style=draw_config, configs=ptrespvspt_configs)

    ptrespvspt_configs = [
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaEB'],       'PfCands_Pi_GENPiEtaEB',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaEE'],       'PfCands_Pi_GENPiEtaEE',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiEtaFwd'],      'PfCands_Pi_GENPiEtaFwd',  {'y_min': 0}),
    ]
    draw_resp_ptVpt(hplot, smps, wc, draw_style=draw_config, configs=ptrespvspt_configs)


    ptrespvseta_configs = [
        (['PfCands',],            ['PFTypeC'],              ['GENPiPt30'],       'PfCands_Pi_GENPiPt30',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiPt30'],       'PfCands_Pi_GENPiPt30',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiPt30'],       'PfCands_Pi_GENPiPt30',  {'y_min': 0}),
    ]
    draw_resp_ptVeta(hplot, smps, wc, draw_style=draw_config, configs=ptrespvseta_configs)

    ptrespvseta_configs = [    
        (['PfCands',],            ['PFTypeC'],              ['GENPiPt30'],       'PfCands_Pi_GENPiPt30',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiPt30'],       'PfCands_Pi_GENPiPt30',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiPt30'],       'PfCands_Pi_GENPiPt30',  {'y_min': 0}),
    ]

    draw_resp_ptVeta_median(hplot, smps, wc, draw_style=draw_config, configs=ptrespvseta_configs)

    ptrespvseta_configs = [    
        (['PfCands',],            ['PFTypeC'],              ['GENPiPt30'],       'PfCands_Pi_GENPiPt30',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiPt30'],       'PfCands_Pi_GENPiPt30',  {'y_min': 0}),
        (['PfCands',],            ['PFTypeC'],              ['GENPiPt30'],       'PfCands_Pi_GENPiPt30',  {'y_min': 0}),
    ]
    draw_resp_ptVeta_sigma(hplot, smps, wc, draw_style=draw_config, configs=ptrespvseta_configs)



def draw_resp_pt(hplot, smps, wc_eff, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            PfResoHistos, 
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
            PfResoHistos, 
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
            PfResoHistos, 
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
            PfResoHistos, 
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
            PfResoHistos, 
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
            PfResoHistos, 
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
            PfResoHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue

        hsets, labels, text = hplot.get_histo(
            PfResoHistos, 
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
            PfResoHistos, 
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
            PfResoHistos, 
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
            PfResoHistos, 
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
            PfResoHistos, 
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
