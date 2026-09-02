import python.histos as histos
from python.draw.drawingTools import *
import python.draw.utilities as draw_utils
from cfg.decoded_in import DecCaloHistos, DecResoHistos
# from cfg.eg_genmatch_draw import draw_effvseta,draw_effvspt
import copy

def what(what):
    match what:
        case 'decoded_calo':
            return [DecCaloHistos], 'decoded_calo', calo_draw
        case 'decoded_calo_reso':
            return [DecResoHistos], 'decoded_calo_reso', calo_draw_reso
        case 'decoded_calo_reso_pi':
            return [DecResoHistos], 'decoded_calo_reso', calo_draw_reso_pi
        case 'decoded_calo_eff_pi':
            return [histos.HistoSetEff], 'decoded_calo_eff', calo_draw_eff_pi
        case 'decoded_calo_eff_em':
            return [histos.HistoSetEff], 'decoded_calo_eff', calo_draw_eff_em
        case 'decoded_calo_reso_em':
            return [DecResoHistos], 'decoded_calo_reso', calo_draw_reso
        case _:
            raise ValueError(f'Unknown draw function: {what}. Available: decoded_calo, decoded_calo_reso, decoded_calo_reso_pi, decoded_calo_reso_em')




draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
# draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #it{#scale[1.]{Phase-2 Simulation}}'),
#                     (0.69, 0.91, '#it{14TeV}, 200 PU')]


def calo_draw_eff_em(hplot, smps, wc):
    configs = [
        (['PU200'], ['DecEmCaloBarrel',],  ['all'], ['GENEtaEB'], 'EmBarrel_all_GENEtaEB', {'y_min': 0.0}),

    ]
    draw_eff_eta(hplot, smps, wc, draw_config, configs)
    draw_eff_phi(hplot, smps, wc, draw_config, configs)

def calo_draw_eff_pi(hplot, smps, wc):
    configs = [
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEB'], 'HadBarrel_all_GENPi', {'y_min': 0.0}),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaPos'], 'HadBarrel_all_GENPiEtaPos', {'y_min': 0.0}),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaNeg'], 'HadBarrel_all_GENPiEtaNeg', {'y_min': 0.0}),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBPt30'], 'HadBarrel_all_GENPiPt30', {'y_min': 0.0}),
        # (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPi', 'GENPiEtaPos', 'GENPiEtaNeg'],     'HadBarrel_all_GENPi_PU0', {'y_min': 0.0}),
        # (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPiPt30'], 'HadBarrel_all_GENPiPt30_PU0', {'y_min': 0.0}),

    ]
    draw_eff_eta(hplot, smps, wc, draw_config, configs)
    draw_eff_phi(hplot, smps, wc, draw_config, configs)


def calo_draw_reso_pi(hplot, smps, wc):
    configs = [
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPi', 'GENPiEtaPos', 'GENPiEtaNeg'], 'HadBarrel_all_GENPi', {'y_min': 0.0}),
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiPt30'], 'HadBarrel_all_GENPiPt30', {'y_min': 0.0}),
        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPi', 'GENPiEtaPos', 'GENPiEtaNeg'],     'HadBarrel_all_GENPi_PU0', {'y_min': 0.0}),
        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPiPt30'], 'HadBarrel_all_GENPiPt30_PU0', {'y_min': 0.0}),

    ]
    draw_reso_eta(hplot, smps, wc, draw_config, configs)
    draw_reso_phi(hplot, smps, wc, draw_config, configs)


    configs = [
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiAll'], 'HadBarrel_all_GENPiAll', {'x_min': -1.5, 'x_max': 1.5}),
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPi'], 'HadBarrel_all_GENPi', {'x_min': -1.5, 'x_max': 1.5}),
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiPt30'], 'HadBarrel_all_GENPiPt30', {'x_min': -1.5, 'x_max': 1.5}),

        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPiAll'], 'HadBarrel_all_GENPiAll_PU0', {'x_min': -1.5, 'x_max': 1.5}),
        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPi'], 'HadBarrel_all_GENPi_PU0', {'x_min': -1.5, 'x_max': 1.5}),
        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPiPt30'], 'HadBarrel_all_GENPiPt30_PU0', {'x_min': -1.5, 'x_max': 1.5}),
    ]

    draw_reso_etaVeta(hplot, smps, wc, draw_config, configs)


    configs = [
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiAll'],  'HadBarrel_all_GENPiAll',     {} ),
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPi'],     'HadBarrel_all_GENPi',     {} ),        
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiPt30'], 'HadBarrel_all_GENPiPt30', {}),

        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPiAll'],  'HadBarrel_all_GENPiAll_PU0',     {} ),
        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPi'],     'HadBarrel_all_GENPi_PU0',     {} ),
        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPiPt30'], 'HadBarrel_all_GENPiPt30_PU0', {}),


    ]
    print("Drawing pt response...")
    draw_resp_pt(hplot, smps, wc, draw_config, configs)
    draw_resp_ptVeta(hplot, smps, wc, draw_config, configs)

    configs = [
        # (['DecHadCaloBarrel',],  ['all'], ['GENPiAll'], 'HadBarrel_all_GENPiAll',   {} ),
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPi'], 'HadBarrel_all_GENPi',   {} ),
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBa'], 'HadBarrel_all_GENPiEtaEBa',   {} ),
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBb'], 'HadBarrel_all_GENPiEtaEBb',   {} ),
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBc'], 'HadBarrel_all_GENPiEtaEBc',   {} ),

        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPi'], 'HadBarrel_all_GENPi_PU0',   {} ),
        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBa'], 'HadBarrel_all_GENPiEtaEBa_PU0',   {} ),
        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBb'], 'HadBarrel_all_GENPiEtaEBb_PU0',   {} ),
        (['PU0'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBc'], 'HadBarrel_all_GENPiEtaEBc_PU0',   {} ),


    ]
    draw_resp_ptVpt(hplot, smps, wc, draw_config, configs)

    # draw_dr_reso(hplot, smps, wc, draw_config, configs)





def calo_draw_reso(hplot, smps, wc):
    configs = [
        (['PU200'], ['DecEmCaloBarrel',],  ['all'],  ['GEN', ],           'EmBarrel_all_GEN', {'y_min': 0.0}),
        (['PU200'], ['DecEmCaloBarrel',],  ['all'],  ['GENPt30'],           'EmBarrel_all_GENPt30', {'y_min': 0.0}),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPi'], 'HadBarrel_all_GENPi', {'y_min': 0.0}),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiPt30'], 'HadBarrel_all_GENPiPt30', {'y_min': 0.0}),
    ]
    draw_reso_eta(hplot, smps, wc, draw_config, configs)
    draw_reso_phi(hplot, smps, wc, draw_config, configs)


    configs = [
        (['PU200'], ['DecEmCaloBarrel',],  ['all'],  ['GEN'],           'EmBarrel_all_GEN', {'x_min': -1.5, 'x_max': 1.5}),
        (['PU200'], ['DecEmCaloBarrel',],  ['all'],  ['GENPt30'],           'EmBarrel_all_GENPt30', {'x_min': -1.5, 'x_max': 1.5}),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiAll'], 'HadBarrel_all_GENPiAll', {'x_min': -1.5, 'x_max': 1.5}),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPi'], 'HadBarrel_all_GENPi', {'x_min': -1.5, 'x_max': 1.5}),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiPt30'], 'HadBarrel_all_GENPiPt30', {'x_min': -1.5, 'x_max': 1.5}),
    ]

    draw_reso_etaVeta(hplot, smps, wc, draw_config, configs)


    configs = [
        (['PU200'], ['DecEmCaloBarrel',],  ['all'],  ['GEN'],           'EmBarrel_all_GEN',        {} ),
        (['PU200'], ['DecEmCaloBarrel',],  ['all'],  ['GENPt30'],           'EmBarrel_all_GENPt30',    {} ),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiAll'],  'HadBarrel_all_GENPiAll',     {} ),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPi'],     'HadBarrel_all_GENPi',     {} ),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBa'], 'HadBarrel_all_GENPiEtaEBa',   {} ),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBb'], 'HadBarrel_all_GENPiEtaEBb',   {} ),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBc'], 'HadBarrel_all_GENPiEtaEBc',   {} ),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiPt30'], 'HadBarrel_all_GENPiPt30', {}),


    ]
    print("Drawing pt response...")
    draw_resp_pt(hplot, smps, wc, draw_config, configs)
    draw_resp_ptVeta(hplot, smps, wc, draw_config, configs)

    configs = [
        (['PU200'], ['DecEmCaloBarrel',],  ['all'],  ['GEN'],           'EmBarrel_all_GEN', {} ),
        # (['DecHadCaloBarrel',],  ['all'], ['GENPiAll'], 'HadBarrel_all_GENPiAll',   {} ),
        (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPi'], 'HadBarrel_all_GENPi',   {} ),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBa'], 'HadBarrel_all_GENPiEtaEBa',   {} ),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBb'], 'HadBarrel_all_GENPiEtaEBb',   {} ),
        # (['PU200'], ['DecHadCaloBarrel',],  ['all'], ['GENPiEtaEBc'], 'HadBarrel_all_GENPiEtaEBc',   {} ),

    ]
    draw_resp_ptVpt(hplot, smps, wc, draw_config, configs)

    # draw_dr_reso(hplot, smps, wc, draw_config, configs)


def draw_resp_pt(hplot, smps, wc_eff, draw_style, configs):
    for pu, objs, objs_sel, gen_sel, h_name, opts in configs:
        # draw_style = style_text(pu, draw_style)

        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            DecResoHistos, 
            [s.type for s in smps], 
            pu, 
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
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', 'a.u.')
        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hPtResp_{h_name}', page_creator=wc_eff)

def style_text(pu, draw_config=draw_config):
    newconfig = copy.deepcopy(draw_config)
    # print(pu)
    x_pos, y_pos, _ = newconfig.additional_text[1]
    if 'PU0' in pu and not 'PU200' in pu:
        label = '#it{14TeV}, {0 PU}'
    elif 'PU200' in pu and not 'PU0' in pu:
        label = '#it{14TeV}, {200 PU}'
    else:        
        label = '#it{14TeV}'
    newconfig.additional_text[1] = (x_pos, y_pos, label)
    return newconfig


def draw_resp_ptVpt(hplot, smps, wc_eff, draw_style, configs):
    for pu, objs, objs_sel, gen_sel, h_name, opts in configs:
        # draw_style = style_text(pu, draw_style)
        print(draw_style.additional_text)
        if len(smps) == 0:
            continue

        dm1 = DrawMachine(draw_style)
        dm1.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            DecResoHistos, 
            [s.type for s in smps], 
            pu, 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue



        width = 3
        # FIXME: # of bins is hardcoded for now
        bin_limits=[(i, i+width-1) for i in range(1, 50, width)]
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
            y_min=opts.get('y_min', 0.5), 
            y_max=opts.get('y_max', 2.5), 
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

        dm2 = DrawMachine(draw_style)
        dm2.config.legend_position = (0.6,0.6)

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

        y_max = opts.get('y_max', 0.4)

        if 'Had' in objs[0]:
            y_max = opts.get('y_max', 1.5)

        dm2.draw(
            text=text, 
            x_min=opts.get('x_min'), 
            x_max=opts.get('x_max'), 
            y_min=opts.get('y_min', 0), 
            y_max=y_max, 
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

        dm3 = DrawMachine(draw_style)
        dm3.config.legend_position = (0.6,0.6)

        # hsets, labels, text = hplot.get_histo(
        #     DecResoHistos, 
        #     [s.type for s in smps], 
        #     ['PU200'], 
        #     objs, 
        #     objs_sel, 
        #     gen_sel, debug=False)
        # if not hsets:
        #     print(' -> skip drawing')
        #     continue


        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        for his in hsets:
            dir(his)
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
            y_min=opts.get('y_min', 0), 
            y_max=opts.get('y_max', 3), 
            h_lines=opts.get('h_lines', [1]),
            norm=opts.get('norm', False),
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


def draw_resp_ptVeta(hplot, smps, wc_eff, draw_style, configs):

    for pu, objs, objs_sel, gen_sel, h_name, opts in configs:
        # draw_style = style_text(pu, draw_style)

        if len(smps) == 0:
            continue

        dm1 = DrawMachine(draw_style)
        dm1.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            DecResoHistos, 
            [s.type for s in smps], 
            pu, 
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

        print(f'ops: {opts} y_min: {opts.get("y_min", 0)} y_max: {opts.get("y_max", 3)}')
        dm1.draw(
            text=text, 
            x_min=opts.get('x_min', 0), 
            x_max=opts.get('x_max', 3.5), 
            y_min=opts.get('y_min', 0.5), 
            y_max=opts.get('y_max', 2.5), 
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

        dm2 = DrawMachine(draw_style)
        dm2.config.legend_position = (0.6,0.6)

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
            y_max=opts.get('y_max', 1), 
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

        dm3 = DrawMachine(draw_style)
        dm3.config.legend_position = (0.6,0.6)

        hsets, labels, text = hplot.get_histo(
            DecResoHistos, 
            [s.type for s in smps], 
            pu, 
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
            x_min=opts.get('x_min', -3.5), 
            x_max=opts.get('x_max', 3.5), 
            y_min=opts.get('y_min', 0.), 
            y_max=opts.get('y_max', 3.), 
            v_lines=opts.get('v_lines', [-3, -2.4, -1.5, 1.5, 2.4, 3]),
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





def draw_reso_phi(hplot, smps, wc_eff, draw_style, configs):
    for pu, objs, objs_sel, gen_sel, h_name, opts in configs:
        # draw_style = style_text(pu, draw_style)

        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            DecResoHistos, 
            [s.type for s in smps], 
            pu, 
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

def draw_eff_eta(hplot, smps, wc_eff, draw_style, configs):
    for pu, objs, objs_sel, gen_sel, h_name, opts in configs:
        # draw_style = style_text(pu, draw_style)

        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            histos.HistoSetEff, 
            [s.type for s in smps], 
            pu, 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_eff.h_eta.CreateGraph() for his in hsets], labels=labels)

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
            h_lines=opts.get('h_lines', [1]),
            norm=opts.get('norm', False),
            options=opts.get('options', ''),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', ''),
            x_axis_label=opts.get('x_axis_label', '')

        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hEtaEff_{h_name}', page_creator=wc_eff)


def draw_eff_phi(hplot, smps, wc_eff, draw_style, configs):
    for pu, objs, objs_sel, gen_sel, h_name, opts in configs:
        # draw_style = style_text(pu, draw_style)

        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            histos.HistoSetEff, 
            [s.type for s in smps], 
            pu, 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_eff.h_phi.CreateGraph() for his in hsets], labels=labels)

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
            h_lines=opts.get('h_lines', [1]),
            norm=opts.get('norm', False),
            options=opts.get('options', ''),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', ''),
            x_axis_label=opts.get('x_axis_label', '')

        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hPhiEff_{h_name}', page_creator=wc_eff)




def draw_reso_eta(hplot, smps, wc_eff, draw_style, configs):
    for pu, objs, objs_sel, gen_sel, h_name, opts in configs:
        # draw_style = style_text(pu, draw_style)

        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            DecResoHistos, 
            [s.type for s in smps], 
            pu, 
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

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            DecResoHistos, 
            [s.type for s in smps], 
            pu, 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_etaResVeta for his in hsets], labels=labels)

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
            options=opts.get('options', 'COLZ'),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', '#eta^{L1}-#eta^{GEN}'),
            x_axis_label=opts.get('x_axis_label', '#eta^{GEN}')

        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hEtaResVEta_{h_name}', page_creator=wc_eff)


def draw_reso_etaVeta(hplot, smps, wc_eff, draw_style, configs):
    for pu, objs, objs_sel, gen_sel, h_name, opts in configs:
        # draw_style = style_text(pu, draw_style)

        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.6,0.7)

        hsets, labels, text = hplot.get_histo(
            DecResoHistos, 
            [s.type for s in smps], 
            pu, 
            objs, 
            objs_sel, 
            gen_sel, debug=False)
        if not hsets:
            print(' -> skip drawing')
            continue
        # print(f"# of hsets: {len(hsets)}")
        # for hset in hsets:
        #     hset.computeEff(rebin=2)
        dm.addHistos([his.h_etaResVeta for his in hsets], labels=labels)

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
            y_min=opts.get('y_min', -0.4), 
            y_max=opts.get('y_max', 0.4), 
            h_lines=opts.get('h_lines', []),
            norm=opts.get('norm', True),
            options=opts.get('options', 'COLZ'),
            do_ratio=opts.get('do_ratio', False),
            y_min_ratio=opts.get('y_min_ratio', 0.9),
            y_max_ratio=opts.get('y_max_ratio', 1.1),
            h_lines_ratio=opts.get('h_lines_ratio', [0.95, 1., 1.05]),
            y_log=opts.get('y_log', False),
            y_axis_label=opts.get('y_axis_label', '#eta^{L1}-#eta^{GEN}'),
            x_axis_label=opts.get('x_axis_label', '#eta^{GEN}')

        )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hEtaResVEta_{h_name}', page_creator=wc_eff)


def calo_draw(hplot, smps, wc):
    configs = [
        # (['DecEmCaloBarrel',],  ['all'],              ['nomatch'],           'EmBarrel_all', {'y_min': 0.0}),
        # (['DecHadCaloBarrel',],  ['all'],              ['nomatch'],           'HadBarrel_all', {'y_min': 0.0}),
        (['DecEmCaloBarrel',],  ['Pt5'],              ['nomatch'],           'EmBarrel_Pt5', {'y_min': 0.0}),
        (['DecHadCaloBarrel',],  ['Pt5'],              ['nomatch'],           'HadBarrel_Pt5', {'y_min': 0.0}),

    ]
    draw_pt(hplot, smps, wc, draw_config, configs)
    draw_empt(hplot, smps, wc, draw_config, configs)
    draw_eta(hplot, smps, wc, draw_config, configs)
    draw_relIso(hplot, smps, wc, draw_config, configs)
    draw_showershape(hplot, smps, wc, draw_config, configs)
    draw_emid(hplot, smps, wc, draw_config, configs)
    draw_hoe(hplot, smps, wc, draw_config, configs)


def draw_pt(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)

        dm.config.legend_position = (0.5,0.5)
        hsets, labels, text = hplot.get_histo(
            DecCaloHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)

        dm.addHistos([his.h_pt for his in hsets], labels=labels)
        labels_float = [f'{label} (float)' for label in labels]
        dm.addHistos([his.h_clPt for his in hsets], labels=labels_float)
        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                x_min=opts.get('x_min', 0), 
                x_max=opts.get('x_max', 100.), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', True),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                # x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', False),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hPt_{h_name}', page_creator=wc)

        dm2 = DrawMachine(draw_style)
        dm2.config.legend_size = (0.3,0.3)

        dm2.config.legend_position = (0.5,0.5)
        dm2.addHistos([his.h_ptResoEmu for his in hsets], labels=labels)

        # dm.addRatioHisto(0,1)

        dm2.draw(text=text, 
                x_min=opts.get('x_min', -10), 
                x_max=opts.get('x_max', 10.), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', True),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                # x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', False),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm2.toWeb(name=f'hPtResoEmu_{h_name}', page_creator=wc)


def draw_empt(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)

        dm.config.legend_position = (0.5,0.5)
        hsets, labels, text = hplot.get_histo(
            DecCaloHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)

        dm.addHistos([his.h_empt for his in hsets], labels=labels)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                x_min=opts.get('x_min', 0), 
                x_max=opts.get('x_max', 100.), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', True),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                # x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', False),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hEmPt_{h_name}', page_creator=wc)


def draw_eta(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)

        dm.config.legend_position = (0.5,0.5)
        hsets, labels, text = hplot.get_histo(
            DecCaloHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)

        dm.addHistos([his.h_eta for his in hsets], labels=labels)
        labels_float = [f'{label} (float)' for label in labels]
        dm.addHistos([his.h_clEta for his in hsets], labels=labels_float)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                # x_min=opts.get('x_min', 0), 
                # x_max=opts.get('x_max', 100.), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', True),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                # x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', False),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hEta_{h_name}', page_creator=wc)

        dm2 = DrawMachine(draw_style)
        dm2.config.legend_size = (0.3,0.3)

        dm2.config.legend_position = (0.5,0.5)
        dm2.addHistos([his.h_etaResoEmu for his in hsets], labels=labels)

        # dm.addRatioHisto(0,1)

        dm2.draw(text=text, 
                x_min=opts.get('x_min', -10), 
                x_max=opts.get('x_max', 10.), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', True),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                # x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', False),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm2.toWeb(name=f'hEtaResoEmu_{h_name}', page_creator=wc)



def draw_relIso(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)
        dm.config.legend_position = (0.5,0.5)
        hsets, labels, text = hplot.get_histo(
            DecCaloHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)
        
        labels_src = [f'{l} (float)' for l in labels]
        dm.addHistos([his.h_clRelIso for his in hsets], labels=labels_src)
        labels_src = []
        
        for i in range(0,len(hsets)):
            if hasattr(hsets[i], 'h_clRelIsoHack'):
                labels_src.append(f'{labels[i]} (hack)')
        # labels_src = [f'{l} (hack)' for l in labels]
        dm.addHistos([his.h_clRelIsoHack for his in hsets if hasattr(his, 'h_clRelIsoHack')], labels=labels_src)
        labels_src = []

        for i in range(0,len(hsets)):
            if hasattr(hsets[i], 'h_clRelIsoOld'):
                labels_src.append(f'{labels[i]} (buggy)')
        
        # labels_src = [f'{l} (buggy)' for l in labels]
        dm.addHistos([his.h_clRelIsoOld for his in hsets if hasattr(his, 'h_clRelIsoOld')], labels=labels_src)

        labels_dec = [f'{l} (dec)' for l in labels]
        dm.addHistos([his.h_relIso for his in hsets], labels=labels_dec)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                # x_min=opts.get('x_min', 0), 
                # x_max=opts.get('x_max', 100.), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', False),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                # x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', False),
                norm=opts.get('norm', False),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hRelIso_{h_name}', page_creator=wc)

def draw_showershape(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)

        dm.config.legend_position = (0.2,0.5)
        hsets, labels, text = hplot.get_histo(
            DecCaloHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)

        labels_src = [f'{l} (src)' for l in labels]
        dm.addHistos([his.h_clShowerShape for his in hsets], labels=labels_src)
        labels_dec = [f'{l} (dec)' for l in labels]
        dm.addHistos([his.h_showerShape for his in hsets], labels=labels_dec)
        # labels_hw = [f'{l} (hw)' for l in labels]
        # dm.addHistos([his.h_hwShowerShape for his in hsets], labels=labels_hw)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                # x_min=opts.get('x_min', 0), 
                # x_max=opts.get('x_max', 100.), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', 0.01), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', False),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                # x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', False),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hShowerShape_{h_name}', page_creator=wc)

        dm2 = DrawMachine(draw_style)
        dm2.config.legend_size = (0.3,0.3)

        dm2.config.legend_position = (0.5,0.5)
        dm2.addHistos([his.h_showerShapeResoEmu for his in hsets], labels=labels)

        # dm.addRatioHisto(0,1)

        dm2.draw(text=text, 
                x_min=opts.get('x_min', -10), 
                x_max=opts.get('x_max', 10.), 
                y_min=opts.get('y_min', None),
                y_max=opts.get('y_max', None), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', True),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                # x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', False),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm2.toWeb(name=f'hShowerShapeResoEmu_{h_name}', page_creator=wc)


def draw_emid(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)
        dm.config.legend_position = (0.7,0.5)

        hsets, labels, text = hplot.get_histo(
            DecCaloHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)
        
        for i in range(0,6):
            for his in hsets:
                his.h_hwEmIDBits.GetXaxis().SetBinLabel(i+1, f'bit {i}')

        # normalize to the total # entries in bin 6 (overflow) 
        # o be able to compare the relative contribution of each bit 
        # independently of the total # entries in the histogram
        for his in hsets:
            # print(f'Entries in overflow bin for {his.h_hwEmIDBits.GetName()}: {his.h_hwEmIDBits.GetBinContent(7)}')
            nentries = his.h_hwEmIDBits.GetBinContent(7) # get total entries from overflow bin
            if nentries > 0:
                # print(f'Normalizing {his.h_hwEmIDBits.GetName()} by {nentries} entries')
                his.h_hwEmIDBits.Scale(1/nentries)

        dm.addHistos([his.h_hwEmIDBits for his in hsets], labels=labels)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                # x_min=opts.get('x_min', 0), 
                # x_max=opts.get('x_max', 100.), 
                y_min=opts.get('y_min', None),
                # y_max=opts.get('y_max', 0.01), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', False),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                # x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', False),
                norm=opts.get('norm', False),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hHwEmIDBits_{h_name}', page_creator=wc)

def draw_hoe(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name, opts in configs:
        if len(smps) == 0:
            print("no samples")
            continue
        dm = DrawMachine(draw_style)
        dm.config.legend_size = (0.3,0.3)

        dm.config.legend_position = (0.2,0.5)
        hsets, labels, text = hplot.get_histo(
            DecCaloHistos, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, gen_sel, debug=False)

        dm.addHistos([his.h_hoe for his in hsets], labels=labels)

        for i in range(1,len(hsets)):
            dm.addRatioHisto(i,0)

        # dm.addRatioHisto(0,1)

        dm.draw(text=text, 
                # x_min=opts.get('x_min', 0), 
                # x_max=opts.get('x_max', 100.), 
                y_min=opts.get('y_min', None),
                # y_max=opts.get('y_max', 0.01), 
                v_lines=opts.get('v_lines', []),
                h_lines=opts.get('h_lines', []),
                do_ratio=opts.get('do_ratio', True),                
                y_min_ratio=opts.get('y_min_ratio'),
                y_max_ratio=opts.get('y_max_ratio'),
                h_lines_ratio=opts.get('h_lines_ratio'),
                y_axis_label=opts.get('y_axis_label', 'a.u'),
                # x_axis_label=opts.get('x_axis_label', 'ID-score'),
                y_log=opts.get('y_log', False),
                norm=opts.get('norm', True),
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=f'hHoe_{h_name}', page_creator=wc)
