import cfg.eg_genmatch
import python.histos as histos
from python.draw.drawingTools import *




def what(what):
    match what:
        case 'eff':
            return [histos.HistoSetEff], 'eff', jet_eff_draw


draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #it{#scale[1.]{Phase-2 Simulation}}'),
                    (0.69, 0.91, '#it{14TeV}, 200 PU')]


def jet_eff_draw(hplot, smps, wc):

    eff_vs_eta_configs = [    
        (['PFJets',], ['all'],      ['GENJ'],  'hEffVsEta_ak4PFJet_all_GEN_ttbar'),
        (['PFJets',], ['all'],      ['GENJPt30'],  'hEffVsEta_ak4PFJet_all_GENPt30_ttbar'),

    ]
    draw_effvseta(hplot, smps, wc, draw_style=draw_config, configs=eff_vs_eta_configs)

    eff_vs_pt_configs = [    
        (['PFJets',], ['all'],      ['GENJEtaEB'],  'hEffVsEta_ak4PFJet_all_GENEtaEB_ttbar'),
        (['PFJets',], ['all'],      ['GENJEtaEE'],  'hEffVsEta_ak4PFJet_all_GENEtaEE_ttbar'),
        (['PFJets',], ['Pt30', 'Pt40'],      ['GENJEtaEE'],  'hEffVsEta_ak4PFJet_Pt_GENEtaEE_ttbar'),
    ]

    draw_effvspt(hplot, smps, wc, draw_style=draw_config, configs=eff_vs_pt_configs)

    # menu_configs = [
    #     (['EGStaEB',], ['MenuSta'],      ['GENEtaEB'],    'ele'),
    #     (['EGStaEE',], ['MenuSta'],      ['GENEtaEE'],    'ele'),
    #     # (smps_ele, ['TkEmL2',], ['MenuSta'],      ['GENEtaEB'],     'ele'),
    #     # (smps_ele, ['TkEmL2',], ['MenuSta'],      ['GENEtaEE'],     'ele'),
    #     (['TkEmL2',], ['MenuPhoIso'],      ['GENEtaEB'],  'ele'),
    #     (['TkEmL2',], ['MenuPhoIso'],      ['GENEtaEE'],  'ele'),

    #     (['TkEleL2',], ['MenuEleLoose'],      ['GENEtaEB'],  ''),
    #     (['TkEleL2',], ['MenuEleLoose'],      ['GENEtaEE'],  ''),
    #     (['TkEleL2',], ['MenuEleTight'],      ['GENEtaEB'],  ''),
    #     (['TkEleL2',], ['MenuEleTight'],      ['GENEtaEE'],  ''),

    #     (['TkEleL2',], ['MenuEleIsoLoose'],      ['GENEtaEB'],  ''),
    #     (['TkEleL2',], ['MenuEleIsoLoose'],      ['GENEtaEE'],  ''),
    #     (['TkEleL2',], ['MenuEleIsoTight'],      ['GENEtaEB'],  ''),
    #     (['TkEleL2',], ['MenuEleIsoTight'],      ['GENEtaEE'],  ''),
    # ]
    # draw_ton(hplot, smps, wc, draw_style=draw_config, configs=menu_configs)




def draw_effvseta(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, gen_sel, h_name in configs:
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
                x_min=0., x_max=3.2, 
                y_min=0.5, y_max=1.1, v_lines=[1.52, 1.7, 2.4],
                do_ratio=True,
                y_min_ratio=0.8,
                y_max_ratio=1.2,
                h_lines=[1., 0.9],
                h_lines_ratio=[0.9, 1., 1.1],
                y_axis_label='efficiency'
            )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=h_name, page_creator=wc)


def draw_ton(hplot, smps, wc_eff, draw_style, configs):
    print('Computing Turn-on curves...')
    for smp in hplot.data['sample'].unique():
        print ('Sample: {}'.format(smp))
        for pu in hplot.data[(hplot.data['sample'] == smp)].pu.unique():
            for tp in ['PFJets']:
                print ('PU: {}, TP: {}'.format(pu, tp))
                for tp_sel in hplot.data[(hplot.data['sample'] == smp) & (hplot.data.pu == pu) & (hplot.data.tp == tp)].tp_sel.unique():
                    if 'Pt' not in tp_sel:
                        continue
                    # print tp_sel
                    tp_sel_den = tp_sel.split('Pt')[0]
                    if tp_sel_den == '':
                        tp_sel_den = 'all'
                    print ('  {} den -> {}'.format(tp_sel, tp_sel_den)) 
                    for gen_sel in hplot.data[(hplot.data['sample'] == smp) & (hplot.data.pu == pu) & (hplot.data.tp == tp) & (hplot.data.tp_sel == tp_sel)].gen_sel.unique():
                        if gen_sel == 'nomatch' or 'Pt' in gen_sel:
                            continue
                        print(gen_sel)
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
    for objs, objs_sel, gen_sel, h_name in configs:
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
            x_min=0, x_max=100, 
            y_min=0.0, y_max=1.1, 
            h_lines=[1.0, 0.9],
            do_ratio=True,
            y_min_ratio=0.9,
            y_max_ratio=1.1,
            h_lines_ratio=[0.95, 1., 1.05],
            y_axis_label='efficiency'
    )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')

        dm.toWeb(name=h_name, page_creator=wc_eff)

