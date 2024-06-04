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
        (['TkEleL2'], ['SingleTkEle36', 'SingleTkEle36EtaEB', 'SingleTkEle36EtaEE']),
        (['TkEleL2'], ['SingleIsoTkEle28', 'SingleIsoTkEle28EtaEB', 'SingleIsoTkEle28EtaEE']),
        (['TkEleL2'], ['SingleIsoTkEle28Tight', 'SingleIsoTkEle28TightEtaEB', 'SingleIsoTkEle28TightEtaEE']),
        (['TkEmL2'], ['SingleIsoTkPho36', 'SingleIsoTkPho36EtaEB', 'SingleIsoTkPho36EtaEE']),
    ]
    menu_double = [
        (['DoubleTkEleL2'], ['DoubleTkEle25-12']),
        (['DoubleTkEmL2'], ['DoubleIsoTkPho22-12']),
    ]


    for smp in smps:
        for obj, sels in menu_single:
            singleobjcounter(hplot, smp_table, smp, obj, sels)

    for smp in smps:
        for obj, sels in menu_double:
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
        (['TkEleL2'], ['MenuEleLoose'], 'hRate_TkEleL2_MenuEleLoose'),
        (['TkEleL2'], ['MenuEleTight'], 'hRate_TkEleL2_MenuEleTight'),
        (['TkEleL2'], ['MenuEleLooseEtaEE'], 'hRate_TkEleL2_MenuEleLooseEtaEE'),
        (['TkEleL2'], ['MenuEleTightEtaEE'], 'hRate_TkEleL2_MenuEleTightEtaEE'),
        (['TkEleL2'], ['MenuEleLooseEtaEB'], 'hRate_TkEleL2_MenuEleLooseEtaEB'),
        (['TkEleL2'], ['MenuEleTightEtaEB'], 'hRate_TkEleL2_MenuEleTightEtaEB'),
        (['TkEleL2'], ['MenuEleIsoLoose'], 'hRate_TkEleL2_MenuEleIsoLoose'),
        (['TkEleL2'], ['MenuEleIsoTight'], 'hRate_TkEleL2_MenuEleIsoTight'),
        (['TkEleL2'], ['MenuEleIsoTight', 'MenuEleIsoLoose'], 'hRate_TkEleL2_MenuEleIso'),
        (['TkEmL2'], ['MenuPhoIso'], 'hRate_TkEmL2_MenuPhoIso'),
        (['TkEmL2'], ['MenuPhoIsoEtaEE'], 'hRate_TkEmL2_MenuPhoIsoEtaEE'),
        (['TkEmL2'], ['MenuPhoIsoEtaEB'], 'hRate_TkEmL2_MenuPhoIsoEtaEB'),
        (['EGStaEE'], ['MenuSta'], 'hRate_EGStaEE_MenuSta'),
        (['EGStaEB'], ['MenuSta'], 'hRate_EGStaEB_MenuSta'),
        (['DoubleTkEmL2'], ['MenuDoubleIsoTkPho22-X'], 'hRate_DoubleTkEmL2_DoubleIsoTkPho22-X')

    ]
    draw_rate(hplot, smps, wc, draw_style=draw_config, configs=menu)


def draw_rate(hplot, smps, wc, draw_style, configs):
    for objs, objs_sel, h_name in configs:
        if len(smps) == 0:
            continue

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.4, 0.4)

        hsets, labels, text = hplot.get_histo(histos.RateHistos, smps, 'PU200', objs, objs_sel, None)
        dm.addHistos([his.h_pt for his in hsets], labels=labels)
        # print(hsets[0].h_pt.GetName())
        dm.addRatioHisto(1,0)
    #     dm.addRatioHisto(2,0)

        dm.draw(
            text=text,
            y_min=0.1, y_max=40000,
            x_min=0, x_max=100,
            y_min_ratio=0.8, y_max_ratio=1.2,
            y_log=True, 
            x_axis_label='online p_{T} thresh. [GeV]',
            h_lines=[20,100,1000],
            h_lines_ratio=[0.9, 1, 1.1],
        do_ratio=True)
        dm.toWeb(name=h_name, page_creator=wc)