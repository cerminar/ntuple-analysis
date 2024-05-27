import python.histos as histos
from python.draw.drawingTools import *

histo_class = histos.RateHistos

draw_config=tdr_config
draw_config.marker_size = 1
draw_config.legend_size=(0.5, 0.3)
draw_config.legend_position=(0.12, 0.15)
draw_config.marker_styles.append(10)
draw_config.additional_text = [(0.13, 0.91, '#scale[1.5]{CMS} #scale[1.]{Phase-2 Simulation}'),
                    (0.69, 0.91, '14TeV, 200 PU')]


wc_label = 'rate'

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