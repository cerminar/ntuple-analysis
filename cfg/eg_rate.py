from python import plotters, selections, histos
import python.boost_hist as bh
import cfg.datasets.fastpuppi_collections as coll
import awkward as ak
import math



class RateHistoCounter(histos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_norm = bh.TH1F(f'{name}_norm', '# of events', 1, 1, 2)
            self.h_rate = bh.TH1F(f'{name}_rate', '# passing events; rate [kHz];',  1, 1, 2)

        histos.BaseHistos.__init__(self, name, root_file, debug)

        if root_file is not None:
            for attr_1d in [attr for attr in dir(self) if (attr.startswith('h_') and 'TH1' in getattr(self, attr).ClassName())]:
                setattr(self, f'{attr_1d}_graph', histos.GraphBuilder(self, attr_1d))

        if root_file is not None:
            self.normalize(2760.0*11246/1000)
            # self.h_simenergy = bh.TH1F(name+'_energy', 'Digi sim-energy (GeV)', 100, 0, 2)

    def fill(self, count):
        self.h_rate.fill(1,  weight=count)

    def fill_norm(self, many=1):
        # print (f' fill rate norm: {many}')
        self.h_norm.fill(1,  weight=many)

    def normalize(self, norm):
        nev = self.h_norm.GetBinContent(1)
        # print(f' .      # ev: {nev}')
        if(nev != norm):
            # in this case we don't want the error as the squared sum of the errors. 
            # The weithgs are the real counts
            self.h_rate.SetBinError(1, math.sqrt(self.h_rate.GetBinContent(1)))
            # print(f'NORM: # of counts: {self.h_norm.GetBinContent(1)} error: {self.h_norm.GetBinError(1)}')
            # print(f'RATE: # of counts: {self.h_rate.GetBinContent(1)} error: {self.h_rate.GetBinError(1)}')

            print(f'normalize to {norm}')
            
            self.h_norm.Scale(norm/nev)
            self.h_rate.Scale(norm/nev)


class SingleObjRateHistoCounter(RateHistoCounter):
    def __init__(self, name, root_file=None, debug=False):
        RateHistoCounter.__init__(self, name, root_file, debug)

    def fill(self, df):
        # print(self.name_)
        # # print(f' .  # of surviving entries: ')
        # # print(df['pt'].groupby(level='entry', group_keys=False).nlargest(n=1).count())
        # print(df)
        # print(df.pt)
        # print(ak.count(df.pt, axis=1))
        # print(ak.any(df.pt, axis=1))
        # print(df.pt[ak.any(df.pt, axis=1)])
        # print(ak.any(df.pt, axis=1)*1)

        # print(ak.sum(ak.any(df.pt, axis=1)*1))

        RateHistoCounter.fill(self, ak.sum(ak.any(df.pt, axis=1)*1))


class DoubleObjRateHistoCounter(RateHistoCounter):
    def __init__(self, name, root_file=None, debug=False):
        RateHistoCounter.__init__(self, name, root_file, debug)

    def has_unique_pairs(entry):
        unique_pairs = [(l1, l2) for l1 in entry.loc[entry.index.get_level_values('leg') == 0].index for l2 in entry.loc[entry.index.get_level_values('leg') == 1].index if ((l1[2] != l2[2]) and (l1[1] != l2[1]))]
        return len(unique_pairs) > 0

    def fill(self, df):
        # print(self.name_)
        # print(ak.count(df.leg0, axis=1))
        # print(ak.any(df.pt, axis=1))
        # print(df.pt[ak.any(df.pt, axis=1)])
        # print(ak.any(df.pt, axis=1)*1)
        # print(df[ak.any(df.leg0.pt, axis=1)].show())        

        # print(f' .  # of surviving entries: ')
        # print(df.groupby(level='entry').filter(DoubleObjRateHistoCounter.has_unique_pairs).index.unique('entry').size)
        RateHistoCounter.fill(self, ak.sum(ak.any(df.leg0.pt, axis=1)*1))



class RatePlotter(plotters.BasePlotter):
    def __init__(self, tp_set, tp_selections=[selections.Selection('all')]):
        self.h_rate = {}
        super(RatePlotter, self).__init__(tp_set, tp_selections)

    def book_histos(self):
        self.tp_set.activate()
        tp_name = self.tp_set.name
        for selection in self.tp_selections:
            self.h_rate[selection.name] = histos.RateHistos(name=f'{tp_name}_{selection.name}')

    def fill_histos(self, debug=0):
        # print '------------------'
        # print self.tp_set.name
        for selection in self.tp_selections:
            sel_clusters = self.tp_set.df
            if not selection.all:
                # print(selection)
                sel_clusters = self.tp_set.df[selection.selection(self.tp_set.df)]
            # max_pt_index = ak.argmax(sel_clusters.pt, axis=1, keepdims=True)
            # max_pt_per_event = sel_clusters[max_pt_index]
            self.h_rate[selection.name].fill(sel_clusters)
            self.h_rate[selection.name].fill_norm(self.tp_set.new_read_nentries)


class BaseRateCounter(plotters.BasePlotter):
    def __init__(self, HistoClass, tp_set, tp_selections=[selections.Selection('all')]):
        self.HistoClass = HistoClass
        self.h_rate = {}
        super(BaseRateCounter, self).__init__(tp_set, tp_selections)

    def book_histos(self):
        self.tp_set.activate()
        tp_name = self.tp_set.name
        for selection in self.tp_selections:
            self.h_rate[selection.name] = self.HistoClass(
                name=f'{tp_name}_{selection.name}')

    def fill_histos(self, debug=0):
        # print('------------------')
        # print(f'L1 Obj: {self.tp_set.name}')
        for selection in self.tp_selections:
            # print(f' .  sel: {selection}')
            # print(f' .  # of read entries: {self.tp_set.new_read_nentries}')
            # print(self.tp_set.df.show())
            # print(self.tp_set.df.pt)
            # print(self.tp_set.df.rho)

            sel_clusters = self.tp_set.df[selection.selection(self.tp_set.df)]
            # print(sel_clusters)
            self.h_rate[selection.name].fill(sel_clusters)
            self.h_rate[selection.name].fill_norm(self.tp_set.new_read_nentries)


class RateCounter(BaseRateCounter):
    def __init__(self, tp_set, tp_selections=[selections.Selection('all')]):
        self.h_rate = {}
        super(RateCounter, self).__init__(SingleObjRateHistoCounter, tp_set, tp_selections)


class DoubleObjRateCounter(BaseRateCounter):
    def __init__(self, tp_set, tp_selections=[selections.Selection('all')]):
        self.h_rate = {}
        super(DoubleObjRateCounter, self).__init__(DoubleObjRateHistoCounter, tp_set, tp_selections)


egid_eta_selections = (selections.Selector('^IDTightS|all')*selections.Selector('^Eta[F]$|^Eta[AF][ABCD]*[CD]$'))()
egid_etatk_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Eta[F]$|^Eta[AF][ABCD]*[C]$'))()
egid_iso_etatk_selections = (selections.Selector('^IDTight[EP]$|all')*
                             (selections.Selector('^IsoEleEE|^IsoPhoEE|all')*selections.Selector('^EtaEE$|all')
                              + selections.Selector('^IsoEleEB|^IsoPhoEB|all')*selections.Selector('^EtaEB$|all')))()
# for sel in egid_iso_etatk_selections:
#     print(sel)
egid_menu_ele_selections = (selections.Selector('^MenuEle')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_pho_selections = (selections.Selector('^MenuPho|^MenuSta')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_sta_selections = (selections.Selector('^MenuSta')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_dipho_selections = (selections.Selector('^MenuDoubleIsoTkPho22'))()


egid_menu_ele_rate_selections = (selections.Selector('^SingleIsoTkEle|^SingleTkEle')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_pho_rate_selections = (selections.Selector('^SingleIsoTkPho|^SingleEGEle')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_sta_rate_selections = (selections.Selector('^SingleEGEle')*selections.Selector('^EtaE[BE]$|all'))()

egid_menu_diele_rate_selections = (selections.Selector('^DoubleTkEle'))()
egid_menu_dipho_rate_selections = (selections.Selector('^DoubleIsoTkPho'))()
egid_menu_dista_rate_selections = (selections.Selector('^DoubleStaEG'))()
egid_menu_ditkelesta_rate_selections = (selections.Selector('^DoubleIsoTkEleStaEG'))()

egid_eta_ee_selections = (selections.Selector('^IDTightS|all')*selections.Selector('^Eta[A][BCD]*[CD]$'))()
egid_eta_eetk_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Eta[A][BCD]*[C]$'))()
egid_iso_eta_eetk_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Iso|all')*selections.Selector('^Eta[A][BCD]*[C]$'))()

egid_iso_eta_eetk_selections_comp = (selections.Selector('^IDTight[E]|^IDCompWP|all')*selections.Selector('^Eta[AB][BCD]*[C]$'))()

egid_eta_eb_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Eta[F]$|all'))()
egid_iso_eta_eb_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Iso|all')*selections.Selector('^Eta[F]$|all'))()


# for sel in egid_iso_etatk_selections:
#     print(sel)

eg_emuCTl1_sta_plotters = [
    RatePlotter(
        coll.EGStaEE, egid_eta_ee_selections),
    RatePlotter(
        coll.EGStaEB, egid_eta_eb_selections),
]

eg_emuCTl1_pho_plotters = [
    RatePlotter(
        coll.TkEmEE, egid_iso_eta_eetk_selections),
    RatePlotter(
        coll.TkEmEB, egid_iso_eta_eb_selections),
]

eg_emuCTl1_ele_plotters = [
    RatePlotter(
        coll.TkEleEE, egid_iso_eta_eetk_selections_comp),
    RatePlotter(
        coll.TkEleEB, egid_iso_eta_eb_selections),
]

eg_emuCTl1_plotters = []
eg_emuCTl1_plotters.extend(eg_emuCTl1_sta_plotters)
eg_emuCTl1_plotters.extend(eg_emuCTl1_pho_plotters)
eg_emuCTl1_plotters.extend(eg_emuCTl1_ele_plotters)


eg_emuCTl2_plotters = [
    RatePlotter(
        coll.TkEmL2, egid_iso_etatk_selections),
    RatePlotter(
        coll.TkEleL2, egid_iso_etatk_selections),
]

eg_emuCTl1_ell_plotters = [
    RatePlotter(
        coll.TkEleEllEE, egid_iso_eta_eetk_selections_comp),
]


eg_emuCTl2_ell_plotters = [
    RatePlotter(
        coll.TkEmL2Ell, egid_iso_etatk_selections),
    RatePlotter(
        coll.TkEleL2Ell, egid_iso_etatk_selections),
]

eg_menuSta_plotters = [
    RatePlotter(
        coll.EGStaEE, egid_menu_sta_selections),
    RatePlotter(
        coll.EGStaEB, egid_menu_sta_selections),
]

eg_menuCTl2_plotters = [
    RatePlotter(
        coll.TkEmL2, egid_menu_pho_selections),
    RatePlotter(
        coll.TkEleL2, egid_menu_ele_selections),
    RatePlotter(
        coll.DoubleTkEmL2, egid_menu_dipho_selections),
    
]

eg_menuCTl2_ell_plotters = [
    RatePlotter(
        coll.TkEmL2Ell, egid_menu_pho_selections),
    RatePlotter(
        coll.TkEleL2Ell, egid_menu_ele_selections),
]


eg_menuCTl2_rate = [
    RateCounter(
        coll.TkEmL2, egid_menu_pho_rate_selections),
    RateCounter(
        coll.TkEleL2, egid_menu_ele_rate_selections),
    # RateCounter(
    #     coll.TkEleL2Ell, egid_menu_ele_rate_selections),
    DoubleObjRateCounter(
        coll.DoubleTkEleL2, egid_menu_diele_rate_selections),
    DoubleObjRateCounter(
        coll.DoubleTkEmL2, egid_menu_dipho_rate_selections)
]

eg_menuSta_rate = [
    RateCounter(
        coll.EGSta, egid_menu_sta_rate_selections),
    # RateCounter(
    #     coll.TkEleL2Ell, egid_menu_ele_rate_selections),
    DoubleObjRateCounter(
        coll.DoubleEGSta, egid_menu_dista_rate_selections),
    DoubleObjRateCounter(
        coll.DoubleTkEleEGSta, egid_menu_ditkelesta_rate_selections),

]


egid_ctl2_pho_selections = (
    selections.Selector('^MenuPho|^MenuSta')*('^EtaE[BE]$|all')+
    selections.Selector('^L2IDPho')*('^L2Iso|^IsoPho9[468]$|all')*('^EtaE[BE]$|^EtaEE[abc]|all'))()
egid_ctl2_dipho_selections = (selections.Selector('^MenuDoubleIsoTkPho22|^MenuDoubleIsoOneTkPho22|^MenuDoubleIso9[02468]TkPho22|^MenuDoubleTkPho22'))()

eg_ctl2_pho_iso_plotters = [
    RatePlotter(
        coll.TkEmL2IsoWP, egid_ctl2_pho_selections),
    RatePlotter(
        coll.DoubleTkEmL2IsoWP, egid_ctl2_dipho_selections),
    
]
