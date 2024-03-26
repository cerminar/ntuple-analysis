from python import collections, plotters, selections, l1THistos
import python.boost_hist as bh
import awkward as ak
import math

class RateHistos(l1THistos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_norm = bh.TH1F(
                f'{name}_norm', 
                '# of events', 
                1, 1, 2)
            self.h_pt = bh.TH1F(
                f'{name}_pt', 
                'rate above p_{T} thresh.; p_{T} [GeV]; rate [kHz];', 
                100, 0, 100)
            # self.h_ptVabseta = bh.TH2F(name+'_ptVabseta', 'Candidate p_{T} vs |#eta|; |#eta|; p_{T} [GeV];', 34, 1.4, 3.1, 100, 0, 100)

        l1THistos.BaseHistos.__init__(self, name, root_file, debug)

        if root_file is not None:
            for attr_1d in [attr for attr in dir(self) if (attr.startswith('h_') and 'TH1' in getattr(self, attr).ClassName())]:
                setattr(self, f'{attr_1d}_graph', l1THistos.GraphBuilder(self, attr_1d))

        if root_file is not None:
            self.normalize(2760.0*11246/1000)
            # self.h_simenergy = bh.TH1F(name+'_energy', 'Digi sim-energy (GeV)', 100, 0, 2)

    def fill(self, data):
        # print(self.h_pt.axes[0])
        pt_max = ak.max(data.pt, axis=1)
        for thr,bin_center in zip(self.h_pt.axes[0].edges, self.h_pt.axes[0].centers, strict=False):
        # for thr,bin_center in zip(self.h_pt.axes[0].edges[1:], self.h_pt.axes[0].centers):
            self.h_pt.fill(bin_center, weight=ak.sum(pt_max>=thr))

        # for ptf in range(0, int(pt)+1):
        #     self.h_pt.Fill(ptf)
        # self.h_ptVabseta.Fill(abs(eta), pt)

    def fill_norm(self, many=1):
        # print (f' fill rate norm: {many}')
        self.h_norm.fill(1, weight=many)

    def normalize(self, norm):
        nev = self.h_norm.GetBinContent(1)
        if(nev != norm):
            print(f'normalize # ev {nev} to {norm}')
            self.h_norm.Scale(norm/nev)
            self.h_pt.Scale(norm/nev)
            # self.h_ptVabseta.Scale(norm/nev)


class RateHistoCounter(l1THistos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_norm = bh.TH1F(f'{name}_norm', '# of events', 1, 1, 2)
            self.h_rate = bh.TH1F(f'{name}_rate', '# passing events; rate [kHz];',  1, 1, 2)

        l1THistos.BaseHistos.__init__(self, name, root_file, debug)

        if root_file is not None:
            for attr_1d in [attr for attr in dir(self) if (attr.startswith('h_') and 'TH1' in getattr(self, attr).ClassName())]:
                setattr(self, f'{attr_1d}_graph', l1THistos.GraphBuilder(self, attr_1d))

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
            self.h_rate[selection.name] = RateHistos(name=f'{tp_name}_{selection.name}')

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




simeg_rate_ee_selections = (selections.Selector('^EGq[4-5]$')*('^Eta[^DA][BC]*[BCD]$|all'))()
emueg_rate_ee_selections = (selections.Selector('^EGq[1,3]$')*('^Eta[A][BC]*[C]$')*('^Iso|all'))()

emueg_fw_rate_ee_selections = (selections.Selector('^EGq[1,3]$')*('^Eta[A][BC]*[BCD]$|all')*('^Iso|all'))()
emueg_rate_eb_selections = (selections.Selector('^LooseTkID$|all')*('^Eta[F]$')*('^Iso|all'))()


# print(egid_eta_selections)

# sim_eg_match_ee_selections = (selections.Selector('^EGq[4-5]$')*('^Pt[1-3][0]$|all'))()
# gen_ee_tk_selections = (selections.Selector('GEN$')*('Ee$')*('^Eta[A-C]$|EtaBC$|all')+selections.Selector('GEN$')*('Ee$')*('Pt15|Pt30'))()
# gen_ee_selections = (selections.Selector('GEN$')*('Ee')*('^Eta[BC]+[CD]$|^Eta[A-D]$|all')+selections.Selector('GEN$')*('Ee')*('^Pt15|^Pt30'))()
tp_plotters = [
    RatePlotter(collections.hgc_cl3d, selections.tp_rate_selections),
    # RatePlotter(collections.cl3d_hm_calib, selections.tp_rate_selections),
    # RatePlotter(collections.cl3d_hm_shapeDr, selections.tp_rate_selections),
    # RatePlotter(collections.cl3d_hm_shapeDr_calib, selections.tp_rate_selections),
    # RatePlotter(collections.cl3d_hm_shapeDr_calib_new, selections.tp_rate_selections),
    # RatePlotter(collections.cl3d_hm_shapeDtDu_calib, selections.tp_rate_selections),
    # RatePlotter(collections.cl3d_hm_emint, selections.tp_rate_selections),
]


eg_emu_plotters = [
    RatePlotter(
        collections.TkEmEE, emueg_rate_ee_selections),
    RatePlotter(
        collections.TkEmEB, selections.barrel_rate_selections),
    RatePlotter(
        collections.TkEleEE, emueg_rate_ee_selections),
    RatePlotter(
        collections.TkEleEB, selections.barrel_rate_selections),
    # RatePlotter(
    #     collections.tkem_EE_pfnf, selections.eg_id_iso_eta_ee_selections),
    # RatePlotter(
    #     collections.tkem_EB_pfnf, selections.barrel_rate_selections),
]

egid_eta_selections = (selections.Selector('^IDTightS|all')*selections.Selector('^Eta[F]$|^Eta[AF][ABCD]*[CD]$'))()
egid_etatk_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Eta[F]$|^Eta[AF][ABCD]*[C]$'))()
egid_iso_etatk_selections = (selections.Selector('^IDTight[EP]|all')*selections.Selector('^Iso|all')*selections.Selector('^Eta[F]$|^Eta[AF][ABCD]*[C]$'))()

egid_menu_ele_selections = (selections.Selector('^MenuEle')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_pho_selections = (selections.Selector('^MenuPho|^MenuSta')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_sta_selections = (selections.Selector('^MenuSta')*selections.Selector('^EtaE[BE]$|all'))()


egid_menu_ele_rate_selections = (selections.Selector('^SingleIsoTkEle|^SingleTkEle')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_pho_rate_selections = (selections.Selector('^SingleIsoTkPho|^SingleEGEle')*selections.Selector('^EtaE[BE]$|all'))()
egid_menu_diele_rate_selections = (selections.Selector('^DoubleTkEle'))()
egid_menu_dipho_rate_selections = (selections.Selector('^DoubleIsoTkPho'))()


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
        collections.EGStaEE, egid_eta_ee_selections),
    RatePlotter(
        collections.EGStaEB, egid_eta_eb_selections),
]

eg_emuCTl1_pho_plotters = [
    RatePlotter(
        collections.TkEmEE, egid_iso_eta_eetk_selections),
    RatePlotter(
        collections.TkEmEB, egid_iso_eta_eb_selections),
]

eg_emuCTl1_ele_plotters = [
    RatePlotter(
        collections.TkEleEE, egid_iso_eta_eetk_selections_comp),
    RatePlotter(
        collections.TkEleEB, egid_iso_eta_eb_selections),
]

eg_emuCTl1_plotters = []
eg_emuCTl1_plotters.extend(eg_emuCTl1_sta_plotters)
eg_emuCTl1_plotters.extend(eg_emuCTl1_pho_plotters)
eg_emuCTl1_plotters.extend(eg_emuCTl1_ele_plotters)


eg_emuCTl2_plotters = [
    RatePlotter(
        collections.TkEmL2, egid_iso_etatk_selections),
    RatePlotter(
        collections.TkEleL2, egid_iso_etatk_selections),
]

eg_emuCTl1_ell_plotters = [
    RatePlotter(
        collections.TkEleEllEE, egid_iso_eta_eetk_selections_comp),
]


eg_emuCTl2_ell_plotters = [
    RatePlotter(
        collections.TkEmL2Ell, egid_iso_etatk_selections),
    RatePlotter(
        collections.TkEleL2Ell, egid_iso_etatk_selections),
]

eg_menuSta_plotters = [
    RatePlotter(
        collections.EGStaEE, egid_menu_sta_selections),
    RatePlotter(
        collections.EGStaEB, egid_menu_sta_selections),
]

eg_menuCTl2_plotters = [
    RatePlotter(
        collections.TkEmL2, egid_menu_pho_selections),
    RatePlotter(
        collections.TkEleL2, egid_menu_ele_selections),
]

eg_menuCTl2_ell_plotters = [
    RatePlotter(
        collections.TkEmL2Ell, egid_menu_pho_selections),
    RatePlotter(
        collections.TkEleL2Ell, egid_menu_ele_selections),
]



eg_menuCTl2_rate = [
    RateCounter(
        collections.TkEmL2, egid_menu_pho_rate_selections),
    RateCounter(
        collections.TkEleL2, egid_menu_ele_rate_selections),
    # RateCounter(
    #     collections.TkEleL2Ell, egid_menu_ele_rate_selections),
    DoubleObjRateCounter(
        collections.DoubleTkEleL2, egid_menu_diele_rate_selections),
    DoubleObjRateCounter(
        collections.DoubleTkEmL2, egid_menu_dipho_rate_selections)
]
