from python import plotters, selections, calibrations, histos
import cfg.datasets.fastpuppi_collections as coll
import python.boost_hist as bh

# ------ Histogram classes ----------------------------------------------

class Cluster3DHistos(histos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            # self.h_npt05 = bh.TH1F(
            #     name+'_npt05', '# 3D Cluster Pt > 0.5 GeV; # 3D clusters in cone;', 1000, 0, 1000)
            # self.h_npt20 = bh.TH1F(
            #     name+'_npt20', '# 3D Cluster Pt > 2.0 GeV; # 3D clusters in cone;', 1000, 0, 1000)
            self.h_pt = bh.TH1F(
                f'{name}_pt', '3D Cluster Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(f'{name}_eta', '3D Cluster eta; #eta;', 100, -4, 4)
            # self.h_energy = bh.TH1F(name+'_energy', '3D Cluster energy (GeV); E [GeV]', 1000, 0, 1000)
            # self.h_nclu = bh.TH1F(name+'_nclu', '3D Cluster # clusters; # 2D components;', 60, 0, 60)
            # self.h_ncluVpt = bh.TH2F(name+'_ncluVpt', '3D Cluster # clusters vs pt; # 2D components; p_{T} [GeV]', 60, 0, 60, 100, 0, 100)
            # self.h_showlenght = bh.TH1F(name+'_showlenght', '3D Cluster showerlenght', 60, 0, 60)
            # self.h_firstlayer = bh.TH1F(name+'_firstlayer', '3D Cluster first layer', 30, 0, 30)
            # self.h_sEtaEtaTot = bh.TH1F(name+'_sEtaEtaTot', '3D Cluster RMS Eta', 100, 0, 0.1)
            # self.h_sEtaEtaMax = bh.TH1F(name+'_sEtaEtaMax', '3D Cluster RMS Eta (max)', 100, 0, 0.1)
            # self.h_sPhiPhiTot = bh.TH1F(name+'_sPhiPhiTot', '3D Cluster RMS Phi', 100, 0, 2)
            # self.h_sPhiPhiMax = bh.TH1F(name+'_sPhiPhiMax', '3D Cluster RMS Phi (max)', 100, 0, 2)
            # self.h_sZZ = bh.TH1F(name+'_sZZ', '3D Cluster RMS Z ???', 100, 0, 10)
            # self.h_eMaxOverE = bh.TH1F(name+'_eMaxOverE', '3D Cluster Emax/E', 100, 0, 1)
            # self.h_HoE = bh.TH1F(name+'_HoE', '3D Cluster H/E', 20, 0, 2)
            # self.h_iso0p2 = bh.TH1F(name+'_iso0p2', '3D Cluster iso DR 0.2(GeV); Iso p_{T} [GeV];', 100, 0, 100)
            # self.h_isoRel0p2 = bh.TH1F(name+'_isoRel0p2', '3D Cluster relative iso DR 0.2; Rel. Iso;', 100, 0, 1)
            # self.h_bdtPU = bh.TH1F(name+'_bdtPU', '3D Cluster bdt PU out; BDT-PU out;', 100, -1, 1)
            # self.h_bdtPi = bh.TH1F(name+'_bdtPi', '3D Cluster bdt Pi out; BDT-Pi out;', 100, -1, 1)
            # self.h_bdtEg = bh.TH1F(name+'_bdtEg', '3D Cluster bdt Pi out; BDT-EG out;', 100, -1, 1)

        histos.BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, cl3ds):
        # self.h_npt05.Fill(len(cl3ds[cl3ds.pt > 0.5].index))
        # self.h_npt20.Fill(len(cl3ds[cl3ds.pt > 2.0].index))

        bh.fill_1Dhist(self.h_pt, cl3ds.pt)
        bh.fill_1Dhist(self.h_eta, cl3ds.eta)
        # bh.fill_1Dhist(self.h_energy, cl3ds.energy)
        # bh.fill_1Dhist(self.h_nclu, cl3ds.nclu)
        # bh.fill_2Dhist(self.h_ncluVpt, cl3ds[['nclu', 'pt']])
        # bh.fill_1Dhist(self.h_showlenght, cl3ds.showerlength)
        # bh.fill_1Dhist(self.h_firstlayer, cl3ds.firstlayer)
        # bh.fill_1Dhist(self.h_sEtaEtaTot, cl3ds.seetot)
        # bh.fill_1Dhist(self.h_sEtaEtaMax, cl3ds.seemax)
        # bh.fill_1Dhist(self.h_sPhiPhiTot, cl3ds.spptot)
        # bh.fill_1Dhist(self.h_sPhiPhiMax, cl3ds.sppmax)
        # bh.fill_1Dhist(self.h_sZZ, cl3ds.szz)
        # bh.fill_1Dhist(self.h_eMaxOverE, cl3ds.emaxe)
        # bh.fill_1Dhist(self.h_HoE, cl3ds.hoe)
        # if 'iso0p2' in cl3ds.fields:
        #     bh.fill_1Dhist(self.h_iso0p2, cl3ds.iso0p2)
        #     bh.fill_1Dhist(self.h_isoRel0p2, cl3ds.isoRel0p2)
        # if 'bdt_pu' in cl3ds.fields:
        #     bh.fill_1Dhist(self.h_bdtPU, cl3ds.bdt_pu)
        # if 'bdt_pi' in cl3ds.fields:
        #     bh.fill_1Dhist(self.h_bdtPi, cl3ds.bdt_pi)
        # bh.fill_1Dhist(self.h_bdtEg, cl3ds.bdteg)


class ResoHistos(histos.BaseResoHistos):
    # @profile
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptResp = bh.TH1F(
                f'{name}_ptResp',
                '3D Cluster Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 2)
            self.h_ptRespVpt = bh.TH2F(
                f'{name}_ptRespVpt',
                '3D Cluster Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 2)
            self.h_ptRespVeta = bh.TH2F(
                f'{name}_ptRespVeta',
                '3D Cluster Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                20, -4, 4, 50, 0, 2)
            self.h_etaRes = bh.TH1F(
                f'{name}_etaRes',
                '3D Cluster eta reso; #eta^{L1}-#eta^{GEN}',
                100, -0.15, 0.15)
            self.h_phiRes = bh.TH1F(
                f'{name}_phiRes',
                '3D Cluster phi reso; #phi^{L1}-#phi^{GEN}',
                100, -0.15, 0.15)
            self.h_drRes = bh.TH1F(
                f'{name}_drRes',
                '3D Cluster DR reso; #DeltaR^{L1}-#DeltaR^{GEN}',
                100, 0, 0.1)

        histos.BaseResoHistos.__init__(self, name, root_file)

    def fill(self, reference, target):
        bh.fill_1Dhist(self.h_ptResp, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVeta, reference.eta, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVpt, reference.pt, target.pt/reference.pt)
        if 'caloeta' in reference.fields:
            bh.fill_1Dhist(self.h_etaRes, target.eta - reference.caloeta)
            bh.fill_1Dhist(self.h_phiRes, target.phi - reference.calophi)

# ------ Plotter classes ------------------------------------------------

class Cl3DPlotter(plotters.GenericDataFramePlotter):
    def __init__(self, data_set, data_selections=[selections.Selection('all')]):
        super(Cl3DPlotter, self).__init__(Cluster3DHistos, data_set, data_selections)


class Cl3DGenMatchPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')],
                 pt_bins=None):
        super(Cl3DGenMatchPlotter, self).__init__(Cluster3DHistos, ResoHistos,
                                                  data_set, gen_set,
                                                  data_selections, gen_selections,
                                                  gen_eta_phi_columns=['caloeta', 'calophi'],
                                                  pt_bins=pt_bins)

class HGCCl3DGenMatchPtWPSPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set, gen_selections):
        super(HGCCl3DGenMatchPtWPSPlotter, self).__init__(
            Cluster3DHistos, ResoHistos,
            data_set, gen_set,
            [], gen_selections)

    def book_histos(self):
        calib_mgr = calibrations.CalibManager()
        rate_pt_wps = calib_mgr.get_pt_wps()
        self.data_selections = calibrations.rate_pt_wps_selections(
            rate_pt_wps, self.data_set.name, 'pt_em')
        plotters.GenericGenMatchPlotter.book_histos(self)

class HGCCl3DRatePlotter(plotters.BasePlotter):
    def __init__(self, tp_set, tp_selections=[selections.Selection('all')]):
        self.h_rate = {}
        super(HGCCl3DRatePlotter, self).__init__(tp_set, tp_selections)

    def book_histos(self):
        self.tp_set.activate()
        tp_name = self.tp_set.name
        for selection in self.tp_selections:
            self.h_rate[selection.name] = histos.RateHistos(name=f'{tp_name}_{selection.name}', var='ptEm')

    def fill_histos(self, debug=0):
            for selection in self.tp_selections:
                sel_clusters = self.tp_set.df
                if not selection.all:
                    sel_clusters = self.tp_set.df[selection.selection(self.tp_set.df)]
                self.h_rate[selection.name].fill(sel_clusters)
                self.h_rate[selection.name].fill_norm(self.tp_set.new_read_nentries)



gen_ee_tk_selections = (selections.Selector('GEN$')*('Ee$')*('^Eta[A-C]$|EtaBC$|all')+selections.Selector('GEN$')*('Ee$')*('Pt15|Pt30'))()
gen_ee_selections = (selections.Selector('GEN$')*('^Eta[ABC]+[CD]$|^Eta[A-D]$|all')+selections.Selector('GEN$')*('^Pt15|^Pt30'))()
gen_selections = (selections.Selector('GEN$')*('^Eta[ABC]+[CD]$|^Eta[A-D,F]$|all')+selections.Selector('GEN$')*('^Pt15|^Pt30'))()

ctl2_eg_selections = (selections.Selector('^IDTightE$|all')*('^EtaE[EB]$|all')+selections.Selector('^Pt15|^Pt30'))()



hgc_tp_selections = (selections.Selector('^IDEm*|all')*('IDPuVeto|all')*('^Pt[2,3][0]$|all'))()

# hgc_tp_selections = (selections.Selector('^EgBdt*|^Em|all')*('PUId|all'))()

# hgc_tp_selections = (selections.Selector('^Eta[BC]+[CD]$|^Eta[A]$|all'))()
hgc_tp_rate_selections = (selections.Selector('^IDEm*|all')*('IDPuVeto|all')*('^Eta[ABC]+[CD]$|all'))()

tkcl3dmatch_selections = (selections.Selector('PUId')*('^EgBdtLE|all')*('^Pt[1,2,5]$|all')*('^MTkPt[2-5]|all'))()

hgc_tp_id_selections = (selections.Selector('^IDTightEm$|^IDLooseEm$|all')+selections.Selector('^EgBdt|all'))()


double_gen_selections = [
    selections.build_DiObj_selection('DoubleGENEtaEB', 'GENEtaEB',
                          (selections.Selector('GEN$')*('^EtaEB$')).one(),
                          (selections.Selector('GEN$')*('^EtaEB$')).one()),
    selections.build_DiObj_selection('DoubleGENEtaEE', 'GENEtaEE',
                          (selections.Selector('GEN$')*('^EtaEE$')).one(),
                          (selections.Selector('GEN$')*('^EtaEE$')).one()),
    selections.build_DiObj_selection('DoubleGEN', 'GEN',
                          (selections.Selector('GEN$')).one(),
                          (selections.Selector('GEN$')).one()),

]


# *('PUId|all')

# print('\n'.join([str(sel) for sel in hgc_tp_rate_selections]))
hgc_tp_unmatched = [
    Cl3DPlotter(coll.hgc_cl3d, hgc_tp_selections)
]


hgc_tp_genmatched = [
    Cl3DGenMatchPlotter(
        coll.hgc_cl3d, coll.gen,
        hgc_tp_selections, gen_ee_selections)
]


hgc_tp_rate = [
    HGCCl3DRatePlotter(
        coll.hgc_cl3d, hgc_tp_rate_selections),
]

hgc_tp_rate_pt_wps = [
    HGCCl3DGenMatchPtWPSPlotter(
        coll.hgc_cl3d, coll.gen,
        gen_ee_selections)
]

# for sel in tkcl3dmatch_selections:
#     print(sel)

hgc_tp_tkmatch_genmatched = [
    Cl3DGenMatchPlotter(
        coll.tkCl3DMatch, coll.gen,
        tkcl3dmatch_selections, gen_ee_selections)
]


# zprime_eff_pt_bins = list(range(0,100, 10))+list(range(100,500, 100))+list(range(500, 1000, 250))+list(range(1000, 2000, 500))


# hgc_tp_highpt_genmatched = [
#     plotters.GenPlotter(
#         coll.gen_ele,
#         gen_ee_selections,
#         pt_bins=range(0,4000, 5)),
#     plotters.DiObjMassPlotter(
#         collections.DoubleSimEle,
#         double_gen_selections
#     ),
#     Cl3DGenMatchPlotter(
#         coll.hgc_cl3d, coll.gen_ele,
#         hgc_tp_id_selections, gen_ee_selections,
#         pt_bins=zprime_eff_pt_bins),
#     plotters.EGGenMatchPlotter(
#         collections.TkEleL2, coll.gen_ele,
#         ctl2_eg_selections, gen_selections,
#         pt_bins=zprime_eff_pt_bins),


# ]

# for sel in gen_selections:
#     print (sel)
