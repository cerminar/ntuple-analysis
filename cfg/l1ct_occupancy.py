from python import histos, plotters, selections, pf_regions
import cfg.datasets.fastpuppi_collections as coll
import python.boost_hist as bh
import awkward as ak
import numpy as np


class CorrOccupancyHistos(histos.BaseHistos):
    class CorrBoardOccupancyHistos:
        def __init__(self, name, board):
            self.h_totOcc = bh.TH1F(
                f'{name}_{board}totOcc', 
                f'{board} total occupancy; {board} total occ.', 
                500, 0, 500)
            self.h_regOcc = bh.TH1F(
                f'{name}_{board}regOcc', 
                f'{board} reg occupancy; {board} reg. occ.', 
                100, 0, 100)
            self.h_maxOcc = bh.TH1F(
                f'{name}_{board}maxOcc', 
                f'{board} max occupancy; {board} max occ.', 
                100, 0, 100)
            self.h_maxMult = bh.TH1F(
                f'{name}_{board}maxMult', 
                f'{board} max multiplicity per cluster; {board} max mult. per cluster', 
                100, 0, 100)
            self.eta_regions_idx = pf_regions.regions[board]
            self.max_count = None
            self.tot_count = None
            # self.max_mult_percluster = 0

        def fillRegion(self, ieta, occupancy, multpercluster):
            if(ieta in self.eta_regions_idx):
                self.h_regOcc.fill(occupancy)
                if self.max_count is None:
                    # print(f'--------- REset Board: {self.eta_regions_idx}')
                    self.max_count = occupancy
                else:
                    # print(f'Occ: {occupancy}')
                    # print(f'Old Max: {self.max_count}')
                    self.max_count = np.maximum(self.max_count, occupancy)
                    # print(f'New Max: {self.max_count}')

                if self.tot_count is None:
                    self.tot_count = occupancy
                else:
                    self.tot_count = self.tot_count + occupancy
                # if multpercluster > self.max_mult_percluster:
                #     self.max_mult_percluster = multpercluster

        def fillBoard(self):
            self.h_maxOcc.fill(self.max_count)
            self.h_totOcc.fill(self.tot_count)
            # self.h_maxMult.Fill(self.max_mult_percluster)
            self.max_count = None
            self.tot_count = None
            # self.max_mult_percluster = 0


    def __init__(self, name, root_file=None, debug=False):
        if not root_file:

            # self.h_avgOcc = ROOT.TProfile2D(
            #     f'{name}_avgOcc',
            #     'region avg occ; #eta, #phi;',
            #     pf_regions.regionizer.n_eta_regions(),
            #     array('d', pf_regions.regionizer.eta_boundaries_fiducial_),
            #     pf_regions.regionizer.n_phi_regions(),
            #     array('d', pf_regions.regionizer.phi_boundaries_fiducial_))

            self.board_histos = []
            for board in ['ALL', 'BRL', 'HGCNoTk', 'HGC']:
                bhs = CorrOccupancyHistos.CorrBoardOccupancyHistos(name, board)
                setattr(self, f'h_{board}totOcc', bhs.h_totOcc)
                setattr(self, f'h_{board}regOcc', bhs.h_regOcc)
                setattr(self, f'h_{board}maxOcc', bhs.h_maxOcc)
                setattr(self, f'h_{board}maxMult', bhs.h_maxMult)
                self.board_histos.append(bhs)

        histos.BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, objects):
        # print(objects.show())
        # print(objects.type.show())

        for ieta in range(pf_regions.regionizer.n_eta_regions()):
            for iphi in range(pf_regions.regionizer.n_phi_regions()):
                objs_in_region = objects[objects[f'eta_reg_{ieta}'] & objects[f'phi_reg_{iphi}']]
                occupancy = ak.count(objs_in_region, axis=1)
                nmatch_percluster = 0
                # if 'clidx' in objs_in_region.columns and not objs_in_region.empty:
                #     nmatch_percluster = objs_in_region.clidx.value_counts().iloc[0]
                # occupancy = objs_in_region.shape[0]

                # self.h_avgOcc.Fill(pf_regions.regionizer.eta_centers[ieta],
                #                    pf_regions.regionizer.phi_centers[iphi],
                #                    occupancy)
                for bhs in self.board_histos:
                    bhs.fillRegion(ieta, occupancy, nmatch_percluster)

        for bhs in self.board_histos:
            bhs.fillBoard()


class CorrOccupancyPlotter(plotters.BasePlotter):
    def __init__(self, tp_set, tp_selections=[selections.Selection('all')]):
        self.h_occ = {}
        super(CorrOccupancyPlotter, self).__init__(tp_set, tp_selections)

    def book_histos(self):
        self.tp_set.activate()
        tp_name = self.tp_set.name
        for selection in self.tp_selections:
            self.h_occ[selection.name] = CorrOccupancyHistos(
                name=f'{tp_name}_{selection.name}')

    def fill_histos(self, debug=0):
        for tp_sel in self.data_selections:
            # print(tp_sel)
            if tp_sel.all:
                objects = self.data_set.df
            else:
                objects = self.data_set.df[tp_sel.selection(self.data_set.df)]
            self.h_occ[tp_sel.name].fill(objects)




# ------ Plotter instances
sm = selections.SelectionManager()

multiclassID_sel = [
    selections.Selection('IDPuVetoMC', 'Pass MC PU veto',  lambda ar: ar.multiClassPuIdScore < 0.4878136),
    selections.Selection('IDEmMC', 'Pass MC EM ID',  lambda ar: ar.multiClassEmIdScore > 0.115991354),

]
selections.Selector.selection_primitives = sm.selections.copy()


pfin_hgc_tp_selections = (selections.Selector('^IDEm*|all')*('IDPuVeto*|all')*('^Pt[1,2,5]$|all'))()
pfin_tkcl3dmatch_selections = (selections.Selector('PUId')*('^EgBdtLE|all')*('^Pt[1,2,5]$|all')*('^MTkPt[1-5]|all'))()
pfin_eb_selections = (selections.Selector('^Pt[1,2,5]$'))()
pfin_tk_selections = (selections.Selector('^TkPt'))()


# pfeg_tp_input_selections = (Selector('^PFinH')*('^Pt[1,2,5]$|all')*('^Em$|all'))()
# pfeg_ee_input_selections = (Selector('^PFinH')*('^Pt[1,2,5]$|all')*('^EGq[1]$|all'))()
# pftkinput_selections = (Selector('^PFinBRL|^PFinHGC$')*('^TkPt'))()



l1tcorr_input_occ = [
    CorrOccupancyPlotter(
        coll.hgc_cl3d_pfinputs,
        pfin_hgc_tp_selections),
    # CorrOccupancyPlotter(
    #     coll.EGStaEB_pfinputs,
    #     pfin_eb_selections),
    # CorrOccupancyPlotter(
    #     coll.tk_pfinputs,
    #     pfin_tk_selections),
]



l1tcorr_tkcl3dmatch_input_occ = [
    CorrOccupancyPlotter(
        coll.tkCl3DMatch,
        pfin_tkcl3dmatch_selections),
    CorrOccupancyPlotter(
        coll.hgc_cl3d_pfinputs,
        pfin_hgc_tp_selections),
    CorrOccupancyPlotter(
        coll.tk_pfinputs,
        pfin_tk_selections),
]


# for sel in pfin_tkcl3dmatch_selections:
#     print(sel)

# # print('\n'.join([str(sel) for sel in hgc_tp_rate_selections]))
# hgc_tp_unmatched = [
#     plotters.Cl3DPlotter(coll.hgc_cl3d, hgc_tp_selections)
# ]


# hgc_tp_genmatched = [
#     plotters.Cl3DGenMatchPlotter(
#         coll.hgc_cl3d, coll.sim_parts,
#         hgc_tp_selections, gen_ee_selections)
# ]


# hgc_tp_rate = [
#     plotters.HGCCl3DRatePlotter(
#         coll.hgc_cl3d, hgc_tp_rate_selections),
# ]

# hgc_tp_rate_pt_wps = [
#     plotters.HGCCl3DGenMatchPtWPSPlotter(
#         coll.hgc_cl3d, coll.sim_parts,
#         gen_ee_selections)
# ]


# correlator_occupancy_plotters = [
#     plotters.CorrOccupancyPlotter(
#         coll.tk_pfinputs, selections.pftkinput_selections),
#     plotters.CorrOccupancyPlotter(
#         coll.egs_EE_pf_reg, selections.pfinput_regions),
#     plotters.CorrOccupancyPlotter(
#         coll.tkeles_EE_pf_reg, selections.pfinput_regions),
#     plotters.CorrOccupancyPlotter(
#         coll.tkeles_EB_pf_reg, selections.pfinput_regions),
#     plotters.CorrOccupancyPlotter(
#         coll.tkem_EE_pf_reg, selections.pfinput_regions),
#     plotters.CorrOccupancyPlotter(
#         coll.tkem_EB_pf_reg, selections.pfinput_regions),
#     plotters.CorrOccupancyPlotter(
#         coll.eg_EE_pfinputs, selections.pfeg_ee_input_selections),
#     plotters.CorrOccupancyPlotter(
#         coll.eg_EB_pfinputs, selections.pfeg_eb_input_selections),
#     plotters.CorrOccupancyPlotter(
#         coll.cl3d_hm_pfinputs, selections.pfeg_tp_input_selections),
# ]

for sel in pfin_hgc_tp_selections:
    print (sel)