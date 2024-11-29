from python import plotters, selections, calibrations, histos
import python.boost_hist as bh
import cfg.datasets.fastpuppi_collections as coll
import awkward as ak
import math

# ------ Histogram classes ----------------------------------------------

class EGHistos(histos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(f'{name}_pt', 'EG Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(f'{name}_eta', 'EG eta; #eta;', 100, -4, 4)
            self.h_energy = bh.TH1F(f'{name}_energy', 'EG energy (GeV); E [GeV]', 1000, 0, 1000)
            self.h_hwQual = bh.TH1F(f'{name}_hwQual', 'EG energy (GeV); hwQual', 5, 0, 5)
            self.h_tkIso = bh.TH1F(f'{name}_tkIso', 'Iso; rel-iso_{tk}', 100, 0, 2)
            self.h_pfIso = bh.TH1F(f'{name}_pfIso', 'Iso; rel-iso_{pf}', 100, 0, 2)
            self.h_puppiIso = bh.TH1F(f'{name}_puppiIso', 'Iso; rel-iso_{puppi}', 100, 0, 2)

            self.h_tkIsoPV = bh.TH1F(f'{name}_tkIsoPV', 'Iso; rel-iso^{PV}_{tk}', 100, 0, 2)
            self.h_pfIsoPV = bh.TH1F(f'{name}_pfIsoPV', 'Iso; rel-iso^{PV}_{pf}', 100, 0, 2)

            self.h_n = bh.TH1F(f'{name}_n', '# objects per event', 100, 0, 100)
            self.h_idScore = bh.TH1F(f'{name}_idScore', 'ID BDT Score', 50, -1, 1)

        histos.BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, egs):
        weight = None
        if 'weight' in egs.fields:
            weight = egs.weight

        bh.fill_1Dhist(hist=self.h_pt,     array=egs.pt,     weights=weight)
        bh.fill_1Dhist(hist=self.h_eta,    array=egs.eta,    weights=weight)
        # bh.fill_1Dhist(hist=self.h_energy, array=egs.energy, weights=weight)
        bh.fill_1Dhist(hist=self.h_hwQual, array=egs.hwQual, weights=weight)
        if 'tkIso' in egs.fields:
            bh.fill_1Dhist(hist=self.h_tkIso, array=egs.tkIso, weights=weight)
        if 'pfIso' in egs.fields:
            bh.fill_1Dhist(hist=self.h_pfIso, array=egs.pfIso, weights=weight)
        if 'puppiIso' in egs.fields:
            bh.fill_1Dhist(hist=self.h_puppiIso, array=egs.puppiIso, weights=weight)
        if 'tkIsoPV' in egs.fields:
            bh.fill_1Dhist(hist=self.h_tkIsoPV, array=egs.tkIsoPV, weights=weight)
            bh.fill_1Dhist(hist=self.h_pfIsoPV, array=egs.pfIsoPV, weights=weight)
        if 'compBDTScore' in egs.fields:
            bh.fill_1Dhist(hist=self.h_compBdt, array=egs.compBDTScore, weights=weight)
        if 'idScore' in egs.fields:
            bh.fill_1Dhist(hist=self.h_idScore, array=egs.idScore, weights=weight)
        # print(ak.count(egs.pt, axis=1))
        # print(egs.pt.type.show())
        # print(ak.count(egs.pt, axis=1).type.show())
        self.h_n.fill(ak.count(egs.pt, axis=1))
        # bh.fill_1Dhist(hist=self.h_n, array=ak.count(egs.pt, axis=1), weights=weight)
        # self.h_n.Fill()

class EGResoHistos(histos.BaseResoHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:

            self.h_ptResVpt = bh.TH2F(
                f'{name}_ptResVpt',
                'EG Pt reso. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}-p_{T}^{GEN} [GeV];',
                50, 0, 100,
                100, -10, 10)
            self.h_ptRes = bh.TH1F(
                f'{name}_ptRes',
                'EG Pt res.; (p_{T}^{L1}-p_{T}^{GEN})/p_{T}^{GEN}',
                100, -1, 1)
            self.h_ptResp = bh.TH1F(
                f'{name}_ptResp',
                'EG Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 3)
            self.h_ptRespVpt = bh.TH2F(
                f'{name}_ptRespVpt',
                'EG Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100,
                100, 0, 3)
            self.h_ptRespVeta = bh.TH2F(
                f'{name}_ptRespVeta',
                'EG Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                50, -4, 4,
                100, 0, 3)

            self.h_etaRes = bh.TH1F(
                f'{name}_etaRes',
                'EG eta reso; #eta^{L1}-#eta^{GEN}',
                100, -0.1, 0.1)
            self.h_phiRes = bh.TH1F(
                f'{name}_phiRes',
                'EG phi reso; #phi^{L1}-#phi^{GEN}',
                100, -0.1, 0.1)

            self.h_exetaRes = bh.TH1F(
                f'{name}_exetaRes',
                'EG eta reso; #eta^{L1}-#eta^{GEN}_{calo}',
                100, -0.1, 0.1)
            self.h_exphiRes = bh.TH1F(
                f'{name}_exphiRes',
                'EG phi reso; #phi^{L1}-#phi^{GEN}_{calo}',
                100, -0.1, 0.1)

            self.h_dzRes = bh.TH1F(
                f'{name}_dzRes',
                '#DeltaZ_{0} res; #DeltaZ_{0}^{L1}-#DeltaZ_{0}^{GEN}',
                100, -10, 10)

        histos.BaseResoHistos.__init__(self, name, root_file, debug)

    def fill(self, reference, target):
        # FIXME: weights

        bh.fill_1Dhist(self.h_ptRes, (target.pt-reference.pt)/reference.pt)
        bh.fill_2Dhist(self.h_ptResVpt, reference.pt, target.pt-reference.pt)
        bh.fill_1Dhist(self.h_ptResp, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVeta, reference.eta, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVpt, reference.pt, target.pt/reference.pt)
        bh.fill_1Dhist(self.h_etaRes, target.eta - reference.eta)
        bh.fill_1Dhist(self.h_phiRes, target.phi - reference.phi)
        bh.fill_1Dhist(self.h_exetaRes, target.eta - reference.caloeta)
        bh.fill_1Dhist(self.h_exphiRes, target.phi - reference.calophi)

        # if 'tkZ0' in target.columns:
        #     self.h_dzRes.Fill(target_line.tkZ0 - reference.ovz)


# ------ Plotter classes ------------------------------------------------

class EGGenMatchPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')],
                 gen_eta_phi_columns=('caloeta', 'calophi'),
                 pt_bins=None):
        super(EGGenMatchPlotter, self).__init__(EGHistos, EGResoHistos,
                                                data_set, gen_set,
                                                data_selections, gen_selections,
                                                gen_eta_phi_columns=gen_eta_phi_columns,
                                                pt_bins=pt_bins)


class EGGenMatchPtWPSPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set, gen_selections):
        super(EGGenMatchPtWPSPlotter, self).__init__(
            EGHistos, EGResoHistos,
            data_set, gen_set,
            [], gen_selections)

    def book_histos(self):
        calib_mgr = calibrations.CalibManager()
        rate_pt_wps = calib_mgr.get_calib('rate_pt_wps')
        self.data_selections = selections.rate_pt_wps_selections(
            rate_pt_wps, self.data_set.name)
        plotters.GenericGenMatchPlotter.book_histos(self)



# ------ Plotter instances



# FIXME: should become in newer versions
# l1tc_match_ee_selections = (selections.Selector('^EGq[1,3]$')*('^Pt[1-2][0]$|all'))()
l1tc_match_ee_selections = (selections.Selector('^EGq[1,3]$')*('^Pt[1-2][0]$|all'))()

# l1tc_eg_genmatched = [
#     EGGenMatchPlotter(
#         collections.EGStaEE, collections.sim_parts,
#         l1tc_match_ee_selections, gen_ee_selections),
#     EGGenMatchPlotter(
#         collections.TkEleEE, collections.sim_parts,
#         l1tc_match_ee_selections, gen_ee_tk_selections),
#     EGGenMatchPlotter(
#         collections.TkEleEB, collections.sim_parts,
#         selections.eg_id_pt_eb_selections, selections.gen_eb_selections),
#     EGGenMatchPlotter(
#         collections.TkEmEE, collections.sim_parts,
#         l1tc_match_ee_selections, gen_ee_tk_selections),
#     EGGenMatchPlotter(
#         collections.TkEmEB, collections.sim_parts,
#         selections.eg_id_pt_eb_selections, selections.gen_eb_selections),
# ]

# FIXME: this one can be dropped in newer versions
l1tc_fw_match_ee_selections = (selections.Selector('^EGq[2,4]or[3,5]$')*('^Pt[1-2][0]$|all'))()

# l1tc_rate_pt_wps = [
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.EGStaEE, collections.sim_parts,
#         gen_ee_selections),
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.TkEleEE, collections.sim_parts,
#         gen_ee_tk_selections),
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.TkEleEB, collections.sim_parts,
#         selections.gen_eb_selections),
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.TkEmEE, collections.sim_parts,
#         gen_ee_tk_selections),
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.TkEmEB, collections.sim_parts,
#         selections.gen_eb_selections),
# ]




gen_selections = (selections.Selector('GEN$')*('^Eta[F]$|^Eta[AF][ABCD]*[C]$|all')+selections.Selector('GEN$')*('^Pt15|^Pt30'))()

# gen_menu_selections = (selections.Selector('GEN$')*('^EtaE[BE]$|all')+selections.Selector('GEN$')*('^Pt10to25$|^Pt25'))()
gen_menu_selections = (selections.Selector('GEN$')*('^EtaE[BE]$|^EtaEE[abc]$|all')+selections.Selector('GEN$')*('^Pt15$|^Pt30$|^Pt10to25$'))()

# for sels in [gen_selections, selections.gen_selections]:
#     print('--------------------')
#     print(f'# of sels: {len(sels)}')
#     for sel in sels:
#         print(sel)

egid_sta_selections = (selections.Selector('^IDTightS|all')*('^Pt[1-3][0]$|all'))()
# egid_iso_tkele_selections = (selections.Selector('^IDTight[E]|all')*('^Pt[1-3][0]$|all')*('^Iso0p[1-2]|all'))()
# egid_iso_tkpho_selections = (selections.Selector('^IDTight[P]|all')*('^Pt[1-3][0]$|all')*('^Iso0p[1-2]|all'))()
egid_iso_tkele_selections = (selections.Selector('^IDTight[E]$|all')*('^Pt[1-3][0]$|all'))()
egid_iso_tkpho_selections = (selections.Selector('^IDTight[P]$|all')*('^Pt[1-3][0]$|all'))()
egid_iso_tkele_comp_selections = (selections.Selector('^IDTight[E]$|^IDComp|all')*('^Pt[1-3][0]$|all'))()


gen_ee_tk_selections = (selections.Selector('GEN$')*('^Eta[A-C]$|EtaBC$|all')+selections.Selector('GEN$')*('Pt15|Pt30'))()
gen_ee_eb_tk_selections = (selections.Selector('^GEN$')*('^Pt5to1[05]$|^Pt30|all')+selections.Selector('^GEN$')*('^EtaE[EB]$|^EtaEE[abc]$'))()
gen_ee_selections = (selections.Selector('^GEN$')*('^Pt5to1[05]$|^Pt30|all')+selections.Selector('^GEN$')*('^EtaEE[abc]$|^EtaEEFwd$'))()
gen_eb_selections = (selections.Selector('^GEN$')*('^Pt5to1[05]$|^Pt30|all')+selections.Selector('^GEN$')*('^EtaEB$'))()

for sel in gen_eb_selections:
    print (sel)

ctl1_tkeg = [
    EGGenMatchPlotter(
        coll.TkEleEE, coll.gen,
        egid_iso_tkele_selections, gen_ee_tk_selections),
    EGGenMatchPlotter(
        coll.TkEleEB, coll.gen,
        egid_iso_tkele_selections, gen_eb_selections),
]

ctl2_tkeg = [
    EGGenMatchPlotter(
        coll.TkEmL2, coll.gen,
        egid_iso_tkpho_selections, gen_ee_eb_tk_selections),
    EGGenMatchPlotter(
        coll.TkEleL2, coll.gen,
        egid_iso_tkele_selections, gen_ee_eb_tk_selections,
        gen_eta_phi_columns=('eta', 'phi')),
]

egsta = [
    EGGenMatchPlotter(
        coll.EGStaEB, coll.gen,
        egid_sta_selections, gen_eb_selections),
    EGGenMatchPlotter(
        coll.EGStaEE, coll.gen,
        egid_sta_selections, gen_ee_selections),
]


# l1tc_emu_genmatched = [
#     # EGGenMatchPlotter(
#     #     collections.EGStaEE, collections.sim_parts,
#     #     egid_sta_selections, gen_ee_selections),
#     # EGGenMatchPlotter(
#     #     collections.EGStaEB, collections.sim_parts,
#     #     egid_sta_selections, selections.gen_eb_selections),
#     EGGenMatchPlotter(
#         collections.TkEleEE, collections.gen_ele,
#         egid_iso_tkele_selections, gen_ee_tk_selections),
#     EGGenMatchPlotter(
#         collections.TkEleEB, collections.gen_ele,
#         egid_iso_tkele_selections, selections.gen_eb_selections),
#     EGGenMatchPlotter(
#         collections.TkEmEE, collections.gen_ele,
#         egid_iso_tkpho_selections, gen_ee_tk_selections),
#     EGGenMatchPlotter(
#         collections.TkEmEB, collections.gen_ele,
#         egid_iso_tkpho_selections, selections.gen_eb_selections),
#     EGGenMatchPlotter(
#         collections.TkEmL2, collections.gen_ele,
#         egid_iso_tkpho_selections, selections.gen_selections),
#     EGGenMatchPlotter(
#         collections.TkEleL2, collections.gen_ele,
#         egid_iso_tkele_selections, selections.gen_selections),

# ]


# l1tc_l1emu_eb_genmatched = [
#     EGGenMatchPlotter(
#         collections.EGStaEB, collections.sim_parts,
#         egid_sta_selections, selections.gen_eb_selections),
#     EGGenMatchPlotter(
#         collections.TkEleEB, collections.sim_parts,
#         egid_iso_tkele_selections, selections.gen_eb_selections),
#     EGGenMatchPlotter(
#         collections.TkEmEB, collections.sim_parts,
#         egid_iso_tkpho_selections, selections.gen_eb_selections),
# ]

# l1tc_l1emu_ee_genmatched = [
#     # EGGenMatchPlotter(
#     #     collections.EGStaEE, collections.sim_parts,
#     #     egid_sta_selections, gen_ee_selections),
#     EGGenMatchPlotter(
#         collections.TkEleEE, collections.sim_parts,
#         egid_iso_tkele_comp_selections, gen_ee_tk_selections),
#     # EGGenMatchPlotter(
#     #     collections.TkEmEE, collections.sim_parts,
#     #     egid_iso_tkpho_selections, gen_ee_tk_selections),
# ]

# l1tc_l1emu_ee_ell_genmatched = [
#     EGGenMatchPlotter(
#         collections.TkEleEllEE, collections.sim_parts,
#         egid_iso_tkele_comp_selections, gen_ee_tk_selections),
# ]


# l1tc_l2emu_genmatched = [
#     EGGenMatchPlotter(
#         collections.TkEmL2, collections.gen_ele,
#         egid_iso_tkpho_selections, selections.gen_selections),
#     EGGenMatchPlotter(
#         collections.TkEleL2, collections.gen_ele,
#         egid_iso_tkele_selections, selections.gen_selections),
# ]

# l1tc_l2emu_ell_genmatched = [
#     # EGGenMatchPlotter(
#     #     collections.TkEmL2Ell, collections.sim_parts,
#     #     egid_iso_tkpho_selections, selections.gen_selections),
#     EGGenMatchPlotter(
#         collections.TkEleL2Ell, collections.sim_parts,
#         egid_iso_tkele_selections, selections.gen_selections),
# ]

do_tons = False
# *('^EtaE[BE]$|all')
egid_menu_ele_selections = (selections.Selector('^SingleIsoTkEle|^SingleTkEle|^MenuEle'))()
egid_menu_pho_selections = (selections.Selector('^SingleIsoTkPho|^SingleEGEle|^MenuSta|^MenuPho'))()
egid_menu_sta_selections = (selections.Selector('^MenuSta|all'))()

if do_tons:
    print('Menu Turn-on selections are enabled!')
    egid_menu_ele_selections.extend((selections.Selector('^MenuEle')*('^Pt[2-4]0$'))())
    egid_menu_pho_selections.extend((selections.Selector('^MenuPho')*('^Pt[2-4]0$'))())
    egid_menu_sta_selections.extend((selections.Selector('^MenuSta')*('^Pt[2-4]0$'))())





ctl2_tkeg_menu = [    
    EGGenMatchPlotter(
        coll.TkEmL2, coll.gen,
        egid_menu_pho_selections, gen_menu_selections),
    EGGenMatchPlotter(
        coll.TkEleL2, coll.gen,
        egid_menu_ele_selections, gen_menu_selections, 
        gen_eta_phi_columns=('eta', 'phi')
        ),
]
egid_menu_ele_ton_selections = []
egid_menu_pho_ton_selections = []
egid_menu_sta_ton_selections = []

egid_menu_ele_ton_selections.extend(egid_menu_ele_selections)
egid_menu_pho_ton_selections.extend(egid_menu_pho_selections)
egid_menu_sta_ton_selections.extend(egid_menu_sta_selections)

egid_menu_ele_ton_selections.extend((selections.Selector('^MenuEle')*('^Pt[2-4]0$'))())
egid_menu_pho_ton_selections.extend((selections.Selector('^MenuPho')*('^Pt[2-4]0$'))())
egid_menu_sta_ton_selections.extend((selections.Selector('^MenuSta')*('^Pt[2-4]0$'))())

egid_menu_ele_ton_selections = selections.prune(egid_menu_ele_ton_selections)
egid_menu_pho_ton_selections = selections.prune(egid_menu_pho_ton_selections)
egid_menu_sta_ton_selections = selections.prune(egid_menu_sta_ton_selections)


ctl2_tkeg_menu_tons = [    
    EGGenMatchPlotter(
        coll.TkEmL2, coll.gen,
        egid_menu_pho_ton_selections, gen_menu_selections),
    EGGenMatchPlotter(
        coll.TkEleL2, coll.gen,
        egid_menu_ele_ton_selections, gen_menu_selections, 
        gen_eta_phi_columns=('eta', 'phi')
        ),
]

egsta_menu = [
    EGGenMatchPlotter(
        coll.EGStaEE, coll.gen,
        egid_menu_sta_selections, gen_menu_selections),
    EGGenMatchPlotter(
        coll.EGStaEB, coll.gen,
        egid_menu_sta_selections, gen_menu_selections),
]



egid_ctl2_pho_selections = (
    selections.Selector('^L2IDPho')*('^L2Iso|^IsoPho9[02468]$|all') + 
    selections.Selector('^Iso@9[02468]TkPho[12]2$|IsoTkPho[12]2$'))()
gen_ctl2_selections = (selections.Selector('GEN$')*('^EtaE[BE]$|^EtaEE[abc]$|all')+selections.Selector('GEN$')*('^Pt15$|^Pt30$|^Pt10to25$'))()


ctl2_tkem_iso = [
    EGGenMatchPlotter(
    coll.TkEmL2IsoWP, coll.gen,
    egid_ctl2_pho_selections, gen_ctl2_selections),
]


# l1tc_l2emu_ell_singlelepton_genmatched = [
#     EGGenMatchPlotter(
#         collections.TkEleL2Ell, collections.gen_ele,
#         egid_menu_ele_rate_selections, gen_menu_selections),
# ]


# l1tc_l2emu_singlelepton_rate_pt_wps = [
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.TkEmL2, collections.sim_parts,
#         gen_selections),
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.TkEleL2, collections.sim_parts,
#         gen_selections),
# ]

# l1tc_emu_rate_pt_wps = [
#     # plotters.EGGenMatchPtWPSPlotter(
#     #     collections.EGStaEE, collections.sim_parts,
#     #     gen_ee_selections),
#     # plotters.EGGenMatchPtWPSPlotter(
#     #     collections.TkEleEE, collections.sim_parts,
#     #     gen_ee_tk_selections),
#     # # plotters.EGGenMatchPtWPSPlotter(
#     #     collections.TkEleEB, collections.sim_parts,
#     #     selections.gen_eb_selections),
#     # plotters.EGGenMatchPtWPSPlotter(
#     #     collections.TkEmEE, collections.sim_parts,
#     #     gen_ee_tk_selections),
#     # plotters.EGGenMatchPtWPSPlotter(
#     #     collections.TkEmEB, collections.sim_parts,
#     #     selections.gen_eb_selections),
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.TkEleEE, collections.sim_parts,
#         gen_selections),
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.TkEleEllEE, collections.sim_parts,
#         gen_selections),
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.TkEmL2, collections.sim_parts,
#         gen_selections),
#     plotters.EGGenMatchPtWPSPlotter(
#         collections.TkEleL2, collections.sim_parts,
#         gen_selections),
# ]

# for sel in egid_menu_ele_rate_selections:
#     print(sel)
