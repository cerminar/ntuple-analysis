from python import plotters, selections, calibrations, histos
import python.boost_hist as bh
import cfg.datasets.fastpuppi_collections as coll
import awkward as ak
import numpy as np
import math


class DecResoHistos(histos.BaseResoHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptResp = bh.TH1F(
                f'{name}_ptResp',
                'Track Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 3)
            self.h_ptRespVpt = bh.TH2F(
                f'{name}_ptRespVpt',
                'Track Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 3)
            self.h_ptRespVeta = bh.TH2F(
                f'{name}_ptRespVeta',
                'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 4, 100, 0, 3)
            self.h_etaRes = bh.TH1F(
                f'{name}_etaRes',
                'Track eta reso',
                100, -0.4, 0.4)
            self.h_etaResVeta = bh.TH2F(
                f'{name}_etaResVeta',
                'Track eta reso vs #eta; #eta^{GEN}; #Delta#eta;',
                50, -4, 4, 100, -0.4, 0.4)
            self.h_phiRes = bh.TH1F(
                f'{name}_phiRes',
                'Track phi reso',
                100, -0.4, 0.4)
            self.h_drRes = bh.TH1F(
                f'{name}_drRes',
                'Track DR reso',
                100, 0, 0.4)
            self.h_nMatch = bh.TH1F(
                f'{name}_nMatch',
                '# matches',
                100, 0, 100)

        histos.BaseResoHistos.__init__(self, name, root_file, debug)

    def fill(self, reference, target):
        bh.fill_1Dhist(self.h_ptResp, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVeta, np.abs(reference.eta), target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVpt, reference.pt, target.pt/reference.pt)
        bh.fill_1Dhist(self.h_etaRes, (target.eta - reference.eta))
        bh.fill_2Dhist(self.h_etaResVeta, reference.eta, (target.eta - reference.eta))
        bh.fill_1Dhist(self.h_phiRes, (target.phi - reference.phi))
        bh.fill_1Dhist(self.h_drRes, np.sqrt((reference.phi-target.phi)**2+(reference.eta-target.eta)**2))

    def fill_nMatch(self, n_matches):
        self.h_nMatch.Fill(n_matches)


# ------ Histogram classes ----------------------------------------------
class DecCaloHistos(histos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(f'{name}_pt',
                                'Pt (GeV); p_{T} [GeV]', 100, 0, 500)
            self.h_eta = bh.TH1F(f'{name}_eta',
                                 'eta; #eta;', 100, -4, 4)

            self.h_phi = bh.TH1F(f'{name}_phi',
                                 'phi; #phi;', 100, -4, 4)

            self.h_clPt = bh.TH1F(f'{name}_clPt',
                                'Pt (GeV); p_{T}^{float} [GeV]', 100, 0, 500)

            self.h_clEta = bh.TH1F(f'{name}_clEta',
                                 'eta; #eta^{float};', 100, -4, 4)

            self.h_clPhi = bh.TH1F(f'{name}_clPhi',
                                 'phi; #phi^{float};', 100, -4, 4)

            self.h_ptResoEmu = bh.TH1F(f'{name}_ptResoEmu',
                                '#Delta Pt (GeV); p_{T}^{float} - p_{T}^{EMU} [GeV]', 40, -10, 10)
            self.h_etaResoEmu = bh.TH1F(f'{name}_etaResoEmu',
                                '#Delta eta; #eta^{float} - #eta^{EMU};', 50, -0.1, 0.1)
            self.h_phiResoEmu = bh.TH1F(f'{name}_phiResoEmu',
                                '#Delta phi; #phi^{float} - #phi^{EMU};', 50, -0.1, 0.1)

            self.h_hwQual = bh.TH1F(f'{name}_hwQual',
                                 'hwQual; hwQual;', 10, 0, 10)
            self.h_puIdProb = bh.TH1F(f'{name}_puIdProb',
                                 'puIdProb; puIdProb;', 100, 0, 1)
            self.h_piIdProb = bh.TH1F(f'{name}_piIdProb',
                                    'piIdProb; piIdProb;', 100, 0, 1)
            self.h_emIdProb = bh.TH1F(f'{name}_emIdProb',
                                    'emIdProb; emIdProb;', 100, 0, 1)
            self.h_puIdProbVeta = bh.TH2F(f'{name}_puIdProbVeta',
                                    'puIdProbVeta; #eta; puIdProb;', 50, 0, 4, 100, 0, 1)
            self.h_puIdProbVpt = bh.TH2F(f'{name}_puIdProbVpt',
                                    'puIdProbVeta; p_{T}; puIdProb;', 100, 0, 500, 100, 0, 1)

            self.h_empt = bh.TH1F(f'{name}_empt',
                                'EM pt (GeV); p_{T}^{EM} [GeV]', 100, 0, 500)

            self.h_srrTot = bh.TH1F(f'{name}_srrTot',
                                'srrTot; srrTot;', 100, 0.001, 0.1)
            self.h_hwSrrTot = bh.TH1F(f'{name}_hwSrrTot',
                                    'hwSrrTot; hwSrrTot;', 100, 0.1, 5)
            self.h_meanz = bh.TH1F(f'{name}_meanz',
                                'meanz; meanz;', 100, 300, 400)
            self.h_hwMeanZ = bh.TH1F(f'{name}_hwMeanZ',
                                    'hwMeanZ; hwMeanZ;', 50, 0, 50)
            self.h_hoe = bh.TH1F(f'{name}_hoe',
                                'hoe; hoe;', 50, 0, 2)

            self.h_clRelIso = bh.TH1F(f'{name}_clRelIso',
                                    'clRelIso; clRelIso;', 32, 0, 16)
            self.h_clRelIsoHack = bh.TH1F(f'{name}_clRelIsoHack',
                                    'clRelIso; clRelIso;', 32, 0, 16)
            self.h_clRelIsoOld = bh.TH1F(f'{name}_clRelIsoOld',
                                    'clRelIso; clRelIso;', 32, 0, 16)


            self.h_hwEmID = bh.TH1F(f'{name}_hwEmID',
                                    'hwEmID; hwEmID;', 10, 0, 10)
            # self.h_hwEmIDBits = bh.TH1F_category(f'{name}_hwEmIDBits',
            #                         'hwEmID bits; hwEmID;', ['bit 0', 'bit 1: STA', 'bit 2: Ele', 'bit 3: pho', 'bit 4', 'bit 5'])
            self.h_hwEmIDBits = bh.TH1F(f'{name}_hwEmIDBits',
                                    'hwEmID bits; hwEmID;', 6, 0, 6)
            
            self.h_clShowerShape = bh.TH1F_binned(f'{name}_clShowerShape',
                                        'clShowerShape; clShowerShape;', 6, 0, 1)

            self.h_showerShape = bh.TH1F_binned(f'{name}_showerShape',
                                        'showerShape; showerShape;', 6, 0, 1)
            self.h_hwShowerShape = bh.TH1F_binned(f'{name}_hwShowerShape',
                                        'hwShowerShape; hwShowerShape;', 6, 0, 1)

            self.h_showerShapeResoEmu = bh.TH1F(f'{name}_showerShapeResoEmu',
                                        'showerShape; showerShape^(float) - showershape^{EMU};', 128, -1, 1)
            
            self.h_hwRelIso = bh.TH1F_binned(f'{name}_hwRelIso',
                                    'hwRelIso; hwRelIso;', 6, 0, 1)
            self.h_relIso = bh.TH1F(f'{name}_relIso',
                                    'relIso; relIso;', 32, 0, 16)


            self.h_clShowerlength = bh.TH1F(f'{name}_clShowerlength',
                                          'clShowerlength; clShowerlength;', 50, 0, 50)
            self.h_clCoreshowerlength = bh.TH1F(f'{name}_clCoreshowerlength',
                                              'clCoreshowerlength; clCoreshowerlength;', 50, 0, 50)
            self.h_clEmf = bh.TH1F(f'{name}_clEmf',
                                  'clEmf; clEmf;', 100, 0, 300)
            self.h_clHwEmf = bh.TH1F(f'{name}_clHwEmf',
                                    'clHwEmf; clHwEmf;', 100, 0, 300)
            self.h_clAbseta = bh.TH1F(f'{name}_clAbseta',
                                     'clAbseta; clAbseta;', 100, 0, 5)
            self.h_clHwAbseta = bh.TH1F(f'{name}_clHwAbseta',
                                       'clHwAbseta; clHwAbseta;', 100, 0, 500)
            self.h_clHwMeanz = bh.TH1F(f'{name}_clHwMeanz',
                                       'clHwMeanz; clHwMeanz;', 100, 0, 200)
            self.h_clSigmaetaeta = bh.TH1F(f'{name}_clSigmaetaeta',
                                          'clSigmaetaeta; clSigmaetaeta;', 100, 0, 0.1)
            self.h_clHwSigmaetaeta = bh.TH1F(f'{name}_clHwSigmaetaeta',
                                            'clHwSigmaetaeta; clHwSigmaetaeta;', 24, 0, 24)
            self.h_clSigmaphiphi = bh.TH1F(f'{name}_clSigmaphiphi',
                                          'clSigmaphiphi; clSigmaphiphi;', 100, 0, 0.1)
            self.h_clHwSigmaphiphi = bh.TH1F(f'{name}_clHwSigmaphiphi',
                                            'clHwSigmaphiphi; clHwSigmaphiphi;', 100, 0, 100)
            self.h_clSigmazz = bh.TH1F(f'{name}_clSigmazz',
                                     'clSigmazz; clSigmazz;', 100, 0, 100)
            self.h_clHwSigmazz = bh.TH1F(f'{name}_clHwSigmazz',
                                        'clHwSigmazz; clHwSigmazz;', 100, 0, 100)



        histos.BaseHistos.__init__(self, name, root_file, debug)


    def fill(self, objs):
        bh.fill_1Dhist(self.h_pt, objs.pt)
        bh.fill_1Dhist(self.h_eta, objs.eta)
        bh.fill_1Dhist(self.h_phi, objs.phi)
        bh.fill_1Dhist(self.h_hwQual, objs.hwQual)

        if 'clPt' in objs.fields:
            bh.fill_1Dhist(self.h_clPt, objs.clPt)
            bh.fill_1Dhist(self.h_clEta, objs.clEta)
            bh.fill_1Dhist(self.h_clPhi, objs.clPhi)        
            bh.fill_1Dhist(self.h_ptResoEmu, objs.clPt - objs.pt)
            bh.fill_1Dhist(self.h_etaResoEmu, objs.clEta - objs.eta)
            bh.fill_1Dhist(self.h_phiResoEmu, objs.clPhi - objs.phi)

        # bh.fill_1Dhist(self.h_puIdProb, tracks.PuIdProb)
        # bh.fill_1Dhist(self.h_piIdProb, tracks.piIdProb)
        # bh.fill_1Dhist(self.h_emIdProb, tracks.EmIdProb)
        bh.fill_2Dhist(self.h_puIdProbVeta, np.abs(objs.eta), objs.PuIdProb)
        bh.fill_2Dhist(self.h_puIdProbVpt, objs.pt, objs.PuIdProb)

        if 'empt' in objs.fields:
            bh.fill_1Dhist(self.h_empt, objs.empt)
        if 'srrTot' in objs.fields:
            bh.fill_1Dhist(self.h_srrTot, objs.srrTot)
        if 'hwSrrTot' in objs.fields:
            bh.fill_1Dhist(self.h_hwSrrTot, objs.hwSrrTot)
        if 'meanz' in objs.fields:
            bh.fill_1Dhist(self.h_meanz, objs.meanz)
        if 'hwMeanZ' in objs.fields:
            bh.fill_1Dhist(self.h_hwMeanZ, objs.hwMeanZ)
        if 'hoe' in objs.fields:
            bh.fill_1Dhist(self.h_hoe, objs.hoe)
        if 'piIdProb' in objs.fields:
            bh.fill_1Dhist(self.h_piIdProb, objs.piIdProb)
        if 'PuIdProb' in objs.fields:
            bh.fill_1Dhist(self.h_puIdProb, objs.PuIdProb)
        if 'EmIdProb' in objs.fields:
            bh.fill_1Dhist(self.h_emIdProb, objs.EmIdProb)
        if 'clRelIso' in objs.fields:
            bh.fill_1Dhist(self.h_clRelIso, objs.clRelIso)
        if 'clRelIsoHack' in objs.fields:
            bh.fill_1Dhist(self.h_clRelIsoOld, objs.clRelIsoOld)
        if 'clRelIsoHack' in objs.fields:
            bh.fill_1Dhist(self.h_clRelIsoHack, objs.clRelIsoHack)
        if 'clShowerShape' in objs.fields:
            bh.fill_1Dhist(self.h_clShowerShape, objs.clShowerShape)
            bh.fill_1Dhist(self.h_showerShapeResoEmu, objs.clShowerShape - objs.showerShape)
        if 'hwEmID' in objs.fields:
            bh.fill_1Dhist(self.h_hwEmID, objs.hwEmID)
        if 'IDTightEle' in objs.fields:
            bh.fill_1Dhist(self.h_hwEmIDBits, objs.IDTightSTA*0 + ~objs.IDTightSTA*-1)            
            bh.fill_1Dhist(self.h_hwEmIDBits, objs.IDTightEle*1 + ~objs.IDTightEle*-1)
            bh.fill_1Dhist(self.h_hwEmIDBits, objs.IDTightPho*2 + ~objs.IDTightPho*-1)
            bh.fill_1Dhist(self.h_hwEmIDBits, ak.full_like(objs.IDTightEle, 10, dtype=int)) # to set the overflow bin to the total entries for normalization purposes

        if 'showerShape' in objs.fields:
            bh.fill_1Dhist(self.h_showerShape, objs.showerShape)
        if 'hwShowerShape' in objs.fields:
            bh.fill_1Dhist(self.h_hwShowerShape, objs.hwShowerShape)
        if 'hwRelIso' in objs.fields:
            bh.fill_1Dhist(self.h_hwRelIso, objs.hwRelIso)
        if 'relIso' in objs.fields:
            bh.fill_1Dhist(self.h_relIso, objs.relIso)
        if 'clShowerlength' in objs.fields:
            bh.fill_1Dhist(self.h_clShowerlength, objs.clShowerlength)
        if 'clCoreshowerlength' in objs.fields:
            bh.fill_1Dhist(self.h_clCoreshowerlength, objs.clCoreshowerlength)
        if 'clEmf' in objs.fields:
            bh.fill_1Dhist(self.h_clEmf, objs.clEmf)
        if 'clHwEmf' in objs.fields:
            bh.fill_1Dhist(self.h_clHwEmf, objs.clHwEmf)
        if 'clAbseta' in objs.fields:
            bh.fill_1Dhist(self.h_clAbseta, objs.clAbseta)
        if 'clHwAbseta' in objs.fields:
            bh.fill_1Dhist(self.h_clHwAbseta, objs.clHwAbseta)
        if 'clHwMeanz' in objs.fields:
            bh.fill_1Dhist(self.h_clHwMeanz, objs.clHwMeanz)
        if 'clSigmaetaeta' in objs.fields:
            bh.fill_1Dhist(self.h_clSigmaetaeta, objs.clSigmaetaeta)
        if 'clHwSigmaetaeta' in objs.fields:
            bh.fill_1Dhist(self.h_clHwSigmaetaeta, objs.clHwSigmaetaeta)
        if 'clSigmaphiphi' in objs.fields:
            bh.fill_1Dhist(self.h_clSigmaphiphi, objs.clSigmaphiphi)
        if 'clHwSigmaphiphi' in objs.fields:  
            bh.fill_1Dhist(self.h_clHwSigmaphiphi, objs.clHwSigmaphiphi)
        if 'clSigmazz' in objs.fields:
            bh.fill_1Dhist(self.h_clSigmazz, objs.clSigmazz)
        if 'clHwSigmazz' in objs.fields:
            bh.fill_1Dhist(self.h_clHwSigmazz, objs.clHwSigmazz)


# ------ Plotter classes ------------------------------------------------
class DecCaloGenMatchPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')],
                 gen_eta_phi_columns=('caloeta', 'calophi'),
                 pt_bins=None):
        super(DecCaloGenMatchPlotter, self).__init__(DecCaloHistos, DecResoHistos,
                                                data_set, gen_set,
                                                data_selections, gen_selections,
                                                gen_eta_phi_columns=gen_eta_phi_columns,
                                                pt_bins=pt_bins,
                                                drcut=0.1)


class DecCaloPlotter(plotters.GenericDataFramePlotter):
    def __init__(self, eg_set, eg_selections=[selections.Selection('all')]):
        super(DecCaloPlotter, self).__init__(DecCaloHistos, eg_set, eg_selections)


# ------ Plotter instances

gen_em_selections = (selections.Selector('^GEN$')*('^EtaE[EB]$|^EtaEEFwd$|^EtaFwd|all')+selections.Selector('^GEN$')*('Pt30'))()
gen_em_ee_selections = (selections.Selector('^GEN$')*('^EtaE[E]$|^EtaEEFwd$|^EtaFwd|all')+selections.Selector('^GEN$')*('Pt30'))()
gen_em_eb_selections = (selections.Selector('^GEN$')*('^EtaEB[abc]$|^EtaEB$')+selections.Selector('^GEN$')*('Pt30'))()

gen_pi_selections = (selections.Selector('^GENPiAll$|GENPi$')*('^EtaE[EB]$|^EtaEEFwd$|^EtaFwd|all')+selections.Selector('GENPi$')*('Pt30'))()
gen_pi_ee_selections = (selections.Selector('^GENPi$')*('^EtaE[E]$|^EtaEEFwd$|^EtaFwd|all')+selections.Selector('GENPi$')*('Pt30'))()
gen_pi_eb_selections = (selections.Selector('^GENPi$')*('^EtaEB[abc]$|^EtaEB$|^EtaPos$|^EtaNeg$|all')+selections.Selector('GENPi$')*('Pt30'))()

pf_selections = (selections.Selector('^PFType[CNEPH]$|all$'))()

decHad_ee_selections = (selections.Selector('^IDHgc|all$')*('^Pt[5]$'))()
decHad_eb_selections = (selections.Selector('^Pt[5]$|all$'))()


decEm_selections = (selections.Selector('^Pt[5]$|all$'))()

# pf = [
#     PfGenMatchPlotter(
#         coll.pf_cands, coll.gen_pi,
#         pf_selections, gen_pi_selections),
#     PfGenMatchPlotter(
#         coll.pf_cands, coll.gen,
#         pf_selections, gen_em_selections),
    
# ]

decoded = [
    # DecCaloGenMatchPlotter(
    #     coll.decHadCaloEndcap, coll.gen_pi,
    #     decHad_ee_selections, gen_pi_ee_selections),
    # DecCaloGenMatchPlotter(
    #     coll.decHadCaloEndcap, coll.gen,
    #     decHad_ee_selections, gen_em_ee_selections),

    # DecCaloGenMatchPlotter(
    #     coll.decHadCaloBarrel, coll.gen_pi,
    #     decHad_eb_selections, gen_pi_eb_selections),
    # DecCaloGenMatchPlotter(
    #     coll.decEmCaloBarrel, coll.gen,
    #     decEm_selections, gen_em_eb_selections),

    DecCaloPlotter(
        coll.decHadCaloEndcap, decHad_ee_selections),
    DecCaloPlotter(
        coll.decEmCaloBarrel, decEm_selections),        
    DecCaloPlotter(
        coll.decHadCaloBarrel, decHad_eb_selections),        

]


decoded_pi = [
    DecCaloGenMatchPlotter(
        coll.decHadCaloEndcap, coll.gen_pi,
        decHad_ee_selections, gen_pi_ee_selections),
    # DecCaloGenMatchPlotter(
    #     coll.decHadCaloEndcap, coll.gen,
    #     decHad_ee_selections, gen_em_ee_selections),
    DecCaloGenMatchPlotter(
        coll.decHadCaloBarrel, coll.gen_pi,
        decHad_eb_selections, gen_pi_eb_selections),
    DecCaloPlotter(
        coll.decHadCaloEndcap, decHad_ee_selections),
    DecCaloPlotter(
        coll.decHadCaloBarrel, decHad_eb_selections),        

]

decoded_em = [
    DecCaloGenMatchPlotter(
        coll.decHadCaloEndcap, coll.gen,
        decHad_ee_selections, gen_em_ee_selections),
    DecCaloGenMatchPlotter(
        coll.decEmCaloBarrel, coll.gen,
        decEm_selections, gen_em_eb_selections),
    DecCaloPlotter(
        coll.decHadCaloEndcap, decHad_ee_selections),
    DecCaloPlotter(
        coll.decEmCaloBarrel, decEm_selections),        

]
