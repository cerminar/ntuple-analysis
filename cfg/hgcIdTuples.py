from python import plotters, selections, calibrations, histos
import cfg.datasets.fastpuppi_collections as coll
import python.boost_hist as bh

import awkward as ak

class DecodedHadHistos(histos.BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            # self.h_npt05 = bh.TH1F(
            #     name+'_npt05', '# 3D Cluster Pt > 0.5 GeV; # 3D clusters in cone;', 1000, 0, 1000)
            # self.h_npt20 = bh.TH1F(
            #     name+'_npt20', '# 3D Cluster Pt > 2.0 GeV; # 3D clusters in cone;', 1000, 0, 1000)
            self.h_pt = bh.TH1F(
                f'{name}_pt', 'Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(f'{name}_eta', 'eta; #eta;', 100, -4, 4)
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

class HGCIdMatchTuples(histos.BaseUpTuples):
    def __init__(self, name, root_file=None, debug=False):
        histos.BaseUpTuples.__init__(
            self, "CompCatData", name, root_file, debug)

    def fill(self, reference, target):
        # print(self.t_name)
        # print(target.fields)
        target_vars = ['hwEta', 'hwPhi', 'hwQual', 'eta', 'phi', 'pt', 'empt', 'srrTot',
       'hwSrrTot', 'meanz', 'hwMeanZ', 'hoe', 'piIdProb', 'PuIdProb',
       'EmIdProb', 'caloIso', 'showerShape', 'showerlength',
       'coreshowerlength', 'emf', 'hwEmf', 'abseta', 'hwAbseta', 'hwFPMeanz',
       'sigmaetaeta', 'hwSigmaetaeta', 'sigmaphiphi', 'hwSigmaphiphi',
       'sigmazz', 'hwSigmazz']
        reference_vars = [
            'pt',
            'eta',
            'phi',
            'pdgid',
            'caloeta', 
            'calophi']
        tree_data = {}
        for var in target_vars:
            tree_data[var] = ak.flatten(ak.drop_none(target[var]))
        for var in reference_vars:
            tree_data[f'gen_{var}'] = ak.flatten(ak.drop_none(reference[var]))
        # print(reference.fields)
        # tree_data[f'gen_dz'] = ak.flatten(ak.drop_none(np.abs(reference.ovz-target.tkZ0)))
        
        histos.BaseUpTuples.fill(self, tree_data)


class HGCIdTuples(histos.BaseUpTuples):
    def __init__(self, name, root_file=None, debug=False):
        histos.BaseUpTuples.__init__(
            self, "CompData", name, root_file, debug)

    def fill(self, data):
        vars = ['hwEta', 'hwPhi', 'hwQual', 'eta', 'phi', 'pt', 'empt', 'srrTot',
       'hwSrrTot', 'meanz', 'hwMeanZ', 'hoe', 'piIdProb', 'PuIdProb',
       'EmIdProb', 'caloIso', 'showerShape', 'showerlength',
       'coreshowerlength', 'emf', 'hwEmf', 'abseta', 'hwAbseta', 'hwFPMeanz',
       'sigmaetaeta', 'hwSigmaetaeta', 'sigmaphiphi', 'hwSigmaphiphi',
       'sigmazz', 'hwSigmazz']
        tree_data = {}
        for var in vars:
            if var in data.fields:
                tree_data[var] = ak.flatten(ak.drop_none(data[var]))
        histos.BaseUpTuples.fill(self, tree_data)





class HGCIdMatchTuplesPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')]):
        super(HGCIdMatchTuplesPlotter, self).__init__(DecodedHadHistos, HGCIdMatchTuples,
                                                data_set, gen_set,
                                                data_selections, gen_selections, 
                                                drcut=0.1)

class HGCIdTuplesPlotter(plotters.GenericDataFramePlotter):
    def __init__(self, obj_set, obj_selections=[selections.Selection('all')]):
        super(HGCIdTuplesPlotter, self).__init__(HGCIdTuples, obj_set, obj_selections)


comp_selections = (selections.Selector('all')&('^EtaEE$|all'))()
sim_eg_selections = (selections.Selector('^GEN$'))()
sim_pi_selections = (selections.Selector('^GENPi$'))()


egid_plotters = [
    HGCIdMatchTuplesPlotter(coll.decHadCaloEndcap, coll.gen, comp_selections, sim_eg_selections)
]

piid_plotters = [
    # plotters.HGCIdTuplesPlotter(collections.hgc_cl3d, comp_selections),
    HGCIdMatchTuplesPlotter(coll.decHadCaloEndcap, coll.gen_pi, comp_selections, sim_pi_selections)
]

pu_plotters = [
    HGCIdTuplesPlotter(coll.decHadCaloEndcap, comp_selections),
]


# for sel in sim_selections:
#     print(sel)