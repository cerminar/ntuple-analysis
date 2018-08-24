
import ROOT
import root_numpy as rnp
import numpy as np
import pandas as pd


class HistoManager(object):
    class __TheManager:
        def __init__(self):
            self.val = None
            self.histoList = list()

        def __str__(self):
            return `self` + self.val

        def addHistos(self, histo):
            # print 'ADD histo: {}'.format(histo)
            self.histoList.append(histo)

        def writeHistos(self):
            for histo in self.histoList:
                histo.write()

    instance = None

    def __new__(cls):
        if not HistoManager.instance:
            HistoManager.instance = HistoManager.__TheManager()
        return HistoManager.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


class BaseHistos():
    def __init__(self, name, root_file=None):
        self.name_ = name
        # print 'BOOK histo: {}'.format(self)
        if root_file is not None:
            root_file.cd()
            histo_names = [histo.GetName() for histo in root_file.GetListOfKeys() if name+'_' in histo.GetName()]
            # print histo_names
            for histo_name in histo_names:
                hinst = root_file.Get(histo_name)
                attr_name = 'h_'+histo_name.split(name+'_')[1]
                setattr(self, attr_name, hinst)
#            self.h_test = root_file.Get('h_EleReso_ptRes')
            # print 'XXXXXX'+str(self.h_test)
        else:
            for histo in [a for a in dir(self) if a.startswith('h_')]:
                getattr(self, histo).Sumw2()
            hm = HistoManager()
            hm.addHistos(self)

    def write(self):
        for histo in [a for a in dir(self) if a.startswith('h_')]:
            getattr(self, histo).Write()

    def annotateTitles(self, annotation):
        for histo_name in [a for a in dir(self) if a.startswith('h_')]:
            histo = getattr(self, histo_name)
            histo.SetTitle('{} ({})'.format(histo.GetTitle(), annotation))
    # def normalize(self, norm):
    #     className = self.__class__.__name__
    #     ret = className()
    #     return ret

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, self.name_)


class GenPartHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        self.h_pt = ROOT.TH1F(name+'_pt', 'Gen Part Pt (GeV)', 100, 0, 100)
        self.h_energy = ROOT.TH1F(name+'_energy', 'Gen Part Energy (GeV)', 100, 0, 1000)
        BaseHistos.__init__(self, name, root_file)

    def fill(self, gps):
        rnp.fill_hist(self.h_pt, gps.pt)
        rnp.fill_hist(self.h_energy, gps.energy)

    def write(self):
        for histo in [a for a in dir(self) if a.startswith('h_')]:
            getattr(self, histo).Write()


class GenParticleHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_eta = ROOT.TH1F(name+'_eta', 'Gen Part eta; #eta;', 50, -3, 3)
            self.h_pt = ROOT.TH1F(name+'_pt', 'Gen Part P_{T} (GeV); p_{T} [GeV];', 50, 0, 100)
            self.h_energy = ROOT.TH1F(name+'_energy', 'Gen Part Energy (GeV); E [GeV];', 100, 0, 1000)
            self.h_reachedEE = ROOT.TH1F(name+'_reachedEE', 'Gen Part reachedEE', 4, 0, 4)
            self.h_fBrem = ROOT.TH1F(name+'_fBrem', 'Brem. p_{T} fraction', 30, -1, 1)
        BaseHistos.__init__(self, name, root_file)

    def fill(self, particles):
        rnp.fill_hist(self.h_eta, particles.eta)
        rnp.fill_hist(self.h_pt, particles.pt)
        rnp.fill_hist(self.h_energy, particles.energy)
        rnp.fill_hist(self.h_reachedEE, particles.reachedEE)
        rnp.fill_hist(self.h_fBrem, particles.fbrem)


class DigiHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_layer = ROOT.TH1F(name+'_layer', 'Digi layer #', 60, 0, 60)
            # self.h_simenergy = ROOT.TH1F(name+'_energy', 'Digi sim-energy (GeV)', 100, 0, 2)
        BaseHistos.__init__(self, name, root_file)

    def fill(self, digis):
        rnp.fill_hist(self.h_layer, digis.layer)
        # rnp.fill_hist(self.h_simenergy, digis.simenergy)


class RateHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_norm = ROOT.TH1F(name+'_norm', '# of events', 1, 1, 2)
            self.h_pt = ROOT.TH1F(name+'_pt', '# events; p_{T} [GeV];', 100, 0, 100)
            # self.h_simenergy = ROOT.TH1F(name+'_energy', 'Digi sim-energy (GeV)', 100, 0, 2)
        BaseHistos.__init__(self, name, root_file)

    def fill(self, pt):
        for ptf in range(0, int(pt)+1):
            self.h_pt.Fill(ptf)

    def fill_norm(self):
        self.h_norm.Fill(1)


class TCHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_energy = ROOT.TH1F(name+'_energy', 'TC energy (GeV)', 100, 0, 2)
            self.h_subdet = ROOT.TH1F(name+'_subdet', 'TC subdet #', 8, 0, 8)
            self.h_mipPt = ROOT.TH1F(name+'_mipPt', 'TC MIP Pt', 50, 0, 10)

            self.h_layer = ROOT.TProfile(name+'_layer', 'TC layer #', 60, 0, 60, 's')
            self.h_absz = ROOT.TH1F(name+'_absz', 'TC z(cm)', 100, 300, 500)
            self.h_wafertype = ROOT.TH1F(name+'_wafertype', 'Wafer type', 10, 0, 10)
            self.h_layerVenergy = ROOT.TH2F(name+'_layerVenergy', "Energy (GeV) vs Layer #", 60, 0, 60, 100, 0, 2)
            self.h_energyVeta = ROOT.TH2F(name+'_energyVeta', "Energy (GeV) vs Eta", 100, -3.5, 3.5, 100, 0, 2)
            self.h_energyVetaL1t5 = ROOT.TH2F(name+'_energyVetaL1t5', "Energy (GeV) vs Eta (layers 1 to 5)", 100, -3.5, 3.5, 100, 0, 2)
            self.h_energyVetaL6t10 = ROOT.TH2F(name+'_energyVetaL6t10', "Energy (GeV) vs Eta (layers 6 to 10)", 100, -3.5, 3.5, 100, 0, 2)
            self.h_energyVetaL11t20 = ROOT.TH2F(name+'_energyVetaL11t20', "Energy (GeV) vs Eta (layers 11 to 20)", 100, -3.5, 3.5, 100, 0, 2)
            self.h_energyVetaL21t60 = ROOT.TH2F(name+'_energyVetaL21t60', "Energy (GeV) vs Eta (layers 21 to 60)", 100, -3.5, 3.5, 100, 0, 2)
            self.h_energyPetaVphi = ROOT.TProfile2D(name+'_energyPetaVphi', "Energy profile (GeV) vs Eta and Phi", 100, -3.5, 3.5, 100, -3.2, 3.2)

        BaseHistos.__init__(self, name, root_file)

    def fill(self, tcs):
        rnp.fill_hist(self.h_energy, tcs.energy)
        rnp.fill_hist(self.h_subdet, tcs.subdet)
        rnp.fill_hist(self.h_mipPt, tcs.mipPt)
        cnt = tcs.layer.value_counts().to_frame(name='counts')
        cnt['layer'] = cnt.index.values
        rnp.fill_profile(self.h_layer, cnt[['layer', 'counts']])
        rnp.fill_hist(self.h_absz, np.fabs(tcs.z))
        rnp.fill_hist(self.h_wafertype, tcs.wafertype)
        rnp.fill_hist(self.h_wafertype, tcs.wafertype)
        # FIXME: should bin this guy in eta bins
        rnp.fill_hist(self.h_layerVenergy, tcs[['layer', 'energy']])
        rnp.fill_hist(self.h_energyVeta, tcs[['eta', 'energy']])
        rnp.fill_hist(self.h_energyVeta, tcs[['eta', 'energy']])
        rnp.fill_hist(self.h_energyVetaL1t5, tcs[(tcs.layer >= 1) & (tcs.layer <= 5)][['eta', 'energy']])
        rnp.fill_hist(self.h_energyVetaL6t10, tcs[(tcs.layer >= 6) & (tcs.layer <= 10)][['eta', 'energy']])
        rnp.fill_hist(self.h_energyVetaL11t20, tcs[(tcs.layer >= 11) & (tcs.layer <= 20)][['eta', 'energy']])
        rnp.fill_hist(self.h_energyVetaL21t60, tcs[(tcs.layer >= 21) & (tcs.layer <= 60)][['eta', 'energy']])
        rnp.fill_profile(self.h_energyPetaVphi, tcs[['eta', 'phi', 'energy']])


class ClusterHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_energy = ROOT.TH1F(name+'_energy', 'Cluster energy (GeV); E [GeV];', 100, 0, 30)
            self.h_layer = ROOT.TH1F(name+'_layer', 'Cluster layer #; layer #;', 60, 0, 60)
            self.h_ncells = ROOT.TH1F(name+'_ncells', 'Cluster # cells; # TC components;', 30, 0, 30)
            self.h_nCoreCells = ROOT.TH1F(name+'_nCoreCells', 'Cluster # cells (core)', 30, 0, 30)

            self.h_layerVenergy = ROOT.TH2F(name+'_layerVenergy', "Cluster Energy (GeV) vs Layer #; layer; E [GeV];", 50, 0, 50, 100, 0, 20)
            self.h_layerVncells = ROOT.TH2F(name+'_layerVncells', "Cluster #cells vs Layer #; layer; # TC components;",  50, 0, 50, 30, 0, 30)
            self.h_layerVnCoreCells = ROOT.TH2F(name+'_layerVnCoreCells', "Cluster #cells vs Layer #",  50, 0, 50, 30, 0, 30)

        BaseHistos.__init__(self, name, root_file)

    def fill(self, clsts):
        rnp.fill_hist(self.h_energy, clsts.energy)
        rnp.fill_hist(self.h_layer, clsts.layer)
        rnp.fill_hist(self.h_ncells, clsts.ncells)
        rnp.fill_hist(self.h_layerVenergy, clsts[['layer', 'energy']])
        rnp.fill_hist(self.h_layerVncells, clsts[['layer', 'ncells']])
        if 'nCoreCells' in clsts.columns:
            rnp.fill_hist(self.h_nCoreCells, clsts.nCoreCells)
            rnp.fill_hist(self.h_layerVnCoreCells, clsts[['layer', 'nCoreCells']])


class Cluster3DHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_npt05 = ROOT.TH1F(name+'_npt05', '# 3D Cluster Pt > 0.5 GeV; # 3D clusters in cone;', 1000, 0, 1000)
            self.h_npt20 = ROOT.TH1F(name+'_npt20', '# 3D Cluster Pt > 2.0 GeV; # 3D clusters in cone;', 1000, 0, 1000)
            self.h_pt = ROOT.TH1F(name+'_pt', '3D Cluster Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = ROOT.TH1F(name+'_eta', '3D Cluster eta; #eta;', 100, -4, 4)
            self.h_energy = ROOT.TH1F(name+'_energy', '3D Cluster energy (GeV); E [GeV]', 1000, 0, 1000)
            self.h_nclu = ROOT.TH1F(name+'_nclu', '3D Cluster # clusters; # 2D components;', 60, 0, 60)
            self.h_ncluVpt = ROOT.TH2F(name+'_ncluVpt', '3D Cluster # clusters vs pt; # 2D components; p_{T} [GeV]', 60, 0, 60, 100, 0, 100)
            self.h_showlenght = ROOT.TH1F(name+'_showlenght', '3D Cluster showerlenght', 60, 0, 60)
            self.h_firstlayer = ROOT.TH1F(name+'_firstlayer', '3D Cluster first layer', 30, 0, 30)
            self.h_sEtaEtaTot = ROOT.TH1F(name+'_sEtaEtaTot', '3D Cluster RMS Eta', 100, 0, 0.1)
            self.h_sEtaEtaMax = ROOT.TH1F(name+'_sEtaEtaMax', '3D Cluster RMS Eta (max)', 100, 0, 0.1)
            self.h_sPhiPhiTot = ROOT.TH1F(name+'_sPhiPhiTot', '3D Cluster RMS Phi', 100, 0, 2)
            self.h_sPhiPhiMax = ROOT.TH1F(name+'_sPhiPhiMax', '3D Cluster RMS Phi (max)', 100, 0, 2)
            self.h_sZZ = ROOT.TH1F(name+'_sZZ', '3D Cluster RMS Z ???', 100, 0, 10)
            self.h_eMaxOverE = ROOT.TH1F(name+'_eMaxOverE', '3D Cluster Emax/E', 100, 0, 1)
            self.h_iso0p2 = ROOT.TH1F(name+'_iso0p2', '3D Cluster iso DR 0.2(GeV); Iso p_{T} [GeV];', 100, 0, 100)
            self.h_isoRel0p2 = ROOT.TH1F(name+'_isoRel0p2', '3D Cluster relative iso DR 0.2; Rel. Iso;', 100, 0, 1)

        BaseHistos.__init__(self, name, root_file)

    def fill(self, cl3ds):
        self.h_npt05.Fill(len(cl3ds[cl3ds.pt > 0.5].index))
        self.h_npt20.Fill(len(cl3ds[cl3ds.pt > 2.0].index))

        rnp.fill_hist(self.h_pt, cl3ds.pt)
        rnp.fill_hist(self.h_eta, cl3ds.eta)
        rnp.fill_hist(self.h_energy, cl3ds.energy)
        rnp.fill_hist(self.h_nclu, cl3ds.nclu)
        rnp.fill_hist(self.h_ncluVpt, cl3ds[['nclu', 'pt']])
        rnp.fill_hist(self.h_showlenght, cl3ds.showerlength)
        rnp.fill_hist(self.h_firstlayer, cl3ds.firstlayer)
        rnp.fill_hist(self.h_sEtaEtaTot, cl3ds.seetot)
        rnp.fill_hist(self.h_sEtaEtaMax, cl3ds.seemax)
        rnp.fill_hist(self.h_sPhiPhiTot, cl3ds.spptot)
        rnp.fill_hist(self.h_sPhiPhiMax, cl3ds.sppmax)
        rnp.fill_hist(self.h_sZZ, cl3ds.szz)
        rnp.fill_hist(self.h_eMaxOverE, cl3ds.emaxe)
        if 'iso0p2' in cl3ds.columns:
            rnp.fill_hist(self.h_iso0p2, cl3ds.iso0p2)
            rnp.fill_hist(self.h_isoRel0p2, cl3ds.isoRel0p2)


class TriggerTowerHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_pt = ROOT.TH1F(name+'_pt', 'Tower Pt (GeV); p_{T} [GeV];', 100, 0, 100)
            self.h_etEm = ROOT.TH1F(name+'_etEm', 'Tower Et EM (GeV)', 100, 0, 100)
            self.h_etHad = ROOT.TH1F(name+'_etHad', 'Tower Et Had (GeV)', 100, 0, 100)
            self.h_HoE = ROOT.TH1F(name+'_HoE', 'Tower H/E', 20, 0, 2)
            self.h_HoEVpt = ROOT.TH2F(name+'_HoEVpt', 'Tower H/E vs Pt (GeV); H/E;', 100, 0, 100, 20, 0, 2)
            self.h_energy = ROOT.TH1F(name+'_energy', 'Tower energy (GeV)', 1000, 0, 1000)
            self.h_eta = ROOT.TH1F(name+'_eta', 'Tower eta; #eta;', 75, -3.169, 3.169)
            self.h_ptVeta = ROOT.TH2F(name+'_ptVeta', 'Tower P_P{T} (GeV) vs #eta; #eta; p_{T} [GeV];',  75, -3.169, 3.169, 100, 0, 100)

        BaseHistos.__init__(self, name, root_file)

    def fill(self, towers):
        rnp.fill_hist(self.h_pt, towers.pt)
        rnp.fill_hist(self.h_etEm, towers.etEm)
        rnp.fill_hist(self.h_etHad, towers.etHad)
        rnp.fill_hist(self.h_HoE, towers.HoE)
        rnp.fill_hist(self.h_HoEVpt, towers[['pt', 'HoE']])
        rnp.fill_hist(self.h_energy, towers.energy)
        rnp.fill_hist(self.h_eta, towers.eta)
        rnp.fill_hist(self.h_ptVeta, towers[['eta', 'pt']])


class TriggerTowerResoHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_ptRes = ROOT.TH1F(name+'_ptRes', 'TT Pt reso (GeV); p_{T}^{RECO}-p_{T}^{GEN} [GeV];', 200, -40, 40)

            self.h_ptResVpt = ROOT.TH2F(name+'_ptResVpt', 'TT Pt reso (GeV) vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{RECO}-p_{T}^{GEN} [GeV];', 50, 0, 100, 200, -40, 40)
            self.h_ptResVeta = ROOT.TH2F(name+'_ptResVeta', 'TT Pt reso (GeV) vs eta; #eta^{GEN}; p_{T}^{RECO}-p_{T}^{GEN} [GeV];', 100, -3.5, 3.5, 200, -40, 40)

            self.h_ptResp = ROOT.TH1F(name+'_ptResp', 'TT Pt resp.; p_{T}^{RECO}/p_{T}^{GEN};', 100, 0, 2)
            self.h_ptRespVpt = ROOT.TH2F(name+'_ptRespVpt', 'TT Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{RECO}/p_{T}^{GEN};', 50, 0, 100, 100, 0, 2)
            self.h_ptRespVeta = ROOT.TH2F(name+'_ptRespVeta', 'TT Pt resp. vs eta; #eta^{GEN}; p_{T}^{RECO}/p_{T}^{GEN};', 100, -3.5, 3.5, 100, 0, 2)

            self.h_energyRes = ROOT.TH1F(name+'_energyRes', 'TT Energy reso (GeV)', 200, -100, 100)
            self.h_energyResVeta = ROOT.TH2F(name+'_energyResVeta', 'TT E reso (GeV) vs eta', 100, -3.5, 3.5, 200, -100, 100)
            # FIXME: add corresponding Pt plots
            self.h_etaRes = ROOT.TH1F(name+'_etaRes', 'TT eta reso', 100, -0.4, 0.4)
            self.h_phiRes = ROOT.TH1F(name+'_phiRes', 'TT phi reso', 100, -0.4, 0.4)
            self.h_etalwRes = ROOT.TH1F(name+'_etalwRes', 'TT eta reso (lw)', 100, -0.4, 0.4)
            self.h_philwRes = ROOT.TH1F(name+'_philwRes', 'TT phi reso (lw)', 100, -0.4, 0.4)

            self.h_drRes = ROOT.TH1F(name+'_drRes', 'TT DR reso', 100, 0, 0.4)
        BaseHistos.__init__(self, name, root_file)

    def fill(self, reference, target):
        self.h_ptRes.Fill(target.pt - reference.pt)
        self.h_ptResVpt.Fill(reference.pt, target.pt - reference.pt)
        self.h_ptResVeta.Fill(reference.eta, target.pt - reference.pt)

        self.h_ptResp.Fill(target.pt/reference.pt)
        self.h_ptRespVpt.Fill(reference.pt, target.pt/reference.pt)
        self.h_ptRespVeta.Fill(reference.eta, target.pt/reference.pt)

        self.h_energyRes.Fill(target.energy - reference.energy)
        self.h_energyResVeta.Fill(reference.eta, target.energy - reference.energy)

        self.h_etaRes.Fill(target.eta - reference.eta)
        self.h_phiRes.Fill(target.phi - reference.phi)
        self.h_drRes.Fill(np.sqrt((reference.phi-target.phi)**2+(reference.eta-target.eta)**2))
        if 'etalw' in target:
            self.h_etalwRes.Fill(target.etalw - reference.eta)
        if 'philw' in target:
            self.h_philwRes.Fill(target.philw - reference.phi)


class ResoHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_ptRes = ROOT.TH1F(name+'_ptRes', '3D Cluster Pt reso (GeV); p_{T}^{RECO} - p_{T}^{GEN} [GeV]', 200, -40, 40)
            self.h_energyRes = ROOT.TH1F(name+'_energyRes', '3D Cluster Energy reso (GeV); E^{RECO} - E^{GEN} [GeV]', 200, -100, 100)
            self.h_ptResVeta = ROOT.TH2F(name+'_ptResVeta', '3D Cluster Pt reso (GeV) vs eta; #eta^{GEN}; p_{T}^{RECO} - p_{T}^{GEN} [GeV];', 100, -3.5, 3.5, 200, -40, 40)
            self.h_energyResVeta = ROOT.TH2F(name+'_energyResVeta', '3D Cluster E reso (GeV) vs eta; #eta^{GEN}; E^{RECO} - E^{GEN} [GeV];', 100, -3.5, 3.5, 200, -100, 100)
            self.h_energyResVnclu = ROOT.TH2F(name+'_energyResVnclu', '3D Cluster E reso (GeV) vs # clusters; # 2D clus.; E^{RECO} - E^{GEN} [GeV];', 50, 0, 50, 200, -100, 100)
            self.h_ptResVpt = ROOT.TH2F(name+'_ptResVpt', '3D Cluster Pt reso (GeV) vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{RECO} - p_{T}^{GEN} [GeV];', 50, 0, 100, 200, -40, 40)
            self.h_ptResVnclu = ROOT.TH2F(name+'_ptResVnclu', '3D Cluster Pt reso (GeV) vs # clusters; # 2D clus.; p_{T}^{RECO} - p_{T}^{GEN} [GeV];', 50, 0, 50, 200, -40, 40)

            self.h_ptResp = ROOT.TH1F(name+'_ptResp', '3D Cluster Pt resp.; p_{T}^{RECO}/p_{T}^{GEN}', 100, 0, 3)
            self.h_ptRespVpt = ROOT.TH2F(name+'_ptRespVpt', '3D Cluster Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{RECO}/p_{T}^{GEN};', 50, 0, 100, 100, 0, 3)
            self.h_ptRespVeta = ROOT.TH2F(name+'_ptRespVeta', '3D Cluster Pt resp. vs #eta; #eta^{GEN}; p_{T}^{RECO}/p_{T}^{GEN};', 50, -4, 4, 100, 0, 3)
            self.h_ptRespVnclu = ROOT.TH2F(name+'_ptRespVnclu', '3D Cluster Pt resp. vs # clus.; # 2D clust. ; p_{T}^{RECO}/p_{T}^{GEN};', 50, 0, 100, 100, 0, 3)

            # FIXME: add corresponding Pt plots
            self.h_coreEnergyResVnclu = ROOT.TH2F(name+'_coreEnergyResVnclu', '3D Cluster E reso (GeV) vs # clusters', 50, 0, 50, 200, -100, 100)
            self.h_corePtResVnclu = ROOT.TH2F(name+'_corePtResVnclu', '3D Cluster Pt reso (GeV) vs # clusters', 50, 0, 50, 200, -40, 40)

            self.h_coreEnergyRes = ROOT.TH1F(name+'_coreEnergyRes', '3D Cluster Energy reso CORE (GeV)', 200, -100, 100)
            self.h_corePtRes = ROOT.TH1F(name+'_corePtRes', '3D Cluster Pt reso CORE (GeV)', 200, -40, 40)

            self.h_centralEnergyRes = ROOT.TH1F(name+'_centralEnergyRes', '3D Cluster Energy reso CENTRAL (GeV)', 200, -100, 100)
            self.h_etaRes = ROOT.TH1F(name+'_etaRes', '3D Cluster eta reso', 100, -0.4, 0.4)
            self.h_phiRes = ROOT.TH1F(name+'_phiRes', '3D Cluster phi reso', 100, -0.4, 0.4)
            self.h_drRes = ROOT.TH1F(name+'_drRes', '3D Cluster DR reso', 100, 0, 0.4)
            self.h_n010 = ROOT.TH1F(name+'_n010', '# of 3D clus in 0.2 cone with pt>0.1GeV', 10, 0, 10)
            self.h_n025 = ROOT.TH1F(name+'_n025', '# of 3D clus in 0.2 cone with pt>0.25GeV', 10, 0, 10)

        BaseHistos.__init__(self, name, root_file)

    def fill(self, reference, target):
        self.h_ptRes.Fill(target.pt - reference.pt)
        self.h_energyRes.Fill(target.energy - reference.energy)
        self.h_ptResVeta.Fill(reference.eta, target.pt - reference.pt)
        self.h_ptResVpt.Fill(reference.pt, target.pt - reference.pt)
        self.h_energyResVeta.Fill(reference.eta, target.energy - reference.energy)
        self.h_energyResVnclu.Fill(target.nclu, target.energy - reference.energy)
        self.h_ptResVnclu.Fill(target.nclu, target.pt - reference.pt)

        self.h_ptResp.Fill(target.pt/reference.pt)
        self.h_ptRespVeta.Fill(reference.eta, target.pt/reference.pt)
        self.h_ptRespVpt.Fill(reference.pt, target.pt/reference.pt)
        self.h_ptRespVnclu.Fill(target.nclu, target.pt/reference.pt)

        if 'energyCore' in target:
            self.h_coreEnergyRes.Fill(target.energyCore - reference.energy)
            self.h_corePtRes.Fill(target.ptCore - reference.pt)

            self.h_coreEnergyResVnclu.Fill(target.nclu, target.energyCore - reference.energy)
            self.h_corePtResVnclu.Fill(target.nclu, target.ptCore - reference.pt)

        if 'energyCentral' in target:
            self.h_centralEnergyRes.Fill(target.energyCentral - reference.energy)
        self.h_etaRes.Fill(target.eta - reference.eta)
        self.h_phiRes.Fill(target.phi - reference.phi)
        self.h_drRes.Fill(np.sqrt((reference.phi-target.phi)**2+(reference.eta-target.eta)**2))

        if 'n010' in target:
            self.h_n010.Fill(target.n010)
        if 'n025' in target:
            self.h_n025.Fill(target.n025)


class Reso2DHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_etaRes = ROOT.TH1F(name+'_etaRes', 'Eta 2D cluster - GEN part', 100, -0.5, 0.5)
            self.h_phiRes = ROOT.TH1F(name+'_phiRes', 'Phi 2D cluster - GEN part', 100, -0.5, 0.5)
            self.h_phiPRes = ROOT.TH1F(name+'_phiPRes', 'Phi (+) 2D cluster - GEN part', 100, -0.5, 0.5)
            self.h_phiMRes = ROOT.TH1F(name+'_phiMRes', 'Phi (-) 2D cluster - GEN part', 100, -0.5, 0.5)
            self.h_xResVlayer = ROOT.TH2F(name+'_xResVlayer', 'X resolution (cm) [(2D clus) - GEN]', 60, 0, 60, 100, -10, 10)
            self.h_yResVlayer = ROOT.TH2F(name+'_yResVlayer', 'Y resolution (cm) [(2D clus) - GEN]', 60, 0, 60, 100, -10, 10)
            self.h_DRRes = ROOT.TH1F(name+'_DRRes', 'DR 2D cluster - GEN part', 100, -0.5, 0.5)

        BaseHistos.__init__(self, name, root_file)

    def fill(self, reference, target):
        rnp.fill_hist(self.h_etaRes, reference.eta-target.eta)

        rnp.fill_hist(self.h_phiRes, reference.phi-target.phi)
        if reference.pdgid < 0:
            rnp.fill_hist(self.h_phiMRes, reference.phi-target.phi)
        elif reference.pdgid > 0:
            rnp.fill_hist(self.h_phiPRes, reference.phi-target.phi)

        rnp.fill_hist(self.h_DRRes, np.sqrt((reference.phi-target.phi)**2+(reference.eta-target.eta)**2))
        if reference.reachedEE == 2:
            if 'x' in target.columns:
                target['xres'] = reference.posx[target.layer-1]-target.x
                # print target[['layer', 'xres']]
                rnp.fill_hist(self.h_xResVlayer, target[['layer', 'xres']])
            if 'y' in target.columns:
                target['yres'] = reference.posy[target.layer-1]-target.y
                # print target[['layer', 'yres']]
                rnp.fill_hist(self.h_yResVlayer, target[['layer', 'yres']])


class GeomHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_maxNNDistVlayer = ROOT.TH2F(name+'_maxNNDistVlayer', 'Max dist between NN vs layer', 60, 0, 60, 100, 0, 10)
            self.h_minNNDistVlayer = ROOT.TH2F(name+'_minNNDistVlayer', 'Max dist between NN vs layer', 60, 0, 60, 100, 0, 10)

            self.h_nTCsPerLayer = ROOT.TH1F(name+'_nTCsPerLayer', '# of Trigger Cells per layer', 60, 0, 60)
            self.h_radiusVlayer = ROOT.TH2F(name+'_radiusVlayer', '# of cells radius vs layer', 60, 0, 60, 200, 0, 200)
        BaseHistos.__init__(self, name, root_file)

    def fill(self, tcs):
        if True:
            ee_tcs = tcs[tcs.subdet == 3]
            for index, tc_geom in ee_tcs.iterrows():
                self.h_maxNNDistVlayer.Fill(tc_geom.layer, np.max(tc_geom.neighbor_distance))
                self.h_minNNDistVlayer.Fill(tc_geom.layer, np.min(tc_geom.neighbor_distance))

        rnp.fill_hist(self.h_nTCsPerLayer, tcs[tcs.subdet == 3].layer)
        rnp.fill_hist(self.h_radiusVlayer, tcs[['layer', 'radius']])


class DensityHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_eDensityVlayer = ROOT.TH2F(name+'_eDensityVlayer', 'E (GeV) Density per layer', 60, 0, 60, 600, 0, 30)
            self.h_nTCDensityVlayer = ROOT.TH2F(name+'_nTCDensityVlayer', '# TC Density per layer', 60, 0, 60, 20, 0, 20)
        elif 'v7' in root_file.GetName() and "NuGun" not in root_file.GetName():
            print "v7 hack"
            self.h_eDensityVlayer = root_file.Get(name+'eDensityVlayer')
            self.h_nTCDensityVlayer = root_file.Get(name+'nTCDensityVlayer')
        BaseHistos.__init__(self, name, root_file)

    def fill(self, layer, energy, nTCs):
        self.h_eDensityVlayer.Fill(layer, energy)
        self.h_nTCDensityVlayer.Fill(layer, nTCs)


# for convenience we define some sets
class HistoSetClusters():
    def __init__(self, name, root_file=None):
        self.htc = TCHistos('h_tc_'+name, root_file)
        self.hcl2d = ClusterHistos('h_cl2d_'+name, root_file)
        self.hcl3d = Cluster3DHistos('h_cl3d_'+name, root_file)
        # if not root_file:
        #     self.htc.annotateTitles(name)
        #     self.hcl2d.annotateTitles(name)
        #     self.hcl3d.annotateTitles(name)

    def fill(self, tcs, cl2ds, cl3ds):
        self.htc.fill(tcs)
        self.hcl2d.fill(cl2ds)
        self.hcl3d.fill(cl3ds)


class HistoSetReso():
    def __init__(self, name, root_file=None):
        self.hreso = ResoHistos('h_reso_'+name, root_file)
        self.hresoCone = ResoHistos('h_resoCone_'+name, root_file)
        self.hreso2D = Reso2DHistos('h_reso2D_'+name, root_file)
        # if not root_file:
        #     self.hreso.annotateTitles(name)
        #     self.hresoCone.annotateTitles(name)
        #     self.hreso2D.annotateTitles(name)


class HistoEff():
    def __init__(self, passed, total):
        # print dir(total)
        for histo in [a for a in dir(total) if a.startswith('h_')]:
            # print histo
            hist_total = getattr(total, histo)
            hist_passed = getattr(passed, histo)
            setattr(self, histo, ROOT.TEfficiency(hist_passed, hist_total))
            # getattr(self, histo).Sumw2()


class HistoSetEff():
    def __init__(self, name, root_file=None):
        self.name = name
        self.h_num = GenParticleHistos('h_effNum_'+name, root_file)
        self.h_den = GenParticleHistos('h_effDen_'+name, root_file)
        self.h_eff = None
        if root_file:
            self.computeEff()

    def fillNum(self, particles):
        self.h_num.fill(particles)

    def fillDen(self, particles):
        self.h_den.fill(particles)

    def computeEff(self):
        # print "Computing eff"
        self.h_eff = HistoEff(passed=self.h_num, total=self.h_den)
        pass
