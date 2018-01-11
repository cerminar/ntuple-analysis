
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

    def __new__(cls): # __new__ always a classmethod
        if not HistoManager.instance:
            HistoManager.instance = HistoManager.__TheManager()
        return HistoManager.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)
    def __setattr__(self, name):
        return setattr(self.instance, name)




class BaseHistos():
    def __init__(self, name, root_file=None):
        # print 'BOOK histo: {}'.format(self)
        if root_file is not None:
            root_file.cd()
            histo_names = [histo.GetName() for histo in root_file.GetListOfKeys() if name+'_' in histo.GetName()]
            print histo_names
            for histo_name in histo_names:
                hinst = root_file.Get(histo_name)
                attr_name = 'h_'+histo_name.split('_')[2]
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


class GenPartHistos(BaseHistos):
    def __init__(self, name):
        self.h_pt = ROOT.TH1F(name+'_pt', 'Gen Part Pt (GeV)', 100, 0, 100)
        self.h_energy = ROOT.TH1F(name+'_energy', 'Gen Part Energy (GeV)', 100, 0, 1000)

        for histo in [a for a in dir(self) if a.startswith('h_')]:
            getattr(self, histo).Sumw2()

    def fill(self, gps):
        rnp.fill_hist(self.h_pt, gps.pt)
        rnp.fill_hist(self.h_energy, gps.energy)

    def write(self):
        for histo in [a for a in dir(self) if a.startswith('h_')]:
            getattr(self, histo).Write()


class GenParticleHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_eta = ROOT.TH1F(name+'_eta', 'Gen Part eta', 100, -3, 3)
            self.h_pt = ROOT.TH1F(name+'_pt', 'Gen Part Pt (GeV)', 100, 0, 100)
            self.h_energy = ROOT.TH1F(name+'_energy', 'Gen Part Energy (GeV)', 100, 0, 1000)
            self.h_reachedEE = ROOT.TH1F(name+'_reachedEE', 'Gen Part reachedEE', 4, 0, 4)
        BaseHistos.__init__(self, name, root_file)

    def fill(self, particles):
        rnp.fill_hist(self.h_eta, particles.eta)
        rnp.fill_hist(self.h_pt, particles.pt)
        rnp.fill_hist(self.h_energy, particles.energy)
        rnp.fill_hist(self.h_reachedEE, particles.reachedEE)


class DigiHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_layer = ROOT.TH1F(name+'_layer', 'Digi layer #', 60, 0, 60)
            self.h_simenergy = ROOT.TH1F(name+'_energy', 'Digi sim-energy (GeV)', 100, 0, 2)
        BaseHistos.__init__(self, name, root_file)

    def fill(self, digis):
        rnp.fill_hist(self.h_layer, digis.layer)
        rnp.fill_hist(self.h_simenergy, digis.simenergy)


class TCHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_energy = ROOT.TH1F(name+'_energy', 'TC energy (GeV)', 100, 0, 2)
            self.h_subdet = ROOT.TH1F(name+'_subdet', 'TC subdet #', 8, 0, 8)
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
            self.h_energy = ROOT.TH1F(name+'_energy', 'Cluster energy (GeV)', 100, 0, 30)
            self.h_layer = ROOT.TH1F(name+'_layer', 'Cluster layer #', 60, 0, 60)
            self.h_ncells = ROOT.TH1F(name+'_ncells', 'Cluster # cells', 30, 0, 30)
            self.h_nCoreCells = ROOT.TH1F(name+'_nCoreCells', 'Cluster # cells (core)', 30, 0, 30)

            self.h_layerVenergy = ROOT.TH2F(name+'_layerVenergy', "Cluster Energy (GeV) vs Layer #", 50, 0, 50, 100, 0, 20)
            self.h_layerVncells = ROOT.TH2F(name+'_layerVncells', "Cluster #cells vs Layer #",  50, 0, 50, 30, 0, 30)
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
            self.h_pt = ROOT.TH1F(name+'_pt', '3D Cluster Pt (GeV)', 100, 0, 100)
            self.h_energy = ROOT.TH1F(name+'_energy', '3D Cluster energy (GeV)', 1000, 0, 1000)
            self.h_nclu = ROOT.TH1F(name+'_nclu', '3D Cluster # clusters', 30, 0, 30)
            self.h_showlenght = ROOT.TH1F(name+'_showlenght', '3D Cluster showerlenght', 60, 0, 60)
            self.h_firstlayer = ROOT.TH1F(name+'_firstlayer', '3D Cluster first layer', 30, 0, 30)
            self.h_sEtaEtaTot = ROOT.TH1F(name+'_sEtaEtaTot', '3D Cluster RMS Eta', 100, 0, 0.1)
            self.h_sEtaEtaMax = ROOT.TH1F(name+'_sEtaEtaMax', '3D Cluster RMS Eta (max)', 100, 0, 0.1)
            self.h_sPhiPhiTot = ROOT.TH1F(name+'_sPhiPhiTot', '3D Cluster RMS Phi', 100, 0, 2)
            self.h_sPhiPhiMax = ROOT.TH1F(name+'_sPhiPhiMax', '3D Cluster RMS Phi (max)', 100, 0, 2)
            self.h_sZZ = ROOT.TH1F(name+'_sZZ', '3D Cluster RMS Z ???', 100, 0, 10)
            self.h_eMaxOverE = ROOT.TH1F(name+'_eMaxOverE', '3D Cluster Emax/E', 100, 0, 1)
        BaseHistos.__init__(self, name, root_file)

    def fill(self, cl3ds):
        rnp.fill_hist(self.h_pt, cl3ds.pt)
        rnp.fill_hist(self.h_energy, cl3ds.energy)
        rnp.fill_hist(self.h_nclu, cl3ds.nclu)
        rnp.fill_hist(self.h_showlenght, cl3ds.showerlength)
        rnp.fill_hist(self.h_firstlayer, cl3ds.firstlayer)
        rnp.fill_hist(self.h_sEtaEtaTot, cl3ds.seetot)
        rnp.fill_hist(self.h_sEtaEtaMax, cl3ds.seemax)
        rnp.fill_hist(self.h_sPhiPhiTot, cl3ds.spptot)
        rnp.fill_hist(self.h_sPhiPhiMax, cl3ds.sppmax)
        rnp.fill_hist(self.h_sZZ, cl3ds.szz)
        rnp.fill_hist(self.h_eMaxOverE, cl3ds.emaxe)


class ResoHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_ptRes = ROOT.TH1F(name+'_ptRes', '3D Cluster Pt reso (GeV)', 200, -100, 100)
            self.h_energyRes = ROOT.TH1F(name+'_energyRes', '3D Cluster Energy reso (GeV)', 200, -100, 100)
            self.h_ptResVeta = ROOT.TH2F(name+'_ptResVeta', '3D Cluster Pt reso (GeV) vs eta', 100, -3.5, 3.5, 200, -100, 100)
            self.h_energyResVeta = ROOT.TH2F(name+'_energyResVeta', '3D Cluster E reso (GeV) vs eta', 100, -3.5, 3.5, 200, -100, 100)
            self.h_energyResVnclu = ROOT.TH2F(name+'_energyResVnclu', '3D Cluster E reso (GeV) vs # clusters', 50, 0, 50, 200, -100, 100)

            self.h_ptResVnclu = ROOT.TH2F(name+'_ptResVnclu', '3D Cluster Pt reso (GeV) vs # clusters', 50, 0, 50, 200, -100, 100)

            # FIXME: add corresponding Pt plots
            self.h_coreEnergyResVnclu = ROOT.TH2F(name+'_coreEnergyResVnclu', '3D Cluster E reso (GeV) vs # clusters', 50, 0, 50, 200, -100, 100)
            self.h_corePtResVnclu = ROOT.TH2F(name+'_corePtResVnclu', '3D Cluster Pt reso (GeV) vs # clusters', 50, 0, 50, 200, -100, 100)

            self.h_coreEnergyRes = ROOT.TH1F(name+'_coreEnergyRes', '3D Cluster Energy reso CORE (GeV)', 200, -100, 100)
            self.h_corePtRes = ROOT.TH1F(name+'_corePtRes', '3D Cluster Pt reso CORE (GeV)', 200, -100, 100)

            self.h_centralEnergyRes = ROOT.TH1F(name+'_centralEnergyRes', '3D Cluster Energy reso CENTRAL (GeV)', 200, -100, 100)

        BaseHistos.__init__(self, name, root_file)

    def fill(self, reference, target):
        self.h_ptRes.Fill(target.pt - reference.pt)
        self.h_energyRes.Fill(target.energy - reference.energy)
        self.h_ptResVeta.Fill(reference.eta, target.pt - reference.pt)
        self.h_energyResVeta.Fill(reference.eta, target.energy - reference.energy)
        self.h_energyResVnclu.Fill(target.nclu, target.energy - reference.energy)
        self.h_ptResVnclu.Fill(target.nclu, target.pt - reference.pt)

        if 'energyCore' in target:
            self.h_coreEnergyRes.Fill(target.energyCore - reference.energy)
            self.h_corePtRes.Fill(target.ptCore - reference.pt)

            self.h_coreEnergyResVnclu.Fill(target.nclu, target.energyCore - reference.energy)
            self.h_corePtResVnclu.Fill(target.nclu, target.ptCore - reference.pt)

        if 'energyCentral' in target:
            self.h_centralEnergyRes.Fill(target.energyCentral - reference.energy)


class Reso2DHistos(BaseHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_etaRes = ROOT.TH1F(name+'_etaRes', 'Eta 2D cluster - GEN part', 100, -0.5, 0.5)
            self.h_phiRes = ROOT.TH1F(name+'_phiRes', 'Phi 2D cluster - GEN part', 100, -0.5, 0.5)
            self.h_phiPRes = ROOT.TH1F(name+'_phiPRes', 'Phi (+) 2D cluster - GEN part', 100, -0.5, 0.5)
            self.h_phiMRes = ROOT.TH1F(name+'_phiMRes', 'Phi (-) 2D cluster - GEN part', 100, -0.5, 0.5)

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

# class HstoSetMatchedClusters():
#     def __init__(self, name):
#
