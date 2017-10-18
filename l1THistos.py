
import ROOT
import root_numpy as rnp
import numpy as np

class GenPartHistos():
    def __init__(cls, name):
        cls.h_pt = ROOT.TH1F(name+'_pt', 'Gen Part Pt (GeV)', 100, 0, 100)
        cls.h_energy = ROOT.TH1F(name+'_energy', 'Gen Part Energy (GeV)', 100, 0, 100)

        for histo in [a for a in dir(cls) if a.startswith('h_')]:
            getattr(cls, histo).Sumw2()

    def fill(cls, pt, energy, weight=1.):
        cls.h_pt.Fill(pt, weight)
        cls.h_energy.Fill(energy, weight)

    def write(cls):
        for histo in [a for a in dir(cls) if a.startswith('h_')]:
            getattr(cls, histo).Write()


class TCHistos():
    def __init__(cls, name, root_file=None):
        if not root_file:
            cls.h_energy = ROOT.TH1F(name+'_energy', 'TC energy Pt (GeV)', 100, 0, 2)
            cls.h_subdet = ROOT.TH1F(name+'_subdet', 'TC subdet #', 8, 0, 8)
            cls.h_layer = ROOT.TH1F(name+'_layer', 'TC layer #', 60, 0, 60)
            cls.h_absz = ROOT.TH1F(name+'_absz', 'TC z(cm)', 100, 300, 500)
            cls.h_wafertype = ROOT.TH1F(name+'_wafertype', 'Wafer type', 10, 0, 10)
            cls.h_layerVenergy = ROOT.TH2F(name+'_layerVenergy', "Energy (GeV) vs Layer #", 100, 0, 2, 50, 0, 50)

            for histo in [a for a in dir(cls) if a.startswith('h_')]:
                getattr(cls, histo).Sumw2()
        else:
            cls.h_energy = root_file.Get(name+'_energy')
            cls.h_subdet = root_file.Get(name+'_subdet')
            cls.h_layer = root_file.Get(name+'_layer')
            cls.h_absz = root_file.Get(name+'_absz')
            cls.h_wafertype = root_file.Get(name+'_wafertype')
            cls.h_layerVenergy = root_file.Get(name+'_layerVenergy')

    def fill(cls, tcs):
        rnp.fill_hist(cls.h_energy, tcs.energy)
        rnp.fill_hist(cls.h_subdet, tcs.subdet)
        rnp.fill_hist(cls.h_layer, tcs.layer)
        rnp.fill_hist(cls.h_absz, np.fabs(tcs.z))
        rnp.fill_hist(cls.h_wafertype, tcs.wafertype)
        rnp.fill_hist(cls.h_wafertype, tcs.wafertype)
        rnp.fill_hist(cls.h_layerVenergy, tcs[['energy', 'layer']])

        # cls.h_layerVenergy.Fill(energy, layer, weight)

    def write(cls):
        for histo in [a for a in dir(cls) if a.startswith('h_')]:
            getattr(cls, histo).Write()


class ClusterHistos():
    def __init__(cls, name, root_file=None):
        if not root_file:
            cls.h_energy = ROOT.TH1F(name+'_energy', 'Cluster energy Pt (GeV)', 100, 0, 20)
            cls.h_layer = ROOT.TH1F(name+'_layer', 'Cluster layer #', 60, 0, 60)
            cls.h_ncells = ROOT.TH1F(name+'_ncells', 'Cluster # cells', 30, 0, 30)
            cls.h_layerVenergy = ROOT.TH2F(name+'_layerVenergy', "Cluster Energy (GeV) vs Layer #", 100, 0, 20, 50, 0, 50)
            cls.h_layerVncells = ROOT.TH2F(name+'_layerVncells', "Cluster #cells vs Layer #", 30, 0, 30, 50, 0, 50)

            for histo in [a for a in dir(cls) if a.startswith('h_')]:
                getattr(cls, histo).Sumw2()
        else:
            cls.h_energy = root_file.Get(name+'_energy')
            cls.h_layer = root_file.Get(name+'_layer')
            cls.h_ncells = root_file.Get(name+'_ncells')
            cls.h_layerVenergy = root_file.Get(name+'_layerVenergy')
            cls.h_layerVncells = root_file.Get(name+'_layerVncells')

    def fill(cls, energy, layer, ncells, weight=1):
        cls.h_energy.Fill(energy, weight)
        cls.h_layer.Fill(layer, weight)
        cls.h_ncells.Fill(ncells, weight)
        cls.h_layerVenergy.Fill(energy, layer, weight)
        cls.h_layerVncells.Fill(ncells, layer, weight)

    def write(cls):
        for histo in [a for a in dir(cls) if a.startswith('h_')]:
            getattr(cls, histo).Write()


class Cluster3DHistos():
    def __init__(cls, name, root_file=None):
        if not root_file:
            cls.h_pt = ROOT.TH1F(name+'_pt', '3D Cluster Pt (GeV)', 100, 0, 100)
            cls.h_energy = ROOT.TH1F(name+'_energy', '3D Cluster energy (GeV)', 100, 0, 100)
            cls.h_nclu = ROOT.TH1F(name+'_nclu', '3D Cluster # clusters', 30, 0, 30)
            cls.h_showlenght = ROOT.TH1F(name+'_showlenght', '3D Cluster showerlenght', 30, 0, 30)
            cls.h_firstlayer = ROOT.TH1F(name+'_firstlayer', '3D Cluster first layer', 30, 0, 30)
            cls.h_sEtaEtaTot = ROOT.TH1F(name+'_sEtaEtaTot', '3D Cluster RMS Eta', 100, 0, 0.1)
            cls.h_sEtaEtaMax = ROOT.TH1F(name+'_sEtaEtaMax', '3D Cluster RMS Eta (max)', 100, 0, 0.1)
            cls.h_sPhiPhiTot = ROOT.TH1F(name+'_sPhiPhiTot', '3D Cluster RMS Phi', 100, 0, 2)
            cls.h_sPhiPhiMax = ROOT.TH1F(name+'_sPhiPhiMax', '3D Cluster RMS Phi (max)', 100, 0, 2)
            cls.h_sZZ = ROOT.TH1F(name+'_sZZ', '3D Cluster RMS Z ???', 100, 0, 10)
            cls.h_eMaxOverE = ROOT.TH1F(name+'_eMaxOverE', '3D Cluster Emax/E', 100, 0, 1)

            for histo in [a for a in dir(cls) if a.startswith('h_')]:
                getattr(cls, histo).Sumw2()
        else:
            cls.h_pt
            cls.h_energy
            cls.h_nclu
            cls.h_showlenght
            cls.h_firstlayer
            cls.h_sEtaEtaTot
            cls.h_sEtaEtaMax
            cls.h_sPhiPhiTot
            cls.h_sPhiPhiMax
            cls.h_sZZ
            cls.h_eMaxOverE

            for histo in [a for a in dir(cls) if a.startswith('h_')]:
                h_method = getattr(cls, histo)
                h_method = root_file.Get(name+'_'+histo.split('_')[1])

    def fill(cls, pt, energy, nclu, showerlenght, firstlayer, seetot, seemax, spptot, sppmax, szz, emaxe, weight=1):
        cls.h_pt.Fill(pt, weight)
        cls.h_energy.Fill(energy, weight)
        cls.h_nclu.Fill(nclu, weight)
        cls.h_showlenght.Fill(showerlenght, weight)
        cls.h_firstlayer .Fill(firstlayer, weight)
        cls.h_sEtaEtaTot.Fill(seetot, weight)
        cls.h_sEtaEtaMax.Fill(seemax, weight)
        cls.h_sPhiPhiTot.Fill(spptot, weight)
        cls.h_sPhiPhiMax.Fill(sppmax, weight)
        cls.h_sZZ.Fill(szz, weight)
        cls.h_eMaxOverE.Fill(emaxe, weight)

    def write(cls):
        for histo in [a for a in dir(cls) if a.startswith('h_')]:
            getattr(cls, histo).Write()
