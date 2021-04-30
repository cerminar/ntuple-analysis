from __future__ import print_function
import ROOT
import root_numpy as rnp
import numpy as np
from array import array
# import pandas as pd
import uuid
import math

import python.pf_regions as pf_regions

stuff = []


class HistoLazyFiller(object):
    def __init__(self, dataframe):
        self.df = dataframe
        self.manager = ROOT.FillerManager()
        self.columns = dataframe.columns
        
    def fill1d_lazy(self, histo, col_name, sel_name):
        if not self.manager.knowsVariable(col_name):
            # print (" - add variable {}".format(col_name))
            self.manager.addVariable(col_name, self.df[col_name].values)
    
        if not self.manager.knowsSelection(sel_name):
            print ("*** [HistoLazyFiller] ERROR: selection: {} not known!".format(sel_name))
            raise ValueError('[HistoLazyFiller] selection {} nott known'.format(sel_name))

        self.manager.add_1Dhisto(histo, col_name, sel_name)
    
    def add_selection(self, sel_name, sel_values):
        self.manager.addFilter(sel_name, sel_values)

    def fill(self):
        self.manager.fill()
    

class HistoManager(object):
    class __TheManager:
        def __init__(self):
            self.val = None
            self.histoList = list()

        def __str__(self):
            return 'self' + self.val

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
    def __init__(self, name, root_file=None, debug=False):
        self.name_ = name
        # print name
        # print self.__class__.__name__
        # # print 'BOOK histo: {}'.format(self)
        if root_file is not None:
            root_file.cd()
            # print 'class: {}'.format(self.__class__.__name__)
            # ROOT.gDirectory.pwd()
            file_dir = root_file.GetDirectory(self.__class__.__name__)
            # print '# keys in dir: {}'.format(len(file_dir.GetListOfKeys()))
            # file_dir.cd()
            selhistos = [(histo.ReadObj(), histo.GetName())
                         for histo in file_dir.GetListOfKeys()
                         if histo.GetName().startswith(name+'_')]
                         # if name+'_' in histo.GetName()]
            if debug:
                print(selhistos)
            for hinst, histo_name in selhistos:
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
        if self.__class__.__name__ not in ROOT.gDirectory.GetListOfKeys():
            ROOT.gDirectory.mkdir(self.__class__.__name__)
        newdir = ROOT.gDirectory.GetDirectory(self.__class__.__name__)
        newdir.cd()
        for histo in [a for a in dir(self) if a.startswith('h_')]:
            # print ("Writing {}".format(histo))
            getattr(self, histo).Write("", ROOT.TObject.kOverwrite)
        ROOT.gDirectory.cd('..')

    # def normalize(self, norm):
    #     className = self.__class__.__name__
    #     ret = className()
    #     return ret

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, self.name_)


class GraphBuilder:
    def __init__(self, h_obj, h_name):
        self.h_obj = h_obj
        self.h_name = h_name

    def get_graph(self, name, title, x, y, ex_l, ex_h, ey_l, ey_h):
        global stuff
        ret = ROOT.TGraphAsymmErrors()
        stuff.append(ret)
        ret.SetName(name)
        ret.SetTitle(title)
        ret.Set(len(x))
        for idx in range(0, len(x)):
            ret.SetPoint(idx, x[idx], y[idx])
            ret.SetPointError(idx, ex_l[idx], ex_h[idx], ey_l[idx], ey_h[idx])
            ret.SetMarkerStyle(2)
        return ret

    def __call__(self, name, title, function):
        h2d = getattr(self.h_obj, self.h_name)
        x, y, ex_l, ex_h, ey_l, ey_h = function(h2d)
        h_title_parts = h2d.GetTitle().split(';')
        g_title = '{};; {}'.format(h_title_parts[0], title)
        if len(h_title_parts) == 2:
            g_title = '{}; {}; {}'.format(h_title_parts, title)
        elif len(h_title_parts) == 3:
            g_title += '{}; {}; {} ({})'.format(h_title_parts[:2], title, h_title_parts[2])

        graph = self.get_graph('{}_{}'.format(h2d.GetName(), name),
                               g_title, x, y, ex_l, ex_h, ey_l, ey_h)
        g_attr_name = 'g_{}_{}'.format(self.h_name.split('h_')[1], name)
        setattr(self.h_obj, g_attr_name, graph)
        return graph

    def Write(self, name='', options=None):
        return


class BaseResoHistos(BaseHistos):
    """ Base class for resolution histogram classes.

        The class adds a special method to produce a graph out of each
        2D histograms of the class via e special <histoname>_graph method.
        The interface of this method is actually defined by the __call__
        method of the GraphBuilder class.
        If the method is called, the newly created graph is also added permanently
        to the class members and can be reused later.
        Example:
        def computeEResolution():
            ....
        hreso.h_energyResVenergy_graph('sigmaEOE', '#sigma_{E}/E', computeEResolution)
        will create the graph accessible with:
        hreso.g_energyResVenergy_sigmaEOE
        )
    """
    def __init__(self, name, root_file=None, debug=False):
        BaseHistos.__init__(self, name, root_file, debug)
        if root_file is not None or True:
            # print dir(self)
            for attr_2d in [attr for attr in dir(self) if (attr.startswith('h_') and 'TH2' in getattr(self, attr).ClassName())]:
                setattr(self, attr_2d+'_graph', GraphBuilder(self, attr_2d))


class BaseTuples(BaseHistos):
    def __init__(self, tuple_suffix, tuple_variables,
                 name, root_file=None, debug=False):
        if not root_file:
            self.t_values = ROOT.TNtuple(
                '{}_{}'.format(name, tuple_suffix),
                '{}_{}'.format(name, tuple_suffix),
                tuple_variables)
        BaseHistos.__init__(self, name, root_file, debug)

    def write(self):
        if self.__class__.__name__ not in ROOT.gDirectory.GetListOfKeys():
            ROOT.gDirectory.mkdir(self.__class__.__name__)
        newdir = ROOT.gDirectory.GetDirectory(self.__class__.__name__)
        newdir.cd()
        self.t_values.Write()
        ROOT.gDirectory.cd('..')
        return


class GenPartHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        self.h_pt = ROOT.TH1F(name+'_pt', 'Gen Part Pt (GeV)', 100, 0, 100)
        self.h_energy = ROOT.TH1F(name+'_energy', 'Gen Part Energy (GeV)', 100, 0, 1000)
        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, gps):
        rnp.fill_hist(self.h_pt, gps.pt)
        rnp.fill_hist(self.h_energy, gps.energy)

    def write(self):
        for histo in [a for a in dir(self) if a.startswith('h_')]:
            getattr(self, histo).Write()


class GenParticleHistos(BaseHistos):
    def __init__(self, name, root_file=None, extended_range=False, debug=False):
        if not root_file:
            pt_bins = [y for y in range(0, 202, 2)]
            if(extended_range):
                # print ('Booking: {} with extended range'.format(name))
                pt_bins = [y for y in range(0, 200, 2)] + \
                          [y for y in range(200, 400, 50)] + \
                          [y for y in range(400, 1100, 100)]
            n_pt_bins = len(pt_bins) - 1
            # print ('bins: {}'.format(pt_bins))
            # print ("# bins: {}".format(n_pt_bins))

            self.h_eta = ROOT.TH1F(name+'_eta', 'Gen Part eta; #eta^{GEN};', 50, -3, 3)
            self.h_abseta = ROOT.TH1F(name+'_abseta', 'Gen Part |eta|; |#eta^{GEN}|;', 40, 0, 4)
            self.h_pt = ROOT.TH1F(name+'_pt', 'Gen Part P_{T} (GeV); p_{T}^{GEN} [GeV];', n_pt_bins, array('d', pt_bins))
            self.h_energy = ROOT.TH1F(name+'_energy', 'Gen Part Energy (GeV); E [GeV];', 100, 0, 1000)
            self.h_reachedEE = ROOT.TH1F(name+'_reachedEE', 'Gen Part reachedEE', 4, 0, 4)
            self.h_fBrem = ROOT.TH1F(name+'_fBrem', 'Brem. p_{T} fraction', 30, 0, 1)
        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, particles):
        particles_weight = particles.weight
        rnp.fill_hist(hist=self.h_eta,
                      array=particles.eta,
                      weights=particles_weight)
        rnp.fill_hist(hist=self.h_abseta,
                      array=particles.abseta,
                      weights=particles_weight)
        rnp.fill_hist(hist=self.h_pt,
                      array=particles.pt,
                      weights=particles_weight)
        rnp.fill_hist(hist=self.h_energy,
                      array=particles.energy,
                      weights=particles_weight)
        rnp.fill_hist(hist=self.h_reachedEE,
                      array=particles.reachedEE,
                      weights=particles_weight)
        rnp.fill_hist(hist=self.h_fBrem,
                      array=particles.fbrem,
                      weights=particles_weight)

    def fill_lazy(self, filler, sel_name):
        filler.fill1d_lazy(self.h_eta, 'eta', sel_name)
        filler.fill1d_lazy(self.h_abseta, 'abseta', sel_name)
        filler.fill1d_lazy(self.h_pt, 'pt', sel_name)
        filler.fill1d_lazy(self.h_energy, 'energy', sel_name)
        filler.fill1d_lazy(self.h_reachedEE, 'reachedEE', sel_name)
        filler.fill1d_lazy(self.h_fBrem, 'fbrem', sel_name)


class DigiHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_layer = ROOT.TH1F(name+'_layer', 'Digi layer #', 60, 0, 60)
            # self.h_simenergy = ROOT.TH1F(name+'_energy', 'Digi sim-energy (GeV)', 100, 0, 2)
        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, digis):
        rnp.fill_hist(self.h_layer, digis.layer)
        # rnp.fill_hist(self.h_simenergy, digis.simenergy)


class RateHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_norm = ROOT.TH1F(name+'_norm', '# of events', 1, 1, 2)
            self.h_pt = ROOT.TH1F(name+'_pt', 'rate above p_{T} thresh.; p_{T} [GeV]; rate [kHz];', 100, 0, 100)
            self.h_ptVabseta = ROOT.TH2F(name+'_ptVabseta', 'Candidate p_{T} vs |#eta|; |#eta|; p_{T} [GeV];', 34, 1.4, 3.1, 100, 0, 100)

        BaseHistos.__init__(self, name, root_file, debug)

        if root_file is not None or True:
            for attr_1d in [attr for attr in dir(self) if (attr.startswith('h_') and 'TH1' in getattr(self, attr).ClassName())]:
                setattr(self, attr_1d+'_graph', GraphBuilder(self, attr_1d))
        
        if root_file is not None:
            self.normalize(31000)
            # self.h_simenergy = ROOT.TH1F(name+'_energy', 'Digi sim-energy (GeV)', 100, 0, 2)

    def fill(self, pt, eta):
        for ptf in range(0, int(pt)+1):
            self.h_pt.Fill(ptf)
        self.h_ptVabseta.Fill(abs(eta), pt)

    def fill_norm(self, many=1):
        # print (f' fill rate norm: {many}')
        self.h_norm.Fill(1, many)

    def normalize(self, norm):
        nev = self.h_norm.GetBinContent(1)
        if(nev != norm):
            print('normalize to {}'.format(norm))
            self.h_norm.Scale(norm/nev)
            self.h_pt.Scale(norm/nev)
            self.h_ptVabseta.Scale(norm/nev)

    def fill_many(self, data):
        ROOT.fill1D_rate(self.h_pt, data.pt.values)


class TCHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_energy = ROOT.TH1F(name+'_energy', 'TC energy (GeV)', 100, 0, 2)
            self.h_subdet = ROOT.TH1F(name+'_subdet', 'TC subdet #', 8, 0, 8)
            self.h_mipPt = ROOT.TH1F(name+'_mipPt', 'TC MIP Pt', 50, 0, 10)

            self.h_layer = ROOT.TProfile(name+'_layer', 'TC layer #', 60, 0, 60, 's')
            self.h_absz = ROOT.TH1F(name+'_absz', 'TC z(cm)', 100, 300, 500)
            self.h_wafertype = ROOT.TH1F(name+'_wafertype', 'Wafer type', 10, 0, 10)
            self.h_layerVenergy = ROOT.TH2F(name+'_layerVenergy', "Energy (GeV) vs Layer #", 60, 0, 60, 100, 0, 2)
            self.h_energyVeta = ROOT.TH2F(name+'_energyVeta', "Energy (GeV) vs Eta", 100, -3.5, 3.5, 100, 0, 2)
            # self.h_energyVetaL1t5 = ROOT.TH2F(name+'_energyVetaL1t5', "Energy (GeV) vs Eta (layers 1 to 5)", 100, -3.5, 3.5, 100, 0, 2)
            # self.h_energyVetaL6t10 = ROOT.TH2F(name+'_energyVetaL6t10', "Energy (GeV) vs Eta (layers 6 to 10)", 100, -3.5, 3.5, 100, 0, 2)
            # self.h_energyVetaL11t20 = ROOT.TH2F(name+'_energyVetaL11t20', "Energy (GeV) vs Eta (layers 11 to 20)", 100, -3.5, 3.5, 100, 0, 2)
            # self.h_energyVetaL21t60 = ROOT.TH2F(name+'_energyVetaL21t60', "Energy (GeV) vs Eta (layers 21 to 60)", 100, -3.5, 3.5, 100, 0, 2)
            self.h_energyPetaVphi = ROOT.TProfile2D(name+'_energyPetaVphi', "Energy profile (GeV) vs Eta and Phi", 100, -3.5, 3.5, 100, -3.2, 3.2)

        BaseHistos.__init__(self, name, root_file, debug)

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
        # rnp.fill_hist(self.h_energyVetaL1t5, tcs[(tcs.layer >= 1) & (tcs.layer <= 5)][['eta', 'energy']])
        # rnp.fill_hist(self.h_energyVetaL6t10, tcs[(tcs.layer >= 6) & (tcs.layer <= 10)][['eta', 'energy']])
        # rnp.fill_hist(self.h_energyVetaL11t20, tcs[(tcs.layer >= 11) & (tcs.layer <= 20)][['eta', 'energy']])
        # rnp.fill_hist(self.h_energyVetaL21t60, tcs[(tcs.layer >= 21) & (tcs.layer <= 60)][['eta', 'energy']])
        rnp.fill_profile(self.h_energyPetaVphi, tcs[['eta', 'phi', 'energy']])


class ClusterHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_energy = ROOT.TH1F(name+'_energy', 'Cluster energy (GeV); E [GeV];', 100, 0, 30)
            self.h_layer = ROOT.TH1F(name+'_layer', 'Cluster layer #; layer #;', 60, 0, 60)
            # self.h_nCoreCells = ROOT.TH1F(name+'_nCoreCells', 'Cluster # cells (core)', 30, 0, 30)

            self.h_layerVenergy = ROOT.TH2F(name+'_layerVenergy', "Cluster Energy (GeV) vs Layer #; layer; E [GeV];", 50, 0, 50, 100, 0, 20)
            self.h_ncells = ROOT.TH1F(name+'_ncells', 'Cluster # cells; # TC components;', 30, 0, 30)
            self.h_layerVncells = ROOT.TH2F(name+'_layerVncells', "Cluster #cells vs Layer #; layer; # TC components;",  50, 0, 50, 30, 0, 30)
            # self.h_layerVnCoreCells = ROOT.TH2F(name+'_layerVnCoreCells', "Cluster #cells vs Layer #",  50, 0, 50, 30, 0, 30)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, clsts):
        rnp.fill_hist(self.h_energy, clsts.energy)
        rnp.fill_hist(self.h_layer, clsts.layer)
        rnp.fill_hist(self.h_layerVenergy, clsts[['layer', 'energy']])
        # if 'ncells' in clsts.columns:
        rnp.fill_hist(self.h_ncells, clsts.ncells)
        rnp.fill_hist(self.h_layerVncells, clsts[['layer', 'ncells']])
        # if 'nCoreCells' in clsts.columns:
        #     rnp.fill_hist(self.h_nCoreCells, clsts.nCoreCells)
        #     rnp.fill_hist(self.h_layerVnCoreCells, clsts[['layer', 'nCoreCells']])


class Cluster3DHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_npt05 = ROOT.TH1F(
                name+'_npt05', '# 3D Cluster Pt > 0.5 GeV; # 3D clusters in cone;', 1000, 0, 1000)
            self.h_npt20 = ROOT.TH1F(
                name+'_npt20', '# 3D Cluster Pt > 2.0 GeV; # 3D clusters in cone;', 1000, 0, 1000)
            self.h_pt = ROOT.TH1F(
                name+'_pt', '3D Cluster Pt (GeV); p_{T} [GeV]', 100, 0, 100)
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
            self.h_HoE = ROOT.TH1F(name+'_HoE', '3D Cluster H/E', 20, 0, 2)
            self.h_iso0p2 = ROOT.TH1F(name+'_iso0p2', '3D Cluster iso DR 0.2(GeV); Iso p_{T} [GeV];', 100, 0, 100)
            self.h_isoRel0p2 = ROOT.TH1F(name+'_isoRel0p2', '3D Cluster relative iso DR 0.2; Rel. Iso;', 100, 0, 1)
            self.h_bdtPU = ROOT.TH1F(name+'_bdtPU', '3D Cluster bdt PU out; BDT-PU out;', 100, -1, 1)
            self.h_bdtPi = ROOT.TH1F(name+'_bdtPi', '3D Cluster bdt Pi out; BDT-Pi out;', 100, -1, 1)
            self.h_bdtEg = ROOT.TH1F(name+'_bdtEg', '3D Cluster bdt Pi out; BDT-EG out;', 100, -1, 1)

        BaseHistos.__init__(self, name, root_file, debug)

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
        rnp.fill_hist(self.h_HoE, cl3ds.hoe)
        if 'iso0p2' in cl3ds.columns:
            rnp.fill_hist(self.h_iso0p2, cl3ds.iso0p2)
            rnp.fill_hist(self.h_isoRel0p2, cl3ds.isoRel0p2)
        if 'bdt_pu' in cl3ds.columns:
            rnp.fill_hist(self.h_bdtPU, cl3ds.bdt_pu)
        if 'bdt_pi' in cl3ds.columns:
            rnp.fill_hist(self.h_bdtPi, cl3ds.bdt_pi)
        rnp.fill_hist(self.h_bdtEg, cl3ds.bdteg)

    def fill_lazy(self, filler, sel_name):
        # self.h_npt05.Fill(len(cl3ds[cl3ds.pt > 0.5].index))
        # self.h_npt20.Fill(len(cl3ds[cl3ds.pt > 2.0].index))

        filler.fill1d_lazy(self.h_pt, 'pt', sel_name)
        filler.fill1d_lazy(self.h_eta, 'eta', sel_name)
        filler.fill1d_lazy(self.h_energy, 'energy', sel_name)
        filler.fill1d_lazy(self.h_nclu, 'nclu', sel_name)
        # filler.fill2d_lazy(self.h_ncluVpt, cl3ds[['nclu', 'pt']], sel_name)
        filler.fill1d_lazy(self.h_showlenght, 'showerlength', sel_name)
        filler.fill1d_lazy(self.h_firstlayer, 'firstlayer', sel_name)
        filler.fill1d_lazy(self.h_sEtaEtaTot, 'seetot', sel_name)
        filler.fill1d_lazy(self.h_sEtaEtaMax, 'seemax', sel_name)
        filler.fill1d_lazy(self.h_sPhiPhiTot, 'spptot', sel_name)
        filler.fill1d_lazy(self.h_sPhiPhiMax, 'sppmax', sel_name)
        filler.fill1d_lazy(self.h_sZZ, 'szz', sel_name)
        filler.fill1d_lazy(self.h_eMaxOverE, 'emaxe', sel_name)
        filler.fill1d_lazy(self.h_HoE, 'hoe', sel_name)
        if 'iso0p2' in filler.columns:
            filler.fill1d_lazy(self.h_iso0p2, 'iso0p2', sel_name)
            filler.fill1d_lazy(self.h_isoRel0p2, 'isoRel0p2', sel_name)
        if 'bdt_pu' in filler.columns:
            filler.fill1d_lazy(self.h_bdtPU, 'bdt_pu', sel_name)
        if 'bdt_pi' in filler.columns:
            filler.fill1d_lazy(self.h_bdtPi, 'bdt_pi', sel_name)
        filler.fill1d_lazy(self.h_bdtEg, 'bdteg', sel_name)



class EGHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = ROOT.TH1F(name+'_pt', 'EG Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = ROOT.TH1F(name+'_eta', 'EG eta; #eta;', 100, -4, 4)
            self.h_energy = ROOT.TH1F(name+'_energy', 'EG energy (GeV); E [GeV]', 1000, 0, 1000)
            self.h_hwQual = ROOT.TH1F(name+'_hwQual', 'EG energy (GeV); hwQual', 5, 0, 5)
            self.h_tkIso = ROOT.TH1F(name+'_tkIso', 'Iso; rel-iso_{tk}', 100, 0, 2)
            self.h_pfIso = ROOT.TH1F(name+'_pfIso', 'Iso; rel-iso_{pf}', 100, 0, 2)
            self.h_tkIsoPV = ROOT.TH1F(name+'_tkIsoPV', 'Iso; rel-iso^{PV}_{tk}', 100, 0, 2)
            self.h_pfIsoPV = ROOT.TH1F(name+'_pfIsoPV', 'Iso; rel-iso^{PV}_{pf}', 100, 0, 2)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, egs):
        weight = None
        if 'weight' in egs.columns:
            weight = egs.weight
        else:
            weight = np.ones(egs.shape[0])

        rnp.fill_hist(hist=self.h_pt,     array=egs.pt,     weights=weight)
        rnp.fill_hist(hist=self.h_eta,    array=egs.eta,    weights=weight)
        rnp.fill_hist(hist=self.h_energy, array=egs.energy, weights=weight)
        rnp.fill_hist(hist=self.h_hwQual, array=egs.hwQual, weights=weight)
        if 'tkIso' in egs.columns:
            rnp.fill_hist(hist=self.h_tkIso, array=egs.tkIso, weights=weight)
            rnp.fill_hist(hist=self.h_pfIso, array=egs.pfIso, weights=weight)
        if 'tkIsoPV' in egs.columns:
            rnp.fill_hist(hist=self.h_tkIsoPV, array=egs.tkIsoPV, weights=weight)
            rnp.fill_hist(hist=self.h_pfIsoPV, array=egs.pfIsoPV, weights=weight)
    
    def fill_lazy(self, filler, sel_name):
        filler.fill1d_lazy(self.h_pt, 'pt', sel_name)
        filler.fill1d_lazy(self.h_eta, 'eta', sel_name)
        filler.fill1d_lazy(self.h_energy, 'energy', sel_name)
        filler.fill1d_lazy(self.h_hwQual, 'hwQual', sel_name)
        if 'tkIso' in filler.columns:
            filler.fill1d_lazy(self.h_tkIso, 'tkIso', sel_name)
            filler.fill1d_lazy(self.h_pfIso, 'pfIso', sel_name)
        if 'tkIsoPV' in filler.columns:
            filler.fill1d_lazy(self.h_tkIsoPV, 'tkIsoPV', sel_name)
            filler.fill1d_lazy(self.h_pfIsoPV, 'pfIsoPV', sel_name)
    
    def add_histos(self):
        self.h_pt.Add(self.h_pt_temp.GetValue())
        self.h_eta.Add(self.h_eta_temp.GetValue())
        self.h_energy.Add(self.h_energy_temp.GetValue())
        self.h_hwQual.Add(self.h_hwQual_temp.GetValue())
        # self.h_tkIso = ROOT.TH1F(name+'_tkIso', 'Iso; rel-iso_{tk}', 100, 0, 2)
        # self.h_pfIso = ROOT.TH1F(name+'_pfIso', 'Iso; rel-iso_{pf}', 100, 0, 2)
        # self.h_tkIsoPV = ROOT.TH1F(name+'_tkIsoPV', 'Iso; rel-iso^{PV}_{tk}', 100, 0, 2)
        # self.h_pfIsoPV = ROOT.TH1F(name+'_pfIsoPV', 'Iso; rel-iso^{PV}_{pf}', 100, 0, 2)

            
class TkEleHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = ROOT.TH1F(name+'_pt', 'Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = ROOT.TH1F(name+'_eta', 'eta; #eta;', 100, -2.5, 2.5)
            self.h_energy = ROOT.TH1F(name+'_energy', 'energy (GeV); E [GeV]', 1000, 0, 1000)
            self.h_hwQual = ROOT.TH1F(name+'_hwQual', 'quality; hwQual', 10, 0, 10)
            self.h_tkpt = ROOT.TH1F(name+'_tkpt', 'Tk Pt (GeV); p_{T}^{L1Tk} [GeV]', 100, 0, 100)
            self.h_dpt = ROOT.TH1F(name+'_dpt', 'Delta Tk Pt (GeV); #Delta p_{T}^{L1Tk-Calo} [GeV]', 100, -50, 50)
            self.h_tkchi2 = ROOT.TH1F(name+'_tkchi2', 'Tk chi2; #Chi^{2}', 1000, 0, 1000)
            self.h_ptVtkpt = ROOT.TH2F(name+'_ptVtkpt', 'TkEG Pt (GeV) vs TkPt; p_{T}^{Tk} [GeV]; p_{T}^{EG} [GeV]', 100, 0, 100, 100, 0, 100)
            self.h_tkIso = ROOT.TH1F(name+'_tkIso', 'Iso; rel-iso_{tk}', 100, 0, 2)
            self.h_pfIso = ROOT.TH1F(name+'_pfIso', 'Iso; rel-iso_{pf}', 100, 0, 2)



        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tkegs):
        rnp.fill_hist(self.h_pt, tkegs.pt)
        rnp.fill_hist(self.h_eta, tkegs.eta)
        rnp.fill_hist(self.h_energy, tkegs.energy)
        rnp.fill_hist(self.h_hwQual, tkegs.hwQual)
        rnp.fill_hist(self.h_tkpt, tkegs.tkPt)
        rnp.fill_hist(self.h_dpt, tkegs.dpt)
        rnp.fill_hist(self.h_tkchi2, tkegs.tkChi2)
        # rnp.fill_hist(self.h_ptVtkpt, tkegs[['tkPt', 'pt']])
        if 'tkIso' in tkegs.columns:
            rnp.fill_hist(self.h_tkIso, tkegs.tkIso)
            rnp.fill_hist(self.h_pfIso, tkegs.pfIso)

    def fill_lazy(self, filler, sel_name):
        filler.fill1d_lazy(self.h_pt, 'pt', sel_name)
        filler.fill1d_lazy(self.h_eta, 'eta', sel_name)
        filler.fill1d_lazy(self.h_energy, 'energy', sel_name)
        filler.fill1d_lazy(self.h_hwQual, 'hwQual', sel_name)
        filler.fill1d_lazy(self.h_tkpt, 'tkPt', sel_name)
        filler.fill1d_lazy(self.h_dpt, 'dpt', sel_name)
        filler.fill1d_lazy(self.h_tkchi2, 'tkChi2', sel_name)
        filler.fill1d_lazy(self.h_tkIso, 'tkIso', sel_name)
        filler.fill1d_lazy(self.h_pfIso, 'pfIso', sel_name)




class TkEmHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = ROOT.TH1F(name+'_pt', 'Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = ROOT.TH1F(name+'_eta', 'eta; #eta;', 100, -2.5, 2.5)
            self.h_energy = ROOT.TH1F(name+'_energy', 'energy (GeV); E [GeV]', 1000, 0, 1000)
            self.h_hwQual = ROOT.TH1F(name+'_hwQual', 'quality; hwQual', 10, 0, 10)
            self.h_tkIso = ROOT.TH1F(name+'_tkIso', 'Iso; rel-iso_{tk}', 100, 0, 2)
            self.h_pfIso = ROOT.TH1F(name+'_pfIso', 'Iso; rel-iso_{pf}', 100, 0, 2)
            self.h_tkIsoPV = ROOT.TH1F(name+'_tkIsoPV', 'Iso; rel-iso^{PV}_{tk}', 100, 0, 2)
            self.h_pfIsoPV = ROOT.TH1F(name+'_pfIsoPV', 'Iso; rel-iso^{PV}_{pf}', 100, 0, 2)



        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tkegs):
        rnp.fill_hist(self.h_pt, tkegs.pt)
        rnp.fill_hist(self.h_eta, tkegs.eta)
        rnp.fill_hist(self.h_energy, tkegs.energy)
        rnp.fill_hist(self.h_hwQual, tkegs.hwQual)
        rnp.fill_hist(self.h_tkIso, tkegs.tkIso)
        rnp.fill_hist(self.h_pfIso, tkegs.pfIso)
        rnp.fill_hist(self.h_tkIsoPV, tkegs.tkIsoPV)
        rnp.fill_hist(self.h_pfIsoPV, tkegs.pfIsoPV)

    def fill_lazy(self, filler, sel_name):
        filler.fill1d_lazy(self.h_pt, 'pt', sel_name)
        filler.fill1d_lazy(self.h_eta, 'eta', sel_name)
        filler.fill1d_lazy(self.h_energy, 'energy', sel_name)
        filler.fill1d_lazy(self.h_hwQual, 'hwQual', sel_name)
        filler.fill1d_lazy(self.h_tkIso, 'tkIso', sel_name)
        filler.fill1d_lazy(self.h_pfIso, 'pfIso', sel_name)
        filler.fill1d_lazy(self.h_tkIsoPV, 'tkIsoPV', sel_name)
        filler.fill1d_lazy(self.h_pfIsoPV, 'pfIsoPV', sel_name)


class TkEGHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt     = ROOT.TH1F(name+'_pt', 'TkEG Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta    = ROOT.TH1F(name+'_eta', 'TkEG eta; #eta;', 100, -4, 4)
            self.h_energy = ROOT.TH1F(name+'_energy', 'TkEG energy (GeV); E [GeV]', 1000, 0, 1000)
            self.h_hwQual = ROOT.TH1F(name+'_hwQual', 'TkEG energy (GeV); hwQual', 5, 0, 5)

            self.h_tkpt    = ROOT.TH1F(name+'_tkpt', 'TkEG Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_tketa = ROOT.TH1F(name+'_tketa', 'TkEG eta; #eta;', 100, -4, 4)
            self.h_tkchi2 = ROOT.TH1F(name+'_tkchi2', 'TkEG chi2; #Chi^{2}', 1000, 0, 1000)
            self.h_tkchi2Red = ROOT.TH1F(name+'_tkchi2Red', 'TkEG chi2 red; reduced #Chi^{2}', 100, 0, 100)
            self.h_tknstubs = ROOT.TH1F(name+'_tknstubs', 'TkEG # stubs; # stubs', 10, 0, 10)
            self.h_tkz0 = ROOT.TH1F(name+'_tkz0', 'TkEG z0; z_{0} [cm]', 100, -10, 10)
            self.h_tkchi2RedVeta = ROOT.TH2F(name+'_tkchi2RedVeta', 'TkEG chi2 red. v eta; #eta; red. #Chi^{2}', 100, -4, 4, 100, 0, 100)
            self.h_tknstubsVeta = ROOT.TH2F(name+'_tknstubsVeta', 'TkEG # stubs vs eta; #eta; # stubs', 100, -4, 4, 10, 0, 10)
            self.h_tkz0Veta = ROOT.TH2F(name+'_tkz0Veta', 'TkEG z0 vs eta; #eta; z_{0} [cm]', 100, -4, 4, 100, -10, 10)
            self.h_dphi  = ROOT.TH1F(name+'_dphi', 'TkEG #Delta #phi; #Delta #phi [rad]', 100, -0.2, 0.2)
            self.h_dphiVpt  = ROOT.TH2F(name+'_dphiVpt', 'TkEG #Delta #phi vs p_{T}^{EG}; p_{T}^{EG} [GeV]; #Delta #phi [rad]', 100, 0, 100, 100, -0.2, 0.2)
            self.h_deta = ROOT.TH1F(name+'_deta', 'TkEG #Delta #eta; #Delta #eta', 100, -0.2, 0.2)
            self.h_detaVpt = ROOT.TH2F(name+'_detaVpt', 'TkEG #Delta #eta vs p_{T}^{EG}; p_{T}^{EG} [GeV]; #Delta #eta', 100, 0, 100, 100, -0.2, 0.2)

            self.h_dr = ROOT.TH1F(name+'_dr', 'TkEG #Delta R; #Delta R', 100, 0, 0.2)
            self.h_ptVtkpt = ROOT.TH2F(name+'_ptVtkpt', 'TkEG Pt (GeV) vs TkPt; p_{T}^{Tk} [GeV]; p_{T}^{EG} [GeV]', 100, 0, 100, 100, 0, 100)



        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tkegs):
        rnp.fill_hist(self.h_pt, tkegs.pt)
        rnp.fill_hist(self.h_eta, tkegs.eta)
        rnp.fill_hist(self.h_energy, tkegs.energy)
        rnp.fill_hist(self.h_hwQual, tkegs.hwQual)
        rnp.fill_hist(self.h_tkpt, tkegs.tkpt)
        rnp.fill_hist(self.h_tketa, tkegs.tketa)
        rnp.fill_hist(self.h_tkchi2, tkegs.tkchi2)
        rnp.fill_hist(self.h_tkchi2Red, tkegs.tkchi2Red)
        rnp.fill_hist(self.h_tknstubs, tkegs.tknstubs)
        rnp.fill_hist(self.h_tkz0, tkegs.tkz0)
        rnp.fill_hist(self.h_tkchi2RedVeta, tkegs[['eta', 'tkchi2Red']])
        rnp.fill_hist(self.h_tknstubsVeta, tkegs[['eta', 'tknstubs']])
        rnp.fill_hist(self.h_tkz0Veta, tkegs[['eta', 'tkz0']])
        rnp.fill_hist(self.h_dphi, tkegs.dphi)
        rnp.fill_hist(self.h_deta, tkegs.deta)
        rnp.fill_hist(self.h_dphiVpt, tkegs[['pt', 'dphi']])
        rnp.fill_hist(self.h_detaVpt, tkegs[['pt', 'deta']])
        rnp.fill_hist(self.h_dr, tkegs.dr)
        rnp.fill_hist(self.h_ptVtkpt, tkegs[['tkpt', 'pt']])


class TrackHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = ROOT.TH1F(name+'_pt', 'Track Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = ROOT.TH1F(name+'_eta', 'Track eta; #eta;', 100, -4, 4)
            self.h_chi2 = ROOT.TH1F(name+'_chi2', 'Track chi2; #Chi^{2}', 1000, 0, 1000)
            self.h_chi2Red = ROOT.TH1F(name+'_chi2Red', 'Track chi2 red; red. #Chi^{2}', 100, 0, 100)
            self.h_nstubs = ROOT.TH1F(name+'_nstubs', 'Track # stubs; # stubs', 10, 0, 10)
            self.h_z0 = ROOT.TH1F(name+'_z0', 'Track z0; z_{0} [cm]', 100, -10, 10)
            self.h_chi2RedVeta = ROOT.TH2F(name+'_chi2RedVeta', 'Track chi2 red. v eta; #eta; red. #Chi^{2}', 100, -4, 4, 100, 0, 100)
            self.h_nstubsVeta = ROOT.TH2F(name+'_nstubsVeta', 'Track # stubs vs eta; #eta; # stubs', 100, -4, 4, 10, 0, 10)
            self.h_z0Veta = ROOT.TH2F(name+'_z0Veta', 'Track z0 vs eta; #eta; z_{0} [cm]', 100, -4, 4, 100, -10, 10)
            self.h_chi2RedVpt = ROOT.TH2F(name+'_chi2RedVpt', 'Track chi2 red. v pT; p_{T} [GeV]; red. #Chi^{2}', 100, 0, 100, 100, 0, 100)
            self.h_nstubsVpt = ROOT.TH2F(name+'_nstubsVpt', 'Track # stubs vs pT; p_{T} [GeV]; # stubs', 100, 0, 100, 10, 0, 10)
            self.h_z0Vpt = ROOT.TH2F(name+'_z0Vpt', 'Track z0 vs pT; p_{T} [GeV]; z_{0} [cm]', 100, 0, 100, 100, -10, 10)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tracks):
        rnp.fill_hist(self.h_pt, tracks.pt)
        rnp.fill_hist(self.h_eta, tracks.eta)
        rnp.fill_hist(self.h_chi2, tracks.chi2)
        rnp.fill_hist(self.h_chi2Red, tracks.chi2Red)
        rnp.fill_hist(self.h_nstubs, tracks.nStubs)
        rnp.fill_hist(self.h_z0, tracks.z0)
        rnp.fill_hist(self.h_chi2RedVeta, tracks[['eta', 'chi2Red']])
        rnp.fill_hist(self.h_nstubsVeta, tracks[['eta', 'nStubs']])
        rnp.fill_hist(self.h_z0Veta, tracks[['eta', 'z0']])
        rnp.fill_hist(self.h_chi2RedVpt, tracks[['pt', 'chi2Red']])
        rnp.fill_hist(self.h_nstubsVpt, tracks[['pt', 'nStubs']])
        rnp.fill_hist(self.h_z0Vpt, tracks[['pt', 'z0']])


class TriggerTowerHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = ROOT.TH1F(name+'_pt', 'Tower Pt (GeV); p_{T} [GeV];', 100, 0, 100)
            self.h_etEm = ROOT.TH1F(name+'_etEm', 'Tower Et EM (GeV)', 100, 0, 100)
            self.h_etHad = ROOT.TH1F(name+'_etHad', 'Tower Et Had (GeV)', 100, 0, 100)
            self.h_HoE = ROOT.TH1F(name+'_HoE', 'Tower H/E', 20, 0, 2)
            self.h_HoEVpt = ROOT.TH2F(name+'_HoEVpt', 'Tower H/E vs Pt (GeV); H/E;', 50, 0, 100, 20, 0, 2)
            self.h_energy = ROOT.TH1F(name+'_energy', 'Tower energy (GeV)', 1000, 0, 1000)
            self.h_eta = ROOT.TH1F(name+'_eta', 'Tower eta; #eta;', 75, -3.169, 3.169)
            self.h_ieta = ROOT.TH1F(name+'_ieta', 'Tower eta; i#eta;', 18, 0, 18)

            self.h_ptVeta = ROOT.TH2F(name+'_ptVeta', 'Tower P_P{T} (GeV) vs #eta; #eta; p_{T} [GeV];',  75, -3.169, 3.169, 100, 0, 10)
            self.h_etVieta = ROOT.TH2F(name+'_etVieta', 'Tower E_{T} (GeV) vs ieta; i#eta; E_{T} [GeV];',  18, 0, 18, 100, 0, 10)
            self.h_etEmVieta = ROOT.TH2F(name+'_etEmVieta', 'Tower E_{T} EM (GeV) vs ieta; i#eta; E_{T}^{EM} [GeV];',  18, 0, 18, 100, 0, 10)
            self.h_etHadVieta = ROOT.TH2F(name+'_etHadVieta', 'Tower E_{T} Had (GeV) vs ieta; i#eta; E_{T}^{HAD} [GeV];',  18, 0, 18, 100, 0, 10)
            self.h_sumEt = ROOT.TH1F(name+'_sumEt', 'Tower SumEt (GeV); E_{T}^{TOT} [GeV];', 200, 0, 400)
            self.h_sumEtCentral = ROOT.TH1F(name+'_sumEtCentral', 'Tower SumEt (GeV) (central); E_{T}^{TOT} [GeV];', 200, 0, 400)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, towers):
        rnp.fill_hist(self.h_pt, towers.pt)
        rnp.fill_hist(self.h_etEm, towers.etEm)
        rnp.fill_hist(self.h_etHad, towers.etHad)
        rnp.fill_hist(self.h_HoE, towers.HoE)
        rnp.fill_hist(self.h_HoEVpt, towers[['pt', 'HoE']])
        rnp.fill_hist(self.h_energy, towers.energy)
        rnp.fill_hist(self.h_eta, towers.eta)
        rnp.fill_hist(self.h_ieta, towers.iEta)
        rnp.fill_hist(self.h_ptVeta, towers[['eta', 'pt']])
        rnp.fill_hist(self.h_etVieta, towers[['iEta', 'pt']])
        rnp.fill_hist(self.h_etEmVieta, towers[['iEta', 'etEm']])
        rnp.fill_hist(self.h_etHadVieta, towers[['iEta', 'etHad']])
        if not towers.empty:
            self.h_sumEt.Fill(towers.momentum.sum().Pt())
            central_towers = towers[(towers.iEta != 0) & (towers.iEta != 17)]
            if not central_towers.empty:
                self.h_sumEtCentral.Fill(central_towers.momentum.sum().Pt())


class TriggerTowerResoHistos(BaseResoHistos):
    def __init__(self, name, root_file=None):
        if not root_file:
            self.h_ptRes = ROOT.TH1F(name+'_ptRes', 'TT Pt reso (GeV); (p_{T}^{L1}-p_{T}^{GEN})/p_{T}^{GEN};', 100, -1, 2)

            self.h_ptResVpt = ROOT.TH2F(name+'_ptResVpt', 'TT Pt reso (GeV) vs pt (GeV); p_{T}^{GEN} [GeV]; (p_{T}^{L1}-p_{T}^{GEN})/p_{T}^{GEN};', 50, 0, 100, 100, -1, 2)
            self.h_ptResVeta = ROOT.TH2F(name+'_ptResVeta', 'TT Pt reso (GeV) vs eta; #eta^{GEN}; (p_{T}^{L1}-p_{T}^{GEN})/p_{T}^{GEN};', 100, -3.5, 3.5, 100, -1, 2)

            self.h_ptResp = ROOT.TH1F(name+'_ptResp', 'TT Pt resp.; p_{T}^{L1}/p_{T}^{GEN};', 100, 0, 2)
            self.h_ptRespVpt = ROOT.TH2F(name+'_ptRespVpt', 'TT Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};', 50, 0, 100, 100, 0, 2)
            self.h_ptRespVeta = ROOT.TH2F(name+'_ptRespVeta', 'TT Pt resp. vs |#eta|; |#eta^{GEN}|; p_{T}^{L1}/p_{T}^{GEN};', 34, 1.4, 3.1, 100, 0, 2)

            self.h_energyRes = ROOT.TH1F(name+'_energyRes', 'TT Energy reso (GeV)', 200, -100, 100)
            self.h_energyResVeta = ROOT.TH2F(name+'_energyResVeta', 'TT E reso (GeV) vs eta', 100, -3.5, 3.5, 200, -100, 100)
            # FIXME: add corresponding Pt plots
            self.h_etaRes = ROOT.TH1F(name+'_etaRes', 'TT eta reso; #eta^{L1}-#eta^{GEN}', 100, -0.4, 0.4)
            self.h_phiRes = ROOT.TH1F(name+'_phiRes', 'TT phi reso; #phi^{L1}-#phi^{GEN}', 100, -0.4, 0.4)
            self.h_etalwRes = ROOT.TH1F(name+'_etalwRes', 'TT eta reso (lw)', 100, -0.4, 0.4)
            self.h_philwRes = ROOT.TH1F(name+'_philwRes', 'TT phi reso (lw)', 100, -0.4, 0.4)

            self.h_drRes = ROOT.TH1F(name+'_drRes', 'TT DR reso', 100, 0, 0.1)
        BaseResoHistos.__init__(self, name, root_file)

    def fill(self, reference, target):
        self.h_ptRes.Fill((target.pt - reference.pt)/reference.pt)
        self.h_ptResVpt.Fill(reference.pt, (target.pt - reference.pt)/reference.pt)
        self.h_ptResVeta.Fill(reference.eta, (target.pt - reference.pt)/reference.pt)

        self.h_ptResp.Fill(target.pt/reference.pt)
        self.h_ptRespVpt.Fill(reference.pt, target.pt/reference.pt)
        self.h_ptRespVeta.Fill(abs(reference.eta), target.pt/reference.pt)

        self.h_energyRes.Fill(target.energy - reference.energy)
        self.h_energyResVeta.Fill(reference.eta, target.energy - reference.energy)

        self.h_etaRes.Fill(target.eta - reference.eta)
        self.h_phiRes.Fill(target.phi - reference.phi)
        self.h_drRes.Fill(np.sqrt((reference.phi-target.phi)**2+(reference.eta-target.eta)**2))
        if 'etalw' in target:
            self.h_etalwRes.Fill(target.etalw - reference.eta)
        if 'philw' in target:
            self.h_philwRes.Fill(target.philw - reference.phi)


class ResoHistos(BaseResoHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptRes = ROOT.TH1F(
                name+'_ptRes', '3D Cluster Pt reso (GeV); (p_{T}^{L1} - p_{T}^{GEN})/p_{T}^{GEN}',
                100, -1, 1)
            # self.h_energyRes = ROOT.TH1F(name+'_energyRes', '3D Cluster Energy reso (GeV); E^{L1} - E^{GEN} [GeV]', 200, -100, 100)
            self.h_ptResVeta = ROOT.TH2F(
                name+'_ptResVeta', '3D Cluster Pt reso (GeV) vs eta; #eta^{GEN}; p_{T}^{L1} - p_{T}^{GEN} [GeV];',
                100, -3.5, 3.5, 200, -40, 40)
            self.h_energyResVenergy = ROOT.TH2F(
                name+'_energyResVenergy',
                '3D Cluster E reso vs E; E^{GEN} [GeV]; (E^{L1} - E^{GEN})/E^{GEN};',
                100, 0, 1000, 100, -1.5, 1.5)
            self.h_energyResVeta = ROOT.TH2F(
                name+'_energyResVeta',
                '3D Cluster E reso (GeV) vs eta; #eta^{GEN}; (E^{L1} - E^{GEN})/E^{GEN};',
                100, -3.5, 3.5, 100, -1.5, 1.5)
            # self.h_energyResVnclu = ROOT.TH2F(name+'_energyResVnclu', '3D Cluster E reso (GeV) vs # clusters; # 2D clus.; E^{L1} - E^{GEN} [GeV];', 50, 0, 50, 200, -100, 100)
            self.h_ptResVpt = ROOT.TH2F(
                name+'_ptResVpt',
                '3D Cluster Pt reso (GeV) vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1} - p_{T}^{GEN} [GeV];',
                50, 0, 100, 200, -40, 40)
            # self.h_ptResVnclu = ROOT.TH2F(name+'_ptResVnclu', '3D Cluster Pt reso (GeV) vs # clusters; # 2D clus.; p_{T}^{L1} - p_{T}^{GEN} [GeV];', 50, 0, 50, 200, -40, 40)

            self.h_ptResp = ROOT.TH1F(
                name+'_ptResp',
                '3D Cluster Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 2)
            self.h_ptRespVpt = ROOT.TH2F(
                name+'_ptRespVpt',
                '3D Cluster Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 2)
            self.h_ptRespVeta = ROOT.TH2F(
                name+'_ptRespVeta',
                '3D Cluster Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                50, -4, 4, 100, 0, 2)
            self.h_ptRespVnclu = ROOT.TH2F(
                name+'_ptRespVnclu',
                '3D Cluster Pt resp. vs # clus.; # 2D clust. ; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 2)
            self.h_ptRespVetaVptL1 = ROOT.TH3F(
                name+'_ptRespVetaVptL1',
                '3D Cluster Pt resp. vs #eta and vs pT; #eta^{L1}; p_{T}^{L1} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                30, 1, 4, 50, 0, 100, 100, 0, 3)

            self.h_ptemResp = ROOT.TH1F(
                name+'_ptemResp',
                '3D Cluster Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 3)
            self.h_ptemRespVpt = ROOT.TH2F(
                name+'_ptemRespVpt',
                '3D Cluster Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 3)

            # self.h_coreEnergyResVnclu = ROOT.TH2F(name+'_coreEnergyResVnclu', '3D Cluster E reso (GeV) vs # clusters', 50, 0, 50, 200, -100, 100)
            # self.h_corePtResVnclu = ROOT.TH2F(name+'_corePtResVnclu', '3D Cluster Pt reso (GeV) vs # clusters', 50, 0, 50, 200, -40, 40)
            #
            # self.h_coreEnergyRes = ROOT.TH1F(name+'_coreEnergyRes', '3D Cluster Energy reso CORE (GeV)', 200, -100, 100)
            # self.h_corePtRes = ROOT.TH1F(name+'_corePtRes', '3D Cluster Pt reso CORE (GeV)', 200, -40, 40)

            # self.h_centralEnergyRes = ROOT.TH1F(name+'_centralEnergyRes', '3D Cluster Energy reso CENTRAL (GeV)', 200, -100, 100)
            self.h_etaRes = ROOT.TH1F(
                name+'_etaRes',
                '3D Cluster eta reso; #eta^{L1}-#eta^{GEN}',
                100, -0.15, 0.15)
            self.h_phiRes = ROOT.TH1F(
                name+'_phiRes',
                '3D Cluster phi reso; #phi^{L1}-#phi^{GEN}',
                100, -0.15, 0.15)
            self.h_drRes = ROOT.TH1F(
                name+'_drRes',
                '3D Cluster DR reso; #DeltaR^{L1}-#DeltaR^{GEN}',
                100, 0, 0.1)
            self.h_n010 = ROOT.TH1F(
                name+'_n010',
                '# of 3D clus in 0.2 cone with pt>0.1GeV',
                10, 0, 10)
            self.h_n025 = ROOT.TH1F(
                name+'_n025',
                '# of 3D clus in 0.2 cone with pt>0.25GeV',
                10, 0, 10)

        BaseResoHistos.__init__(self, name, root_file)

    def fill(self, reference, target):
        target_line = target.iloc[0]
        target_pt = target_line.pt
        target_ptem = target_line.ptem
        target_energy = target_line.energy
        target_eta = target_line.eta
        target_phi = target_line.phi
        target_nclu = target_line.nclu

        reference_pt, reference_energy, reference_eta, reference_phi = \
            reference.pt, reference.energy, reference.eta, reference.phi

        self.h_ptRes.Fill((target_pt - reference_pt)/reference_pt)
        # self.h_energyRes.Fill(target_energy - reference.energy)
        self.h_ptResVeta.Fill(reference_eta, target_pt - reference_pt)
        self.h_ptResVpt.Fill(reference_pt, target_pt - reference_pt)
        self.h_energyResVeta.Fill(reference_eta, (target_energy - reference_energy)/reference_energy)
        self.h_energyResVenergy.Fill(reference_energy, (target_energy - reference_energy)/reference_energy)
        # self.h_energyResVnclu.Fill(target.nclu, target_energy - reference.energy)
        # self.h_ptResVnclu.Fill(target.nclu, target_pt - reference_pt)

        self.h_ptResp.Fill(target_pt/reference_pt)
        self.h_ptRespVeta.Fill(reference_eta, target_pt/reference_pt)
        self.h_ptRespVpt.Fill(reference_pt, target_pt/reference_pt)
        self.h_ptRespVnclu.Fill(target_nclu, target_pt/reference_pt)
        self.h_ptRespVetaVptL1.Fill(abs(target_eta), target_pt, target_pt/reference_pt)

        self.h_ptemResp.Fill(target_ptem/reference_pt)
        self.h_ptemRespVpt.Fill(reference_pt, target_ptem/reference_pt)

        # if 'energyCore' in target:
        #     self.h_coreEnergyRes.Fill(target_energyCore - reference.energy)
        #     self.h_corePtRes.Fill(target_ptCore - reference_pt)
        #
        #     self.h_coreEnergyResVnclu.Fill(target.nclu, target_energyCore - reference.energy)
        #     self.h_corePtResVnclu.Fill(target.nclu, target_ptCore - reference_pt)

        # if 'energyCentral' in target:
        #     self.h_centralEnergyRes.Fill(target_energyCentral - reference.energy)
        if 'exeta' in reference:
            self.h_etaRes.Fill(target_eta - reference.exeta)
            self.h_phiRes.Fill(target_phi - reference.exphi)
            self.h_drRes.Fill(np.sqrt((reference.exphi-target_phi)**2+(reference.exeta-target_eta)**2))

        if 'n010' in target:
            self.h_n010.Fill(target_line.n010)
        if 'n025' in target:
            self.h_n025.Fill(target_line.n025)


class Reso2DHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            # self.h_etaRes = ROOT.TH1F(name+'_etaRes', 'Eta 2D cluster - GEN part', 100, -0.5, 0.5)
            # self.h_phiRes = ROOT.TH1F(name+'_phiRes', 'Phi 2D cluster - GEN part', 100, -0.5, 0.5)
            # self.h_phiPRes = ROOT.TH1F(name+'_phiPRes', 'Phi (+) 2D cluster - GEN part', 100, -0.5, 0.5)
            # self.h_phiMRes = ROOT.TH1F(name+'_phiMRes', 'Phi (-) 2D cluster - GEN part', 100, -0.5, 0.5)
            self.h_xResVlayer = ROOT.TH2F(name+'_xResVlayer', 'X resolution (cm) [(2D clus) - GEN]', 60, 0, 60, 100, -10, 10)
            self.h_yResVlayer = ROOT.TH2F(name+'_yResVlayer', 'Y resolution (cm) [(2D clus) - GEN]', 60, 0, 60, 100, -10, 10)
            # self.h_DRRes = ROOT.TH1F(name+'_DRRes', 'DR 2D cluster - GEN part', 100, -0.5, 0.5)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, reference, target):
        # rnp.fill_hist(self.h_etaRes, reference.eta-target.eta)
        #
        # rnp.fill_hist(self.h_phiRes, reference.phi-target.phi)
        # if reference.pdgid < 0:
        #     rnp.fill_hist(self.h_phiMRes, reference.phi-target.phi)
        # elif reference.pdgid > 0:
        #     rnp.fill_hist(self.h_phiPRes, reference.phi-target.phi)

        # rnp.fill_hist(self.h_DRRes, np.sqrt((reference.phi-target.phi)**2+(reference.eta-target.eta)**2))
        if reference.reachedEE == 2:
            if 'x' in target.columns:
                target['xres'] = reference.posx[target.layer-1]-target.x
                # print target[['layer', 'xres']]
                rnp.fill_hist(self.h_xResVlayer, target[['layer', 'xres']])
            if 'y' in target.columns:
                target['yres'] = reference.posy[target.layer-1]-target.y
                # print target[['layer', 'yres']]
                rnp.fill_hist(self.h_yResVlayer, target[['layer', 'yres']])
            # print target[['layer', 'xres', 'yres']]


class GeomHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_maxNNDistVlayer = ROOT.TH2F(name+'_maxNNDistVlayer', 'Max dist between NN vs layer', 60, 0, 60, 100, 0, 10)
            self.h_minNNDistVlayer = ROOT.TH2F(name+'_minNNDistVlayer', 'Max dist between NN vs layer', 60, 0, 60, 100, 0, 10)

            self.h_nTCsPerLayer = ROOT.TH1F(name+'_nTCsPerLayer', '# of Trigger Cells per layer', 60, 0, 60)
            self.h_radiusVlayer = ROOT.TH2F(name+'_radiusVlayer', '# of cells radius vs layer', 60, 0, 60, 200, 0, 200)
        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tcs):
        if True:
            ee_tcs = tcs[tcs.subdet == 3]
            for index, tc_geom in ee_tcs.iterrows():
                self.h_maxNNDistVlayer.Fill(tc_geom.layer, np.max(tc_geom.neighbor_distance))
                self.h_minNNDistVlayer.Fill(tc_geom.layer, np.min(tc_geom.neighbor_distance))

        rnp.fill_hist(self.h_nTCsPerLayer, tcs[tcs.subdet == 3].layer)
        rnp.fill_hist(self.h_radiusVlayer, tcs[['layer', 'radius']])


class DensityHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_eDensityVlayer = ROOT.TH2F(name+'_eDensityVlayer', 'E (GeV) Density per layer', 60, 0, 60, 600, 0, 30)
            self.h_nTCDensityVlayer = ROOT.TH2F(name+'_nTCDensityVlayer', '# TC Density per layer', 60, 0, 60, 20, 0, 20)
        elif 'v7' in root_file.GetName() and "NuGun" not in root_file.GetName():
            print("v7 hack")
            self.h_eDensityVlayer = root_file.Get(name+'eDensityVlayer')
            self.h_nTCDensityVlayer = root_file.Get(name+'nTCDensityVlayer')
        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, layer, energy, nTCs):
        self.h_eDensityVlayer.Fill(layer, energy)
        self.h_nTCDensityVlayer.Fill(layer, nTCs)


# for convenience we define some sets
class HistoSetClusters():
    def __init__(self, name, root_file=None, debug=False):
        self.htc = TCHistos('h_tc_'+name, root_file, debug)
        self.hcl2d = ClusterHistos('h_cl2d_'+name, root_file, debug)
        self.hcl3d = Cluster3DHistos('h_cl3d_'+name, root_file, debug)
        # if not root_file:
        #     self.htc.annotateTitles(name)
        #     self.hcl2d.annotateTitles(name)
        #     self.hcl3d.annotateTitles(name)

    def fill(self, tcs, cl2ds, cl3ds):
        self.htc.fill(tcs)
        self.hcl2d.fill(cl2ds)
        self.hcl3d.fill(cl3ds)


class HistoSetReso():
    def __init__(self, name, root_file=None, debug=False):
        self.hreso = ResoHistos('h_reso_'+name, root_file, debug)
        self.hresoCone = None
        self.hreso2D = None
        # self.hresoCone = ResoHistos('h_resoCone_'+name, root_file)
        # self.hreso2D = Reso2DHistos('h_reso2D_'+name, root_file)
        # if not root_file:
        #     self.hreso.annotateTitles(name)
        #     self.hresoCone.annotateTitles(name)
        #     self.hreso2D.annotateTitles(name)


class HistoEff():
    def __init__(self, passed, total, rebin=None, debug=False):
        # print dir(total)
        for histo in [a for a in dir(total) if a.startswith('h_')]:
            if debug:
                print(histo)
            hist_total = getattr(total, histo)
            hist_passed = getattr(passed, histo)
            if rebin is None:
                setattr(self, histo, ROOT.TEfficiency(hist_passed, hist_total))
            else:
                setattr(self, histo, ROOT.TEfficiency(hist_passed.Rebin(rebin,
                                                                        '{}_rebin{}'.format(hist_passed.GetName(), rebin)),
                                                      hist_total.Rebin(rebin,
                                                                       '{}_rebin{}'.format(hist_total.GetName(), rebin))))
            # getattr(self, histo).Sumw2()


class HistoSetEff():
    def __init__(self, name, root_file=None, extended_range=False, debug=False):
        self.name = name
        self.h_num = GenParticleHistos('h_effNum_'+name, root_file, extended_range, debug)
        self.h_den = GenParticleHistos('h_effDen_'+name, root_file, extended_range, debug)
        self.h_eff = None
        self.h_ton = None

        if root_file:
            self.computeEff(debug=debug)

    def fillNum(self, particles):
        self.h_num.fill(particles)

    def fillDen(self, particles):
        self.h_den.fill(particles)

    def computeEff(self, rebin=None, debug=False):
        # print "Computing eff"
        if self.h_eff is None or rebin is not None:
            self.h_eff = HistoEff(passed=self.h_num, total=self.h_den, rebin=rebin, debug=debug)

    def computeTurnOn(self, denominator, debug=False):
            self.h_ton = HistoEff(passed=self.h_num, total=denominator, debug=debug)


class TrackResoHistos(BaseResoHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptResVpt = ROOT.TH2F(
                name+'_ptResVpt',
                'Track Pt reso. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}-p_{T}^{GEN} [GeV];',
                50, 0, 100, 100, -20, 20)
            self.h_ptResp = ROOT.TH1F(
                name+'_ptResp',
                'Track Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 3)
            self.h_ptRespVpt = ROOT.TH2F(
                name+'_ptRespVpt',
                'Track Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 3)
            self.h_ptRespVeta = ROOT.TH2F(
                name+'_ptRespVeta',
                'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                50, -4, 4, 100, 0, 3)
            self.h_etaRes = ROOT.TH1F(
                name+'_etaRes',
                'Track eta reso',
                100, -0.4, 0.4)
            self.h_phiRes = ROOT.TH1F(
                name+'_phiRes',
                'Track phi reso',
                100, -0.4, 0.4)
            self.h_drRes = ROOT.TH1F(
                name+'_drRes',
                'Track DR reso',
                100, 0, 0.4)
            self.h_nMatch = ROOT.TH1F(
                name+'_nMatch',
                '# matches',
                100, 0, 100)

            # self.h_pt2stResVpt = ROOT.TH2F(name+'_pt2stResVpt', 'EG Pt 2stubs reso. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}-p_{T}^{GEN} [GeV];',
            #                                50, 0, 100, 100, -20, 20)
            #
            # self.h_pt2stResp = ROOT.TH1F(name+'_pt2stResp', 'Track Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
            #                              100, 0, 3)
            # self.h_pt2stRespVpt = ROOT.TH2F(name+'_pt2stRespVpt', 'Track Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
            #                                 50, 0, 100, 100, 0, 3)
            # self.h_pt2stRespVeta = ROOT.TH2F(name+'_pt2stRespVeta', 'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
            #                                  50, -4, 4, 100, 0, 3)

        BaseResoHistos.__init__(self, name, root_file, debug)

    def fill(self, reference, target):
        # target_pt, target_eta, target_phi = \
        #     target[['pt', 'eta', 'phi']].values[0]
        # reference_pt, reference_eta, reference_phi = \
        #     reference[['pt', 'eta', 'phi']].values
        target_line = target.iloc[0]

        target_pt = target_line.pt
        target_eta = target_line.eta
        target_phi = target_line.phi
        reference_pt = reference.pt
        reference_eta = reference.eta
        reference_phi = reference.phi

        self.h_ptResVpt.Fill(reference_pt, target_pt-reference_pt)
        self.h_ptResp.Fill(target_pt/reference_pt)
        self.h_ptRespVeta.Fill(reference_eta, target_pt/reference_pt)
        self.h_ptRespVpt.Fill(reference_pt, target_pt/reference_pt)

        # self.h_pt2stResVpt.Fill(reference.pt, target.pt2stubs-reference.pt)
        # self.h_pt2stResp.Fill(target.pt2stubs/reference.pt)
        # self.h_pt2stRespVeta.Fill(reference.eta, target.pt2stubs/reference.pt)
        # self.h_pt2stRespVpt.Fill(reference.pt, target.pt2stubs/reference.pt)

        self.h_etaRes.Fill(target_eta - reference_eta)
        self.h_phiRes.Fill(target_phi - reference_phi)
        self.h_drRes.Fill(np.sqrt((reference_phi-target_phi)**2+(reference_eta-target_eta)**2))

    def fill_nMatch(self, n_matches):
        self.h_nMatch.Fill(n_matches)


class EGResoHistos(BaseResoHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:

            self.h_ptResVpt = ROOT.TH2F(name+'_ptResVpt', 'EG Pt reso. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}-p_{T}^{GEN} [GeV];', 50, 0, 100, 100, -10, 10)
            self.h_ptRes = ROOT.TH1F(name+'_ptRes', 'EG Pt res.; (p_{T}^{L1}-p_{T}^{GEN})/p_{T}^{GEN}', 100, -1, 1)

            self.h_ptResp = ROOT.TH1F(name+'_ptResp', 'EG Pt resp.; p_{T}^{L1}/p_{T}^{GEN}', 100, 0, 3)
            self.h_ptRespVpt = ROOT.TH2F(name+'_ptRespVpt', 'EG Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};', 50, 0, 100, 100, 0, 3)
            self.h_ptRespVeta = ROOT.TH2F(name+'_ptRespVeta', 'EG Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};', 50, -4, 4, 100, 0, 3)

            self.h_etaRes = ROOT.TH1F(name+'_etaRes', 'EG eta reso; #eta^{L1}-#eta^{GEN}', 100, -0.1, 0.1)
            self.h_phiRes = ROOT.TH1F(name+'_phiRes', 'EG phi reso; #phi^{L1}-#phi^{GEN}', 100, -0.1, 0.1)

            self.h_exetaRes = ROOT.TH1F(name+'_exetaRes', 'EG eta reso; #eta^{L1}-#eta^{GEN}_{calo}', 100, -0.1, 0.1)
            self.h_exphiRes = ROOT.TH1F(name+'_exphiRes', 'EG phi reso; #phi^{L1}-#phi^{GEN}_{calo}', 100, -0.1, 0.1)

        BaseResoHistos.__init__(self, name, root_file, debug)

    def fill(self, reference, target):
        # FIXME: reference is a series while target is a DF -> moving to a series saves a lot of time
        target_line = target.iloc[0]
        target_pt = target_line.pt
        target_eta = target_line.eta
        target_phi = target_line.phi

        reference_pt = reference.pt
        reference_eta = reference.eta
        reference_phi = reference.phi
        reference_exeta = reference.exeta
        reference_exphi = reference.exphi
        reference_weight = reference.weight

        self.h_ptRes.Fill((target_pt-reference_pt)/reference_pt, reference_weight)
        self.h_ptResVpt.Fill(reference_pt, target_pt-reference_pt, reference_weight)
        self.h_ptResp.Fill(target_pt/reference_pt, reference_weight)
        self.h_ptRespVeta.Fill(reference_eta, target_pt/reference_pt, reference_weight)
        self.h_ptRespVpt.Fill(reference_pt, target_pt/reference_pt, reference_weight)
        self.h_etaRes.Fill(target_eta - reference_eta, reference_weight)
        self.h_phiRes.Fill(target_phi - reference_phi, reference_weight)

        self.h_exetaRes.Fill(target_eta - reference_exeta, reference_weight)
        self.h_exphiRes.Fill(target_phi - reference_exphi, reference_weight)


class ClusterConeHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptRel = ROOT.TH1F(name+'_ptRel',
                                     'Pt best/Pt other; p_{T}^{best}/p_{T}^{other}', 100, 0, 5)
            self.h_ptRelVpt = ROOT.TH2F(name+'_ptRelVpt', 'Pt best/Pt other vs pt (GeV); p_{T}^{best} [GeV]; p_{T}^{best}/p_{T}^{other};', 50, 0, 100, 100, 0, 5)
            self.h_deltaEta = ROOT.TH1F(name+'_deltaEta', '#Delta eta; #eta^{best}-#eta^{other}', 100, -0.4, 0.4)
            self.h_deltaPhi = ROOT.TH1F(name+'_deltaPhi', '#Delta phi; #phi^{best}-#phi^{other}', 100, -0.4, 0.4)
            self.h_deltaPhiVq = ROOT.TH2F(name+'_deltaPhiVq', '#Delta phi; #phi^{best}-#phi^{other}; GEN charge;', 100, -0.4, 0.4, 3, -1, 2)

            self.h_deltaR = ROOT.TH1F(name+'_deltaR', '#Delta R (best-other); #Delta R (best, other)', 100, 0, 0.4)
            self.h_n = ROOT.TH1F(name+'_n', '# other clusters in cone; # others', 20, 0, 20)
        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, reference, target, charge):
        self.h_ptRel.Fill(target.pt/reference.pt)
        self.h_ptRelVpt.Fill(reference.pt, target.pt/reference.pt)
        self.h_deltaEta.Fill(target.eta - reference.eta)
        self.h_deltaPhi.Fill(target.phi - reference.phi)
        # self.h_deltaPhi.Fill(target.phi - reference.phi)
        self.h_deltaPhiVq.Fill(target.phi - reference.phi, charge)
        self.h_deltaR.Fill(np.sqrt((reference.phi-target.phi)**2+(reference.eta-target.eta)**2))

    def fill_n(self, num):
        self.h_n.Fill(num)


class IsoTuples(BaseTuples):
    def __init__(self, name, root_file=None, debug=False):
        BaseTuples.__init__(
            self, 'iso', 'pid_gen:e_gen:pt_gen:eta_gen:e:pt:eta:tkIso:pfIso:tkIsoPV:pfIsoPV',
            name, root_file, debug)

    def fill(self, reference, target):
        values_fill = []

        if reference is not None:
            values_fill.append(reference.pid)
            values_fill.append(reference.energy)
            values_fill.append(reference.pt)
            values_fill.append(reference.eta)
        else:
            values_fill.extend([-1]*4)

        values_fill.append(target.energy)
        values_fill.append(target.pt)
        values_fill.append(target.eta)
        values_fill.append(target.tkIso)
        values_fill.append(target.pfIso)
        if 'tkIsoPV' in target.keys():
            values_fill.append(target.tkIsoPV)
            values_fill.append(target.pfIsoPV)
        else:
            values_fill.append(-1)
            values_fill.append(-1)
            
        self.t_values.Fill(array('f', values_fill))


class ResoTuples(BaseTuples):
    def __init__(self, name, root_file=None, debug=False):
        BaseTuples.__init__(
            self, 'reso', 'e_gen:pt_gen:eta_gen:e:pt:eta',
            name, root_file, debug)

    def fill(self, reference, target):
        values_fill = []

        values_fill.append(reference.energy)
        values_fill.append(reference.pt)
        values_fill.append(reference.eta)
        values_fill.append(target.energy)
        values_fill.append(target.pt)
        values_fill.append(target.eta)
        self.t_values.Fill(array('f', values_fill))


class CalibrationHistos(BaseTuples):
    def __init__(self, name, root_file=None, debug=False):
        BaseTuples.__init__(
            self, 'calib',
            'e1:e3:e5:e7:e9:e11:e13:e15:e17:e19:e21:e23:e25:e27:Egen:eta:pt:ptgen',
            name, root_file, debug)

    def fill(self, reference, target):
        # cluster_data = []
        # self.data.append(target.iloc[0]['layer_energy'])
        # self.reference.append(reference.energy)
        values_fill = []
        values_fill.extend(target.iloc[0]['layer_energy'])
        values_fill.append(reference.energy)
        values_fill.append(target.eta)
        values_fill.append(target.pt)
        values_fill.append(reference.pt)
        self.t_values.Fill(array('f', values_fill))


class CorrOccupancyHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:

            self.h_avgOcc = ROOT.TProfile2D(
                name+'_avgOcc',
                'region avg occ; #eta, #phi;',
                pf_regions.regionizer.n_eta_regions(),
                array('d', pf_regions.regionizer.eta_boundaries_fiducial_),
                pf_regions.regionizer.n_phi_regions(),
                array('d', pf_regions.regionizer.phi_boundaries_fiducial_))

            self.h_maxOcc = ROOT.TH1F(name+'_maxOcc', 'max occupancy;', 100, 0, 100)
            self.h_totOcc = ROOT.TH1F(name+'_totOcc', 'total occupancy;', 500, 0, 500)
            self.h_regOcc = ROOT.TH1F(name+'_regOcc', 'reg occupancy;', 100, 0, 100)

            self.eta_regions_idx = range(0, pf_regions.regionizer.n_eta_regions())
            for key in ['all', 'BRL', 'HGCNoTk', 'HGC', 'HF']:
                if key in name:
                    self.eta_regions_idx = pf_regions.regions[key]

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, objects):
        max_count = 0
        tot_count = 0
        for ieta in self.eta_regions_idx:
            for iphi in range(0, pf_regions.regionizer.n_phi_regions()):
                occupancy = len(objects[objects['eta_reg_{}'.format(ieta)] & objects['phi_reg_{}'.format(iphi)]].index)

                # print occupancy
                self.h_avgOcc.Fill(pf_regions.regionizer.eta_centers[ieta],
                                   pf_regions.regionizer.phi_centers[iphi],
                                   occupancy)
                self.h_regOcc.Fill(occupancy)
                if occupancy > max_count:
                    max_count = occupancy
                tot_count += occupancy
        # print objects[:10]
        # print max_count
        self.h_maxOcc.Fill(max_count)
        self.h_totOcc.Fill(tot_count)


class TCClusterMatchHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:

            self.h_dEtaVdPhi = ROOT.TH2F(name+'_dEtaVdPhi',
                                         '#Delta#eta vs #Delta#phi; #Delta#phi [rad]; #Delta#eta;',
                                         100, -0.1, 0.1, 100, -0.1, 0.1)
            self.h_dEtaRMSVenergy = ROOT.TProfile(name+'_dEtaRMSVenergy',
                                                  'RMS(#Delta#eta) vs energy; E [GeV]; RMS(#Delta#eta);',
                                                  100, 0, 1000)
            self.h_dEtaRMSVpt = ROOT.TProfile(name+'_dEtaRMSVpt',
                                              'RMS(#Delta#eta) vs pt; p_{T} [GeV]; RMS(#Delta#eta);',
                                              100, 0, 100)

            self.h_dPhiRMSVenergy = ROOT.TProfile(name+'_dPhiRMSVenergy',
                                                  'RMS(#Delta#phi) vs energy; E [GeV]; RMS(#Delta#phi);',
                                                  100, 0, 1000)
            self.h_dPhiRMSVpt = ROOT.TProfile(name+'_dPhiRMSVpt',
                                              'RMS(#Delta#phi) vs pt; p_{T} [GeV]; RMS(#Delta#phi);',
                                              100, 0, 100)

            self.h_dRhoRMSVenergy = ROOT.TProfile(name+'_dRhoRMSVenergy',
                                         'RMS(#Delta#rho) vs energy; E [GeV]; RMS(#Delta#rho);',
                                         100, 0, 1000)
            self.h_dRhoRMSVpt = ROOT.TProfile(name+'_dRhoRMSVpt',
                                              'RMS(#Delta#rho) vs pt; p_{T} [GeV]; RMS(#Delta#rho);',
                                              100, 0, 100)
            self.h_dRho = ROOT.TH1F(name+'_dRho',
                                    '#Delta#rho; #Delta#rho;',
                                    100, 0, 0.1)
            self.h_dRho2 = ROOT.TH1F(name+'_dRho2',
                                    '#Delta#rho (E fraction weighted); #Delta#rho;',
                                    100, 0, 0.1)

            self.h_dRhoVlayer = ROOT.TH2F(name+'_dRhoVlayer',
                                    '#Delta#rho; layer #; #Delta#rho;',
                                    60, 0, 60, 100, 0, 0.1)
            self.h_dRhoVabseta = ROOT.TH2F(name+'_dRhoVabseta',
                                    '#Delta#rho; |#eta|; #Delta#rho;',
                                    100, 1.4, 3.1, 100, 0, 0.1)
            # self.h_dRhoVfbrem = ROOT.TH2F(name+'_dRhoVfbrem',
            #                         '#Delta#rho vs f_{brem}; f_{brem}; #Delta#rho;',
            #                         100, 0, 1, 100, 0, 0.1)
            self.h_dtVlayer = ROOT.TH2F(name+'_dtVlayer',
                                    '#Deltat vs layer; layer #; #Deltat;',
                                    60, 0, 60, 100, -0.05, 0.05)
            self.h_duVlayer = ROOT.TH2F(name+'_duVlayer',
                                    '#Delta#rho; layer #; #Deltau;',
                                    60, 0, 60, 100, -0.05, 0.05)

            self.h_dtVlayer2 = ROOT.TH2F(name+'_dtVlayer2',
                                    '#Deltat vs layer; layer #; #Deltat;',
                                    60, 0, 60, 100, -0.05, 0.05)
            self.h_duVlayer2 = ROOT.TH2F(name+'_duVlayer2',
                                    '#Delta#rho; layer #; #Deltau;',
                                    60, 0, 60, 100, -0.05, 0.05)

            self.h_dtVdu = ROOT.TH2F(name+'_dtVdu',
                                    '#Deltat vs #Deltau; #Deltat [cm]; #Deltau [cm];',
                                    100, -0.05, 0.05, 100, -0.05, 0.05)
            self.h_dtVdu2 = ROOT.TH2F(name+'_dtVdu2',
                                    '#Deltat vs #Deltau (E fract. weighted); #Deltat [cm]; #Deltau [cm];',
                                    100, -0.05, 0.05, 100, -0.05, 0.05)
            # self.h_fbremVabseta = ROOT.TH2F(name+'_fbremVabseta',
            #                         'f_{brem} vs |#eta|; |#eta|; f_{brem};',
            #                         100, 1.4, 3.1, 100, 0, 1)


        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tcs, cluster):
        rnp.fill_hist(self.h_dEtaVdPhi, tcs[['delta_phi', 'delta_eta']])
        # print tcs.dr
        # print tcs.delta_eta.std(), tcs.delta_phi.std(), tcs.dr.std()

        self.h_dEtaRMSVenergy.Fill(cluster.energy, tcs.delta_eta.std())
        self.h_dEtaRMSVpt.Fill(cluster.pt, tcs.delta_eta.std())
        self.h_dPhiRMSVenergy.Fill(cluster.energy, tcs.delta_phi.std())
        self.h_dPhiRMSVpt.Fill(cluster.pt, tcs.delta_phi.std())
        self.h_dRhoRMSVenergy.Fill(cluster.energy, tcs.dr.std())
        self.h_dRhoRMSVpt.Fill(cluster.pt, tcs.dr.std())

        rnp.fill_hist(self.h_dRho, tcs.dr)
        rnp.fill_hist(self.h_dRho2, tcs.dr, tcs.ef)
        rnp.fill_hist(self.h_dRhoVlayer, tcs[['layer', 'dr']])
        rnp.fill_hist(self.h_dtVlayer2, tcs[['layer', 'dt']], tcs['ef'])
        rnp.fill_hist(self.h_duVlayer2, tcs[['layer', 'du']], tcs['ef'])

        rnp.fill_hist(self.h_dtVlayer, tcs[['layer', 'dt']])
        rnp.fill_hist(self.h_duVlayer, tcs[['layer', 'du']])

        rnp.fill_hist(self.h_dRhoVabseta, tcs[['abseta_cl', 'dr']])
        # rnp.fill_hist(self.h_dRhoVfbrem, tcs[['fbrem_cl', 'dr']])
        rnp.fill_hist(self.h_dtVdu, tcs[['dt', 'du']])
        rnp.fill_hist(self.h_dtVdu2, tcs[['dt', 'du']], tcs['ef'])
        # self.h_fbremVabseta.Fill(cluster.abseta, cluster.fbrem)

# if __name__ == "__main__":
#     import sys
#     def createHisto(Class):
#         return Class(name='pippo_{}'.format(Class))
#
#     @profile
#     def createAll():
#         histos = []
#         histos.append(createHisto(Reso2DHistos))
#         # print sys.getsizeof(createHisto(Reso2DHistos))
#         histos.append(createHisto(GenParticleHistos))
#         histos.append(createHisto(RateHistos))
#         histos.append(createHisto(TCHistos))
#         histos.append(createHisto(ClusterHistos))
#         createHisto(Cluster3DHistos)
#         createHisto(TriggerTowerHistos)
#         createHisto(TriggerTowerResoHistos)
#         createHisto(ResoHistos)
#
#         return histos
#
#
#
#     createAll()
