from __future__ import print_function
import ROOT
# import root_numpy as rnp
import numpy as np
from array import array
# import pandas as pd
import uproot as up
import awkward as ak
import python.boost_hist as bh
import python.pf_regions as pf_regions
from scipy.special import expit
import hist

stuff = []

class HistoManager(object):
    class __TheManager:
        def __init__(self):
            self.val = None
            self.histoList = list()
            self.file = None

        def __str__(self):
            return 'self' + self.val



        def addHistos(self, histo):
            # print 'ADD histo: {}'.format(histo)
            self.histoList.append(histo)

        def writeHistos(self):
            for histo in self.histoList:
                histo.write(self.file)

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
            if debug:
                print(selhistos)
            for hinst, histo_name in selhistos:
                attr_name = 'h_'+histo_name.split(name+'_')[1]
                setattr(self, attr_name, hinst)
#            self.h_test = root_file.Get('h_EleReso_ptRes')
            # print 'XXXXXX'+str(self.h_test)
        else:
            # for histo in [a for a in dir(self) if a.startswith('h_')]:
                # FIXME
                # getattr(self, histo).Sumw2()
            hm = HistoManager()
            hm.addHistos(self)

    def write(self, upfile):
        dir_name = self.__class__.__name__
        for histo in [a for a in dir(self) if a.startswith('h_')]:
            writeable_hist = getattr(self, histo)
            # print (f"Writing {histo} class {writeable_hist.__class__.__name__}")
            if 'GraphBuilder' in writeable_hist.__class__.__name__ :
                continue
            elif 'TH1' in writeable_hist.__class__.__name__ or 'TH2' in writeable_hist.__class__.__name__:
                # print('start')
                # FIXME: this somehow fails randomply. ROOT not lining the right python???
                upfile[f'{dir_name}/{writeable_hist.GetName()}'] = writeable_hist
                # print('ok')
            else:
                up_writeable_hist = up.to_writable(writeable_hist)
                upfile[f'{dir_name}/{writeable_hist.label}'] = up_writeable_hist

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
            g_title = '{}; {}; {}'.format(
                h_title_parts[0],
                h_title_parts[1],
                title)
        elif len(h_title_parts) == 3:
            g_title += '{}; {}; {} ({})'.format(
                h_title_parts[0],
                h_title_parts[1],
                title,
                h_title_parts[2])

        graph = self.get_graph('{}_{}'.format(h2d.GetName(), name),
                               g_title, x, y, ex_l, ex_h, ey_l, ey_h)
        g_attr_name = 'g_{}_{}'.format(self.h_name.split('h_')[1], name)
        setattr(self.h_obj, g_attr_name, graph)
        return graph

    def Write(self, name='', options=None):
        return


class BaseResoHistos(BaseHistos):
    """

        Base class for resolution histogram classes.

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
        --------

    """

    def __init__(self, name, root_file=None, debug=False):
        BaseHistos.__init__(self, name, root_file, debug)
        if root_file is not None:
            # print dir(self)
            for attr_2d in [attr for attr in dir(self) if (attr.startswith('h_') and 'TH2' in getattr(self, attr).__class__.__name__)]:
                setattr(self, attr_2d+'_graph', GraphBuilder(self, attr_2d))


class BaseUpTuples(BaseHistos):
    def __init__(self, tuple_suffix, name, root_file=None, debug=False):
        self.t_name = '{}_{}'.format(name, tuple_suffix)
        self.init_ = False
        # if not root_file:
        #     self.t_name = '{}_{}'.format(name, tuple_suffix)
        if root_file:
            # print(root_file)
            # print(root_file.GetName())
            upfile = up.open(root_file.GetName(), num_workers=1)
            # print(upfile.keys())
            dir_name = self.__class__.__name__
            self.tree = upfile[f'{dir_name}/{self.t_name}']
        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, data):
        dir_name = self.__class__.__name__
        hm = HistoManager()
        obj_name = f'{dir_name}/{self.t_name}'
        # print( hm.file.keys())
        # print(f'OBJECT: {obj_name}')
        if self.init_:
            # print('extending')
            hm.file[f'{dir_name}/{self.t_name}'].extend(data)
        else:
            # print('creating')
            # hm.file.mktree(f'{dir_name}/{self.t_name}')
            hm.file[f'{dir_name}/{self.t_name}'] = data
            self.init_ = True

    def write(self, upfile):
        return


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


class CompCatTuples(BaseUpTuples):
    def __init__(self, name, root_file=None, debug=False):
        BaseUpTuples.__init__(
            self, "CompCatData", name, root_file, debug)

    def fill(self, reference, target):
        # print(self.t_name)
        # print(target.fields)
        target_vars = [
            'pt',
            'eta',
            'phi',
            'compChi2RZ',
            'compChi2RPhi',
            'compChi2Bend',
            'compNstubs',
            'tkPt',
            'compDphi',
            'compDeta',
            'compDpt',
            'compSrrtot',
            'compHoe',
            'compMeanz',
            'compBDTScore']
        rference_vars = [
            'pt',
            'eta',
            'phi',]
        # FIXME: add dz0 gen-track
        tree_data = {}
        for var in target_vars:
            tree_data[var] = ak.flatten(ak.drop_none(target[var]))
        for var in rference_vars:
            tree_data[f'gen_{var}'] = ak.flatten(ak.drop_none(reference[var]))
        # print(reference.fields)
        tree_data[f'gen_dz'] = ak.flatten(ak.drop_none(np.abs(reference.ovz-target.tkZ0)))
        
        BaseUpTuples.fill(self, tree_data)


class CompTuples(BaseUpTuples):
    def __init__(self, name, root_file=None, debug=False):
        BaseUpTuples.__init__(
            self, "CompData", name, root_file, debug)

    def fill(self, data):
# Index(['pt', 'energy', 'eta', 'phi', 'tkIso', 'pfIso', 'puppiIso', 'tkChi2',
#        'tkPt', 'tkZ0', 'compBDTScore', 'compBdt', 'compHoe', 'compSrrtot',
#        'compDeta', 'compDphi', 'compDpt', 'compMeanz', 'compNstubs',
#        'compChi2RPhi', 'compChi2RZ', 'compChi2Bend', 'dpt', 'hwQual',
#        'IDTightSTA', 'IDTightEle', 'IDTightPho', 'IDNoBrem', 'IDBrem'],
#       dtype='object')
        # FIXME: here we do the selection of the tree branches and other manipulations
        vars = [
            'pt',
            'eta',
            'phi',
            'compChi2RZ',
            'compChi2RPhi',
            'compChi2Bend',
            'compNstubs',
            'tkPt',
            'compDphi',
            'compDeta',
            'compDpt',
            'compSrrtot',
            'compHoe',
            'compMeanz',
            'compBDTScore']
        tree_data = {}
        for var in vars:
            if var in data.fields:
                tree_data[var] = data[var]
        BaseUpTuples.fill(self, tree_data)



class HGCIdTuples(BaseUpTuples):
    def __init__(self, name, root_file=None, debug=False):
        BaseUpTuples.__init__(
            self, "CompData", name, root_file, debug)

    def fill(self, data):
# Index(['pt', 'energy', 'eta', 'phi', 'tkIso', 'pfIso', 'puppiIso', 'tkChi2',
#        'tkPt', 'tkZ0', 'compBDTScore', 'compBdt', 'compHoe', 'compSrrtot',
#        'compDeta', 'compDphi', 'compDpt', 'compMeanz', 'compNstubs',
#        'compChi2RPhi', 'compChi2RZ', 'compChi2Bend', 'dpt', 'hwQual',
#        'IDTightSTA', 'IDTightEle', 'IDTightPho', 'IDNoBrem', 'IDBrem'],
#       dtype='object')
        # FIXME: here we do the selection of the tree branches and other manipulations
        vars = ['rho', 'phi', 'eta', 'coreshowerlength', 'ebm0', 'ebm1',
       'firstlayer', 'hbm', 'hwQual', 'maxlayer', 'nTcs', 'showerlength',
       'emax1layers', 'emax3layers', 'emax5layers', 'emaxe', 'eot',
       'first1layers', 'first3layers', 'first5layers', 'firstHcal1layers',
       'firstHcal3layers', 'firstHcal5layers', 'hoe', 'last1layers',
       'last3layers', 'last5layers', 'layer10', 'layer50', 'layer90', 'meanz',
       'ntc67', 'ntc90', 'ptEm', 'seemax', 'seetot', 'sppmax', 'spptot',
       'srrmax', 'srrmean', 'srrtot', 'szz', 'varEtaEta', 'varPhiPhi', 'varRR',
       'varZZ', 'pfPuIdPass', 'pfEmIdPass', 'pfPuIdScore', 'pfEmIdScore',
       'egEmIdScore', 'IDTightEm', 'IDLooseEm', 'eMax']
        tree_data = {}
        for var in vars:
            if var in data.fields:
                tree_data[var] = data[var]
        BaseUpTuples.fill(self, tree_data)

class HGCIdMatchTuples(BaseUpTuples):
    def __init__(self, name, root_file=None, debug=False):
        BaseUpTuples.__init__(
            self, "CompCatData", name, root_file, debug)

    def fill(self, reference, target):
        # print(self.t_name)
        # print(target.fields)
        target_vars = ['rho', 'phi', 'eta', 'coreshowerlength', 'ebm0', 'ebm1',
       'firstlayer', 'hbm', 'hwQual', 'maxlayer', 'nTcs', 'showerlength',
       'emax1layers', 'emax3layers', 'emax5layers', 'emaxe', 'eot',
       'first1layers', 'first3layers', 'first5layers', 'firstHcal1layers',
       'firstHcal3layers', 'firstHcal5layers', 'hoe', 'last1layers',
       'last3layers', 'last5layers', 'layer10', 'layer50', 'layer90', 'meanz',
       'ntc67', 'ntc90', 'ptEm', 'seemax', 'seetot', 'sppmax', 'spptot',
       'srrmax', 'srrmean', 'srrtot', 'szz', 'varEtaEta', 'varPhiPhi', 'varRR',
       'varZZ', 'pfPuIdPass', 'pfEmIdPass', 'pfPuIdScore', 'pfEmIdScore',
       'egEmIdScore', 'IDTightEm', 'IDLooseEm', 'eMax']
        rference_vars = [
            'rho',
            'eta',
            'phi',
            'pdgid',
            'caloeta', 
            'calophi']
        # FIXME: add dz0 gen-track
        tree_data = {}
        for var in target_vars:
            tree_data[var] = ak.flatten(ak.drop_none(target[var]))
        for var in rference_vars:
            tree_data[f'gen_{var}'] = ak.flatten(ak.drop_none(reference[var]))
        # print(reference.fields)
        # tree_data[f'gen_dz'] = ak.flatten(ak.drop_none(np.abs(reference.ovz-target.tkZ0)))
        
        BaseUpTuples.fill(self, tree_data)



class GenPartHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        self.h_pt = bh.TH1F(name+'_pt', 'Gen Part Pt (GeV)', 100, 0, 100)
        self.h_energy = bh.TH1F(name+'_energy', 'Gen Part Energy (GeV)', 100, 0, 1000)
        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, gps):
        bh.fill_1Dhist(self.h_pt, gps.pt)
        bh.fill_1Dhist(self.h_energy, gps.energy)

    # def write(self):
    #     for histo in [a for a in dir(self) if a.startswith('h_')]:
    #         getattr(self, histo).Write()





class GenParticleHistos(BaseHistos):
    def __init__(self, name, root_file=None, pt_bins=None, debug=False):
        if not root_file:
            self.h_eta = bh.TH1F(name+'_eta', 'Gen Part eta; #eta^{GEN};', 50, -3, 3)
            self.h_abseta = bh.TH1F(name+'_abseta', 'Gen Part |eta|; |#eta^{GEN}|;', 40, 0, 4)

            if pt_bins is None:
                self.h_pt = bh.TH1F(name+'_pt', 'Gen Part P_{T} (GeV); p_{T}^{GEN} [GeV];', 50, 0, 100)
            else:
                self.h_pt = hist.Hist(
                    hist.axis.Variable(pt_bins, name='p_{T}^{GEN} [GeV]'),
                    label=name+'_pt', 
                    name='Gen Part P_{T} (GeV)',
                    storage=hist.storage.Weight()
                    )
            # print ('bins: {}'.format(pt_bins))
            # print ("# bins: {}".format(n_pt_bins))

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, particles):
        weights = None
        if 'weights' in particles.fields:
            weights = particles.weights

        bh.fill_1Dhist(hist=self.h_eta,
                       array=particles.eta)
        bh.fill_1Dhist(hist=self.h_abseta,
                       array=particles.abseta)
        bh.fill_1Dhist(hist=self.h_pt,
                       array=particles.pt)


class GenParticleExtraHistos(GenParticleHistos):
    def __init__(self, name, root_file=None, pt_bins=None, debug=False):
        if not root_file:
            self.h_n = bh.TH1F(name+'_#', 'Gen Part #; #', 10, 0, 10)
            self.h_pdgid = bh.TH1F(name+'_pdgid', 'Gen Part pdgid; pdgid;', 100, -50, 50)

        GenParticleHistos.__init__(self, name, root_file, pt_bins, debug)

    def fill(self, particles):
        weights = None
        if 'weights' in particles.fields:
            weights = particles.weights

        self.h_n.fill(ak.count(particles.pt, axis=1))
        bh.fill_1Dhist(hist=self.h_pdgid,
                       array=particles.pdgid)


class DigiHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_layer = bh.TH1F(name+'_layer', 'Digi layer #', 60, 0, 60)
            # self.h_simenergy = bh.TH1F(name+'_energy', 'Digi sim-energy (GeV)', 100, 0, 2)
        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, digis):
        bh.fill_1Dhist(self.h_layer, digis.layer)
        # rnp.fill_hist(self.h_simenergy, digis.simenergy)


class RateHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_norm = bh.TH1F(name+'_norm', '# of events', 1, 1, 2)
            self.h_pt = bh.TH1F(name+'_pt', 'rate above p_{T} thresh.; p_{T} [GeV]; rate [kHz];', 100, 0, 100)
            # self.h_ptVabseta = bh.TH2F(name+'_ptVabseta', 'Candidate p_{T} vs |#eta|; |#eta|; p_{T} [GeV];', 34, 1.4, 3.1, 100, 0, 100)

        BaseHistos.__init__(self, name, root_file, debug)

        if root_file is not None:
            for attr_1d in [attr for attr in dir(self) if (attr.startswith('h_') and 'TH1' in getattr(self, attr).ClassName())]:
                setattr(self, attr_1d+'_graph', GraphBuilder(self, attr_1d))

        if root_file is not None:
            self.normalize(2760.0*11246/1000)
            # self.h_simenergy = bh.TH1F(name+'_energy', 'Digi sim-energy (GeV)', 100, 0, 2)

    def fill(self, data):
        # print(self.h_pt.axes[0])
        pt_max = ak.max(data.pt, axis=1)
        for thr,bin_center in zip(self.h_pt.axes[0].edges, self.h_pt.axes[0].centers):
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


class RateHistoCounter(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_norm = bh.TH1F(name+'_norm', '# of events', 1, 1, 2)
            self.h_rate = bh.TH1F(name+'_rate', '# passing events; rate [kHz];',  1, 1, 2)

        BaseHistos.__init__(self, name, root_file, debug)

        if root_file is not None or True:
            for attr_1d in [attr for attr in dir(self) if (attr.startswith('h_') and 'TH1' in getattr(self, attr).ClassName())]:
                setattr(self, attr_1d+'_graph', GraphBuilder(self, attr_1d))

        if root_file is not None:
            self.normalize(31000)
            # self.h_simenergy = bh.TH1F(name+'_energy', 'Digi sim-energy (GeV)', 100, 0, 2)

    def fill(self, count):
        self.h_rate.Fill(1, count)
    
    def fill_norm(self, many=1):
        # print (f' fill rate norm: {many}')
        self.h_norm.Fill(1, many)

    def normalize(self, norm):
        nev = self.h_norm.GetBinContent(1)
        # print(f' .      # ev: {nev}')
        if(nev != norm):
            print('normalize to {}'.format(norm))
            self.h_norm.Scale(norm/nev)
            self.h_rate.Scale(norm/nev)


class SingleObjRateHistoCounter(RateHistoCounter):
    def __init__(self, name, root_file=None, debug=False):
        RateHistoCounter.__init__(self, name, root_file, debug)

    def fill(self, df):
        # print(f' .  # of surviving entries: ')
        # print(df['pt'].groupby(level='entry', group_keys=False).nlargest(n=1).count())
        RateHistoCounter.fill(self, df['pt'].groupby(level='entry', group_keys=False).nlargest(n=1).count())
    

class DoubleObjRateHistoCounter(RateHistoCounter):
    def __init__(self, name, root_file=None, debug=False):
        RateHistoCounter.__init__(self, name, root_file, debug)

    def has_unique_pairs(entry):
        unique_pairs = [(l1, l2) for l1 in entry.loc[entry.index.get_level_values('leg') == 0].index for l2 in entry.loc[entry.index.get_level_values('leg') == 1].index if ((l1[2] != l2[2]) and (l1[1] != l2[1]))]
        return len(unique_pairs) > 0

    def fill(self, df):
        # print(f' .  # of surviving entries: ')
        # print(df.groupby(level='entry').filter(DoubleObjRateHistoCounter.has_unique_pairs).index.unique('entry').size)
        RateHistoCounter.fill(self, df.groupby(level='entry').filter(DoubleObjRateHistoCounter.has_unique_pairs).index.unique('entry').size)


class TCHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_energy = bh.TH1F(name+'_energy', 'TC energy (GeV)', 100, 0, 2)
            self.h_subdet = bh.TH1F(name+'_subdet', 'TC subdet #', 8, 0, 8)
            self.h_mipPt = bh.TH1F(name+'_mipPt', 'TC MIP Pt', 50, 0, 10)

            self.h_layer = ROOT.TProfile(name+'_layer', 'TC layer #', 60, 0, 60, 's')
            self.h_absz = bh.TH1F(name+'_absz', 'TC z(cm)', 100, 300, 500)
            self.h_wafertype = bh.TH1F(name+'_wafertype', 'Wafer type', 10, 0, 10)
            self.h_layerVenergy = bh.TH2F(name+'_layerVenergy', "Energy (GeV) vs Layer #", 60, 0, 60, 100, 0, 2)
            self.h_energyVeta = bh.TH2F(name+'_energyVeta', "Energy (GeV) vs Eta", 100, -3.5, 3.5, 100, 0, 2)
            # self.h_energyVetaL1t5 = bh.TH2F(name+'_energyVetaL1t5', "Energy (GeV) vs Eta (layers 1 to 5)", 100, -3.5, 3.5, 100, 0, 2)
            # self.h_energyVetaL6t10 = bh.TH2F(name+'_energyVetaL6t10', "Energy (GeV) vs Eta (layers 6 to 10)", 100, -3.5, 3.5, 100, 0, 2)
            # self.h_energyVetaL11t20 = bh.TH2F(name+'_energyVetaL11t20', "Energy (GeV) vs Eta (layers 11 to 20)", 100, -3.5, 3.5, 100, 0, 2)
            # self.h_energyVetaL21t60 = bh.TH2F(name+'_energyVetaL21t60', "Energy (GeV) vs Eta (layers 21 to 60)", 100, -3.5, 3.5, 100, 0, 2)
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
            self.h_energy = bh.TH1F(name+'_energy', 'Cluster energy (GeV); E [GeV];', 100, 0, 30)
            self.h_layer = bh.TH1F(name+'_layer', 'Cluster layer #; layer #;', 60, 0, 60)
            # self.h_nCoreCells = bh.TH1F(name+'_nCoreCells', 'Cluster # cells (core)', 30, 0, 30)

            self.h_layerVenergy = bh.TH2F(name+'_layerVenergy', "Cluster Energy (GeV) vs Layer #; layer; E [GeV];", 50, 0, 50, 100, 0, 20)
            self.h_ncells = bh.TH1F(name+'_ncells', 'Cluster # cells; # TC components;', 30, 0, 30)
            self.h_layerVncells = bh.TH2F(name+'_layerVncells', "Cluster #cells vs Layer #; layer; # TC components;",  50, 0, 50, 30, 0, 30)
            # self.h_layerVnCoreCells = bh.TH2F(name+'_layerVnCoreCells', "Cluster #cells vs Layer #",  50, 0, 50, 30, 0, 30)

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
            # self.h_npt05 = bh.TH1F(
            #     name+'_npt05', '# 3D Cluster Pt > 0.5 GeV; # 3D clusters in cone;', 1000, 0, 1000)
            # self.h_npt20 = bh.TH1F(
            #     name+'_npt20', '# 3D Cluster Pt > 2.0 GeV; # 3D clusters in cone;', 1000, 0, 1000)
            self.h_pt = bh.TH1F(
                name+'_pt', '3D Cluster Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(name+'_eta', '3D Cluster eta; #eta;', 100, -4, 4)
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

        BaseHistos.__init__(self, name, root_file, debug)

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


class EGHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(name+'_pt', 'EG Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(name+'_eta', 'EG eta; #eta;', 100, -4, 4)
            self.h_energy = bh.TH1F(name+'_energy', 'EG energy (GeV); E [GeV]', 1000, 0, 1000)
            self.h_hwQual = bh.TH1F(name+'_hwQual', 'EG energy (GeV); hwQual', 5, 0, 5)
            self.h_tkIso = bh.TH1F(name+'_tkIso', 'Iso; rel-iso_{tk}', 100, 0, 2)
            self.h_pfIso = bh.TH1F(name+'_pfIso', 'Iso; rel-iso_{pf}', 100, 0, 2)
            self.h_tkIsoPV = bh.TH1F(name+'_tkIsoPV', 'Iso; rel-iso^{PV}_{tk}', 100, 0, 2)
            self.h_pfIsoPV = bh.TH1F(name+'_pfIsoPV', 'Iso; rel-iso^{PV}_{pf}', 100, 0, 2)
            self.h_n = bh.TH1F(name+'_n', '# objects per event', 100, 0, 100)
            self.h_compBdt = bh.TH1F(name+'_compBdt', 'BDT Score Comp ID', 50, 0, 1)

        BaseHistos.__init__(self, name, root_file, debug)

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
        if 'tkIsoPV' in egs.fields:
            bh.fill_1Dhist(hist=self.h_tkIsoPV, array=egs.tkIsoPV, weights=weight)
            bh.fill_1Dhist(hist=self.h_pfIsoPV, array=egs.pfIsoPV, weights=weight)
        if 'compBDTScore' in egs.fields:
            bh.fill_1Dhist(hist=self.h_compBdt, array=egs.compBDTScore, weights=weight)
        if 'idScore' in egs.fields:
            bh.fill_1Dhist(hist=self.h_compBdt, array=expit(egs.idScore), weights=weight)
        # print(ak.count(egs.pt, axis=1))
        # print(egs.pt.type.show())
        # print(ak.count(egs.pt, axis=1).type.show())
        self.h_n.fill(ak.count(egs.pt, axis=1))
        # bh.fill_1Dhist(hist=self.h_n, array=ak.count(egs.pt, axis=1), weights=weight)
        # self.h_n.Fill()


    def add_histos(self):
        self.h_pt.Add(self.h_pt_temp.GetValue())
        self.h_eta.Add(self.h_eta_temp.GetValue())
        self.h_energy.Add(self.h_energy_temp.GetValue())
        self.h_hwQual.Add(self.h_hwQual_temp.GetValue())
        # self.h_tkIso = bh.TH1F(name+'_tkIso', 'Iso; rel-iso_{tk}', 100, 0, 2)
        # self.h_pfIso = bh.TH1F(name+'_pfIso', 'Iso; rel-iso_{pf}', 100, 0, 2)
        # self.h_tkIsoPV = bh.TH1F(name+'_tkIsoPV', 'Iso; rel-iso^{PV}_{tk}', 100, 0, 2)
        # self.h_pfIsoPV = bh.TH1F(name+'_pfIsoPV', 'Iso; rel-iso^{PV}_{pf}', 100, 0, 2)


class DecTkHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(
                name+'_pt',
                'Pt (GeV); p_{T} [GeV]',
                100, 0, 100)
            self.h_deltaPt = bh.TH1F(
                name+'_deltaPt',
                'Pt (GeV); p_{T}^{decoded}-p_{T}^{float}  [GeV]',
                100, -10, 10)
            self.h_deltaPtVeta = bh.TH2F(
                name+'_deltaPtVeta',
                'Pt (GeV); #eta^{float}; p_{T}^{decoded}-p_{T}^{float}  [GeV]',
                50, -2.5, 2.5,
                50, -0.25, 0.25)
            self.h_deltaPtVabseta = bh.TH2F(
                name+'_deltaPtVabseta',
                'Pt (GeV); |#eta^{float}|; p_{T}^{decoded}-p_{T}^{float}  [GeV]',
                50, 0, 2.5,
                50, -0.25, 0.25)
            self.h_eta = bh.TH1F(
                name+'_eta',
                '#eta; #eta;',
                100, -4, 4)
            self.h_z0 = bh.TH1F(
                name+'_z0',
                'z0; z_{0} [cm];',
                100, -10, 10)
            self.h_deltaZ0 = bh.TH1F(
                name+'_deltaZ0',
                '#Delta z0; z0^{decoded}-z0^{float};',
                50, -0.2, 0.2)
            self.h_deltaZ0Veta = bh.TH2F(
                name+'_deltaZ0Veta',
                '#Delta z0; #eta^{float}; z0^{decoded}-z0^{float};',
                100, -2.5, 2.5,
                50, -0.05, 0.05)
            self.h_deltaEta = bh.TH1F(
                name+'_deltaEta',
                '#Delta #eta_{@vtx}; #eta_{@vtx}^{decoded}-#eta_{@vtx}^{float};',
                100, -1, 1)
            self.h_deltaEtaVabseta = bh.TH2F(
                name+'_deltaEtaVabseta',
                '#Delta #eta_{@vtx} vs |#eta^{float}|; |#eta^{float}|; #eta_{@vtx}^{decoded}-#eta_{@vtx}^{float};',
                25, 0, 2.5,
                100, -0.004, 0.004)
            self.h_deltaEtaVeta = bh.TH2F(
                name+'_deltaEtaVeta',
                '#Delta #eta_{@vtx} vs #eta^{float}; #eta^{float}; #eta_{@vtx}^{decoded}-#eta_{@vtx}^{float};',
                50, -2.5, 2.5,
                50, -0.004, 0.004)
            self.h_deltaCaloEta = bh.TH1F(
                name+'_deltaCaloEta',
                '#Delta #eta_{@calo}; #eta_{@calo}^{decoded}-#eta_{@calo}^{float};',
                100, -1, 1)
            self.h_deltaCaloEtaVabseta = bh.TH2F(
                name+'_deltaCaloEtaVabseta',
                '#Delta #eta_{@calo} vs |#eta^{float}|; |#eta^{float}|; #eta_{@calo}^{decoded}-#eta_{@calo}^{float};',
                50, 0, 2.5,
                100, -0.04, 0.04)
            self.h_deltaCaloEtaVeta = bh.TH2F(
                name+'_deltaCaloEtaVeta',
                '#Delta #eta_{@calo} vs #eta^{float}; #eta^{float}; #eta_{@calo}^{decoded}-#eta_{@calo}^{float};',
                100, -2.5, 2.5,
                100, -0.04, 0.04)
            self.h_deltaCaloPhi = bh.TH1F(
                name+'_deltaCaloPhi',
                '#Delta #phi_{@calo}; #phi_{@calo}^{decoded}-#phi_{@calo}^{float};',
                100, -1, 1)
            self.h_deltaCaloPhiVabseta = bh.TH2F(
                name+'_deltaCaloPhiVabseta',
                '#Delta #phi_{@calo} vs |#eta^{float}|; |#phi^{float}|; #phi_{@calo}^{decoded}-#phi_{@calo}^{float};',
                100, 0, 2.5,
                100, -0.1, 0.1)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, egs):
        bh.fill_1Dhist(self.h_pt, egs.pt)
        bh.fill_1Dhist(self.h_deltaPt, egs.deltaPt)
        bh.fill_2Dhist(self.h_deltaPtVeta, egs.simeta, egs.deltaPt)
        bh.fill_2Dhist(self.h_deltaPtVabseta, egs.simabseta, egs.deltaPt)
        bh.fill_1Dhist(self.h_eta, egs.eta)
        bh.fill_1Dhist(self.h_z0, egs.z0)
        bh.fill_1Dhist(self.h_deltaZ0, egs.deltaZ0)
        bh.fill_2Dhist(self.h_deltaZ0Veta, egs.simeta, egs.deltaZ0)
        bh.fill_1Dhist(self.h_deltaEta, egs.deltaEta)
        bh.fill_2Dhist(self.h_deltaEtaVabseta, egs.simabseta, egs.deltaEta)
        bh.fill_2Dhist(self.h_deltaEtaVeta, egs.simeta, egs.deltaEta)
        bh.fill_1Dhist(self.h_deltaCaloEta, egs.deltaCaloEta)
        bh.fill_2Dhist(self.h_deltaCaloEtaVabseta, egs.simabseta, egs.deltaCaloEta)
        bh.fill_2Dhist(self.h_deltaCaloEtaVeta, egs.simeta, egs.deltaCaloEta)
        bh.fill_1Dhist(self.h_deltaCaloPhi, egs.deltaCaloPhi)
        bh.fill_2Dhist(self.h_deltaCaloPhiVabseta, egs.simabseta, egs.deltaCaloPhi)


class TkEleHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(name+'_pt', 'Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(name+'_eta', 'eta; #eta;', 100, -2.5, 2.5)
            self.h_energy = bh.TH1F(name+'_energy', 'energy (GeV); E [GeV]', 1000, 0, 1000)
            self.h_hwQual = bh.TH1F(name+'_hwQual', 'quality; hwQual', 10, 0, 10)
            self.h_tkpt = bh.TH1F(name+'_tkpt', 'Tk Pt (GeV); p_{T}^{L1Tk} [GeV]', 100, 0, 100)
            self.h_dpt = bh.TH1F(name+'_dpt', 'Delta Tk Pt (GeV); #Delta p_{T}^{L1Tk-Calo} [GeV]', 100, -50, 50)
            self.h_tkchi2 = bh.TH1F(name+'_tkchi2', 'Tk chi2; #Chi^{2}', 1000, 0, 1000)
            self.h_ptVtkpt = bh.TH2F(name+'_ptVtkpt', 'TkEG Pt (GeV) vs TkPt; p_{T}^{Tk} [GeV]; p_{T}^{EG} [GeV]', 100, 0, 100, 100, 0, 100)
            self.h_tkIso = bh.TH1F(name+'_tkIso', 'Iso; rel-iso_{tk}', 100, 0, 2)
            self.h_pfIso = bh.TH1F(name+'_pfIso', 'Iso; rel-iso_{pf}', 100, 0, 2)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tkegs):
        bh.fill_1Dhist(self.h_pt, tkegs.pt)
        bh.fill_1Dhist(self.h_eta, tkegs.eta)
        bh.fill_1Dhist(self.h_energy, tkegs.energy)
        bh.fill_1Dhist(self.h_hwQual, tkegs.hwQual)
        bh.fill_1Dhist(self.h_tkpt, tkegs.tkPt)
        bh.fill_1Dhist(self.h_dpt, tkegs.dpt)
        bh.fill_1Dhist(self.h_tkchi2, tkegs.tkChi2)
        bh.fill_1Dhist(self.h_tkIso, tkegs.tkIso)
        bh.fill_1Dhist(self.h_pfIso, tkegs.pfIso)



class TkEmHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(name+'_pt', 'Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(name+'_eta', 'eta; #eta;', 100, -2.5, 2.5)
            self.h_energy = bh.TH1F(name+'_energy', 'energy (GeV); E [GeV]', 1000, 0, 1000)
            self.h_hwQual = bh.TH1F(name+'_hwQual', 'quality; hwQual', 10, 0, 10)
            self.h_tkIso = bh.TH1F(name+'_tkIso', 'Iso; rel-iso_{tk}', 100, 0, 2)
            self.h_pfIso = bh.TH1F(name+'_pfIso', 'Iso; rel-iso_{pf}', 100, 0, 2)
            self.h_tkIsoPV = bh.TH1F(name+'_tkIsoPV', 'Iso; rel-iso^{PV}_{tk}', 100, 0, 2)
            self.h_pfIsoPV = bh.TH1F(name+'_pfIsoPV', 'Iso; rel-iso^{PV}_{pf}', 100, 0, 2)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tkegs):
        bh.fill_1Dhist(self.h_pt, tkegs.pt)
        bh.fill_1Dhist(self.h_eta, tkegs.eta)
        bh.fill_1Dhist(self.h_energy, tkegs.energy)
        bh.fill_1Dhist(self.h_hwQual, tkegs.hwQual)
        bh.fill_1Dhist(self.h_tkIso, tkegs.tkIso)
        bh.fill_1Dhist(self.h_pfIso, tkegs.pfIso)
        bh.fill_1Dhist(self.h_tkIsoPV, tkegs.tkIsoPV)
        bh.fill_1Dhist(self.h_pfIsoPV, tkegs.pfIsoPV)


class TkEGHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(name+'_pt', 'TkEG Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(name+'_eta', 'TkEG eta; #eta;', 100, -4, 4)
            self.h_energy = bh.TH1F(name+'_energy', 'TkEG energy (GeV); E [GeV]', 1000, 0, 1000)
            self.h_hwQual = bh.TH1F(name+'_hwQual', 'TkEG energy (GeV); hwQual', 5, 0, 5)

            self.h_tkpt = bh.TH1F(name+'_tkpt', 'TkEG Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_tketa = bh.TH1F(name+'_tketa', 'TkEG eta; #eta;', 100, -4, 4)
            self.h_tkchi2 = bh.TH1F(name+'_tkchi2', 'TkEG chi2; #Chi^{2}', 1000, 0, 1000)
            self.h_tkchi2Red = bh.TH1F(name+'_tkchi2Red', 'TkEG chi2 red; reduced #Chi^{2}', 100, 0, 100)
            self.h_tknstubs = bh.TH1F(name+'_tknstubs', 'TkEG # stubs; # stubs', 10, 0, 10)
            self.h_tkz0 = bh.TH1F(name+'_tkz0', 'TkEG z0; z_{0} [cm]', 100, -10, 10)
            self.h_tkchi2RedVeta = bh.TH2F(name+'_tkchi2RedVeta', 'TkEG chi2 red. v eta; #eta; red. #Chi^{2}', 100, -4, 4, 100, 0, 100)
            self.h_tknstubsVeta = bh.TH2F(name+'_tknstubsVeta', 'TkEG # stubs vs eta; #eta; # stubs', 100, -4, 4, 10, 0, 10)
            self.h_tkz0Veta = bh.TH2F(name+'_tkz0Veta', 'TkEG z0 vs eta; #eta; z_{0} [cm]', 100, -4, 4, 100, -10, 10)
            self.h_dphi = bh.TH1F(name+'_dphi', 'TkEG #Delta #phi; #Delta #phi [rad]', 100, -0.2, 0.2)
            self.h_dphiVpt = bh.TH2F(name+'_dphiVpt', 'TkEG #Delta #phi vs p_{T}^{EG}; p_{T}^{EG} [GeV]; #Delta #phi [rad]', 100, 0, 100, 100, -0.2, 0.2)
            self.h_deta = bh.TH1F(name+'_deta', 'TkEG #Delta #eta; #Delta #eta', 100, -0.2, 0.2)
            self.h_detaVpt = bh.TH2F(name+'_detaVpt', 'TkEG #Delta #eta vs p_{T}^{EG}; p_{T}^{EG} [GeV]; #Delta #eta', 100, 0, 100, 100, -0.2, 0.2)

            self.h_dr = bh.TH1F(name+'_dr', 'TkEG #Delta R; #Delta R', 100, 0, 0.2)
            self.h_ptVtkpt = bh.TH2F(name+'_ptVtkpt', 'TkEG Pt (GeV) vs TkPt; p_{T}^{Tk} [GeV]; p_{T}^{EG} [GeV]', 100, 0, 100, 100, 0, 100)

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
            self.h_pt = bh.TH1F(name+'_pt', 
                                'Track Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(name+'_eta', 
                                 'Track eta; #eta;', 100, -4, 4)
            self.h_chi2 = bh.TH1F(name+'_chi2', 
                                  'Track chi2; #Chi^{2}', 1000, 0, 1000)
            self.h_chi2Red = bh.TH1F(name+'_chi2Red', 
                                     'Track chi2 red; red. #Chi^{2}', 100, 0, 100)
            self.h_nstubs = bh.TH1F(name+'_nstubs', 
                                    'Track # stubs; # stubs', 10, 0, 10)
            self.h_z0 = bh.TH1F(name+'_z0', 
                                'Track z0; z_{0} [cm]', 100, -10, 10)
            self.h_chi2RedVeta = bh.TH2F(name+'_chi2RedVeta', 
                                         'Track chi2 red. v eta; #eta; red. #Chi^{2}', 100, -4, 4, 100, 0, 100)
            self.h_nstubsVeta = bh.TH2F(name+'_nstubsVeta', 
                                        'Track # stubs vs eta; #eta; # stubs', 100, -4, 4, 10, 0, 10)
            self.h_z0Veta = bh.TH2F(name+'_z0Veta', 
                                    'Track z0 vs eta; #eta; z_{0} [cm]', 100, -4, 4, 100, -10, 10)
            self.h_chi2RedVpt = bh.TH2F(name+'_chi2RedVpt', 
                                        'Track chi2 red. v pT; p_{T} [GeV]; red. #Chi^{2}', 100, 0, 100, 100, 0, 100)
            self.h_nstubsVpt = bh.TH2F(name+'_nstubsVpt', 
                                       'Track # stubs vs pT; p_{T} [GeV]; # stubs', 100, 0, 100, 10, 0, 10)
            self.h_z0Vpt = bh.TH2F(name+'_z0Vpt', 
                                   'Track z0 vs pT; p_{T} [GeV]; z_{0} [cm]', 100, 0, 100, 100, -10, 10)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tracks):
        bh.fill_1Dhist(self.h_pt, tracks.pt)
        bh.fill_1Dhist(self.h_eta, tracks.eta)
        bh.fill_1Dhist(self.h_chi2, tracks.chi2)
        bh.fill_1Dhist(self.h_chi2Red, tracks.chi2Red)
        bh.fill_1Dhist(self.h_nstubs, tracks.nStubs)
        bh.fill_1Dhist(self.h_z0, tracks.z0)
        bh.fill_2Dhist(self.h_chi2RedVeta, tracks.eta, tracks.chi2Red)
        bh.fill_2Dhist(self.h_nstubsVeta, tracks.eta, tracks.nStubs)
        bh.fill_2Dhist(self.h_z0Veta, tracks.eta, tracks.z0)
        bh.fill_2Dhist(self.h_chi2RedVpt, tracks.pt, tracks.chi2Red)
        bh.fill_2Dhist(self.h_nstubsVpt, tracks.pt, tracks.nStubs)
        bh.fill_2Dhist(self.h_z0Vpt, tracks.pt, tracks.z0)



class JetHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(name+'_pt', 
                                'Track Pt (GeV); p_{T} [GeV]', 100, 0, 100)
            self.h_eta = bh.TH1F(name+'_eta', 
                                 'Track eta; #eta;', 100, -4, 4)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, tracks):
        bh.fill_1Dhist(self.h_pt, tracks.pt)
        bh.fill_1Dhist(self.h_eta, tracks.eta)




class TriggerTowerHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_pt = bh.TH1F(name+'_pt', 'Tower Pt (GeV); p_{T} [GeV];', 100, 0, 100)
            self.h_etEm = bh.TH1F(name+'_etEm', 'Tower Et EM (GeV)', 100, 0, 100)
            self.h_etHad = bh.TH1F(name+'_etHad', 'Tower Et Had (GeV)', 100, 0, 100)
            self.h_HoE = bh.TH1F(name+'_HoE', 'Tower H/E', 20, 0, 2)
            self.h_HoEVpt = bh.TH2F(name+'_HoEVpt', 'Tower H/E vs Pt (GeV); H/E;', 50, 0, 100, 20, 0, 2)
            self.h_energy = bh.TH1F(name+'_energy', 'Tower energy (GeV)', 1000, 0, 1000)
            self.h_eta = bh.TH1F(name+'_eta', 'Tower eta; #eta;', 75, -3.169, 3.169)
            self.h_ieta = bh.TH1F(name+'_ieta', 'Tower eta; i#eta;', 18, 0, 18)

            self.h_ptVeta = bh.TH2F(name+'_ptVeta', 'Tower P_P{T} (GeV) vs #eta; #eta; p_{T} [GeV];',  75, -3.169, 3.169, 100, 0, 10)
            self.h_etVieta = bh.TH2F(name+'_etVieta', 'Tower E_{T} (GeV) vs ieta; i#eta; E_{T} [GeV];',  18, 0, 18, 100, 0, 10)
            self.h_etEmVieta = bh.TH2F(name+'_etEmVieta', 'Tower E_{T} EM (GeV) vs ieta; i#eta; E_{T}^{EM} [GeV];',  18, 0, 18, 100, 0, 10)
            self.h_etHadVieta = bh.TH2F(name+'_etHadVieta', 'Tower E_{T} Had (GeV) vs ieta; i#eta; E_{T}^{HAD} [GeV];',  18, 0, 18, 100, 0, 10)
            self.h_sumEt = bh.TH1F(name+'_sumEt', 'Tower SumEt (GeV); E_{T}^{TOT} [GeV];', 200, 0, 400)
            self.h_sumEtCentral = bh.TH1F(name+'_sumEtCentral', 'Tower SumEt (GeV) (central); E_{T}^{TOT} [GeV];', 200, 0, 400)

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
            self.h_ptRes = bh.TH1F(name+'_ptRes', 'TT Pt reso (GeV); (p_{T}^{L1}-p_{T}^{GEN})/p_{T}^{GEN};', 100, -1, 2)

            self.h_ptResVpt = bh.TH2F(name+'_ptResVpt', 'TT Pt reso (GeV) vs pt (GeV); p_{T}^{GEN} [GeV]; (p_{T}^{L1}-p_{T}^{GEN})/p_{T}^{GEN};', 50, 0, 100, 100, -1, 2)
            self.h_ptResVeta = bh.TH2F(name+'_ptResVeta', 'TT Pt reso (GeV) vs eta; #eta^{GEN}; (p_{T}^{L1}-p_{T}^{GEN})/p_{T}^{GEN};', 100, -3.5, 3.5, 100, -1, 2)

            self.h_ptResp = bh.TH1F(name+'_ptResp', 'TT Pt resp.; p_{T}^{L1}/p_{T}^{GEN};', 100, 0, 2)
            self.h_ptRespVpt = bh.TH2F(name+'_ptRespVpt', 'TT Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};', 50, 0, 100, 100, 0, 2)
            self.h_ptRespVeta = bh.TH2F(name+'_ptRespVeta', 'TT Pt resp. vs |#eta|; |#eta^{GEN}|; p_{T}^{L1}/p_{T}^{GEN};', 34, 1.4, 3.1, 100, 0, 2)

            self.h_energyRes = bh.TH1F(name+'_energyRes', 'TT Energy reso (GeV)', 200, -100, 100)
            self.h_energyResVeta = bh.TH2F(name+'_energyResVeta', 'TT E reso (GeV) vs eta', 100, -3.5, 3.5, 200, -100, 100)
            # FIXME: add corresponding Pt plots
            self.h_etaRes = bh.TH1F(name+'_etaRes', 'TT eta reso; #eta^{L1}-#eta^{GEN}', 100, -0.4, 0.4)
            self.h_phiRes = bh.TH1F(name+'_phiRes', 'TT phi reso; #phi^{L1}-#phi^{GEN}', 100, -0.4, 0.4)
            self.h_etalwRes = bh.TH1F(name+'_etalwRes', 'TT eta reso (lw)', 100, -0.4, 0.4)
            self.h_philwRes = bh.TH1F(name+'_philwRes', 'TT phi reso (lw)', 100, -0.4, 0.4)

            self.h_drRes = bh.TH1F(name+'_drRes', 'TT DR reso', 100, 0, 0.1)
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
    # @profile
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            # self.h_ptRes = bh.TH1F(
            #     name+'_ptRes', '3D Cluster Pt reso (GeV); (p_{T}^{L1} - p_{T}^{GEN})/p_{T}^{GEN}',
            #     100, -1, 1)
            # # self.h_energyRes = bh.TH1F(name+'_energyRes', '3D Cluster Energy reso (GeV); E^{L1} - E^{GEN} [GeV]', 200, -100, 100)
            # self.h_ptResVeta = bh.TH2F(
            #     name+'_ptResVeta', '3D Cluster Pt reso (GeV) vs eta; #eta^{GEN}; p_{T}^{L1} - p_{T}^{GEN} [GeV];',
            #     50, -3.5, 3.5, 20, -40, 40)
            # # self.h_energyResVenergy = bh.TH2F(
            # #     name+'_energyResVenergy',
            # #     '3D Cluster E reso vs E; E^{GEN} [GeV]; (E^{L1} - E^{GEN})/E^{GEN};',
            # #     100, 0, 1000, 100, -1.5, 1.5)
            # # self.h_energyResVeta = bh.TH2F(
            # #     name+'_energyResVeta',
            # #     '3D Cluster E reso (GeV) vs eta; #eta^{GEN}; (E^{L1} - E^{GEN})/E^{GEN};',
            # #     100, -3.5, 3.5, 100, -1.5, 1.5)
            # # self.h_energyResVnclu = bh.TH2F(name+'_energyResVnclu', '3D Cluster E reso (GeV) vs # clusters; # 2D clus.; E^{L1} - E^{GEN} [GeV];', 50, 0, 50, 200, -100, 100)
            # self.h_ptResVpt = bh.TH2F(
            #     name+'_ptResVpt',
            #     '3D Cluster Pt reso (GeV) vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1} - p_{T}^{GEN} [GeV];',
            #     50, 0, 100, 200, -40, 40)
            # self.h_ptResVnclu = bh.TH2F(name+'_ptResVnclu', '3D Cluster Pt reso (GeV) vs # clusters; # 2D clus.; p_{T}^{L1} - p_{T}^{GEN} [GeV];', 50, 0, 50, 200, -40, 40)

            self.h_ptResp = bh.TH1F(
                name+'_ptResp',
                '3D Cluster Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 2)
            self.h_ptRespVpt = bh.TH2F(
                name+'_ptRespVpt',
                '3D Cluster Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 2)
            self.h_ptRespVeta = bh.TH2F(
                name+'_ptRespVeta',
                '3D Cluster Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                20, -4, 4, 50, 0, 2)
            # self.h_ptRespVnclu = bh.TH2F(
            #     name+'_ptRespVnclu',
            #     '3D Cluster Pt resp. vs # clus.; # 2D clust. ; p_{T}^{L1}/p_{T}^{GEN};',
            #     50, 0, 100, 100, 0, 2)
            # self.h_ptRespVetaVptL1 = ROOT.TH3F(
            #     name+'_ptRespVetaVptL1',
            #     '3D Cluster Pt resp. vs #eta and vs pT; #eta^{L1}; p_{T}^{L1} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
            #     30, 1, 4, 50, 0, 100, 100, 0, 3)

            # self.h_ptemResp = bh.TH1F(
            #     name+'_ptemResp',
            #     '3D Cluster Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
            #     100, 0, 3)
            # self.h_ptemRespVpt = bh.TH2F(
            #     name+'_ptemRespVpt',
            #     '3D Cluster Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
            #     50, 0, 100, 100, 0, 3)

            # self.h_coreEnergyResVnclu = bh.TH2F(name+'_coreEnergyResVnclu', '3D Cluster E reso (GeV) vs # clusters', 50, 0, 50, 200, -100, 100)
            # self.h_corePtResVnclu = bh.TH2F(name+'_corePtResVnclu', '3D Cluster Pt reso (GeV) vs # clusters', 50, 0, 50, 200, -40, 40)
            #
            # self.h_coreEnergyRes = bh.TH1F(name+'_coreEnergyRes', '3D Cluster Energy reso CORE (GeV)', 200, -100, 100)
            # self.h_corePtRes = bh.TH1F(name+'_corePtRes', '3D Cluster Pt reso CORE (GeV)', 200, -40, 40)

            # self.h_centralEnergyRes = bh.TH1F(name+'_centralEnergyRes', '3D Cluster Energy reso CENTRAL (GeV)', 200, -100, 100)
            self.h_etaRes = bh.TH1F(
                name+'_etaRes',
                '3D Cluster eta reso; #eta^{L1}-#eta^{GEN}',
                100, -0.15, 0.15)
            self.h_phiRes = bh.TH1F(
                name+'_phiRes',
                '3D Cluster phi reso; #phi^{L1}-#phi^{GEN}',
                100, -0.15, 0.15)
            self.h_drRes = bh.TH1F(
                name+'_drRes',
                '3D Cluster DR reso; #DeltaR^{L1}-#DeltaR^{GEN}',
                100, 0, 0.1)
            # self.h_n010 = bh.TH1F(
            #     name+'_n010',
            #     '# of 3D clus in 0.2 cone with pt>0.1GeV',
            #     10, 0, 10)
            # self.h_n025 = bh.TH1F(
            #     name+'_n025',
            #     '# of 3D clus in 0.2 cone with pt>0.25GeV',
            #     10, 0, 10)

        BaseResoHistos.__init__(self, name, root_file)

    def fill(self, reference, target):
        bh.fill_1Dhist(self.h_ptResp, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVeta, reference.eta, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVpt, reference.pt, target.pt/reference.pt)
        if 'caloeta' in reference.fields:
            bh.fill_1Dhist(self.h_etaRes, target.eta - reference.caloeta)
            bh.fill_1Dhist(self.h_phiRes, target.phi - reference.calophi)
            # self.h_drRes.Fill(np.sqrt((reference.exphi-target_phi)**2+(reference.exeta-target_eta)**2))

        # if 'n010' in target:
        #     self.h_n010.Fill(target_line.n010)
        # if 'n025' in target:
        #     self.h_n025.Fill(target_line.n025)


class Reso2DHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            # self.h_etaRes = bh.TH1F(name+'_etaRes', 'Eta 2D cluster - GEN part', 100, -0.5, 0.5)
            # self.h_phiRes = bh.TH1F(name+'_phiRes', 'Phi 2D cluster - GEN part', 100, -0.5, 0.5)
            # self.h_phiPRes = bh.TH1F(name+'_phiPRes', 'Phi (+) 2D cluster - GEN part', 100, -0.5, 0.5)
            # self.h_phiMRes = bh.TH1F(name+'_phiMRes', 'Phi (-) 2D cluster - GEN part', 100, -0.5, 0.5)
            self.h_xResVlayer = bh.TH2F(name+'_xResVlayer', 'X resolution (cm) [(2D clus) - GEN]', 60, 0, 60, 100, -10, 10)
            self.h_yResVlayer = bh.TH2F(name+'_yResVlayer', 'Y resolution (cm) [(2D clus) - GEN]', 60, 0, 60, 100, -10, 10)
            # self.h_DRRes = bh.TH1F(name+'_DRRes', 'DR 2D cluster - GEN part', 100, -0.5, 0.5)

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
            self.h_maxNNDistVlayer = bh.TH2F(name+'_maxNNDistVlayer', 'Max dist between NN vs layer', 60, 0, 60, 100, 0, 10)
            self.h_minNNDistVlayer = bh.TH2F(name+'_minNNDistVlayer', 'Max dist between NN vs layer', 60, 0, 60, 100, 0, 10)

            self.h_nTCsPerLayer = bh.TH1F(name+'_nTCsPerLayer', '# of Trigger Cells per layer', 60, 0, 60)
            self.h_radiusVlayer = bh.TH2F(name+'_radiusVlayer', '# of cells radius vs layer', 60, 0, 60, 200, 0, 200)
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
            self.h_eDensityVlayer = bh.TH2F(name+'_eDensityVlayer', 'E (GeV) Density per layer', 60, 0, 60, 600, 0, 30)
            self.h_nTCDensityVlayer = bh.TH2F(name+'_nTCDensityVlayer', '# TC Density per layer', 60, 0, 60, 20, 0, 20)
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
    def __init__(self, name, root_file=None, pt_bins=None, debug=False):
        self.name = name
        self.h_num = GenParticleHistos('h_effNum_'+name, root_file, pt_bins=pt_bins, debug=debug)
        self.h_den = GenParticleHistos('h_effDen_'+name, root_file, pt_bins=pt_bins, debug=debug)
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
            self.h_ptResp = bh.TH1F(
                name+'_ptResp',
                'Track Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 3)
            self.h_ptRespVpt = bh.TH2F(
                name+'_ptRespVpt',
                'Track Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 3)
            self.h_ptRespVeta = bh.TH2F(
                name+'_ptRespVeta',
                'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                50, -4, 4, 100, 0, 3)
            self.h_etaRes = bh.TH1F(
                name+'_etaRes',
                'Track eta reso',
                100, -0.4, 0.4)
            self.h_phiRes = bh.TH1F(
                name+'_phiRes',
                'Track phi reso',
                100, -0.4, 0.4)
            self.h_drRes = bh.TH1F(
                name+'_drRes',
                'Track DR reso',
                100, 0, 0.4)
            self.h_nMatch = bh.TH1F(
                name+'_nMatch',
                '# matches',
                100, 0, 100)

            # self.h_pt2stResVpt = bh.TH2F(name+'_pt2stResVpt', 'EG Pt 2stubs reso. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}-p_{T}^{GEN} [GeV];',
            #                                50, 0, 100, 100, -20, 20)
            #
            # self.h_pt2stResp = bh.TH1F(name+'_pt2stResp', 'Track Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
            #                              100, 0, 3)
            # self.h_pt2stRespVpt = bh.TH2F(name+'_pt2stRespVpt', 'Track Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
            #                                 50, 0, 100, 100, 0, 3)
            # self.h_pt2stRespVeta = bh.TH2F(name+'_pt2stRespVeta', 'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
            #                                  50, -4, 4, 100, 0, 3)

        BaseResoHistos.__init__(self, name, root_file, debug)

    def fill(self, reference, target):
        bh.fill_1Dhist(self.h_ptResp, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVeta, reference.eta, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVpt, reference.pt, target.pt/reference.pt)

        # self.h_pt2stResVpt.Fill(reference.pt, target.pt2stubs-reference.pt)
        # self.h_pt2stResp.Fill(target.pt2stubs/reference.pt)
        # self.h_pt2stRespVeta.Fill(reference.eta, target.pt2stubs/reference.pt)
        # self.h_pt2stRespVpt.Fill(reference.pt, target.pt2stubs/reference.pt)

        bh.fill_1Dhist(self.h_etaRes, (target.eta - reference.eta))
        bh.fill_1Dhist(self.h_phiRes, (target.phi - reference.phi))
        bh.fill_1Dhist(self.h_drRes, np.sqrt((reference.phi-target.phi)**2+(reference.eta-target.eta)**2))

    def fill_nMatch(self, n_matches):
        self.h_nMatch.Fill(n_matches)


class JetResoHistos(BaseResoHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptResp = bh.TH1F(
                name+'_ptResp',
                'Track Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 3)
            self.h_ptRespVpt = bh.TH2F(
                name+'_ptRespVpt',
                'Track Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 3)
            self.h_ptRespVeta = bh.TH2F(
                name+'_ptRespVeta',
                'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                50, -4, 4, 100, 0, 3)
            self.h_etaRes = bh.TH1F(
                name+'_etaRes',
                'Track eta reso',
                100, -0.4, 0.4)
            self.h_phiRes = bh.TH1F(
                name+'_phiRes',
                'Track phi reso',
                100, -0.4, 0.4)
            self.h_drRes = bh.TH1F(
                name+'_drRes',
                'Track DR reso',
                100, 0, 0.4)
            self.h_nMatch = bh.TH1F(
                name+'_nMatch',
                '# matches',
                100, 0, 100)

        BaseResoHistos.__init__(self, name, root_file, debug)

    def fill(self, reference, target):
        bh.fill_1Dhist(self.h_ptResp, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVeta, reference.eta, target.pt/reference.pt)
        bh.fill_2Dhist(self.h_ptRespVpt, reference.pt, target.pt/reference.pt)
        bh.fill_1Dhist(self.h_etaRes, (target.eta - reference.eta))
        bh.fill_1Dhist(self.h_phiRes, (target.phi - reference.phi))
        bh.fill_1Dhist(self.h_drRes, np.sqrt((reference.phi-target.phi)**2+(reference.eta-target.eta)**2))

    def fill_nMatch(self, n_matches):
        self.h_nMatch.Fill(n_matches)


class DecTkResoHistos(BaseResoHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptResVpt = bh.TH2F(
                name+'_ptResVpt',
                'Track Pt reso. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}-p_{T}^{GEN} [GeV];',
                50, 0, 100, 100, -20, 20)
            self.h_ptResp = bh.TH1F(
                name+'_ptResp',
                'Track Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 3)
            self.h_ptRespVpt = bh.TH2F(
                name+'_ptRespVpt',
                'Track Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100, 100, 0, 3)
            self.h_ptRespVeta = bh.TH2F(
                name+'_ptRespVeta',
                'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                50, -4, 4, 100, 0, 3)
            self.h_etaRes = bh.TH1F(
                name+'_etaRes',
                'Track eta reso',
                100, -0.15, 0.15)
            self.h_etaResVabseta = bh.TH2F(
                name+'_etaResVabseta',
                '#eta_{@vtx} reso; |#eta^{GEN}|; #eta_{@vtx}^{L1} vs #eta_{@vtx}^{GEN}',
                50, 0, 2.5,
                100, -0.1, 0.1)
            self.h_etaResVeta = bh.TH2F(
                name+'_etaResVeta',
                '#eta_{@vtx} reso; #eta^{GEN}; #eta_{@vtx}^{L1} vs #eta_{@vtx}^{GEN}',
                200, -2.5, 2.5,
                100, -0.1, 0.1)
            self.h_phiRes = bh.TH1F(
                name+'_phiRes',
                'Track phi reso',
                100, -0.4, 0.4)
            self.h_caloEtaRes = bh.TH1F(
                name+'_caloEtaRes',
                '#eta_{@calo} reso; #eta_{@calo}^{L1} vs #eta_{@calo}^{GEN}',
                100, -0.15, 0.15)
            self.h_caloEtaResVabseta = bh.TH2F(
                name+'_caloEtaResVabseta',
                '#eta_{@calo} reso; |#eta^{GEN}|; #eta_{@calo}^{L1} vs #eta_{@calo}^{GEN}',
                50, 0, 2.5,
                100, -0.1, 0.1)
            self.h_caloEtaResVeta = bh.TH2F(
                name+'_caloEtaResVeta',
                '#eta_{@calo} reso; #eta^{GEN}; #eta_{@calo}^{L1} vs #eta_{@calo}^{GEN}',
                200, -2.5, 2.5,
                100, -0.1, 0.1)
            self.h_caloPhiRes = bh.TH1F(
                name+'_caloPhiRes',
                '#phi_{@calo} reso; #phi_{@calo}^{L1} vs #phi_{@calo}^{GEN}',
                100, -0.4, 0.4)
            self.h_caloPhiResVabseta = bh.TH2F(
                name+'_caloPhiResVabseta',
                '#phi_{@calo} reso; |#eta^{GEN}|; #phi_{@calo}^{L1} vs #phi_{@calo}^{GEN}',
                50, 0, 3,
                100, -0.4, 0.4)
            self.h_dzRes = bh.TH1F(
                name+'_dzRes',
                '#DeltaZ_{0} res; #DeltaZ_{0}^{L1}-#DeltaZ_{0}^{GEN}',
                100, -10, 10)

            # self.h_caloPhiResVeta = bh.TH2F(
            #     name+'_caloPhiResVabseta',
            #     '#phi_{@calo} reso; #eta^{GEN}; #phi_{@calo}^{L1} vs #phi_{@calo}^{GEN}',
            #     50, 0, 3,
            #     100, -0.4, 0.4)
            self.h_nMatch = bh.TH1F(
                name+'_nMatch',
                '# matches',
                100, 0, 100)

            # self.h_pt2stResVpt = bh.TH2F(name+'_pt2stResVpt', 'EG Pt 2stubs reso. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}-p_{T}^{GEN} [GeV];',
            #                                50, 0, 100, 100, -20, 20)
            #
            # self.h_pt2stResp = bh.TH1F(name+'_pt2stResp', 'Track Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
            #                              100, 0, 3)
            # self.h_pt2stRespVpt = bh.TH2F(name+'_pt2stRespVpt', 'Track Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
            #                                 50, 0, 100, 100, 0, 3)
            # self.h_pt2stRespVeta = bh.TH2F(name+'_pt2stRespVeta', 'Track Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
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
        self.h_etaResVabseta.Fill(reference.abseta, target_eta - reference_eta)
        self.h_etaResVeta.Fill(reference.eta, target_eta - reference_eta)

        self.h_phiRes.Fill(target_phi - reference_phi)
        self.h_caloEtaRes.Fill(target_line.caloeta - reference.exeta)
        self.h_caloPhiRes.Fill(target_line.calophi - reference.exphi)
        self.h_caloEtaResVabseta.Fill(reference.abseta, target_line.caloeta - reference.exeta)
        self.h_caloPhiResVabseta.Fill(reference.abseta, target_line.calophi - reference.exphi)
        self.h_caloEtaResVeta.Fill(reference_eta, target_line.caloeta - reference.exeta)
        # self.h_caloPhiResVeta.Fill(reference_eta, target_line.calophi - reference.exphi)
        self.h_dzRes.Fill(target_line.z0 - reference.ovz)

    def fill_nMatch(self, n_matches):
        self.h_nMatch.Fill(n_matches)


class EGResoHistos(BaseResoHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:

            self.h_ptResVpt = bh.TH2F(
                name+'_ptResVpt',
                'EG Pt reso. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}-p_{T}^{GEN} [GeV];',
                50, 0, 100,
                100, -10, 10)
            self.h_ptRes = bh.TH1F(
                name+'_ptRes',
                'EG Pt res.; (p_{T}^{L1}-p_{T}^{GEN})/p_{T}^{GEN}',
                100, -1, 1)
            self.h_ptResp = bh.TH1F(
                name+'_ptResp',
                'EG Pt resp.; p_{T}^{L1}/p_{T}^{GEN}',
                100, 0, 3)
            self.h_ptRespVpt = bh.TH2F(
                name+'_ptRespVpt',
                'EG Pt resp. vs pt (GeV); p_{T}^{GEN} [GeV]; p_{T}^{L1}/p_{T}^{GEN};',
                50, 0, 100,
                100, 0, 3)
            self.h_ptRespVeta = bh.TH2F(
                name+'_ptRespVeta',
                'EG Pt resp. vs #eta; #eta^{GEN}; p_{T}^{L1}/p_{T}^{GEN};',
                50, -4, 4,
                100, 0, 3)

            self.h_etaRes = bh.TH1F(
                name+'_etaRes',
                'EG eta reso; #eta^{L1}-#eta^{GEN}',
                100, -0.1, 0.1)
            self.h_phiRes = bh.TH1F(
                name+'_phiRes',
                'EG phi reso; #phi^{L1}-#phi^{GEN}',
                100, -0.1, 0.1)

            self.h_exetaRes = bh.TH1F(
                name+'_exetaRes',
                'EG eta reso; #eta^{L1}-#eta^{GEN}_{calo}',
                100, -0.1, 0.1)
            self.h_exphiRes = bh.TH1F(
                name+'_exphiRes',
                'EG phi reso; #phi^{L1}-#phi^{GEN}_{calo}',
                100, -0.1, 0.1)

            self.h_dzRes = bh.TH1F(
                name+'_dzRes',
                '#DeltaZ_{0} res; #DeltaZ_{0}^{L1}-#DeltaZ_{0}^{GEN}',
                100, -10, 10)

        BaseResoHistos.__init__(self, name, root_file, debug)

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


class ClusterConeHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_ptRel = bh.TH1F(name+'_ptRel',
                                     'Pt best/Pt other; p_{T}^{best}/p_{T}^{other}', 100, 0, 5)
            self.h_ptRelVpt = bh.TH2F(name+'_ptRelVpt', 'Pt best/Pt other vs pt (GeV); p_{T}^{best} [GeV]; p_{T}^{best}/p_{T}^{other};', 50, 0, 100, 100, 0, 5)
            self.h_deltaEta = bh.TH1F(name+'_deltaEta', '#Delta eta; #eta^{best}-#eta^{other}', 100, -0.4, 0.4)
            self.h_deltaPhi = bh.TH1F(name+'_deltaPhi', '#Delta phi; #phi^{best}-#phi^{other}', 100, -0.4, 0.4)
            self.h_deltaPhiVq = bh.TH2F(name+'_deltaPhiVq', '#Delta phi; #phi^{best}-#phi^{other}; GEN charge;', 100, -0.4, 0.4, 3, -1, 2)

            self.h_deltaR = bh.TH1F(name+'_deltaR', '#Delta R (best-other); #Delta R (best, other)', 100, 0, 0.4)
            self.h_n = bh.TH1F(name+'_n', '# other clusters in cone; # others', 20, 0, 20)
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
            self, 'iso', 'pid_gen:e_gen:pt_gen:eta_gen:z0_gen:e:pt:eta:tkIso:pfIso:tkIsoPV:pfIsoPV',
            name, root_file, debug)

    def fill(self, reference, target):
        values_fill = []

        if reference is not None:
            values_fill.append(reference.pid)
            values_fill.append(reference.energy)
            values_fill.append(reference.pt)
            values_fill.append(reference.eta)
            values_fill.append(reference.ovz)
        else:
            values_fill.extend([-1]*5)

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


class ResoTuples(BaseUpTuples):
    def __init__(self, name, root_file=None, debug=False):
        BaseUpTuples.__init__(
            self, "ResoData", name, root_file, debug)

    def fill(self, reference, target):
        # print(self.t_name)
        # print(target.fields)
        target_vars = [
            'pt',
            'eta',
            'phi',
            'energy',
            'hwQual']
        rference_vars = [
            'pt',
            'eta',
            'phi',
            'energy']
        tree_data = {}
        for var in target_vars:
            tree_data[var] = ak.flatten(ak.drop_none(target[var]))
        for var in rference_vars:
            tree_data[f'gen_{var}'] = ak.flatten(ak.drop_none(reference[var]))
        # print(reference.fields)
        # tree_data[f'gen_dz'] = ak.flatten(ak.drop_none(np.abs(reference.ovz-target.tkZ0)))
        
        BaseUpTuples.fill(self, tree_data)


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
    class CorrBoardOccupancyHistos(object):
        def __init__(self, name, board):
            self.h_totOcc = bh.TH1F(f'{name}_{board}totOcc', f'{board} total occupancy; {board} total occ.', 500, 0, 500)
            self.h_regOcc = bh.TH1F(f'{name}_{board}regOcc', f'{board} reg occupancy; {board} reg. occ.', 100, 0, 100)
            self.h_maxOcc = bh.TH1F(f'{name}_{board}maxOcc', f'{board} max occupancy; {board} max occ.', 100, 0, 100)
            self.h_maxMult = bh.TH1F(f'{name}_{board}maxMult', f'{board} max multiplicity per cluster; {board} max mult. per cluster', 100, 0, 100)
            self.eta_regions_idx = pf_regions.regions[board]
            self.max_count = 0
            self.tot_count = 0
            self.max_mult_percluster = 0

        def fillRegion(self, ieta, occupancy, multpercluster):
            if(ieta in self.eta_regions_idx):
                if occupancy > self.max_count:
                    self.max_count = occupancy
                self.tot_count += occupancy
                self.h_regOcc.Fill(occupancy)
                if multpercluster > self.max_mult_percluster:
                    self.max_mult_percluster = multpercluster

        def fillBoard(self):
            self.h_maxOcc.Fill(self.max_count)
            self.h_totOcc.Fill(self.tot_count)
            self.h_maxMult.Fill(self.max_mult_percluster)
            self.max_count = 0
            self.tot_count = 0
            self.max_mult_percluster = 0


    def __init__(self, name, root_file=None, debug=False):
        if not root_file:

            self.h_avgOcc = ROOT.TProfile2D(
                name+'_avgOcc',
                'region avg occ; #eta, #phi;',
                pf_regions.regionizer.n_eta_regions(),
                array('d', pf_regions.regionizer.eta_boundaries_fiducial_),
                pf_regions.regionizer.n_phi_regions(),
                array('d', pf_regions.regionizer.phi_boundaries_fiducial_))

            self.board_histos = []
            for board in ['ALL', 'BRL', 'HGCNoTk', 'HGC']:
                bhs = CorrOccupancyHistos.CorrBoardOccupancyHistos(name, board)
                setattr(self, f'h_{board}totOcc', bhs.h_totOcc)
                setattr(self, f'h_{board}regOcc', bhs.h_regOcc)
                setattr(self, f'h_{board}maxOcc', bhs.h_maxOcc)
                setattr(self, f'h_{board}maxMult', bhs.h_maxMult)
                self.board_histos.append(bhs)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, objects):
        for ieta in range(0, pf_regions.regionizer.n_eta_regions()):
            for iphi in range(0, pf_regions.regionizer.n_phi_regions()):
                objs_in_region = objects[objects['eta_reg_{}'.format(ieta)] & objects['phi_reg_{}'.format(iphi)]]
                nmatch_percluster = 0
                if 'clidx' in objs_in_region.columns and not objs_in_region.empty:
                    nmatch_percluster = objs_in_region.clidx.value_counts().iloc[0]
                occupancy = objs_in_region.shape[0]
                
                self.h_avgOcc.Fill(pf_regions.regionizer.eta_centers[ieta],
                                   pf_regions.regionizer.phi_centers[iphi],
                                   occupancy)
                for bhs in self.board_histos:
                    bhs.fillRegion(ieta, occupancy, nmatch_percluster)
                        
        for bhs in self.board_histos:
            bhs.fillBoard()
            

# for convenience we define some sets
class HistoSetOccupancy():
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






class TCClusterMatchHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_dEtaVdPhi = bh.TH2F(
                name+'_dEtaVdPhi',
                '#Delta#eta vs #Delta#phi; #Delta#phi [rad]; #Delta#eta;',
                100, -0.1, 0.1, 100, -0.1, 0.1)
            self.h_dEtaRMSVenergy = ROOT.TProfile(
                name+'_dEtaRMSVenergy',
                'RMS(#Delta#eta) vs energy; E [GeV]; RMS(#Delta#eta);',
                100, 0, 1000)
            self.h_dEtaRMSVpt = ROOT.TProfile(
                name+'_dEtaRMSVpt',
                'RMS(#Delta#eta) vs pt; p_{T} [GeV]; RMS(#Delta#eta);',
                100, 0, 100)
            self.h_dPhiRMSVenergy = ROOT.TProfile(
                name+'_dPhiRMSVenergy',
                'RMS(#Delta#phi) vs energy; E [GeV]; RMS(#Delta#phi);',
                100, 0, 1000)
            self.h_dPhiRMSVpt = ROOT.TProfile(
                name+'_dPhiRMSVpt',
                'RMS(#Delta#phi) vs pt; p_{T} [GeV]; RMS(#Delta#phi);',
                100, 0, 100)
            self.h_dRhoRMSVenergy = ROOT.TProfile(
                name+'_dRhoRMSVenergy',
                'RMS(#Delta#rho) vs energy; E [GeV]; RMS(#Delta#rho);',
                100, 0, 1000)
            self.h_dRhoRMSVpt = ROOT.TProfile(
                name+'_dRhoRMSVpt',
                'RMS(#Delta#rho) vs pt; p_{T} [GeV]; RMS(#Delta#rho);',
                100, 0, 100)
            self.h_dRho = bh.TH1F(
                name+'_dRho',
                '#Delta#rho; #Delta#rho;',
                100, 0, 0.1)
            self.h_dRho2 = bh.TH1F(
                name+'_dRho2',
                '#Delta#rho (E fraction weighted); #Delta#rho;',
                100, 0, 0.1)

            self.h_dRhoVlayer = bh.TH2F(
                name+'_dRhoVlayer',
                '#Delta#rho; layer #; #Delta#rho;',
                60, 0, 60, 100, 0, 0.1)
            self.h_dRhoVabseta = bh.TH2F(
                name+'_dRhoVabseta',
                '#Delta#rho; |#eta|; #Delta#rho;',
                100, 1.4, 3.1, 100, 0, 0.1)
            # self.h_dRhoVfbrem = bh.TH2F(name+'_dRhoVfbrem',
            #                         '#Delta#rho vs f_{brem}; f_{brem}; #Delta#rho;',
            #                         100, 0, 1, 100, 0, 0.1)
            self.h_dtVlayer = bh.TH2F(
                name+'_dtVlayer',
                '#Deltat vs layer; layer #; #Deltat;',
                60, 0, 60, 100, -0.05, 0.05)
            self.h_duVlayer = bh.TH2F(
                name+'_duVlayer',
                '#Delta#rho; layer #; #Deltau;',
                60, 0, 60, 100, -0.05, 0.05)

            self.h_dtVlayer2 = bh.TH2F(
                name+'_dtVlayer2',
                '#Deltat vs layer; layer #; #Deltat;',
                60, 0, 60, 100, -0.05, 0.05)
            self.h_duVlayer2 = bh.TH2F(
                name+'_duVlayer2',
                '#Delta#rho; layer #; #Deltau;',
                60, 0, 60, 100, -0.05, 0.05)

            self.h_dtVdu = bh.TH2F(
                name+'_dtVdu',
                '#Deltat vs #Deltau; #Deltat [cm]; #Deltau [cm];',
                100, -0.05, 0.05, 100, -0.05, 0.05)
            self.h_dtVdu2 = bh.TH2F(
                name+'_dtVdu2',
                '#Deltat vs #Deltau (E fract. weighted); #Deltat [cm]; #Deltau [cm];',
                100, -0.05, 0.05, 100, -0.05, 0.05)
            # self.h_fbremVabseta = bh.TH2F(name+'_fbremVabseta',
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



class QuantizationHistos(BaseHistos):
    def __init__(self, name, features=None, root_file=None, debug=False):
        if not root_file:
            self.features = features
            self.h_features = bh.TH2F_category(
                name+'_features', 
                'features; feature; value',
                 self.features,
                 1000, -1000, 1000)
            self.h_featuresLog2 = bh.TH2F_category(
                name+'_featuresLog2', 
                'featuresLog2; features; log_{2}(value)',
                 self.features,
                 64, -32, 32)
            # for bin,ft in enumerate(features):
            #     self.h_features.GetXaxis().SetBinLabel(bin+1, ft)
            #     self.h_featuresLog2.GetXaxis().SetBinLabel(bin+1, ft)

        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, df):
        fill = df
        # print(df.fields)
        for bin,ft in enumerate(self.features):
            fill[f'{ft}_bin'] = [ft]
            # print(fill[ft])
            # print(ak.flatten(fill[ft]))
            fill[f'{ft}_log2'] = np.log2(fill[ft])
            # print(fill[[f'{ft}_bin', ft]])
            # print(ak.flatten(fill[[f'{ft}_bin', ft]]))

            bh.fill_2Dhist(self.h_features, fill[f'{ft}_bin'], fill[ft]) 
            bh.fill_2Dhist(self.h_featuresLog2, fill[f'{ft}_bin'], fill[f'{ft}_log2']) 


class DiObjMassHistos(BaseHistos):
    def __init__(self, name, root_file=None, debug=False):
        if not root_file:
            self.h_mass = bh.TH1F(name+'_mass', 'mass (GeV); M(ll) [GeV]', 400, 0, 8000)
        BaseHistos.__init__(self, name, root_file, debug)

    def fill(self, obj_pairs):
        weight = None
        if 'weight' in obj_pairs.fields:
            weight = obj_pairs.weight
        objs_sum = obj_pairs.leg0+obj_pairs.leg1
        # bh.fill_1Dhist(
        #     hist=self.h_mass,     
        #     array=objs_sum[ak.count((objs_sum).pt, axis=1) > 0][:, 0].mass,     
        #     weights=weight)
        self.h_mass.fill(objs_sum[ak.count((objs_sum).pt, axis=1) > 0][:, 0].mass)
        



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
