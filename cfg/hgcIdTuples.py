from python import plotters, selections, calibrations, histos
import cfg.datasets.fastpuppi_collections as coll
from cfg.hgctps import Cluster3DHistos
import awkward as ak


class HGCIdMatchTuples(histos.BaseUpTuples):
    def __init__(self, name, root_file=None, debug=False):
        histos.BaseUpTuples.__init__(
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
        reference_vars = [
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
        histos.BaseUpTuples.fill(self, tree_data)





class HGCIdMatchTuplesPlotter(plotters.GenericGenMatchPlotter):
    def __init__(self, data_set, gen_set,
                 data_selections=[selections.Selection('all')],
                 gen_selections=[selections.Selection('all')]):
        super(HGCIdMatchTuplesPlotter, self).__init__(Cluster3DHistos, HGCIdMatchTuples,
                                                data_set, gen_set,
                                                data_selections, gen_selections, 
                                                drcut=0.1)

class HGCIdTuplesPlotter(plotters.GenericDataFramePlotter):
    def __init__(self, obj_set, obj_selections=[selections.Selection('all')]):
        super(HGCIdTuplesPlotter, self).__init__(HGCIdTuples, obj_set, obj_selections)


comp_selections = (selections.Selector('^Pt15|all')&('^EtaABC$|^EtaBC$|all'))()
sim_eg_selections = (selections.Selector('^GEN$'))()
sim_pi_selections = (selections.Selector('^GENPi$'))()


egid_plotters = [
    HGCIdMatchTuplesPlotter(coll.hgc_cl3d, coll.gen, comp_selections, sim_eg_selections)
]

piid_plotters = [
    # plotters.HGCIdTuplesPlotter(collections.hgc_cl3d, comp_selections),
    HGCIdMatchTuplesPlotter(coll.hgc_cl3d, coll.gen_pi, comp_selections, sim_pi_selections)
]

pu_plotters = [
    HGCIdTuplesPlotter(coll.hgc_cl3d, comp_selections),
]


# for sel in sim_selections:
#     print(sel)