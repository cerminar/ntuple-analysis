"""
Define and instantiate the selections.

The Selection class define via string a selection to be pplied to a certain
DataFrame. The selections are named (the name enters the final histogram name).
Selections can be composed (added). The actual selection syntax follows the
`pandas` `DataFrame` `query` syntax.
"""

import json
import os
import re

import numpy as np
import python.calibrations as calibs

from python import pf_regions
from rich import print as pprint


class PID:
    electron = 11
    photon = 22
    pizero = 111
    pion = 211
    kzero = 130


class SelectionManager:
    """
    SelectionManager.

    Manages the registration of selections to have a global dictionary of the labels for drawing.

    It is a singleton.
    """

    class __TheManager:
        def __init__(self):
            self.selections = []

        def registerSelection(self, selection):
            # print '[EventManager] registering collection: {}'.format(collection.name)
            self.selections.append(selection)

        def get_labels(self):
            label_dict = {}
            for sel in self.selections:
                label_dict[sel.name] = sel.label
            return label_dict

    instance = None

    def __new__(cls):
        if not SelectionManager.instance:
            SelectionManager.instance = SelectionManager.__TheManager()
        return SelectionManager.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


class Selection:
    """
    [Selection] class.

    Args:
    ----
        name (string): the name to be used in the histo name
                       (should not use `-` characters or spaces)

        label (string): used in plot legends, no constraints
        selection (string): see pandas.DataFrame.query syntax

    """

    def __init__(self, name, label='', selection=None):
        self.name = name
        self.label_ = label
        self.selection = selection
        self.all = False
        if self.name == 'all' or selection is None:
            self.selection = lambda ar: True
            self.all = True
        self.hash = hash(selection)
        self.register()

    @property
    def label(self):
        obj_name = 'L1'
        if 'GEN' in self.name:
            obj_name = 'GEN'
        return self.label_.replace('TOBJ', obj_name)

    def register(self):
        selection_manager = SelectionManager()
        selection_manager.registerSelection(self)

    def __and__(self, other):
        """& operation"""
        new_name = name=f'{self.name}{other.name}'
        if self.name == 'all':
            new_name = other.name
        if other.name == 'all':
            new_name = self.name

        new_label = f'{self.label_}, {other.label_}'
        if self.label_ == '':
            new_label = other.label_
        if other.label == '':
            new_label = self.label_

        new_selection = None
        if other.all and not self.all:
            new_selection = self.selection
        elif self.all and not other.all:
            new_selection = other.selection
        elif not self.all and not other.all:
            new_selection = lambda array : self.selection(array) & other.selection(array)

        return Selection(
            name=new_name,
            label=new_label,
            selection=new_selection)


    def __or__(self, other):
        """| operation"""
        if other.all:
            return other.all
        if self.all:
            return self.all
        new_label = f'{self.label_} or {other.label_}'
        if self.label_ == '':
            new_label = other.label_
        if other.label == '':
            new_label = self.label_
        # obj_name = 'L1'
        # if 'GEN' in other.name or 'GEN' in self.name:
        #     obj_name = 'GEN'
        # new_label = new_label.replace('TOBJ', obj_name)
        return Selection(
            name=f'{self.name}Or{other.name}',
            label=new_label,
            selection=lambda array : self.selection(array) | other.selection(array))

    def rename(self, new_name, new_label = None):
        self.name = new_name
        if new_label:
            self.label_ = new_label
        self.register()

    # def __mul__(self, other):
    #     return self.__add__(other)

    # def __add__(self, other):
    #     return self.__and__(other)

    def __str__(self):
        return f'n: {self.name}, \n\t l:{self.label}'

    def __repr__(self):
        return f'<{self.__class__.__name__} {self}> '



def multiply_selections(list1, list2):
    return and_selections(list1, list2)

def and_selections(list1, list2):
    ret = []
    for sel1 in list1:
        for sel2 in list2:
            ret.append(sel1&sel2)
    return ret

def or_selections(list1, list2):
    ret = []
    for sel1 in list1:
        for sel2 in list2:
            ret.append(sel1|sel2)
    return ret


def prune(selection_list):
    sel_names = set()
    ret = []
    for sel in selection_list:
        if sel.name not in sel_names:
            sel_names.add(sel.name)
            ret.append(sel)
    return ret


def build_DiObj_selection(name, label, selection_leg0, selection_leg1, selection_glb=Selection('all')):
    return Selection(
        name,
        label,
        lambda array: selection_leg0.selection(array.leg0) & selection_leg1.selection(array.leg1) & selection_glb.selection(array))
        # FIXME: it was (leg0 sel. | leg1 sel.) instead of &


def fill_isowp_sel(sel_list, wps):
    for iso_cut in wps.keys():
        for pt_cut in wps[iso_cut]:
            iso_var_name = iso_cut.split('0p')[0]
            iso_cut_value = iso_cut.split('0p')[1]
            sel_list.append(
                Selection(
                    f'{iso_cut}Pt{pt_cut}',
                    f'{iso_var_name}<=0.{iso_cut_value} & p_{{T}}>{pt_cut}GeV',
                    f'({iso_var_name}<=0.{iso_cut_value})&(pt>{pt_cut})')
            )


def read_isowp_sel(file_name, obj_name, eta_reg):
    pwd = os.path.dirname(__file__)
    filename = os.path.join(pwd, '..', file_name)
    iso_wps = {}
    with open(filename) as f:
        iso_wps = json.load(f)

    iso_wps_eb = iso_wps[obj_name]
    ret_sel = []
    for iso_var, wps_pt in iso_wps_eb.items():
        for pt_point, wps in wps_pt.items():
            for eff, cut in wps.items():
                eff_str = str(eff).split('.')[1]
                pt_str = pt_point.split(eta_reg)[1]
                wp_name = f'{iso_var}WP{eff_str}{pt_str}'
                wp_label = f'{iso_var} WP{eff_str} @ {pt_str}'
                ret_sel.append(Selection(f'{wp_name}', f'{wp_label}', f'{iso_var}<={cut}'))
    return ret_sel


def read_isoptwp_sel(file_name, obj_name):
    pwd = os.path.dirname(__file__)
    filename = os.path.join(pwd, '..', file_name)
    iso_wps = {}
    with open(filename) as f:
        iso_wps = json.load(f)

    iso_wps_obj = iso_wps[obj_name]
    ret_sel = []
    for iso_sel, wps_pt in iso_wps_obj.items():
        for rate_point, pt_cut in wps_pt.items():
            ret_sel.append((iso_sel, Selection(f'@{rate_point}kHz', f'@{rate_point}kHz', f'pt>={pt_cut}')))
    return ret_sel


def read_iso_flateffwp_sel(file_name):
    pwd = os.path.dirname(__file__)
    filename = os.path.join(pwd, '..', file_name)
    iso_wps = {}
    
    with open(filename) as f:
        iso_wps = json.load(f)
    # pprint(iso_wps)

    'IsoPhoIDTightEB@'
    for eta in ['EE','EB']:
        wps = iso_wps[f'GENEta{eta}']['IDTightPho']
        for (eff, bins) in wps.items():
            pass
        #     print(eff, bins)
        # pprint(wps)
    return

class Selector:
    # common to all instances of the object
    selection_primitives = []

    def __init__(self, selector, primitives=None):
        if primitives is None:
            primitives = Selector.selection_primitives
        self.selections = []
        self.debug = False
        r = re.compile(selector)
        # mgr = SelectionManager()
        self.selections = [sel for sel in primitives if r.match(sel.name)]
        self.selections = prune(self.selections)
        if self.debug:
            print([sel.name for sel in self.selections])

    def __and__(self, match):
        other = None
        if match.__class__ == Selector:
            other = match
        else:
            other = Selector(match)
        self.selections = and_selections(self.selections, other.selections)
        if self.debug:
            print([sel.name for sel in self.selections])
        return self

    def __or__(self, match):
        other = None
        if match.__class__ == Selector:
            other = match
        else:
            other = Selector(match)
        self.selections = or_selections(self.selections, other.selections)
        if self.debug:
            print([sel.name for sel in self.selections])
        return self

    def __mul__(self, match):
        other = None
        if match.__class__ == Selector:
            other = match
        else:
            other = Selector(match)
        self.selections = multiply_selections(self.selections, other.selections)
        if self.debug:
            print([sel.name for sel in self.selections])
        return self

    def __rmul__(self, match):
        return self.__mul__(match)

    def __add__(self, match):
        other = None
        if match.__class__ == Selector:
            other = match
        else:
            other = Selector(match)
        self.selections.extend(other.selections)
        if self.debug:
            print([sel.name for sel in self.selections])
        return self

    def __repr__(self):
        return '<Selector sels=\n{}\n>'.format('\n'.join([str(sel) for sel in self.selections]))

    def __call__(self):
        return self.selections

    def one(self, new_name=None, new_label=None):
        if len(self.selections) != 1:
            print(f'[Selector.one] ERROR: selector returns {len(self.selections)} object and one() called!')
            raise ValueError
        sel = self.selections[0]
        if new_name:
            sel.rename(new_name, new_label)
        return sel


def compare_selections(sel1, sel2):
    if len(sel1) != len(sel2):
        print(f'[DIFF] len 1: {len(sel1)} len2: {len(sel2)}')
        return False

    sel1.sort(key=lambda x: x.name)
    sel2.sort(key=lambda x: x.name)
    ret = True
    for id in range(len(sel1)):
        isDiff = False
        if sel1[id].name != sel2[id].name:
            isDiff = True
        if sel1[id].label != sel2[id].label:
            isDiff = True
        if sel1[id].selection != sel2[id].selection:
            isDiff = True

        if isDiff:
            print(f'[DIFF] \n {sel1[id]} \n {sel2[id]}')
            ret = False

    return ret

def rate_pt_wps_selections(wps, obj, pt_var='pt'):
    data_selections = []
    # gen_selections = []
    sm = selections.SelectionManager()
    if obj in wps.keys():
     #    print(wps[obj])
        for obj_sel_name, pt_wps in wps[obj].items():
            # print(f'WPS for {obj_sel_name}:')
            for rate, pt_cut in wps[obj][obj_sel_name].items():
                # print(f'   rate: {rate}kHz, pt cut: {pt_cut}GeV')
                pt_sel = selections.Selection(
                    f'@{rate}kHz', f'p_{{T}}^{{TOBJ}}>={pt_cut}GeV (@{rate}kHz)', lambda ar, pt_cut=pt_cut : ar.pt >= pt_cut)
                obj_sel = selections.Selector(f'^{obj_sel_name}$', sm.selections)()[0]

                # print(obj_sel*pt_sel)
                data_selections.append(obj_sel&pt_sel)
                # gen_selections.append(selections.Selection('all'))
    return data_selections


# TP selections

tp_pt_sel = [
    # Selection('Pt5to10', '5<=p_{T}^{TOBJ}<10GeV', '(pt >= 5) & (pt < 10)'),
    # Selection('Pt10to20', '10<=p_{T}^{TOBJ}<20GeV', '(pt >= 10) & (pt < 20)'),
    # Selection('Pt10', 'p_{T}^{TOBJ}>=10GeV', 'pt >= 10'),
    Selection('Pt10', 'p_{T}^{TOBJ} #geq 10 GeV', lambda array: array.pt >= 10),
    Selection('Pt20', 'p_{T}^{TOBJ} #geq 20 GeV', lambda array: array.pt >= 20),
    Selection('Pt25', 'p_{T}^{TOBJ} #geq 25 GeV', lambda array: array.pt >= 25),
    Selection('Pt30', 'p_{T}^{TOBJ} #geq 30 GeV', lambda array: array.pt >= 30)
]
tp_pt_sel_ext = [
    Selection('Pt2', 'p_{T}^{TOBJ} #geq 2GeV',  lambda array: array.pt >= 2),
    Selection('Pt3', 'p_{T}^{TOBJ} #geq 3GeV',  lambda array: array.pt >= 3),
    Selection('Pt4', 'p_{T}^{TOBJ} #geq 4GeV',  lambda array: array.pt >= 4),
    Selection('Pt5', 'p_{T}^{TOBJ} #geq 5GeV',  lambda array: array.pt >= 5),
    Selection('Pt10', 'p_{T}^{TOBJ} #geq 10 GeV', lambda array: array.pt >= 10),
    Selection('Pt15', 'p_{T}^{TOBJ} #geq 15 GeV', lambda array: array.pt >= 15),
    Selection('Pt20', 'p_{T}^{TOBJ} #geq 20 GeV', lambda array: array.pt >= 20),

    Selection('Pt23', 'p_{T}^{TOBJ} #geq 23 GeV', lambda array: array.pt >= 23),
    Selection('Pt28', 'p_{T}^{TOBJ} #geq 28 GeV', lambda array: array.pt >= 28),
    Selection('Pt24', 'p_{T}^{TOBJ} #geq 23 GeV', lambda array: array.pt >= 24),
    Selection('Pt25', 'p_{T}^{TOBJ} #geq 25 GeV', lambda array: array.pt >= 25),
    Selection('Pt30', 'p_{T}^{TOBJ} #geq 30 GeV', lambda array: array.pt >= 30),
    Selection('Pt40', 'p_{T}^{TOBJ} #geq 40 GeV', lambda array: array.pt >= 40)
]

tp_tccluster_match_selections = [
    Selection('Pt5to10', '5 <= p_{T}^{TOBJ} < 10GeV', lambda array: (array.pt < 10) & (array.pt >= 5)),
    Selection('Pt5to15', '5 <= p_{T}^{TOBJ} < 15GeV', lambda array: (array.pt < 15) & (array.pt >= 5)),

    Selection('Pt10to20', '10 <= p_{T}^{TOBJ} #leq 20GeV', lambda array: (array.pt < 20) & (array.pt >= 10)),
    Selection('Pt10to25', '10 #leq p_{T}^{TOBJ} < 25GeV', lambda array: (array.pt < 25) & (array.pt >= 10))
]

tp_eta_ee_sel = [
    # Selection('EtaA', '|#eta^{TOBJ}| <= 1.52', 'abs(eta) <= 1.52'),
    # Selection('EtaB', '1.52 < |#eta^{TOBJ}| <= 1.7', '1.52 < abs(eta) <= 1.7'),
    # Selection('EtaC', '1.7 < |#eta^{TOBJ}| <= 2.4', '1.7 < abs(eta) <= 2.4'),
    # Selection('EtaD', '2.4 < |#eta^{TOBJ}| <= 2.8', '2.4 < abs(eta) <= 2.8'),
    # Selection('EtaDE', '2.4 < |#eta^{TOBJ}| <= 3.0', '2.4 < abs(eta) <= 3.0'),
    # Selection('EtaE', '|#eta^{TOBJ}| > 2.8', 'abs(eta) > 2.8'),
    # Selection('EtaAB', '|#eta^{TOBJ}| <= 1.7', 'abs(eta) <= 1.7'),
    # Selection('EtaABC', '|#eta^{TOBJ}| <= 2.4', 'abs(eta) <= 2.4'),
    Selection('EtaBC', '1.52 < |#eta^{TOBJ}| #leq 2.4', lambda array: (abs(array.eta) > 1.52) &  (abs(array.eta) <= 2.4)),
    Selection('EtaBCD', '1.52 < |#eta^{TOBJ}| #leq 2.8', lambda array: (abs(array.eta) > 1.52) &  (abs(array.eta) <= 2.8)),
    # Selection('EtaBCDE', '1.52 < |#eta^{TOBJ}| < 3', '1.52 < abs(eta) < 3')
                    ]

genpart_ele_selections = [
    Selection('Ele', 'e^{#pm}', lambda array: abs(array.pdgid) == PID.electron)]
genpart_photon_selections = [
    Selection('Phot', '#gamma', lambda array: abs(array.pdgid) == PID.photon)]
genpart_pion_selections = [
    Selection('Pion', '#pi', lambda array: abs(array.pdgid) == PID.pion)]


gen_ee_sel = [
    Selection('Ee', '', lambda array: array.reachedEE > 0),
]

eta_sel = [
    Selection('EtaA', '1.49 < |#eta^{TOBJ}| #leq 1.52', lambda array: (abs(array.eta) > 1.49) & (abs(array.eta) <= 1.52)),
    Selection('EtaB', '1.52 < |#eta^{TOBJ}| #leq 1.7', lambda array: (abs(array.eta) > 1.52) & (abs(array.eta) <= 1.7)),
    Selection('EtaC', '1.7 < |#eta^{TOBJ}| #leq 2.4', lambda array: (abs(array.eta) > 1.7) & (abs(array.eta) <= 2.4)),
    Selection('EtaD', '2.4 < |#eta^{TOBJ}| #leq 2.8', lambda array: (abs(array.eta) > 2.4) & (abs(array.eta) <= 2.8)),
    Selection('EtaDE', '2.4 < |#eta^{TOBJ}| #leq 3.0', lambda array: (abs(array.eta) > 2.4) & (abs(array.eta) <= 3.0)),
    Selection('EtaE', '|#eta^{TOBJ}| > 2.8', lambda array: abs(array.eta) > 2.8),
    Selection('EtaAB', '1.49 < |#eta^{TOBJ}| #leq 1.7', lambda array: (abs(array.eta) > 1.49) & (abs(array.eta) <= 1.7)),
    Selection('EtaABC', '1.49 < |#eta^{TOBJ}| #leq 2.4', lambda array: (abs(array.eta) > 1.49) & (abs(array.eta) <= 2.4)),
    Selection('EtaABCD', '1.49 < |#eta^{TOBJ}| #leq 2.8', lambda array: (abs(array.eta) > 1.49) & (abs(array.eta) <= 2.8)),
    Selection('EtaFABCD', '|#eta^{TOBJ}| #leq 2.8', lambda array: abs(array.eta) <= 2.8),
    Selection('EtaFABC', '|#eta^{TOBJ}| #leq 2.4', lambda array: abs(array.eta) <= 2.4),
    Selection('EtaBCDE', '1.52 < |#eta^{TOBJ}|', lambda array: abs(array.eta) > 1.52)
]

gen_pid_sel = [
    Selection('GEN', '', 
            lambda ar: ((np.abs(ar.pdgid) == PID.electron ) | (np.abs(ar.pdgid) == PID.photon)) & (ar.prompt >= 2)),
    Selection('GENPi', '', 
            lambda ar: ((np.abs(ar.pdgid) == PID.pion )) & (ar.prompt >= 2))
]

gen_jet_sel = [
    Selection('GENJ')
]

gen_ele_sel = [
    Selection('GEN11', '', f'((abs(pdgid) == {PID.electron}) & (abs(firstmother_pdgid) == {PID.electron}))')
]
gen_part_fbrem_sel = [
    Selection('all'),
    Selection('BremH', 'f_{BREM}>=0.5', 'fbrem >= 0.5'),
    Selection('BremL', 'f_{BREM}<0.5', 'fbrem < 0.5'),
]


eg_eta_eb_sel = [
    Selection('all'),
    Selection('EtaF', '|#eta^{TOBJ}| <= 1.479', lambda ar: abs(ar.eta) <= 1.479)
    ]
eg_eta_sel = [
    Selection('all'),
    Selection('EtaEB', '|#eta^{TOBJ}| <= 1.479', lambda ar: abs(ar.eta) <= 1.479),
    Selection('EtaEE', '1.479 < |#eta^{TOBJ}| <= 2.4', lambda ar: (abs(ar.eta) > 1.479) & (abs(ar.eta) <= 2.4)),
    Selection('EtaEEa', '1.479 < |#eta^{TOBJ}| <= 1.52', lambda ar: (abs(ar.eta) > 1.479) & (abs(ar.eta) <= 1.52)),
    Selection('EtaEEb', '1.52 < |#eta^{TOBJ}| <= 2.3', lambda ar: (abs(ar.eta) > 1.52) & (abs(ar.eta) <= 2.3)),
    Selection('EtaEEc', '2.3 < |#eta^{TOBJ}| <= 2.4', lambda ar: (abs(ar.eta) > 2.3) & (abs(ar.eta) <= 2.4)),
    Selection('EtaEEFwd', '1.479 < |#eta^{TOBJ}| <= 3.0', lambda ar: (abs(ar.eta) > 1.479) & (abs(ar.eta) <= 3.0)),

]

pfinput_regions = [
    Selection('all'),
    Selection('PFinBRL', 'Barrel', ' | '.join([f'eta_reg_{r}' for r in pf_regions.regions['BRL']])),  # 4 5 6 7 8 9
    Selection('PFinHGC', 'HgCal', ' | '.join([f'eta_reg_{r}' for r in pf_regions.regions['HGC']])),  # 3 10
    Selection('PFinHGCNoTk', 'HgCalNoTk', ' | '.join([f'eta_reg_{r}' for r in pf_regions.regions['HGCNoTk']])),  # 2 11
    Selection('PFinHF', 'HF', ' | '.join([f'eta_reg_{r}' for r in pf_regions.regions['HF']])),  # 0 1 12 13
    ]

pftkinput_quality = [
    Selection('all'),
    Selection('TkPt2Chi2', 'p_{T}^{TOBJ} > 2GeV & #Chi^{2}_{norm} < 15', '(pt > 2) & (chi2Red < 15)'),
    Selection('TkPt3Chi2', 'p_{T}^{TOBJ} > 3GeV & #Chi^{2}_{norm} < 15', '(pt > 3) & (chi2Red < 15)'),
    Selection('TkPt4Chi2', 'p_{T}^{TOBJ} > 4GeV & #Chi^{2}_{norm} < 15', '(pt > 4) & (chi2Red < 15)'),
    Selection('TkPt5Chi2', 'p_{T}^{TOBJ} > 5GeV & #Chi^{2}_{norm} < 15', '(pt > 5) & (chi2Red < 15)'),
    Selection('TkPt2', 'p_{T}^{TOBJ} > 2GeV', '(pt > 2) & (nStubs >= 4)'),
    Selection('TkPt2Chi2Pt5', '(p_{T}^{TOBJ} > 2GeV & #Chi^{2}_{norm} < 15) | p_{T}^{TOBJ} > 5GeV ', '((pt > 2) & (chi2Red < 15) | (pt > 5))  & (nStubs >= 4)'),
    Selection('TkPt2Chi2Pt10', '(p_{T}^{TOBJ} > 2GeV & #Chi^{2}_{norm} < 15) | p_{T}^{TOBJ} > 10GeV ', '((pt > 2) & (chi2Red < 15) | (pt > 10)) & (nStubs >= 4)'),
    Selection('TkPt5', 'p_{T}^{TOBJ} > 5GeV', '(pt > 5)'),
    Selection('TkPt10', 'p_{T}^{TOBJ} > 10GeV', '(pt > 10)'),
    Selection('TkCTL1', '(p_{T}^{TOBJ} > 2GeV & #Chi^{2}_{norm} < 15) | p_{T}^{TOBJ} > 5GeV ', lambda ar: ((ar.pt > 2) & (ar.chi2Red < 15) | (ar.pt > 5)))
    ]

pf_matchedtk_input_quality = [
    Selection('all'),
    Selection('MTkPt2Chi2', 'p_{T}^{Tk} > 2GeV & #Chi^{2}_{norm} < 15', '(tkpt > 2) & (tkchi2red < 15)'),
    Selection('MTkPt3Chi2', 'p_{T}^{Tk} > 3GeV & #Chi^{2}_{norm} < 15', '(tkpt > 3) & (tkchi2red < 15)'),
    Selection('MTkPt4Chi2', 'p_{T}^{Tk} > 4GeV & #Chi^{2}_{norm} < 15', '(tkpt > 4) & (tkchi2red < 15)'),
    Selection('MTkPt5Chi2', 'p_{T}^{Tk} > 5GeV & #Chi^{2}_{norm} < 15', '(tkpt > 5) & (tkchi2red < 15)'),
    Selection('MTkPt2', 'p_{T}^{Tk} > 2GeV', '(tkpt > 2)'),
    Selection('MTkPt2Chi2Pt5', '(p_{T}^{Tk} > 2GeV & #Chi^{2}_{norm} < 15) | p_{T}^{Tk} > 5GeV ', '((tkpt > 2) & (tkchi2red < 15) | (tkpt > 5))'),
    # Selection('MTkPt2Chi2Pt10', '(p_{T}^{Tk} > 2GeV & #Chi^{2}_{norm} < 15) | p_{T}^{Tk} > 10GeV ', '((tkpt > 2) & (tkchi2red < 15) | (tkpt > 10))'),
    Selection('MTkPt5', 'p_{T}^{Tk} > 5GeV', '(tkpt > 5)'),
    # Selection('MTkPt10', 'p_{T}^{Tk} > 10GeV', '(tkpt > 10)'),
    ]



pfeginput_pt = [
    Selection('all'),
    Selection('Pt1', 'p_{T}^{TOBJ}#geq1GeV', lambda ar: ar.pt >= 1),
    Selection('Pt2', 'p_{T}^{TOBJ}#geq2GeV', lambda ar: ar.pt >= 2),
    Selection('Pt5', 'p_{T}^{TOBJ}#geq5GeV', lambda ar: ar.pt >= 5),
]


def scaling_offline2online_pt(obj, sel, pt):
    # abspath
    script_dir = os.path.dirname(os.path.abspath(__file__)) #<-- absolute dir the script is in
    rel_path = "../data/scalings/scaling_ARv38.json"
# abs_file_path = os.path.join(script_dir, rel_path)
    #FIXME: this should come from confing yaml file...
    with open(os.path.join(script_dir, rel_path)) as f:
        scaling_data = json.load(f)

    scaling = scaling_data[obj][sel]
    return (pt-float(scaling['b']))/float(scaling['a'])

menu_thresh_pt = [
    Selection('PtStaEB51', 'p_{T}^{TOBJ}#geq51GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('EGSta', 'EtaEB', 51): ar.pt >= ptcut),
    Selection('PtStaEE51', 'p_{T}^{TOBJ}#geq51GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('EGSta', 'EtaEE', 51): ar.pt >= ptcut),
    Selection('PtStaEB37', 'p_{T}^{TOBJ}#geq37GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('EGSta', 'EtaEB', 37): ar.pt >= ptcut),
    Selection('PtStaEE37', 'p_{T}^{TOBJ}#geq37GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('EGSta', 'EtaEE', 37): ar.pt >= ptcut),
    Selection('PtStaEB24', 'p_{T}^{TOBJ}#geq24GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('EGSta', 'EtaEB', 24): ar.pt >= ptcut),
    Selection('PtStaEE24', 'p_{T}^{TOBJ}#geq24GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('EGSta', 'EtaEE', 24): ar.pt >= ptcut),
    Selection('PtStaEB12', 'p_{T}^{TOBJ}#geq12GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('EGSta', 'EtaEB', 12): ar.pt >= ptcut),
    Selection('PtStaEE12', 'p_{T}^{TOBJ}#geq12GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('EGSta', 'EtaEE', 12): ar.pt >= ptcut),

    Selection('PtEleEB36', 'p_{T}^{TOBJ}#geq36GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEleL2', 'EtaEB', 36): ar.pt >= ptcut),
    Selection('PtEleEE36', 'p_{T}^{TOBJ}#geq36GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEleL2', 'EtaEE', 36): ar.pt >= ptcut),
    Selection('PtEleEB25', 'p_{T}^{TOBJ}#geq25GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEleL2', 'EtaEB', 25): ar.pt >= ptcut),
    Selection('PtEleEE25', 'p_{T}^{TOBJ}#geq25GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEleL2', 'EtaEE', 25): ar.pt >= ptcut),
    Selection('PtEleEB12', 'p_{T}^{TOBJ}#geq12GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEleL2', 'EtaEB', 12): ar.pt >= ptcut),
    Selection('PtEleEE12', 'p_{T}^{TOBJ}#geq12GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEleL2', 'EtaEE', 12): ar.pt >= ptcut),

    Selection('PtIsoEleEB28', 'p_{T}^{TOBJ}#geq28GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEleL2', 'EtaEB', 28): ar.pt >= ptcut),
    Selection('PtIsoEleEE28', 'p_{T}^{TOBJ}#geq28GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEleL2', 'EtaEE', 28): ar.pt >= ptcut),
    Selection('PtIsoEleEB22', 'p_{T}^{TOBJ}#geq22GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEleL2', 'EtaEB', 22): ar.pt >= ptcut),
    Selection('PtIsoEleEE22', 'p_{T}^{TOBJ}#geq22GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEleL2', 'EtaEE', 22): ar.pt >= ptcut),

    Selection('PtIsoPhoEB36', 'p_{T}^{TOBJ}#geq36GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEmL2', 'IsoEtaEB', 36): ar.pt >= ptcut),
    Selection('PtIsoPhoEE36', 'p_{T}^{TOBJ}#geq36GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEmL2', 'IsoEtaEE', 36): ar.pt >= ptcut),
    Selection('PtIsoPhoEB22', 'p_{T}^{TOBJ}#geq22GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEmL2', 'IsoEtaEB', 22): ar.pt >= ptcut),
    Selection('PtIsoPhoEE22', 'p_{T}^{TOBJ}#geq22GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEmL2', 'IsoEtaEE', 22): ar.pt >= ptcut),
    Selection('PtIsoPhoEB12', 'p_{T}^{TOBJ}#geq12GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEmL2', 'IsoEtaEB', 12): ar.pt >= ptcut),
    Selection('PtIsoPhoEE12', 'p_{T}^{TOBJ}#geq12GeV', 
            lambda ar, ptcut=scaling_offline2online_pt('TkEmL2', 'IsoEtaEE', 12): ar.pt >= ptcut),
]


pfeg_ee_input_qual = [
    Selection('EGq1', 'hwQual 1', 'hwQual == 1'),
]

eg_id_eb_sel = [
    Selection('all'),
    Selection('LooseTkID', 'LooseTkID', 'looseTkID')]

eg_id_sel = [
    Selection('all'),
    Selection('IDTightS', 'Tight-STA', lambda array: array.IDTightSTA),
    Selection('IDTightE', 'Tight-TkEle', lambda array: array.IDTightEle),
    Selection('IDTightP', 'Tight-TkEm', lambda array: array.IDTightPho),
    Selection('IDLooseP', 'Loose-TkEm', lambda array: array.IDLoosePho),
    Selection('IDNoBrem', 'NoBrem', lambda array: array.IDNoBrem),
    Selection('IDBrem', 'Brem', lambda array: array.IDNoBrem == False),

    # Selection('IDEleH', 'TkEle ID (H)', ''),

    ]

tp_id_sel = [
    Selection('all'),
    Selection('IDTightEm', 'Tight-EM', lambda array: array.IDTightEm),
    Selection('IDLooseEm', 'Loose-EM', lambda array: array.IDLooseEm),
    ]

dz_sel = [
    Selection('Dz1', '|#DeltaZ|<1cm', lambda array: array.dz < 1)
]

dR_sel = [
    Selection('DRg0p1', '#DeltaR>0.1', lambda array: array.dr >0.1)
]

comp_id_sel = [
    Selection('IDCompWP955', 'CompID WP 0.955', lambda ar: ar.compBDTScore > -0.7318549872638138), #, epsilon_b = 0.0985
    # Selection('IDCompWP950', 'CompID WP 0.950', 'compBDTScore > -0.5871849', #, epsilon_b = 0.0917
    # Selection('IDCompWP940', 'CompID WP 0.940', 'compBDTScore > -0.4392925', #, epsilon_b = 0.0788
    # Selection('IDCompWP930', 'CompID WP 0.930', 'compBDTScore > -0.2919413', #, epsilon_b = 0.0638
    # Selection('IDCompWP920', 'CompID WP 0.920', 'compBDTScore > -0.1440416', #, epsilon_b = 0.0531
    # Selection('IDCompWP910', 'CompID WP 0.910', 'compBDTScore > 0.0825459', # epsilon_b = 0.0437
    Selection('IDCompWP900', 'CompID WP 0.900', lambda ar: ar.compBDTScore > 0.2157780720764229), # epsilon_b = 0.0373
    Selection('IDCompWP800', 'CompID WP 0.800', lambda ar: ar.compBDTScore > 1.694870131268548), # epsilon_b = 0.0081
    # Selection('IDCompWP700', 'CompID WP 0.700', 'compBDTScore > 0.9914881', # epsilon_b = 0.0034
    # Selection('IDCompWP650', 'CompID WP 0.650', 'compBDTScore > 0.9954325', # epsilon_b = 0.0021
    # Selection('IDCompWP600', 'CompID WP 0.600', 'compBDTScore > 0.9958264', # epsilon_b = 0.0017
    # Selection('IDCompWP550', 'CompID WP 0.550', 'compBDTScore > 0.9976058', # epsilon_b = 0.0013
    # Selection('IDCompWP500', 'CompID WP 0.500', 'compBDTScore > 0.9977186', # epsilon_b = 0.0004
    # Selection('IDCompWP450', 'CompID WP 0.450', 'compBDTScore > 0.9978157', # epsilon_b = 0.0004
    # Selection('IDCompWP400', 'CompID WP 0.400', 'compBDTScore > 0.9985109', # epsilon_b = 0.0004
    ]

iso_sel = [
    Selection('Iso0p2', 'iso_{tk}<=0.2', lambda ar: ar.tkIso <= 0.2),
    Selection('Iso0p1', 'iso_{tk}<=0.1', lambda ar: ar.tkIso <= 0.1),
    Selection('Iso0p3', 'iso_{tk}<=0.3', lambda ar: ar.tkIso <= 0.3),
    Selection('IsoEleEB', 'iso_{tk}<=0.13', lambda ar: ar.tkIso <= 0.13),
    Selection('IsoEleEE', 'iso_{tk}<=0.28', lambda ar: ar.tkIso <= 0.28),
    Selection('IsoPhoEB', 'iso_{tk}<=0.25', lambda ar: ar.tkIso <= 0.25),
    Selection('IsoPhoEE', 'iso_{tk}<=0.205', lambda ar: ar.tkIso <= 0.205),    
    # Selection('IsoEleMenu', 'iso_{tk}<=(0.13,0.28)', '((abs(eta) < 1.479) & (tkIso <= 0.13)) | ((abs(eta) > 1.479) & (tkIso <= 0.28))'),
    # Selection('IsoPhoMenu', 'iso_{tk}<=(0.25,0.205)', '((abs(eta) < 1.479) & (tkIso <= 0.25)) | ((abs(eta) > 1.479) & (tkIso <= 0.205))'),
    ]

iso_sel_flateff = []
for eff in [90, 92, 94, 96, 98]:
    iso_sel_flateff.append(
        Selection(f'IsoPho{eff}', f'iso_{{tk}}@{eff}\%', lambda ar, e=eff: getattr(ar, f'tkIso{e}'))
    )

working_points_histomax = {
        'v10_3151': [
                # Low eta
                {
                '900': 0.9903189,
                '950': 0.9646683,
                '975': 0.8292287,
                '995': -0.7099538,
                },
                # High eta
                {
                '900': 0.9932326,
                '950': 0.9611762,
                '975': 0.7616282,
                '995': -0.9163715,
                }
            ]
        }


tight_wp = ['975', '900']
loose_wp = ['995', '950']

version = 'v10_3151'

wps = working_points_histomax[version]
labels = ['LE', 'HE']
wls = zip(wps, labels)
# for i,
tphgc_egbdt_sel = []

for wps,lab in wls:

    for wp,cut in wps.items():
        tphgc_egbdt_sel.append(
            Selection(
                f'EgBdt{lab}{wp}', 
                f'BDT^{{eg}}_{{{lab}}}@{wp}%', 
                lambda x,cut=float(cut) : x.egEmIdScore > cut))

tphgc_pubdt_sel = [
    Selection('IDPuVeto', 'PU Veto', lambda ar: ar.pfPuIdPass),
    Selection('IDEmPf', 'PF Em', lambda ar: ar.pfEmIdPass),
    Selection('IDEmTight', 'EgID Tight', lambda ar: ar.IDTightEm),
    Selection('IDEmLoose', 'EgID Loose', lambda ar: ar.IDLooseEm),
]

# print(tphgc_egbdt_sel)

sm = SelectionManager()
Selector.selection_primitives = sm.selections.copy()


menu_sel = [
    ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightS$'))).one('MenuSta', 'TightID'),
    # ((Selector('^EtaEB')&('^IsoEleEB$'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^IDTightE$'))).one('MenuEleIsoTight', 'Iso TightID'),
    # ((Selector('^EtaEB')&('^IsoEleEB$'))|(Selector('^EtaEE$')&('^IsoEleEE'))).one('MenuEleIsoLoose', 'Iso LooseID'),
    ((Selector('^EtaEB')&('^IDTightE$')&('^IsoEleEB$'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^IDTightE$'))).one('MenuEleIsoTight', 'Iso TightID'),
    ((Selector('^EtaEB')&('^IDTightE$')&('^IsoEleEB$'))|(Selector('^EtaEE$')&('^IsoEleEE'))).one('MenuEleIsoLoose', 'Iso LooseID'),

    ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightE$'))).one('MenuEleTight', 'TightID'),
    ((Selector('^EtaEB')&('^IDTightE$$'))|(Selector('^EtaEE$'))).one('MenuEleLoose', 'LooseID'),
    ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightP'))).one('MenuPho', 'no-iso'),
    ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP'))).one('MenuPhoIso', 'Iso'),
    # Rate selections
    ((Selector('^EtaEB')&('^IDTightE$')&('^IsoEleEB')&('^PtIsoEleEB28'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^PtIsoEleEE28'))).one('SingleIsoTkEle28', 'SingleIsoTkEle28'),
    ((Selector('^EtaEB')&('^IDTightE$')&('^IsoEleEB')&('^PtIsoEleEB28'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^IDTightE$')&('^PtIsoEleEE28'))).one('SingleIsoTkEle28Tight', 'SingleIsoTkEle28Tight'),
    # ((Selector('^EtaEB')&('^IsoEleEB')&('^PtIsoEleEB28'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^PtIsoEleEE28'))).one('SingleIsoTkEle28', 'SingleIsoTkEle28'),
    # ((Selector('^EtaEB')&('^IsoEleEB')&('^PtIsoEleEB28'))|(Selector('^EtaEE$')&('^IsoEleEE')&('^IDTightE$')&('^PtIsoEleEE28'))).one('SingleIsoTkEle28Tight', 'SingleIsoTkEle28Tight'),
    ((Selector('^EtaEB')&('^IDTightE$')&('^PtEleEB36'))|(Selector('^EtaEE$')&('^IDTightE$')&('^PtEleEE36'))).one('SingleTkEle36', 'SingleTkEle36'),
    ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB36'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE36'))).one('SingleIsoTkPho36', 'SingleIsoTkPho36'),
    ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one('SingleIsoTkPho22', 'SingleIsoTkPho22'),
    ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB12'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE12'))).one('SingleIsoTkPho12', 'SingleIsoTkPho12'),
    ((Selector('^EtaEB')&('^IDTightE$')&('^PtStaEB51'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE51'))).one('SingleEGEle51', 'SingleEGEle51'),
    build_DiObj_selection('DoubleIsoTkPho22-12', 'DoubleIsoTkPho22-12',
                        ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
                        ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB12'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE12'))).one()),
    build_DiObj_selection('MenuDoubleIsoTkPho22-X', 'DoubleIsoTkPho22-X',
                        ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
                        ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP'))).one()),
    build_DiObj_selection('DoubleTkEle25-12', 'DoubleTkEle25-12',
                        ((Selector('^EtaEB')&('^IDTightE$')&('^PtEleEB25'))|(Selector('^EtaEE$')&('^PtEleEE25'))).one(),
                        ((Selector('^EtaEB')&('^IDTightE$')&('^PtEleEB12'))|(Selector('^EtaEE$')&('^PtEleEE12'))).one(),
                        Selector('^Dz1$').one()),
    build_DiObj_selection('MenuDoubleTkPho22-X', 'DoubleTkPho22-X',
                        ((Selector('^EtaEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
                        ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightP'))).one()),
    build_DiObj_selection('MenuDoubleIsoOneTkPho22-X', 'DoubleIsoOneTkPho22-X',
                        ((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
                        ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightP'))).one()),
    build_DiObj_selection('DoubleStaEG37-24', 'DoubleStaEG37-24',
                        ((Selector('^EtaEB')&('^IDTightE$')&('^PtStaEB37'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE37'))).one(),
                        ((Selector('^EtaEB')&('^IDTightE$')&('^PtStaEB24'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE24'))).one()),
    build_DiObj_selection('DoubleIsoTkEleStaEG22-12', 'DoubleIsoTkEleStaEG22-12',
                        ((Selector('^EtaEB')&('^IDTightE$')&('^IsoEleEB$')&('^PtIsoEleEB22'))|(Selector('^EtaEE$')&('^IsoEleEE$')&('^PtIsoEleEE22'))).one(),
                        ((Selector('^EtaEB')&('^IDTightE$')&('^IDTightE$')&('^PtStaEB12'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE12'))).one(),
                        Selector('^DRg0p1').one()),
    # build_DiObj_selection('DoubleIsoTkEleStaEG22-12', 'DoubleIsoTkEleStaEG22-12',
    #                     ((Selector('^EtaEB')&('^IsoEleEB$')&('^PtIsoEleEB22'))|(Selector('^EtaEE$')&('^IsoEleEE$')&('^PtIsoEleEE22'))).one(),
    #                     ((Selector('^EtaEB')&('^IDTightE$')&('^PtStaEB12'))|(Selector('^EtaEE$')&('^IDTightS')&('^PtStaEE12'))).one(),
    #                     Selector('^DRg0p1').one()),
                        

]


digen_sele = [
    build_DiObj_selection('DiGEN', 'Di-GEN prompt',
                        (Selector('^GEN$')).one(),
                        (Selector('^GEN$')).one()),
]


for eff in [90, 92, 94, 96, 98]:
    menu_sel.append(build_DiObj_selection(f'MenuDoubleIso{eff}TkPho22-X', f'DoubleIso{eff}TkPho22-X',
                        ((Selector('^EtaEB')&(f'^IsoPho{eff}')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&(f'^IsoPho{eff}')&('^IDTightP')&('^PtIsoPhoEE22'))).one(),
                        ((Selector('^EtaEB')&(f'^IsoPho{eff}')&('^IDTightE$'))|(Selector('^EtaEE$')&(f'^IsoPho{eff}')&('^IDTightP'))).one()))
    
    
    
    menu_sel.append(((Selector('^EtaEB')&(f'^IsoPho{eff}')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&(f'^IsoPho{eff}')&('^IDTightP')&('^PtIsoPhoEE22'))).one(f'Iso@{eff}TkPho22', f'p_{{T}}>22, iso@{eff}'))
    menu_sel.append(((Selector('^EtaEB')&(f'^IsoPho{eff}')&('^IDTightE$')&('^PtIsoPhoEB12'))|(Selector('^EtaEE$')&(f'^IsoPho{eff}')&('^IDTightP')&('^PtIsoPhoEE12'))).one(f'Iso@{eff}TkPho12', f'p_{{T}}>12, iso@{eff}'))
    menu_sel.append(((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB12'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE12'))).one(f'IsoTkPho12', 'p_{T}>12, iso@Menu'))
    menu_sel.append(((Selector('^EtaEB')&('^IsoPhoEB')&('^IDTightE$')&('^PtIsoPhoEB22'))|(Selector('^EtaEE$')&('^IsoPhoEE')&('^IDTightP')&('^PtIsoPhoEE22'))).one(f'IsoTkPho22', 'p_{T}>22, iso@Menu'))



ctl2_sel = [
    
    ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$'))).one('L2IDPhoL', 'ID-loose'),
    ((Selector('^EtaEB')&('^IDTightE$'))|(Selector('^EtaEE$')&('^IDTightP'))).one('L2IDPhoT', 'ID-tight'),
    ((Selector('^EtaEB')&('^IsoPhoEB'))|(Selector('^EtaEE$')&('^IsoPhoEE'))).one('L2Iso', 'iso'),
]



# repeat the call: we want the menu selections to be avaialble via the selectors
Selector.selection_primitives = sm.selections.copy()


# tp_rate_selections = (Selector('^Em|all')*('^Eta[^DA][BC]*[BCD]$|all'))()
# tp_match_selections = (Selector('^Em|all')*('^Pt[1-3]0$|all'))()
# tp_calib_selections = (Selector('^Em|all'))()

# tracks_selections = (Selector('^St[3-4]|all')*('^Pt[2-5]$|^Pt10$|all'))()

# gen_ee_calib_selections = (Selector('GEN$')*('Ee')*('^Eta[BC]+[CD]$|all'))()
# gen_ee_selections = (Selector('GEN$')*('Ee')*('^Eta[BC]+[CD]$|all')+Selector('GEN$')*('Ee')*('^Pt15|^Pt30'))()
# gen_eb_selections = (Selector('^GEN$')*('^Pt15|^Pt30|all')+Selector('^GEN$')*('^EtaF'))()
# gen_ee_extrange_selections = (Selector('GEN$')*('Ee')*('^Eta[BC]+[CD]$|all')+Selector('GEN$')*('Ee')*('^Pt15|^Pt30'))()
# gen_ee_tk_selections = (Selector('GEN$')*('Ee$')*('EtaBC$|all')+Selector('GEN$')*('Ee$')*('Pt15|Pt30'))()
# gen_ele_ee_selections = (Selector('GEN11')*('^Eta[BC]+[CD]$|all')*('^Pt15|all'))()
# gen_ele_ee_tk_selections = (Selector('GEN11')*('^Eta[BC]+[C]$|all')*('^Pt15|all'))()
# gen_selections = (Selector('GEN$')*('^Eta[DF]$|^Eta[BC]+[CD]$|^Pt15$|^Pt30$|all'))()
# genpart_ele_genplotting = (Selector('GEN11$|all'))()
# gen_pid_eta_fbrem_ee_selections = (Selector('^GEN$')*('Ee')*('^Eta[BC]+[BCD]$')*('^Brem[HL]|all'))()
# eg_id_pt_eb_selections = (Selector('^LooseTk|all')*('^Pt[1-2][0]$|all'))()
# eg_id_pt_eb_selections_ext = (Selector('^LooseTk|all')*('^Pt[1-4][0,5]$|all'))()
# eg_id_pt_ee_selections = (Selector('^EGq[4-5]')*('^Pt[1-4][0]$|all'))()
# eg_id_pt_ee_selections_ext = (Selector('^EGq[4-5]')*('^Pt[1-4][0,5]$|all'))()
# gen_pid_ee_selections = (Selector('GEN$')*('Ee$'))()

# simeg_ee_selections = (Selector('^EGq[4-5]$')*('^Pt[1-3][0]$|all'))()
# emueg_ee_selections = (Selector('^EGq[1-2]$')*('^Pt[1-3][0]$|all'))()
# # simeg_rate_ee_selections = (Selector('^EGq[4-5]$')*('^Eta[^DA][BC]*[BCD]$|all'))()
# # emueg_rate_ee_selections = (Selector('^EGq[1-3,6]$|^EGq[1,2]or[3]')*('^Eta[^DA][BC]*[BCD]$|all'))()
# simeg_match_ee_selections = (Selector('^EGq[4-5]$')*('^Pt[1-3][0]$|all'))()
# emueg_match_ee_selections = (Selector('^EGq[1,2]$')*('^Pt[1-2][0]$|all'))()
# eg_id_eta_ee_selections = (Selector('^EGq[4-5]')*('^Eta[BC]+[CD]$|all'))()

# pfeg_tp_input_selections = (Selector('^PFinH')*('^Pt[1,2,5]$|all')*('^Em$|all'))()
# pfeg_ee_input_selections = (Selector('^PFinH')*('^Pt[1,2,5]$|all')*('^EGq[1]$|all'))()
# pfeg_eb_input_selections = (Selector('^PFinB|all')*('^Pt[1,2,5]$'))()
# pftkinput_selections = (Selector('^PFinBRL|^PFinHGC$')*('^TkPt'))()


# egid_ee_selections = (Selector('^EGq[4-5]'))()
# egid_ee_pfnf_selections = (Selector('^EGq[1-2]$'))()

# # EG selection quality and Pt EE

# eg_id_iso_sel = [
#     Selection('all'),
#     Selection('LooseTkID', 'LooseTkID', 'looseTkID'),
#     # Selection('Iso0p1', 'Iso0p1', '((tkIso <= 0.1) & (abs(eta) <= 1.479)) | ((tkIso <= 0.125) & (abs(eta) > 1.479))'),
#     ]

# if False:
#     eg_id_iso_sel.extend(read_isowp_sel('data/iso_wps.json', 'PFTkEmEB', 'EtaF'))

# # for iso_var in ['tkIso']:
# #     for cut in [0.1, 0.2, 0.3, 0.4, 0.5]:
# #         cut_str = str(cut).replace('.', 'p')
# #         eg_id_iso_sel.append(Selection(f'{iso_var}{cut_str}', f'{iso_var}<={cut}', f'{iso_var}<={cut}'))
# #
# # for iso_var in ['tkIsoPV']:
# #     for cut in [0.01, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3]:
# #         cut_str = str(cut).replace('.', 'p')
# #         eg_id_iso_sel.append(Selection(f'{iso_var}{cut_str}', f'{iso_var}<={cut}', f'{iso_var}<={cut}'))


# barrel_rate_selections = multiply_selections(eg_eta_eb_sel, eg_id_iso_sel)
# all_rate_selections = prune(eg_eta_sel+barrel_rate_selections)

# eg_barrel_rate_selections = [sel for sel in barrel_rate_selections if 'Iso' not in sel.name]
# eg_all_rate_selections = [sel for sel in all_rate_selections if 'Iso' not in sel.name]

# # eg_id_pt_eb_selections = []
# # eg_id_pt_eb_selections += multiply_selections(eg_id_eb_sel, tp_pt_sel)

# eg_iso_sel = [
#     Selection('all'),
#     # Selection('Iso0p2', 'Iso0p2', 'tkIso <= 0.2'),
#     # Selection('Iso0p1', 'Iso0p1', 'tkIso <= 0.1'),
#     # Selection('Iso0p3', 'Iso0p3', 'tkIso <= 0.3'),
#     ]

# if False:
#     eg_iso_sel.extend(read_isowp_sel('data/iso_wps.json', 'PFTkEmEE', 'EtaABC'))


# # eg_id_iso_ee_sel = []
# # eg_id_iso_ee_sel += multiply_selections(eg_id_ee_selections, eg_iso_sel)
# # eg_id_iso_eta_ee_selections = []
# # eg_id_iso_eta_ee_selections += multiply_selections(eg_id_iso_ee_sel, tp_eta_ee_sel)
# # eg_id_iso_pt_ee_selections_ext = []
# # eg_id_iso_pt_ee_selections_ext += multiply_selections(eg_id_ee_selections, tp_pt_sel_ext)
# # eg_id_iso_pt_ee_selections_ext += eg_id_iso_ee_sel
# # eg_id_iso_pt_ee_selections_ext = prune(eg_id_iso_pt_ee_selections_ext)

# # print 'eg_id_iso_eta_ee_selections:'
# # print eg_id_iso_eta_ee_selections
# eg_id_iso_pt_eb_selections_ext = []
# # eg_id_iso_pt_eb_selections_ext += tp_pt_sel_ext
# eg_id_iso_pt_eb_selections_ext += multiply_selections(eg_id_pt_eb_selections_ext, eg_id_iso_sel)


# eg_iso_ee_wp = {
#     'tkIso0p2': [27, 16, 8],
#     'tkIsoPV0p06': [27, 19, 11]
# }

# eg_iso_ee_wp_sel = [
#     # Selection('tkIso0p2Pt10', 'tkIso <= 0.2 & p_{T}>10GeV', '(tkIso<=0.2)&(pt>10)'),
#     # Selection('tkIsoPV0p06Pt10', 'tkIsoPV <= 0.06 & p_{T}>10GeV', '(tkIsoPV<=0.06)&(pt>10)')
# ]

# # print(isopt_sels)
# # fill_isowp_sel(eg_iso_ee_wp_sel, eg_iso_ee_wp)

# eg_iso_pt_ee_selections = []
# # eg_iso_pt_eb_selections = []

# if False:
#     for iso_sel_name, pt_sel in read_isoptwp_sel('data/iso_pt_wps.json', 'PFNFtkEmEE'):
#         iso_sel = list(filter(lambda x: x.name == iso_sel_name, eg_id_iso_eta_ee_selections))[0]
#         eg_iso_pt_ee_selections.append(iso_sel+pt_sel)
#         # print(iso_sel+pt_sel)
#     #
#     for iso_sel_name, pt_sel in read_isoptwp_sel('data/iso_pt_wps.json', 'PFNFtkEmEB'):
#         iso_sel = list(filter(lambda x: x.name == iso_sel_name, barrel_rate_selections))[0]
#         eg_iso_pt_eb_selections.append(iso_sel+pt_sel)
# # else:
# #     eg_iso_pt_ee_selections += multiply_selections(eg_id_ee_selections, eg_iso_ee_wp_sel)

# # EG selection quality and Pt EB


if __name__ == '__main__':
    from cfg import *
    print('enter selection name: ')
    selec_name = input()
    sel_list = []
    sel_list = eval(selec_name)
    for sel in sel_list:
        print(sel)
