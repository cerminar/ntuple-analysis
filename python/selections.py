"""
Define and instantiate the selections.

The Selection class define via string a selection to be pplied to a certain
DataFrame. The selections are named (the name enters the final histogram name).
Selections can be composed (added). The actual selection syntax follows the
`pandas` `DataFrame` `query` syntax.
"""

from __future__ import print_function
import json
import os


class PID:
    electron = 11
    photon = 22
    pizero = 111
    pion = 211
    kzero = 130


class SelectionManager(object):
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
        name (string): the name to be used in the histo name
                       (should not use `-` characters or spaces)

        label (string): used in plot legends, no constraints
        selection (string): see pandas.DataFrame.query syntax
    """

    def __init__(self, name, label='', selection=''):
        self.name = name
        self.label = label
        self.selection = selection
        self.hash = hash(selection)
        self.register()

    def register(self):
        selection_manager = SelectionManager()
        selection_manager.registerSelection(self)

    def __add__(self, sel_obj):
        """ & operation """
        if sel_obj.all:
            return self
        if self.all:
            return sel_obj
        new_label = '{}, {}'.format(self.label, sel_obj.label)
        if self.label == '':
            new_label = sel_obj.label
        if sel_obj.label == '':
            new_label = self.label
        return Selection(name='{}{}'.format(self.name, sel_obj.name),
                         label=new_label,
                         selection='({}) & ({})'.format(self.selection, sel_obj.selection))

    def __str__(self):
        return 'n: {}, s: {}, l:{}'.format(self.name, self.selection, self.label)

    def __repr__(self):
        return '<{} n: {}, s: {}, l:{}> '.format(self.__class__.__name__,
                                                 self.name,
                                                 self.selection,
                                                 self.label)

    @property
    def all(self):
        if self.name == 'all':
            return True
        return False


def add_selections(list1, list2):
    ret = []
    for sel1 in list1:
        for sel2 in list2:
            ret.append(sel1+sel2)
    return ret


def prune(selection_list):
    sel_names = set()
    ret = []
    for sel in selection_list:
        if sel.name not in sel_names:
            sel_names.add(sel.name)
            ret.append(sel)
    return ret


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



# TP selections
tp_id_sel = [
    Selection('all', '', ''),
    Selection('Em', 'EGId', 'quality >0'),
]
tp_pt_sel = [
    Selection('all', '', ''),
    # Selection('Pt5to10', '5<=p_{T}^{L1}<10GeV', '(pt >= 5) & (pt < 10)'),
    # Selection('Pt10to20', '10<=p_{T}^{L1}<20GeV', '(pt >= 10) & (pt < 20)'),
    Selection('Pt10', 'p_{T}^{L1}>=10GeV', 'pt >= 10'),
    Selection('Pt10', 'p_{T}^{L1}>=10GeV', 'pt >= 10'),
    Selection('Pt20', 'p_{T}^{L1}>=20GeV', 'pt >= 20'),
    # Selection('Pt25', 'p_{T}^{L1}>=25GeV', 'pt >= 25'),
    Selection('Pt30', 'p_{T}^{L1}>=30GeV', 'pt >= 30')
]
tp_pt_sel_ext = [
    Selection('all', '', ''),
    Selection('Pt10', 'p_{T}^{L1}>=10GeV', 'pt >= 10'),
    Selection('Pt15', 'p_{T}^{L1}>=15GeV', 'pt >= 15'),
    Selection('Pt20', 'p_{T}^{L1}>=20GeV', 'pt >= 20'),
    Selection('Pt25', 'p_{T}^{L1}>=25GeV', 'pt >= 25'),
    Selection('Pt30', 'p_{T}^{L1}>=30GeV', 'pt >= 30'),
    Selection('Pt40', 'p_{T}^{L1}>=40GeV', 'pt >= 40')
]

tp_tccluster_match_selections = [Selection('all', '', ''),
                                 Selection('Pt5to10', '5 <= p_{T}^{L1} < 10GeV', '(pt < 10) & (pt >= 5)'),
                                 Selection('Pt10to20', '10 <= p_{T}^{L1} < 20GeV', '(pt < 20) & (pt >= 10)')
                                 ]
tp_eta_ee_sel = [
    Selection('all', '', ''),
    # Selection('EtaA', '|#eta^{L1}| <= 1.52', 'abs(eta) <= 1.52'),
    # Selection('EtaB', '1.52 < |#eta^{L1}| <= 1.7', '1.52 < abs(eta) <= 1.7'),
    # Selection('EtaC', '1.7 < |#eta^{L1}| <= 2.4', '1.7 < abs(eta) <= 2.4'),
    # Selection('EtaD', '2.4 < |#eta^{L1}| <= 2.8', '2.4 < abs(eta) <= 2.8'),
    # Selection('EtaDE', '2.4 < |#eta^{L1}| <= 3.0', '2.4 < abs(eta) <= 3.0'),
    # Selection('EtaE', '|#eta^{L1}| > 2.8', 'abs(eta) > 2.8'),
    # Selection('EtaAB', '|#eta^{L1}| <= 1.7', 'abs(eta) <= 1.7'),
    Selection('EtaABC', '|#eta^{L1}| <= 2.4', 'abs(eta) <= 2.4'),
    # Selection('EtaBC', '1.52 < |#eta^{L1}| <= 2.4', '1.52 < abs(eta) <= 2.4'),
    # Selection('EtaBCD', '1.52 < |#eta^{L1}| <= 2.8', '1.52 < abs(eta) <= 2.8'),
    # Selection('EtaBCDE', '1.52 < |#eta^{L1}| < 3', '1.52 < abs(eta) < 3')
                     ]


tp_rate_selections = add_selections(tp_id_sel, tp_eta_ee_sel)
tp_match_selections = add_selections(tp_id_sel, tp_pt_sel)
tp_calib_selections = tp_id_sel


genpart_ele_selections = [
    Selection('Ele', 'e^{#pm}', 'abs(pdgid) == {}'.format(PID.electron))]
genpart_photon_selections = [
    Selection('Phot', '#gamma', 'abs(pdgid) == {}'.format(PID.photon))]
genpart_pion_selections = [
    Selection('Pion', '#pi', 'abs(pdgid) == {}'.format(PID.pion))]


gen_ee_sel = [
    Selection('', '', 'reachedEE >0'),
    # # FIXME: remove after test (or pick one)
    # Selection('R0', 'R0', 'reachedEE >0 ')
]
# gen_ee_sel = [
#     Selection('', '', 'reachedEE >0 ')]
gen_eta_ee_sel = [
    # Selection('EtaA', '|#eta^{GEN}| <= 1.52', 'abs(eta) <= 1.52'),
    # Selection('EtaB', '1.52 < |#eta^{GEN}| <= 1.7', '1.52 < abs(eta) <= 1.7'),
    # Selection('EtaC', '1.7 < |#eta^{GEN}| <= 2.4', '1.7 < abs(eta) <= 2.4'),
    # Selection('EtaD', '2.4 < |#eta^{GEN}| <= 2.8', '2.4 < abs(eta) <= 2.8'),
    # Selection('EtaDE', '2.4 < |#eta^{GEN}| <= 3.0', '2.4 < abs(eta) <= 3.0'),
    # Selection('EtaE', '|#eta^{GEN}| > 2.8', 'abs(eta) > 2.8'),
    # Selection('EtaAB', '|#eta^{GEN}| <= 1.7', 'abs(eta) <= 1.7'),
    # Selection('EtaABC', '|#eta^{GEN}| <= 2.4', 'abs(eta) <= 2.4'),
    Selection('EtaBC', '1.52 < |#eta^{GEN}| <= 2.4', '1.52 < abs(eta) <= 2.4'),
    # Selection('EtaBCD', '1.52 < |#eta^{GEN}| <= 2.8', '1.52 < abs(eta) <= 2.8'),
    # Selection('EtaBCDE', '1.52 < |#eta^{GEN}|', '1.52 < abs(eta)')
    ]
gen_eta_eb_sel = [
    Selection('EtaF', '|#eta^{GEN}| <= 1.479', 'abs(eta) <= 1.479')]
gen_eta_sel = [
    Selection('EtaF', '|#eta^{GEN}| <= 1.479', 'abs(eta) <= 1.479'),
    Selection('EtaD', '2.4 < |#eta^{GEN}| <= 2.8', '2.4 < abs(eta) <= 2.8'),
    Selection('EtaBC', '1.52 < |#eta^{GEN}| <= 2.4', '1.52 < abs(eta) <= 2.4'),
    Selection('EtaBCD', '1.52 < |#eta^{GEN}| <= 2.8', '1.52 < abs(eta) <= 2.8')
]


gen_pt_sel = [
    Selection('Pt15', 'p_{T}^{GEN}>=15GeV', 'pt >= 15'),
    # Selection('Pt10to25', '10 #leq p_{T}^{GEN} < 25GeV', '(pt >= 10) & (pt < 25)'),
    # Selection('Pt20', 'p_{T}^{GEN}>=20GeV', 'pt >= 20'),
    Selection('Pt30', 'p_{T}^{GEN}>=30GeV', 'pt >= 30'),
    # Selection('Pt35', 'p_{T}^{GEN}>=35GeV', 'pt >= 35'),
    # Selection('Pt40', 'p_{T}^{GEN}>=40GeV', 'pt >= 40')
]
gen_pt_sel_red = [
    Selection('all'),
    Selection('Pt15', 'p_{T}^{GEN}>=15GeV', 'pt >= 15')
]

gen_pt_upper = [
    Selection('', '', 'pt <= 100')
]

gen_pid_sel = [
    Selection('GEN', '', '(((abs(pdgid) == {}) & (abs(firstmother_pdgid) == {})) | \
                           ((abs(pdgid) == {}) & (abs(firstmother_pdgid) == {})))'.format(
        PID.electron, PID.electron,
        PID.photon, PID.photon))
]
gen_ele_sel = [
    Selection('GEN', '', '((abs(pdgid) == {}) & (abs(firstmother_pdgid) == {}))'.format(
        PID.electron, PID.electron))
]
gen_part_fbrem_sel = [
    Selection('all', '', ''),
    Selection('HBrem', 'f_{BREM} >= 0.5', 'fbrem >= 0.5'),
    Selection('LBrem', 'ff_{BREM} < 0.5', 'fbrem < 0.5'),
]
gen_ele_ee_sel = add_selections(gen_ele_sel, gen_ee_sel)
gen_ele_pt_ee_sel = add_selections(gen_ele_ee_sel, gen_pt_sel)
gen_ele_eta_ee_sel = add_selections(gen_ele_ee_sel, gen_eta_ee_sel)
gen_ele_eta_brem_ee_sel = add_selections(gen_ele_eta_ee_sel, gen_part_fbrem_sel)

gen_pid_ee_sel = add_selections(gen_pid_sel, gen_ee_sel)
gen_pid_pt_ee_sel = add_selections(gen_pid_ee_sel, gen_pt_sel)
gen_pid_eta_ee_sel = add_selections(gen_pid_ee_sel, gen_eta_ee_sel)
gen_pid_eta_fbrem_ee_sel = add_selections(gen_pid_eta_ee_sel, gen_part_fbrem_sel)



gen_ee_selections = []
gen_ee_selections += gen_pid_ee_sel
gen_ee_selections += gen_pid_pt_ee_sel
gen_ee_selections += gen_pid_eta_ee_sel
gen_ee_extrange_selections = gen_ee_selections
gen_ee_selections = add_selections(gen_ee_selections, gen_pt_upper)



# gen_ee_selections += add_selections(gen_pid_eta_ee_sel, gen_pt_sel_red)

gen_ee_tk_selections = [gen_sel for gen_sel in gen_ee_selections if 'EtaD' not in gen_sel.name and 'EtaBCD' not in gen_sel.name]

gen_eb_selections = []
gen_eb_selections += gen_pid_sel
gen_eb_selections += add_selections(gen_pid_sel, gen_pt_sel)
gen_eb_selections += add_selections(gen_pid_sel, gen_eta_eb_sel)


gen_ele_ee_selections = []
gen_ele_ee_selections += gen_ele_ee_sel
gen_ele_ee_selections += gen_ele_pt_ee_sel
# gen_ee_selections += gen_pid_eta_ee_sel
gen_ele_ee_selections += add_selections(gen_ele_eta_ee_sel, gen_pt_sel_red)

gen_ele_ee_tk_selections = [gen_sel for gen_sel in gen_ele_ee_selections if 'EtaD' not in gen_sel.name and 'EtaBCD' not in gen_sel.name]

gen_selections = []
gen_selections += gen_pid_sel
gen_selections += add_selections(gen_pid_sel, gen_pt_sel)
gen_selections += add_selections(gen_pid_sel, gen_eta_sel)


gen_ee_selections_calib = []
gen_ee_selections_calib += gen_pid_ee_sel
gen_ee_selections_calib += gen_pid_eta_ee_sel
# print (gen_pid_eta_ee_sel)
# gen_ee_selections_calib += add_selections([gen_pid_eta_ee_sel[1]], gen_pt_sel)
# gen_ee_selections_calib += gen_pid_eta_ee_sel

# genpart_ele_

genpart_ele_genplotting = [Selection('all')]
genpart_ele_genplotting += gen_ele_ee_sel

# EG selection quality and Pt EE

eg_eta_eb_sel = [
    Selection('all'),
    # Selection('EtaF', '|#eta^{L1}| <= 1.479', 'abs(eta) <= 1.479')
    ]
eg_eta_sel = [
    Selection('all'),
    Selection('EtaF', '|#eta^{L1}| <= 1.479', 'abs(eta) <= 1.479'),
    Selection('EtaA', '|#eta^{L1}| <= 1.52', 'abs(eta) <= 1.52'),
    Selection('EtaBC', '1.52 < |#eta^{L1}| <= 2.4', '1.52 < abs(eta) <= 2.4')
]
eg_id_iso_sel = [
    Selection('all'),
    # Selection('LooseTkID', 'LooseTkID', 'looseTkID'),
    # Selection('Iso0p1', 'Iso0p1', '((tkIso <= 0.1) & (abs(eta) <= 1.479)) | ((tkIso <= 0.125) & (abs(eta) > 1.479))'),
    ]

eg_id_iso_sel.extend(read_isowp_sel('data/iso_wps.json', 'PFTkEmEB', 'EtaF'))

# for iso_var in ['tkIso']:
#     for cut in [0.1, 0.2, 0.3, 0.4, 0.5]:
#         cut_str = str(cut).replace('.', 'p')
#         eg_id_iso_sel.append(Selection(f'{iso_var}{cut_str}', f'{iso_var}<={cut}', f'{iso_var}<={cut}'))
# 
# for iso_var in ['tkIsoPV']:
#     for cut in [0.01, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3]:
#         cut_str = str(cut).replace('.', 'p')
#         eg_id_iso_sel.append(Selection(f'{iso_var}{cut_str}', f'{iso_var}<={cut}', f'{iso_var}<={cut}'))


barrel_rate_selections = add_selections(eg_eta_eb_sel, eg_id_iso_sel)
all_rate_selections = prune(eg_eta_sel+barrel_rate_selections)

eg_barrel_rate_selections = [sel for sel in barrel_rate_selections if 'Iso' not in sel.name]
eg_all_rate_selections = [sel for sel in all_rate_selections if 'Iso' not in sel.name]


eg_id_ee_selections = [
    # Selection('EGq1', 'q1', 'hwQual > 0'),
    # Selection('EGq2', 'hwQual 2', 'hwQual == 2'),
    # Selection('EGq3', 'hwQual 3', 'hwQual == 3'),
    # Selection('EGq4', 'hwQual 4', 'hwQual == 4'),
    Selection('EGq5', 'hwQual 5', 'hwQual == 5')
]

eg_id_pt_ee_selections = []
eg_id_pt_ee_selections += add_selections(eg_id_ee_selections, tp_pt_sel)




eg_id_eb_sel = [
    Selection('all'),
    Selection('LooseTkID', 'LooseTkID', 'looseTkID')]


eg_id_pt_eb_selections = []
eg_id_pt_eb_selections += add_selections(eg_id_eb_sel, tp_pt_sel)


eg_iso_sel = [
    Selection('all'),
    # Selection('Iso0p2', 'Iso0p2', 'tkIso <= 0.2'),
    # Selection('Iso0p1', 'Iso0p1', 'tkIso <= 0.1'),
    # Selection('Iso0p3', 'Iso0p3', 'tkIso <= 0.3'), 
    ]

eg_iso_sel.extend(read_isowp_sel('data/iso_wps.json', 'PFTkEmEE', 'EtaABC'))
# for iso_var in ['tkIso']:
#     for cut in [0.1, 0.2, 0.3, 0.4, 0.5]:
#         cut_str = str(cut).replace('.', 'p')
#         eg_iso_sel.append(Selection(f'{iso_var}{cut_str}', f'{iso_var}<={cut}', f'{iso_var}<={cut}'))
# 
# for iso_var in ['tkIsoPV']:
#     for cut in [0.01, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3]:
#         cut_str = str(cut).replace('.', 'p')
#         eg_iso_sel.append(Selection(f'{iso_var}{cut_str}', f'{iso_var}<={cut}', f'{iso_var}<={cut}'))


eg_id_iso_ee_sel = []
eg_id_iso_ee_sel += add_selections(eg_id_ee_selections, eg_iso_sel)

eg_id_eta_ee_selections = []
eg_id_eta_ee_selections += add_selections(eg_id_ee_selections, tp_eta_ee_sel)

eg_id_pt_ee_selections_ext = []
eg_id_pt_ee_selections_ext += add_selections(eg_id_ee_selections, tp_pt_sel_ext)

eg_id_pt_eb_selections_ext = []
eg_id_pt_eb_selections_ext += add_selections(eg_id_eb_sel, tp_pt_sel_ext)
# eg_id_pt_eb_selections_ext += eg_id_iso_sel
# eg_id_pt_eb_selections_ext = prune(eg_id_pt_eb_selections_ext)

eg_id_iso_eta_ee_selections = []
eg_id_iso_eta_ee_selections += add_selections(eg_id_iso_ee_sel, tp_eta_ee_sel)
eg_id_iso_pt_ee_selections_ext = []
eg_id_iso_pt_ee_selections_ext += add_selections(eg_id_ee_selections, tp_pt_sel_ext)
eg_id_iso_pt_ee_selections_ext += eg_id_iso_ee_sel
eg_id_iso_pt_ee_selections_ext = prune(eg_id_iso_pt_ee_selections_ext)

# print 'eg_id_iso_eta_ee_selections:'
# print eg_id_iso_eta_ee_selections
eg_id_iso_pt_eb_selections_ext = []
# eg_id_iso_pt_eb_selections_ext += tp_pt_sel_ext
eg_id_iso_pt_eb_selections_ext += add_selections(eg_id_pt_eb_selections_ext, eg_id_iso_sel)


tracks_quality_sels = [Selection('all'),
                       Selection('St4', '# stubs > 3', 'nStubs > 3')]
tracks_pt_sels = [Selection('all'),
                  Selection('Pt2', 'p_{T}^{tk} > 2 GeV', 'pt > 2'),
                  Selection('Pt5', 'p_{T}^{tk} > 5 GeV', 'pt > 5'),
                  Selection('Pt10', 'p_{T}^{tk} > 10 GeV', 'pt > 10')]

tracks_selections = []
tracks_selections += add_selections(tracks_quality_sels, tracks_pt_sels)


pfinput_regions = [
    Selection('all'),
    Selection('BRL', 'Barrel', 'eta_reg_4 | eta_reg_5 | eta_reg_6'),  # 4 5 6
    Selection('HGC', 'HgCal', 'eta_reg_3 | eta_reg_7'),  # 3 7
    Selection('HGCNoTk', 'HgCalNoTk', 'eta_reg_2 | eta_reg_8'),  # 2 8
    Selection('HF', 'HF', 'eta_reg_0 | eta_reg_1 | eta_reg_9 | eta_reg_10'),  # 0 1 9 10
    ]

pftkinput_quality = [
    Selection('all'),
    Selection('Pt2Chi2', 'p_{T}^{Tk} > 2GeV & #Chi^{2}_{norm} < 15', '(pt > 2) & (chi2Red < 15) & (nStubs >= 4)'),
    Selection('Pt2', 'p_{T}^{Tk} > 2GeV', '(pt > 2) & (nStubs >= 4)'),
    Selection('Pt2Chi2Pt5', '(p_{T}^{Tk} > 2GeV & #Chi^{2}_{norm} < 15) | p_{T}^{Tk} > 5GeV ', '((pt > 2) & (chi2Red < 15) | (pt > 5))  & (nStubs >= 4)'),
    Selection('Pt2Chi2Pt10', '(p_{T}^{Tk} > 2GeV & #Chi^{2}_{norm} < 15) | p_{T}^{Tk} > 10GeV ', '((pt > 2) & (chi2Red < 15) | (pt > 10)) & (nStubs >= 4)'),
    Selection('Pt5', 'p_{T}^{Tk} > 5GeV', '(pt > 5)'),
    Selection('Pt10', 'p_{T}^{Tk} > 10GeV', '(pt > 10)'),
    ]

pfeginput_pt = [
    Selection('all'),
    Selection('Pt1', 'p_{T}^{Tk} > 1GeV', '(pt > 1)'),
    Selection('Pt2', 'p_{T}^{Tk} > 2GeV', '(pt > 2)'),
    Selection('Pt5', 'p_{T}^{Tk} > 5GeV', '(pt > 5)'),
]

pfeg_ee_input_qual = [
    Selection('EGq1', 'hwQual 1', 'hwQual == 1'),
]


pfeg_tp_input_selections = add_selections(
    pfinput_regions,
    add_selections(
        pfeginput_pt,
        tp_id_sel)
)

pfeg_ee_input_selections = add_selections(
    pfinput_regions,
    add_selections(
        pfeginput_pt,
        pfeg_ee_input_qual)
)

pfeg_eb_input_selections = add_selections(
    pfinput_regions,
    pfeginput_pt
)


pftkinput_selections = []
pftkinput_selections += add_selections(pfinput_regions, pftkinput_quality)

eg_iso_ee_wp = {
    'tkIso0p2': [27, 16, 8],
    'tkIsoPV0p06': [27, 19, 11]
}

eg_iso_ee_wp_sel = [
    # Selection('tkIso0p2Pt10', 'tkIso <= 0.2 & p_{T}>10GeV', '(tkIso<=0.2)&(pt>10)'),
    # Selection('tkIsoPV0p06Pt10', 'tkIsoPV <= 0.06 & p_{T}>10GeV', '(tkIsoPV<=0.06)&(pt>10)')
]

# print(isopt_sels)
# fill_isowp_sel(eg_iso_ee_wp_sel, eg_iso_ee_wp)

eg_iso_pt_ee_selections = []
# for iso_sel_name, pt_sel in read_isoptwp_sel('data/iso_pt_wps.json', 'PFNFtkEmEE'):
#     print (iso_sel_name, pt_sel)
#     iso_sel = list(filter(lambda x: x.name == iso_sel_name, eg_id_iso_eta_ee_selections))[0]
#     eg_iso_pt_ee_selections.append(iso_sel+pt_sel)
#     # print(iso_sel+pt_sel)
# # eg_iso_pt_ee_selections += add_selections(eg_id_ee_selections, eg_iso_ee_wp_sel)
# 
eg_iso_pt_eb_selections = []
# for iso_sel_name, pt_sel in read_isoptwp_sel('data/iso_pt_wps.json', 'PFNFtkEmEB'):
#     iso_sel = list(filter(lambda x: x.name == iso_sel_name, barrel_rate_selections))[0]
#     eg_iso_pt_eb_selections.append(iso_sel+pt_sel)

# EG selection quality and Pt EB

if __name__ == "__main__":
    # for sel in pfeginput_selections:
    #     print(sel)
    # for sel in eg_pt_selections:
    #     print sel.name
    # for sel in tkisoeg_pt_selections:
    #     print sel
    # for sel in gen_ee_selections_tketa:
    #     print sel
    for sel in eg_iso_pt_eb_selections:
        print (sel)
    # for sel in eg_pt_selections_barrel:
    #     print sel
    # for sel in gen_part_barrel_selections:
    #     print sel
    # for sel in gen_selections:
    #     print sel
    # for sel in gen_ee_selections_tketa:
    #     print sel
    # for sel in eg_pt_selections:
    #     print sel
    # for sel in eg_id_iso_eta_ee_selections:
    #     print sel
    # for sel in eg_id_iso_pt_eb_selections_ext:
    #     print sel
    # for sel in eg_id_pt_eb_selections_ext:
    #     print sel
    # for sel in eg_all_rate_selections:
    #     print sel
    # for sel in tp_rate_selections:
    #     print sel
    # for sel in gen_ele_ee_selections:
    #     print sel
