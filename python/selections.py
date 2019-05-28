"""
Define and instantiate the selections.

The Selection class define via string a selection to be pplied to a certain
DataFrame. The selections are named (the name enters the final histogram name).
Selections can be composed (added). The actual selection syntax follows the
`pandas` `DataFrame` `query` syntax.
"""


class PID:
    electron = 11
    photon = 22
    pizero = 111
    pion = 211
    kzero = 130


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


# TP selections
tp_id_selections = [
                    Selection('all', '', ''),
                    Selection('Em', 'EGId', 'quality >0'),
                    # Selection('Emv1', 'EGId V1', '(showerlength > 1) & (bdt_pu > 0.026) & (bdt_pi > -0.03)')
                    ]


tp_pt_selections = [Selection('all', '', ''),
                    Selection('Pt10', 'p_{T}^{L1}>=10GeV', 'pt >= 10'),
                    Selection('Pt20', 'p_{T}^{L1}>=20GeV', 'pt >= 20'),
                    # Selection('Pt25', 'p_{T}^{L1}>=25GeV', 'pt >= 25'),
                    Selection('Pt30', 'p_{T}^{L1}>=30GeV', 'pt >= 30')
                    ]


tp_eta_selections = [Selection('all', '', ''),
                     # Selection('EtaA', '|#eta^{L1}| <= 1.52', 'abs(eta) <= 1.52'),
                     # Selection('EtaB', '1.52 < |#eta^{L1}| <= 1.7', '1.52 < abs(eta) <= 1.7'),
                     # Selection('EtaC', '1.7 < |#eta^{L1}| <= 2.4', '1.7 < abs(eta) <= 2.4'),
                     Selection('EtaD', '2.4 < |#eta^{L1}| <= 2.8', '2.4 < abs(eta) <= 2.8'),
                     # Selection('EtaE', '|#eta^{L1}| > 2.8', 'abs(eta) > 2.8'),
                     # Selection('EtaAB', '|#eta^{L1}| <= 1.7', 'abs(eta) <= 1.7'),
                     # Selection('EtaABC', '|#eta^{L1}| <= 2.4', 'abs(eta) <= 2.4'),
                     Selection('EtaBC', '1.52 < |#eta^{L1}| <= 2.4', '1.52 < abs(eta) <= 2.4'),
                     Selection('EtaBCD', '1.52 < |#eta^{L1}| <= 2.8', '1.52 < abs(eta) <= 2.8'),
                     # Selection('EtaBCDE', '1.52 < |#eta^{L1}|', '1.52 < abs(eta)')
                     ]


tp_rate_selections = add_selections(tp_id_selections, tp_eta_selections)

tp_match_selections = add_selections(tp_id_selections, tp_pt_selections)

genpart_ele_selections = [Selection('Ele', 'e^{#pm}', 'abs(pdgid) == {}'.format(PID.electron))]
genpart_photon_selections = [Selection('Phot', '#gamma', 'abs(pdgid) == {}'.format(PID.photon))]                      #
genpart_pion_selections = [Selection('Pion', '#pi', 'abs(pdgid) == {}'.format(PID.pion))]


gen_ee_selections = [Selection('', '', 'reachedEE == 2')]

gen_eta_selections = [
                      # Selection('EtaA', '|#eta^{GEN}| <= 1.52', 'abs(eta) <= 1.52'),
                      Selection('EtaB', '1.52 < |#eta^{GEN}| <= 1.7', '1.52 < abs(eta) <= 1.7'),
                      Selection('EtaC', '1.7 < |#eta^{GEN}| <= 2.4', '1.7 < abs(eta) <= 2.4'),
                      Selection('EtaD', '2.4 < |#eta^{GEN}| <= 2.8', '2.4 < abs(eta) <= 2.8'),
                      # Selection('EtaE', '|#eta^{GEN}| > 2.8', 'abs(eta) > 2.8'),
                      # Selection('EtaAB', '|#eta^{GEN}| <= 1.7', 'abs(eta) <= 1.7'),
                      # Selection('EtaABC', '|#eta^{GEN}| <= 2.4', 'abs(eta) <= 2.4'),
                      Selection('EtaBC', '1.52 < |#eta^{GEN}| <= 2.4', '1.52 < abs(eta) <= 2.4'),
                      Selection('EtaBCD', '1.52 < |#eta^{GEN}| <= 2.8', '1.52 < abs(eta) <= 2.8'),
                      # Selection('EtaBCDE', '1.52 < |#eta^{GEN}|', '1.52 < abs(eta)')
                      ]


gen_pt_selections = [Selection('Pt10', 'p_{T}^{GEN}>=10GeV', 'pt >= 10'),
                     Selection('Pt20', 'p_{T}^{GEN}>=20GeV', 'pt >= 20'),
                     Selection('Pt30', 'p_{T}^{GEN}>=30GeV', 'pt >= 30'),
                     Selection('Pt40', 'p_{T}^{GEN}>=40GeV', 'pt >= 40')]

# gen_part_selections = [Selection('GEN', '', '(abs(pdgid) == {}) | (abs(pdgid) == {}) | (abs(pdgid) == {})'.format(PID.electron, PID.photon, PID.pion))]
# gen_part_selections = [Selection('GEN', '', '(abs(pdgid) == {}) & (firstmother_pdgid == {})'.format(PID.electron, PID.electron))]
gen_part_selections = [Selection('GEN', '', '((abs(pdgid) == {}) & (firstmother_pdgid == {})) | ((abs(pdgid) == {}) & (firstmother_pdgid == {}))'.format(PID.electron, PID.electron,
                                                                                                                                                         PID.photon, PID.photon))]


# gen_part_selections = [Selection('GEN', '', '(abs(pdgid) == {})'.format(PID.e lectron))]


gen_part_ee_sel = add_selections(gen_part_selections, gen_ee_selections)
gen_part_ee_pt_sel = add_selections(gen_part_ee_sel, gen_pt_selections)
gen_part_ee_eta_sel = add_selections(gen_part_ee_sel, gen_eta_selections)

gen_part_selections_debug = []
gen_part_selections_debug = add_selections(gen_part_ee_sel, [Selection('EtaBCD', '1.52 < |#eta^{GEN}| <= 2.8', '1.52 < abs(eta) <= 2.8')])


gen_part_selections = []
gen_part_selections += gen_part_ee_sel
gen_part_selections += gen_part_ee_pt_sel
gen_part_selections += gen_part_ee_eta_sel

print 'gen_part_selections: {}'.format(len(gen_part_selections))

genpart_ele_ee_selections = []
genpart_ele_ee_selections_tmp = add_selections(genpart_ele_selections, gen_ee_selections)
genpart_ele_ee_selections += genpart_ele_ee_selections_tmp
genpart_ele_ee_selections += add_selections(genpart_ele_ee_selections_tmp, gen_pt_selections)
genpart_ele_ee_selections += add_selections(genpart_ele_ee_selections_tmp, gen_eta_selections)

# print 'genpart_ele_ee_selections: {}'.format(len(genpart_ele_ee_selections))

genpart_photon_ee_selections = []
genpart_photon_ee_selections_tmp = add_selections(genpart_photon_selections, gen_ee_selections)
genpart_photon_ee_selections += genpart_photon_ee_selections_tmp
genpart_photon_ee_selections += add_selections(genpart_photon_ee_selections_tmp, gen_pt_selections)
genpart_photon_ee_selections += add_selections(genpart_photon_ee_selections_tmp, gen_eta_selections)

genpart_pion_ee_selections = []
genpart_pion_ee_selections_tmp = add_selections(genpart_pion_selections, gen_ee_selections)
genpart_pion_ee_selections += genpart_pion_ee_selections_tmp
genpart_pion_ee_selections += add_selections(genpart_pion_ee_selections_tmp, gen_pt_selections)
genpart_pion_ee_selections += add_selections(genpart_pion_ee_selections_tmp, gen_eta_selections)

# genpart_ele_

genpart_ele_genplotting = [Selection('all')]
genpart_ele_genplotting += add_selections(genpart_ele_selections, gen_ee_selections)

eg_qual_selections = [
                      # Selection('EGq1', 'q1', 'hwQual > 0'),
                      Selection('EGq2', 'q2', 'hwQual == 2'),
                      Selection('EGq3', 'q3', 'hwQual == 3')]

iso_selections = [Selection('all'),
                  Selection('Iso0p2', 'Iso0p2', 'tkIso <= 0.2'),
                  Selection('Iso0p3', 'Iso0p3', 'tkIso <= 0.3'), ]


tkisoeg_selections = []
tkisoeg_selections += add_selections(eg_qual_selections, iso_selections)


eg_rate_selections = []
eg_rate_selections += add_selections(eg_qual_selections, tp_eta_selections)
eg_pt_selections = []
eg_pt_selections += add_selections(eg_qual_selections, tp_pt_selections)

tkisoeg_rate_selections = []
tkisoeg_rate_selections += add_selections(tkisoeg_selections, tp_eta_selections)
tkisoeg_pt_selections = []
tkisoeg_pt_selections += add_selections(tkisoeg_selections, tp_pt_selections)
# print 'tkisoeg_rate_selections:'
# print tkisoeg_rate_selections


tkeg_selection = [Selection('all'),
                  Selection('M1', '|#Delta#phi| <0.08 & #DeltaR < 0.07', '(abs(dphi) < 0.08) & (dr < 0.07)'),
                  Selection('M1P2', '|#Delta#phi| <0.08 & #DeltaR < 0.07 & p_{T}^{trk} > 2GeV', '(abs(dphi) < 0.08) & (dr < 0.07) & (tkpt > 2.)'),
                  Selection('M1P5', '|#Delta#phi| <0.08 & #DeltaR < 0.07 & p_{T}^{trk} > 5GeV', '(abs(dphi) < 0.08) & (dr < 0.07) & (tkpt > 5.)'),
                  Selection('M1P10', '|#Delta#phi| <0.08 & #DeltaR < 0.07 & p_{T}^{trk} > 10GeV', '(abs(dphi) < 0.08) & (dr < 0.07) & (tkpt > 10.)'),
                  Selection('M1P2S', '|#Delta#phi| <0.08 & #DeltaR < 0.07 & p_{T}^{trk} > 2GeV', '(abs(dphi) < 0.08) & (dr < 0.07) & (tkpt > 2.) & (tknstubs > 3)'),
                  Selection('M1P5S', '|#Delta#phi| <0.08 & #DeltaR < 0.07 & p_{T}^{trk} > 5GeV', '(abs(dphi) < 0.08) & (dr < 0.07) & (tkpt > 5.) & (tknstubs > 3)'),
                  Selection('M1P10S', '|#Delta#phi| <0.08 & #DeltaR < 0.07 & p_{T}^{trk} > 10GeV', '(abs(dphi) < 0.08) & (dr < 0.07) & (tkpt > 10.) & (tknstubs > 3)'),
                  # Selection('M2', '|#Delta#phi| <0.08 & #DeltaR < 0.05', '(abs(dphi) < 0.08) & (dr < 0.05)'),
                  # Selection('M2P', '|#Delta#phi| <0.08 & #DeltaR < 0.05 & p_{T}^{trk} > 10GeV', '(abs(dphi) < 0.08) & (dr < 0.05) & (tkpt > 10.)'),
                  # Selection('M2S', '|#Delta#phi| <0.08 & #DeltaR < 0.05 & #stubs > 3', '(abs(dphi) < 0.08) & (dr < 0.05) & (tknstubs > 3)'),
                  ]

tkeg_rate_selections = []
tkeg_rate_selections += add_selections(eg_rate_selections, tkeg_selection)
tkeg_qual_selections = []
tkeg_qual_selections += add_selections(eg_qual_selections, tkeg_selection)
tkeg_pt_selections = []
tkeg_pt_selections += add_selections(tkeg_qual_selections, tp_pt_selections)

# === L1 Track selections ===========================================

tracks_quality_sels = [Selection('all'),
                       Selection('St4', '# stubs > 3', 'nStubs > 3')]
tracks_pt_sels = [Selection('all'),
                  Selection('Pt2', 'p_{T}^{tk} > 2 GeV', 'pt > 2'),
                  Selection('Pt10', 'p_{T}^{tk} > 10 GeV', 'pt > 10')]

tracks_selections = []
tracks_selections += add_selections(tracks_quality_sels, tracks_pt_sels)


if __name__ == "__main__":
    for sel in gen_part_selections_debug:
        print sel
