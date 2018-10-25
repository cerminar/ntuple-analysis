# ---------------------------------------------------

class PID:
    electron = 11
    photon = 22
    pizero = 111
    pion = 211
    kzero = 130


class Selection:
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
tp_id_selections = [Selection('all', '', ''),
                    Selection('Em', 'EGId', 'quality >0'),
                    Selection('Emv1', 'EGId V2', '(showerlength > 1) & (bdt_out > 0.02)')]


tp_pt_selections = [Selection('all', '', ''),
                    Selection('Pt10', 'p_{T}^{L1}>=10GeV', 'pt >= 10'),
                    Selection('Pt20', 'p_{T}^{L1}>=20GeV', 'pt >= 20'),
                    Selection('Pt25', 'p_{T}^{L1}>=25GeV', 'pt >= 25'),
                    Selection('Pt30', 'p_{T}^{L1}>=30GeV', 'pt >= 30')]


tp_eta_selections = [Selection('all', '', ''),
                     Selection('EtaA', '|#eta^{L1}| <= 1.52', 'abs(eta) <= 1.52'),
                     Selection('EtaB', '1.52 < |#eta^{L1}| <= 1.7', '1.52 < abs(eta) <= 1.7'),
                     Selection('EtaC', '1.7 < |#eta^{L1}| <= 2.4', '1.7 < abs(eta) <= 2.4'),
                     Selection('EtaD', '2.4 < |#eta^{L1}| <= 2.8', '2.4 < abs(eta) <= 2.8'),
                     Selection('EtaE', '|#eta^{L1}| > 2.8', 'abs(eta) > 2.8'),
                     Selection('EtaAB', '|#eta^{L1}| <= 1.7', 'abs(eta) <= 1.7'),
                     Selection('EtaABC', '|#eta^{L1}| <= 2.4', 'abs(eta) <= 2.4'),
                     Selection('EtaBC', '1.52 < |#eta^{L1}| <= 2.4', '1.52 < abs(eta) <= 2.4'),
                     Selection('EtaBCD', '1.52 < |#eta^{L1}| <= 2.8', '1.52 < abs(eta) <= 2.8'),
                     Selection('EtaBCDE', '1.52 < |#eta^{L1}|', '1.52 < abs(eta)')]


tp_rate_selections = add_selections(tp_id_selections, tp_eta_selections)

tp_match_selections = add_selections(tp_id_selections, tp_pt_selections)

genpart_ele_selections = [Selection('Ele', 'e^{#pm}', 'abs(pdgid) == {}'.format(PID.electron))]
genpart_photon_selections = [Selection('Phot', '#gamma', 'abs(pdgid) == {}'.format(PID.photon))]                      #
genpart_pion_selections = [Selection('Pion', '#pi', 'abs(pdgid) == {}'.format(PID.pion))]


gen_ee_selections = [Selection('', '', 'reachedEE == 2')]

gen_eta_selections = [Selection('EtaA', '|#eta^{GEN}| <= 1.52', 'abs(eta) <= 1.52'),
                      Selection('EtaB', '1.52 < |#eta^{GEN}| <= 1.7', '1.52 < abs(eta) <= 1.7'),
                      Selection('EtaC', '1.7 < |#eta^{GEN}| <= 2.4', '1.7 < abs(eta) <= 2.4'),
                      Selection('EtaD', '2.4 < |#eta^{GEN}| <= 2.8', '2.4 < abs(eta) <= 2.8'),
                      Selection('EtaE', '|#eta^{GEN}| > 2.8', 'abs(eta) > 2.8'),
                      Selection('EtaAB', '|#eta^{GEN}| <= 1.7', 'abs(eta) <= 1.7'),
                      Selection('EtaABC', '|#eta^{GEN}| <= 2.4', 'abs(eta) <= 2.4'),
                      Selection('EtaBC', '1.52 < |#eta^{GEN}| <= 2.4', '1.52 < abs(eta) <= 2.4'),
                      Selection('EtaBCD', '1.52 < |#eta^{GEN}| <= 2.8', '1.52 < abs(eta) <= 2.8'),
                      Selection('EtaBCDE', '1.52 < |#eta^{GEN}|', '1.52 < abs(eta)')]


gen_pt_selections = [Selection('Pt10', 'p_{T}^{GEN}>=10GeV', 'pt >= 10'),
                     Selection('Pt20', 'p_{T}^{GEN}>=20GeV', 'pt >= 20'),
                     Selection('Pt30', 'p_{T}^{GEN}>=30GeV', 'pt >= 30'),
                     Selection('Pt40', 'p_{T}^{GEN}>=40GeV', 'pt >= 40')]

gen_part_selections = [Selection('GEN', '', '(abs(pdgid) == {}) | (abs(pdgid) == {}) | (abs(pdgid) == {})'.format(PID.electron, PID.photon, PID.pion))]

gen_part_ee_sel = add_selections(gen_part_selections, gen_ee_selections)
gen_part_ee_pt_sel = add_selections(gen_part_ee_sel, gen_pt_selections)
gen_part_ee_eta_sel = add_selections(gen_part_ee_sel, gen_eta_selections)

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
genpart_ele_genplotting +=  add_selections(genpart_ele_selections, gen_ee_selections)

eg_qual_selections = [Selection('EGq1', 'EGq1', 'hwQual > 0'),
                      Selection('EGq2', 'EGq2', 'hwQual > 1')]

eg_rate_selections = []
eg_rate_selections += add_selections(eg_qual_selections, tp_eta_selections)
eg_pt_selections = []
eg_pt_selections += add_selections(eg_qual_selections, tp_pt_selections)



class EgammaSet:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.cl3d_df = None

    def set_collections(self, cl3d):
        self.cl3d_df = cl3d


class TPSet:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.tc_df = None
        self.cl2d_df = None
        self.cl3d_df = None

    def set_collections(self, tc_df, cl2d_df, cl3d_df):
        self.tc_df = tc_df
        self.cl2d_df = cl2d_df
        self.cl3d_df = cl3d_df


class GenSet:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.gen_df = None

    def set_collections(self, gen_df):
        self.gen_df = gen_df


class TTSet:
    def __init__(self, name, label):
        self.name = name
        self.label = label
        self.tt_df = None

    def set_collections(self, tt_df):
        self.tt_df = tt_df


tp_def = TPSet('DEF', 'NNDR')
tp_def_calib = TPSet('DEFCalib', 'NNDR + calib. v1')
gen_set = GenSet('GEN', '')
tt_set = TTSet('TT', 'Trigger Towers')
simtt_set = TTSet('SimTT', 'Sim Trigger Towers')
eg_set = EgammaSet('EG', 'EGPhase2')

if __name__ == "__main__":
    for sel in gen_part_selections:
        print sel
