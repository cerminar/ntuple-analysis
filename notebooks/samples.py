import ROOT

version = 'v47'

files = {}
file_keys = {}


class RootFile:
    def __init__(self, file_name):
        global file
        self.file_name = file_name
        if self.file_name not in files.keys():
            print 'get file: {}'.format(self.file_name)
            files[self.file_name] = ROOT.TFile(self.file_name)
        self._file = files[self.file_name]
        self._file_keys = None

    def cd(self):
        self._file.cd()

    def GetListOfKeys(self):
        global file_keys
        if self.file_name not in file_keys.keys():
            print 'get list'
            file_keys[self.file_name] = self._file.GetListOfKeys()
        self._file_keys = file_keys[self.file_name]
        return self._file_keys

class Sample():
    def __init__(self, name, label, version=None):
        self.name = name
        self.label = label
        if version:
            version = '_'+version
        else:
            version = ''
        self.histo_filename = '../plots1/histos_{}{}.root'.format(self.name, version)
        self.histo_file = ROOT.TFile(self.histo_filename)


sample_names = ['ele_flat2to100_PU0',
                'ele_flat2to100_PU200',
                'photonPt35_PU0',
                'photonPt35_PU200']

sample_ele_flat2to100_PU0 = Sample('ele_flat2to100_PU0', 'PU0', version)
sample_ele_flat2to100_PU200 = Sample('ele_flat2to100_PU200', 'PU200', version)

sample_gPt35_PU0 = Sample('photonPt35_PU0', 'Pt35 PU0', version)
sample_gPt35_PU200 = Sample('photonPt35_PU200', 'Pt35 PU200', version)

sample_hadronGun_PU0 = Sample('hadronGun_PU0', 'PU0', version)
sample_hadronGun_PU200 = Sample('hadronGun_PU200', 'PU200', version)

samples_ele = [sample_ele_flat2to100_PU0, sample_ele_flat2to100_PU200]

samples_photon = [sample_gPt35_PU0, sample_gPt35_PU200]
samples_hadrons = [sample_hadronGun_PU0, sample_hadronGun_PU200]

sample_nugunrate = Sample('nugun_alleta_pu200', 'PU200', version)
samples_nugunrates = [sample_nugunrate]


from python.selections import tp_rate_selections
from python.selections import tp_match_selections
from python.selections import gen_part_selections

tpsets = {'DEF': 'NNDR',
          'DEFCalib': 'NNDR Calib v1'}

tpset_selections = {}

gen_selections = {}

def get_label_dict(selections):
    dictionary = {}
    for sel in selections:
        dictionary[sel.name] = sel.label
    return dictionary


tpset_selections.update(get_label_dict(tp_rate_selections))
tpset_selections.update(get_label_dict(tp_match_selections))

gen_selections.update(get_label_dict(gen_part_selections))



tpset_labels = {'DEF': 'NNDR',
                'DEF_em': 'NNDR + EGID',
                'DEF_em_calib': 'NNDR + EGID + calib v1',
                'DEF_emL': 'NNDR + EGIDv1',
                'DEF_emL_calib': 'NNDR + EGIDv1 + calib v1',
                'DEF_pt10': 'NNDR, p_{T}^{L1}>10GeV',
                'DEF_pt20': 'NNDR, p_{T}^{L1}>20GeV',
                'DEF_pt25': 'NNDR, p_{T}^{L1}>25GeV',
                'DEF_pt30': 'NNDR, p_{T}^{L1}>30GeV',
                'DEF_pt10_em': 'NNDR + EGID, p_{T}^{L1}>10GeV',
                'DEF_pt20_em': 'NNDR + EGID, p_{T}^{L1}>20GeV',
                'DEF_pt25_em': 'NNDR + EGID, p_{T}^{L1}>25GeV',
                'DEF_pt30_em': 'NNDR + EGID, p_{T}^{L1}>30GeV',
                'DEF_pt10_emL': 'NNDR + EGIDv1, p_{T}^{L1}>10GeV',
                'DEF_pt20_emL': 'NNDR + EGIDv1, p_{T}^{L1}>20GeV',
                'DEF_pt25_emL': 'NNDR + EGIDv1, p_{T}^{L1}>25GeV',
                'DEF_pt30_emL': 'NNDR + EGIDv1, p_{T}^{L1}>30GeV',
                'DEF_pt10_em_calib': 'NNDR + EGID + calib, p_{T}^{L1}>10GeV',
                'DEF_pt20_em_calib': 'NNDR + EGID + calib, p_{T}^{L1}>20GeV',
                'DEF_pt25_em_calib': 'NNDR + EGID + calib, p_{T}^{L1}>25GeV',
                'DEF_pt30_em_calib': 'NNDR + EGID + calib, p_{T}^{L1}>30GeV',
                'DEF_pt10_emL_calib':  'NNDR + EGIDv1 + calib, p_{T}^{L1}>10GeV',
                'DEF_pt20_emL_calib': 'NNDR + EGIDv1 + calib, p_{T}^{L1}>20GeV',
                'DEF_pt25_emL_calib': 'NNDR + EGIDv1 + calib, p_{T}^{L1}>25GeV',
                'DEF_pt30_emL_calib': 'NNDR + EGIDv1 + calib, p_{T}^{L1}>30GeV'
                }


particle_labels = {'ele': 'all #eta',
                   'elePt20': 'p_{T}^{GEN}>20GeV',
                   'elePt30': 'p_{T}^{GEN}>30GeV',
                   'elePt40': 'p_{T}^{GEN}>40GeV',
                   'eleA': '|#eta^{GEN}| <= 1.52',
                   'eleB': '1.52 < |#eta^{GEN}| <= 1.7',
                   'eleC': '1.7 < |#eta^{GEN}| <= 2.4',
                   'eleD': '2.4 < |#eta^{GEN}| <= 2.8',
                   'eleE': '|#eta^{GEN}| > 2.8',
                   'eleAB': '|#eta^{GEN}| <= 1.7',
                   'eleABC': '|#eta^{GEN}| <= 2.4',
                   'eleBC': '1.52 < |#eta^{GEN}| <= 2.4',
                   'eleBCD': '1.52 < |#eta^{GEN}| <= 2.8',
                   'eleBCDE': '|#eta^{GEN}| > 1.52',
                   'all': 'all #eta^{L1}',
                   'etaA': '|#eta^{L1}| <= 1.52',
                   'etaB': '1.52 < |#eta^{L1}| <= 1.7)',
                   'etaC': '1.7 < |#eta^{L1}| <= 2.4)',
                   'etaD': '2.4 < |#eta^{L1}| <= 2.8)',
                   'etaE': '|#eta^{L1}| > 2.8',
                   'etaAB': '|#eta^{L1}| <= 1.7',
                   'etaABC': '|#eta^{L1}| <= 2.4',
                   'etaBC': '1.52 < |#eta^{L1}| <= 2.4',
                   'etaBCD': '1.52 < |#eta^{L1}| <= 2.8',
                   'etaBCDE': '|#eta^{L1}| > 1.52'}

samples = []
# particles = ''
