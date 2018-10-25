import ROOT

version = 'v51'

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
    def __init__(cls, name, label, version=None):
        cls.name = name
        cls.label = label
        if version:
            version = '_'+version
        else:
            version = ''
        cls.histo_filename = '../plots1/histos_{}{}.root'.format(cls.name, version)
        cls.histo_file = ROOT.TFile(cls.histo_filename, 'r')#RootFile(cls.histo_filename)


# sample_names = ['ele_flat2to100_PU0',
#                 'ele_flat2to100_PU200',
#                 'photonPt35_PU0',
#                 'photonPt35_PU200']


def get_label_dict(selections):
    dictionary = {}
    for sel in selections:
        dictionary[sel.name] = sel.label
    return dictionary



class HProxy:
    def __init__(self, classtype, tp, tp_sel, gen_sel, root_file):
        self.classtype = classtype
        self.tp = tp
        self.tp_sel = tp_sel
        self.gen_sel = gen_sel
        self.root_file = root_file
        self.instance = None


    def get(self):
        if self.instance is None:
            name = '{}_{}_{}'.format(self.tp, self.tp_sel, self.gen_sel)
            if self.gen_sel == None:
                name = '{}_{}'.format(self.tp, self.tp_sel)
            self.instance = self.classtype(name, self.root_file)
        return self.instance

class HPlot:
    def __init__(self, samples, tp_sets, tp_selections, gen_selections):
        self.tp_sets = tp_sets
        self.tp_selections = tp_selections
        self.gen_selections = gen_selections
        self.pus = []
        for sample in samples:
            self.pus.append(sample.label)
        self.data = pd.DataFrame(columns=['sample', 'pu', 'tp', 'tp_sel', 'gen_sel', 'classtype', 'histo'])
        self.labels_dict = {}

        self.labels_dict.update(tp_sets)
        self.labels_dict.update(tp_selections)
        self.labels_dict.update(gen_selections)
        self.labels_dict.update({'PU0': 'PU0', 'PU200': 'PU200'})

    def cache_histo(self,
                    classtype,
                    samples,
                    pus,
                    tps,
                    tp_sels,
                    gen_sels):
        for sample in samples:
            print sample
            for tp in tps:
                for tp_sel in tp_sels:
                    if gen_sels is None:
                        self.data = self.data.append({'sample': 'ele',
                                                        'pu': sample.label,
                                                        'tp': tp,
                                                        'tp_sel': tp_sel,
                                                        'gen_sel': None,
                                                        'classtype': classtype,
                                                        'histo': HProxy(classtype, tp, tp_sel, None, sample.histo_file)}
                                                          , ignore_index=True)
                    else:
                        for gen_sel in gen_sels:
                            print sample, tp, tp_sel, gen_sel
                            self.data = self.data.append({'sample': 'ele',
                                                            'pu': sample.label,
                                                            'tp': tp,
                                                            'tp_sel': tp_sel,
                                                            'gen_sel': gen_sel,
                                                            'classtype': classtype,
                                                            'histo': HProxy(classtype, tp, tp_sel, gen_sel, sample.histo_file)}
                                                              , ignore_index=True)



    def get_histo(self,
                  classtype,
                  sample=None,
                  pu=None,
                  tp=None,
                  tp_sel=None,
                  gen_sel=None):
        histo = None
        labels = []
        text = ''
        query = '(pu == @pu) & (tp == @tp) & (tp_sel == @tp_sel) & (classtype == @classtype)'
        if gen_sel is not None:
            query += '& (gen_sel == @gen_sel)'
        histo_df = self.data.query(query)

        if histo_df.empty:
            print 'No match found for: pu: {}, tp: {}, tp_sel: {}, gen_sel: {}, classtype: {}'.format(pu, tp, tp_sel, gen_sel, classtype)


        field_counts = histo_df.apply(lambda x: len(x.unique()))
        label_fields = []
        text_fields = []
        # print field_counts
        for field in field_counts.iteritems():
            if(field[1] > 1 and field[0] != 'histo' and field[0] != 'sample'):
                label_fields.append(field[0])
            if(field[1] == 1 and field[0] != 'histo' and field[0] != 'classtype' and field[0] != 'sample'):
                if(gen_sel is None and field[0] == 'gen_sel'):
                    continue
                text_fields.append(field[0])

#         print 'label fields: {}'.format(label_fields)
#         print 'text fields: {}'.format(text_fields)

        for item in histo_df[label_fields].iterrows():
            labels.append(', '.join([self.labels_dict[tx] for tx in item[1].values]))

        #print labels
        text = ', '.join([self.labels_dict[fl] for fl in histo_df[text_fields].iloc[0].values])
        histo = [his.get() for his in  histo_df['histo'].values]
        return histo, labels, text



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
from python.selections import genpart_ele_ee_selections

tpsets = {'DEF': 'NNDR',
          'DEFCalib': 'NNDR Calib v1'}

tpset_selections = {}

gen_selections = {}
samples = []


#tpset_selections.update(get_label_dict(tp_rate_selections))
tpset_selections.update(get_label_dict(tp_match_selections))

gen_selections.update(get_label_dict(genpart_ele_ee_selections))
