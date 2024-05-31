import datetime
import gc
import resource

import awkward as ak
import vector

vector.register_awkward()

class TreeReader:
    def __init__(self, entry_range, max_events):
        self.tree = None
        self._branches = []
        # this is the gloabl "entry" across files
        self.global_entry = -1
        # this is the "entry" local to the open file (reset to 0) every new file
        self.file_entry = -1
        self.max_events = max_events
        self.entry_range = entry_range

        self.n_tot_entries = 0

    def setTree(self, uptree):
        self.tree = uptree
        self._branches = []
        branch_blacklist = ['tc_wafer',
                            'tc_cell',
                            'tc_waferu',
                            'tc_waferv',
                            'tc_cellu',
                            'tc_cellv',
                            'gen_PUNumInt',
                            'gen_TrueNumInt',
                            # 'gen_daughters',
                            'simpart_posx', 'simpart_posy', 'simpart_posz',
                            ]
        if len(self._branches) == 0:
            self._branches = [br for br in self.tree.keys() if br not in branch_blacklist]
        print(f'open new tree file with # entries: {self.tree.num_entries}')
        self.file_entry = -1

    def next(self, debug=0):

        if self.max_events != -1:
            if self.n_tot_entries == self.max_events:
                print('END loop for max_event!')
                # we processed the max # of events
                return False
        if self.entry_range[1] != -1:
            if self.global_entry == self.entry_range[1]:
                print('END loop for entry_range')
                return False
        if self.file_entry == self.tree.num_entries-1:
            print('END loop for end_of_file')
            return False

        if self.global_entry == -1:
            self.global_entry = self.entry_range[0]
            self.file_entry = self.entry_range[0]
        else:
            self.file_entry += 1
            self.global_entry += 1

        # entry is the cursor in the file: when we open a new one (not the first) needs to be set to 0 again
        if debug >= 2 or self.global_entry % 1000 == 0:
            self.printEntry()

        self.n_tot_entries += 1
        return True

    def printEntry(self):
        print(f'--- File entry: {self.file_entry}, global entry: {self.global_entry}, tot # events: {self.n_tot_entries} @ {datetime.datetime.now()}, MaxRSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000.0:.2f} Mb')
        # print(self.tree.keys())
        # print(self.tree.arrays(['run', 'lumi', 'event'], library='pd', entry_start=self.file_entry, entry_stop=self.file_entry+1))
        # self.dump_garbage()

    def dump_garbage(self):
        """
        show us what's the garbage about
        """
        # force collection
        print ('\nGARBAGE:')
        gc.collect()

        print ('\nGARBAGE OBJECTS:')
        for x in gc.garbage:
            s = str(x)
            if len(s) > 80: s = s[:80]
            print (type(x),'\n  ', s)


    def getDataFrame(self, prefix, entry_block, fallback=None):
        branches = [br for br in self._branches
                    if br.startswith(f'{prefix}_') and
                    br != f'{prefix}_n']
        names = ['_'.join(br.split('_')[1:]) for br in branches]
        name_map = dict(zip(names, branches))
        if len(branches) == 0:
            if fallback is not None:
                return self.getDataFrame(prefix=fallback, entry_block=entry_block)
            prefs = set([br.split('_')[0] for br in self._branches])
            print(f'stored branch prefixes are: {prefs}')
            raise ValueError(f'[TreeReader::getDataFrame] No branches with prefix: {prefix}')

        akarray = self.tree.arrays(names,
                                   library='ak',
                                   aliases=name_map,
                                   entry_start=self.file_entry,
                                   entry_stop=self.file_entry+entry_block)

        # print(akarray)
        records = {}
        for field in akarray.fields:
            records[field] = akarray[field]

        if 'pt' in names and 'eta' in names and 'phi' in names:
            if 'mass' not in names and 'energy' not in names:
                records['mass'] = 0.*akarray['pt']
            return vector.zip(records)

        return ak.zip(records)

        # FIXME: we should probably do an ak.Record using sometjhing along the lines of:
        # ele_rec = ak.zip({'pt': tkele.pt, 'eta': tkele.eta, 'phi': tkele.phi}, with_name="pippo")
        # this would allow to handle the records and assign behaviours....

        # return akarray

