import uproot4 as up
import pandas as pd
import datetime

class TreeReader(object):
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
                            'gen_TrueNumInt']
        if len(self._branches) == 0:
            self._branches = [br for br in self.tree.keys() if br not in branch_blacklist]
        self.file_entry = -1


    def next(self, debug=0):
        if self.global_entry == -1:
            self.global_entry = self.entry_range[0]
            self.file_entry = self.entry_range[0]
        else:
            self.file_entry += 1
            self.global_entry += 1
            
        if self.max_events != -1:
            if self.n_tot_entries == self.max_events:
                print ('END loop for max_event!')
                # we processed the max # of events
                return False
        if self.entry_range[1] != -1:
            if self.global_entry > self.entry_range[1]:
                print ('END loop for entry_range')
                return False
        if self.file_entry > self.tree.num_entries:
            print ('END loop for end_of_file')
            return False

        # entry is the cursor in the file: when we open a new one (not the first) needs to be set to 0 again
        self.n_tot_entries += 1
        if debug >= 2 or self.global_entry % 100 == 0:
            print ("--- File entry: {}, global entry: {}, tot # events: {} @ {}".format(
                self.file_entry, self.global_entry, self.n_tot_entries, datetime.datetime.now()))
        return True

        
    def getDataFrame(self, prefix, entry_block, fallback=None):
        branches = [br for br in self._branches
                    if br.startswith(prefix+'_') and
                    not br == '{}_n'.format(prefix)]
        names = ['_'.join(br.split('_')[1:]) for br in branches]
        name_map = dict(zip(branches, names))
        
        if len(branches) == 0:
            if fallback is not None:
                return self.getDataFrame(prefix=fallback, entry_block=entry_block)
            return pd.DataFrame()
            
        # FIXME: stride needs to be set somehow
        df = self.tree.arrays(branches, library='pd', entry_start=self.file_entry, entry_stop=self.file_entry+entry_block)
        df.rename(columns=name_map, inplace=True)
        
        return df
