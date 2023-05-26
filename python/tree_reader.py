import pandas as pd
import datetime
import resource
import gc
import awkward as ak
import awkward_pandas as akpd

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
                            'gen_TrueNumInt',
                            'gen_daughters', 
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
        self.n_tot_entries += 1

        # entry is the cursor in the file: when we open a new one (not the first) needs to be set to 0 again
        if debug >= 2 or self.global_entry % 100 == 0:
            self.printEntry()
        return True

    def printEntry(self):
        print("--- File entry: {}, global entry: {}, tot # events: {} @ {}, MaxRSS {:.2f} Mb".format(
            self.file_entry,
            self.global_entry,
            self.n_tot_entries,
            datetime.datetime.now(),
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1E6))
        # print(self.tree.keys())
        # print(self.tree.arrays(['run', 'lumi', 'event'], library='pd', entry_start=self.file_entry, entry_stop=self.file_entry+1))
        # self.dump_garbage()
    
    def dump_garbage(self):
        """
        show us what's the garbage about
        """
            
        # force collection
        print ("\nGARBAGE:")
        gc.collect()

        print ("\nGARBAGE OBJECTS:")
        for x in gc.garbage:
            s = str(x)
            if len(s) > 80: s = s[:80]
            print (type(x),"\n  ", s)


    def getDataFrame(self, prefix, entry_block, fallback=None):
        branches = [br for br in self._branches
                    if br.startswith(prefix+'_') and
                    not br == '{}_n'.format(prefix)]
        names = ['_'.join(br.split('_')[1:]) for br in branches]
        name_map = dict(zip(names, branches))
        if len(branches) == 0:
            if fallback is not None:
                return self.getDataFrame(prefix=fallback, entry_block=entry_block)
            raise ValueError(f'[TreeReader::getDataFrame] No branches with prefix: {prefix}')
        
        akarray = self.tree.arrays(names, 
                                   library='ak', 
                                   aliases=name_map, 
                                   entry_start=self.file_entry, 
                                   entry_stop=self.file_entry+entry_block)
        return akarray
        
