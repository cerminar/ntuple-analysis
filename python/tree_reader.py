import datetime
import gc
import os
import resource
import subprocess

import awkward as ak
import vector

vector.register_awkward()


def get_current_rss_mb():
    """Return current process RSS in MB, or None if unavailable."""
    try:
        rss_kb = subprocess.check_output(
            ['ps', '-o', 'rss=', '-p', str(os.getpid())],
            text=True,
        ).strip()
        return float(rss_kb) / 1024.0 if rss_kb else None
    except Exception:
        return None

class TreeReader:
    def __init__(self, entry_range, max_events, progress_every=10000):
        self.tree = None
        self._branches = []
        # this is the gloabl "entry" across files
        self.global_entry = -1
        # this is the "entry" local to the open file (reset to 0) every new file
        self.file_entry = -1
        self.max_events = max_events
        self.entry_range = entry_range
        self.progress_every = progress_every

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
                if debug >= 1:
                    print('END loop for max_event!')
                # we processed the max # of events
                return False
        if self.entry_range[1] != -1:
            if self.global_entry == self.entry_range[1]:
                if debug >= 1:
                    print('END loop for entry_range')
                return False
        if self.file_entry == self.tree.num_entries-1:
            if debug >= 1:
                print('END loop for end_of_file')
            return False

        if self.global_entry == -1:
            self.global_entry = self.entry_range[0]
            self.file_entry = self.entry_range[0]
        else:
            self.file_entry += 1
            self.global_entry += 1

        # entry is the cursor in the file: when we open a new one (not the first) needs to be set to 0 again
        if debug >= 2:
            self.printEntry(include_event_id=True)
        elif self.progress_every > 0 and self.global_entry % self.progress_every == 0:
            self.printEntry(include_event_id=False)

        self.n_tot_entries += 1
        return True

    def printEntry(self, include_event_id=False):
        msg = (
            f'--- File entry: {self.file_entry}, glb. entry: {self.global_entry}, '
            f'tot evts.: {self.n_tot_entries}'
        )

        if include_event_id:
            evtIndex = self.tree.arrays(
                ['run', 'luminosityBlock', 'event'],
                library='pd',
                entry_start=self.file_entry,
                entry_stop=self.file_entry+1,
            )
            row = evtIndex.iloc[0]
            msg += f' (e:{row["event"]} l:{row["luminosityBlock"]} r:{row["run"]})'

        max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000000.0
        curr_rss_mb = get_current_rss_mb()
        msg += f' @ {datetime.datetime.now().replace(microsecond=0)}, MaxRSS {max_rss_mb:.2f} Mb'
        if curr_rss_mb is not None:
            msg += f', CurrRSS {curr_rss_mb:.2f} Mb'
        print(msg)
        # print(self.tree.keys())
        # print(f"run={row['run']}, lumi={row['luminosityBlock']}, event={row['event']}")
        # print(.to_dict())
        # print(f'')
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
            return ak.zip(records, with_name="Momentum4D")

        return ak.zip(records)

        # FIXME: we should probably do an ak.Record using sometjhing along the lines of:
        # ele_rec = ak.zip({'pt': tkele.pt, 'eta': tkele.eta, 'phi': tkele.phi}, with_name="pippo")
        # this would allow to handle the records and assign behaviours....

        # return akarray

