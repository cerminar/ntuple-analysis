# import math
# import collections

import ROOT
import pandas as pd
import root_numpy as rnp


##########
class HGCalNtuple(object):
    """Class abstracting the whole ntuple/TTree.

    Main benefit is to provide nice interface for
    - iterating over events
    - querying whether hit/seed information exists

    Note that to iteratate over the evets with zip(), you should use
    itertools.izip() instead.
    """

    def __init__(self, fileNames, tree="ana/hgc"):
        """Constructor.

        Arguments:
        fileName -- String for path to the ROOT file
        tree     -- Name of the TTree object inside the ROOT file (default: 'ana/hgc')
        """
        super(HGCalNtuple, self).__init__()
        self._tree = ROOT.TChain(tree)

        for file_name in fileNames:
            protocol = ''
            if '/eos/user/' in file_name:
                protocol = 'root://eosuser.cern.ch/'
            elif '/eos/cms/' in file_name:
                protocol = 'root://eoscms.cern.ch/'

            self._tree.Add(protocol+file_name)

        # print 'Cache size: {}'.format(self._tree.GetCacheSize())

        self._entries = self._tree.GetEntries()

    def setCache(self, learn_events=-1, entry_range=None):
        print 'Resetting cache: {}'.format(self._tree.SetCacheSize(0))
        cachesize = 400000000
        print 'Setting new cache size: {}'.format(self._tree.SetCacheSize(cachesize))
        if learn_events != -1:
            print 'Setting # of entries for cache learning: {} to {}'.format(
                self._tree.SetCacheLearnEntries(learn_events), learn_events)
        else:
            print self._tree.AddBranchToCache("*", True)
            # print self._tree.AddBranchToCache("cl_layer")
            self._tree.StopCacheLearningPhase()

        if entry_range:
            print 'Setting cache entry range: {}'.format(
                self._tree.SetCacheEntryRange(entry_range[0], entry_range[-1]))
        print 'Cache size: {}'.format(self._tree.GetCacheSize())

    def PrintCacheStats(self):
        self._tree.PrintCacheStats('cachedbranches')

    def tree(self):
        return self._tree

    def nevents(self):
        return self._entries

    # def hasRawRecHits(self):
    #     """Returns true if the ntuple has raw RecHit information."""
    #     return hasattr(self._tree, "rechit_raw_pt")

    def __iter__(self):
        """Returns generator for iterating over TTree entries (events)

        Generator returns Event objects.

        """
        for jentry in range(self._entries):
            # get the next tree in the chain and verify
            ientry = self._tree.LoadTree(jentry)
            if ientry < 0:
                break
            # copy next entry into memory and verify
            nb = self._tree.GetEntry(jentry)
            if nb <= 0:
                continue

            yield Event(self._tree, jentry)

    def getEvent(self, index):
        """Returns Event for a given index"""
        ientry = self._tree.LoadTree(index)
        if ientry < 0:
            return None
        # nb = self._tree.GetEntry(index)  # ientry or jentry?
        # if nb <= 0:
        #     None

        return Event(self._tree, index)  # ientry of jentry?


##########
class Event(object):
    """Class abstracting a single event.

    Main benefit is to provide nice interface to get various objects
    or collections of objects.
    """

    def __init__(self, tree, entry):
        """Constructor.

        Arguments:
        tree  -- TTree object
        entry -- Entry number in the tree
        """
        super(Event, self).__init__()
        self._tree = tree
        self._entry = entry

    def entry(self):
        return self._entry

    def event(self):
        """Returns event number."""
        return self._tree.event

    def lumi(self):
        """Returns lumisection number."""
        return self._tree.lumi

    def run(self):
        """Returns run number."""
        return self._tree.run

    def eventId(self):
        """Returns (run, lumi, event) tuple."""
        return (self._tree.run, self._tree.lumi, self._tree.event)

    def eventIdStr(self):
        """Returns 'run:lumi:event' string."""
        return "%d:%d:%d" % self.eventId()

    def getDataFrame(self, prefix):
        branch_blacklist = ['tc_wafer', 'tc_cell']

        df = pd.DataFrame()
        branches = [br.GetName() for br in self._tree.GetListOfBranches() if (
            br.GetName().startswith(prefix+'_') and not br.GetName() == '{}_n'.format(prefix))]
        if len(branches) == 0:
            return df
            
        names = ['_'.join(br.split('_')[1:]) for br in branches]
        nd_array = rnp.tree2array(self._tree, branches=branches,
                                  start=self._entry, stop=self._entry+1, cache_size=400000000)
        for idx, branch in enumerate(branches):
            if branch in branch_blacklist:
                continue
            # print names[idx]
            # print nd_array[branch][0]

            df[names[idx]] = nd_array[branch][0]
        return df

    def getPUInfo(self):
        branches = ['gen_PUNumInt', 'gen_TrueNumInt']
        names = ['PU', 'PUTrue']
        nd_array = rnp.tree2array(self._tree, branches=branches,
                                  start=self._entry, stop=self._entry+1)
        df = pd.DataFrame(columns=names)
        for idx, name in enumerate(names):
            df[name] = [nd_array[0][idx]]
        return df
