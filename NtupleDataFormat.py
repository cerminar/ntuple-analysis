# import math
# import collections

import ROOT


class _Collection(object):
    """Adaptor class representing a collection of objects.

    Concrete collection classes should inherit from this class.

    """

    def __init__(self, tree, sizeBranch, objclass, prefix):
        """Constructor.

        Arguments:
        tree        -- TTree object
        sizeBranch  -- Name of the branch to be used in size()
        objclass    -- Class to be used for the objects in __getitem__()
        """
        super(_Collection, self).__init__()
        self._tree = tree
        self._sizeBranch = sizeBranch
        self._objclass = objclass
        self._prefix = prefix

    def size(self):
        """Number of objects in the collection."""
        return int(getattr(self._tree, self._sizeBranch).size())

    def __len__(self):
        """Number of objects in the collection."""
        return self.size()

    def __getitem__(self, index):
        """Get object 'index' in the collection."""
        return self._objclass(self._tree, index, self._prefix)

    def __iter__(self):
        """Returns generator for the objects."""
        for index in range(self.size()):
            yield self._objclass(self._tree, index, self._prefix)


class _Object(object):
    """Adaptor class representing a single object in a collection.

    The member variables of the object are obtained from the branches
    with common prefix and a given index.

    Concrete object classes should inherit from this class.
    """

    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree   -- TTree object
        index  -- Index for this object
        prefix -- Prefix of the branchs
        """
        super(_Object, self).__init__()
        self._tree = tree
        self._index = int(index)
        self._prefix = prefix

    def __getattr__(self, attr):
        """Return object member variable.

        'attr' is translated as a branch in the TTree (<prefix>_<attr>).
        """
        self._checkIsValid()
        val = getattr(self._tree, self._prefix + "_" + attr)[self._index]
        return lambda: val

    def _checkIsValid(self):
        """Raise an exception if the object index is not valid."""
        if not self.isValid():
            raise Exception("%s is not valid" % self.__class__.__name__)

    def isValid(self):
        """Check if object index is valid."""
        return self._index != -1

    def index(self):
        """Return object index."""
        return self._index


##########
class HGCalNtuple(object):
    """Class abstracting the whole ntuple/TTree.

    Main benefit is to provide nice interface for
    - iterating over events
    - querying whether hit/seed information exists

    Note that to iteratate over the evets with zip(), you should use
    itertools.izip() instead.
    """

    def __init__(self, fileName, tree="ana/hgc"):
        """Constructor.

        Arguments:
        fileName -- String for path to the ROOT file
        tree     -- Name of the TTree object inside the ROOT file (default: 'ana/hgc')
        """
        super(HGCalNtuple, self).__init__()
        self._file = ROOT.TFile.Open(fileName)
        self._tree = self._file.Get(tree)
        self._entries = self._tree.GetEntriesFast()

    def file(self):
        return self._file

    def tree(self):
        return self._tree

    def nevents(self):
        return self._entries

    def hasRawRecHits(self):
        """Returns true if the ntuple has raw RecHit information."""
        return hasattr(self._tree, "rechit_raw_pt")

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
        nb = self._tree.GetEntry(ientry)  # ientry or jentry?
        if nb <= 0:
            None

        return Event(self._tree, ientry)  # ientry of jentry?


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

    def genParticles(self, prefix="genpart"):
        """Returns generator particles object."""
        return GenParticles(self._tree, prefix)

    def primaryVertex(self, prefix="vtx"):
        """Returns PrimaryVertex object."""
        return PrimaryVertex(self._tree, prefix)

    def recHits(self, prefix="rechit"):
        """Returns RecHits object."""
        return RecHits(self._tree, prefix)

    def layerClusters(self, prefix="cluster2d"):
        """Returns LayerClusters object."""
        return LayerClusters(self._tree, prefix)

    def multiClusters(self, prefix="multiclus"):
        """Returns MultiClusters object."""
        return MultiClusters(self._tree, prefix)

    def simClusters(self, prefix="simcluster"):
        """Returns SimClusters object."""
        return SimClusters(self._tree, prefix)

    def pfClusters(self, prefix="pfcluster"):
        """Returns PFClusters object."""
        return PFClusters(self._tree, prefix)

    def pfClustersFromMultiCl(self, prefix="pfclusterFromMultiCl"):
        """Returns PFClusters object."""
        return PFClusters(self._tree, prefix)

    def caloParticles(self, prefix="calopart"):
        """Returns CaloParticles object."""
        return CaloParticles(self._tree, prefix)

    def tracks(self, prefix="track"):
        """Returns Tracks object."""
        return Tracks(self._tree, prefix)

    def electrons(self, prefix="ecalDrivenGsfele"):
        """Returns Electrons object."""
        return Electrons(self._tree, prefix)

##########
class PrimaryVertex(object):
    """Class representing the primary vertex."""

    def __init__(self, tree):
        """Constructor.

        Arguments:
        tree -- TTree object
        """
        super(PrimaryVertex, self).__init__()
        self._tree = tree
        self._prefix = "vtx"

    def __getattr__(self, attr):
        """Return object member variable.

        'attr' is translated as a branch in the TTree (bsp_<attr>).
        """
        val = getattr(self._tree, self._prefix + "_" + attr)
        return lambda: val


##########
class RecHit(_Object):
    """Class representing a RecHit."""

    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree    -- TTree object
        index   -- Index of the RecHit
        prefix -- TBranch prefix
        """
        super(RecHit, self).__init__(tree, index, prefix)

    # def __getattr__(self, attr):
    #     """Custom __getattr__ because of the second index needed to access the branch."""
    #     val = super(SimHitMatchInfo, self).__getattr__(attr)()[self._shindex]
    #     return lambda: val


class RecHits(_Collection):
    """Class presenting a collection of RecHits."""

    def __init__(self, tree, prefix):
        """Constructor.

        Arguments:
        tree -- TTree object
        prefix -- TBranch prefix
        """
        super(RecHits, self).__init__(tree, prefix + "_pt", RecHit, prefix)

##########
class GenParticle(_Object):
    """Class representing a GenParticle."""

    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree  -- TTree object
        index -- Index of the GenParticle
        prefix -- TBranch prefix
        """
        super(GenParticle, self).__init__(tree, index, prefix)

    def _nExtrapolatedLayers(self):
        """Internal function to get the number of layers through which the particle was extrapolated."""
        return self._tree.genpart_posx[self._index].size()

    def nExtrapolatedLayers(self):
        """Returns the number of layers through which the particle was extrapolated."""
        self._checkIsValid()
        return self._nExtrapolatedLayers()

    # def matchedTrackInfos(self):
    #     """Returns a generator for matched tracks.
    #
    #     The generator returns TrackMatchInfo objects.
    #     """
    #     self._checkIsValid()
    #     for imatch in range(self._nMatchedTracks()):
    #         yield TrackMatchInfo(self._tree, self._index, imatch, self._prefix)

    # def simHits(self):
    #     """Returns generator for SimHits."""
    #     self._checkIsValid()
    #     for ihit in self.simHitIdx():
    #         yield SimHit(self._tree, ihit)


class GenParticles(_Collection):
    """Class presenting a collection of GenParticles."""

    def __init__(self, tree, prefix):
        """Constructor.

        Arguments:
        tree -- TTree object
        prefix -- TBranch prefix
        """
        # self.prefix = prefix
        super(GenParticles, self).__init__(tree, prefix + "_pt", GenParticle, prefix)


##########
class LayerCluster(_Object):
    """Class representing a LayerCluster."""

    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree    -- TTree object
        index   -- Index of the LayerCluster
        prefix -- TBranch prefix
        """
        super(LayerCluster, self).__init__(tree, index, prefix)


class LayerClusters(_Collection):
    """Class presenting a collection of LayerClusters."""

    def __init__(self, tree, prefix):
        """Constructor.

        Arguments:
        tree -- TTree object
        prefix -- TBranch prefix
        """
        super(LayerClusters, self).__init__(tree, prefix + "_pt", LayerCluster, prefix)


##########
class MultiCluster(_Object):
    """Class representing a MultiCluster."""

    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree    -- TTree object
        index   -- Index of the MultiCluster
        prefix -- TBranch prefix
        """
        super(MultiCluster, self).__init__(tree, index, prefix)


class MultiClusters(_Collection):
    """Class presenting a collection of MultiClusters."""

    def __init__(self, tree, prefix):
        """Constructor.

        Arguments:
        tree -- TTree object
        prefix -- TBranch prefix
        """
        super(MultiClusters, self).__init__(tree, prefix + "_pt", MultiCluster, prefix)


##########
class Track(_Object):
    """Class representing a Track."""

    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree    -- TTree object
        index   -- Index of the Track
        prefix -- TBranch prefix
        """
        super(Track, self).__init__(tree, index, prefix)


class Tracks(_Collection):
    """Class presenting a collection of Tracks."""

    def __init__(self, tree, prefix):
        """Constructor.

        Arguments:
        tree -- TTree object
        prefix -- TBranch prefix
        """
        super(Tracks, self).__init__(tree, prefix + "_pt", Track, prefix)


##########
class PFCluster(_Object):
    """Class representing a PFCluster."""

    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree    -- TTree object
        index   -- Index of the PFCluster
        prefix -- TBranch prefix
        """
        super(PFCluster, self).__init__(tree, index, prefix)


class PFClusters(_Collection):
    """Class presenting a collection of PFClusters."""

    def __init__(self, tree, prefix):
        """Constructor.

        Arguments:
        tree -- TTree object
        prefix -- TBranch prefix
        """
        super(PFClusters, self).__init__(tree, prefix + "_pt", PFCluster, prefix)


##########
class PFClusterFromMultiCl(_Object):
    """Class representing a PFClusterFromMultiCl. """

    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree    -- TTree object
        index   -- Index of the PFCluster
        prefix -- TBranch prefix
        """
        super(PFClusterFromMultiCl, self).__init__(tree, index, prefix)

    def hits(self):
        """Loop over all RecHits associated to the PFCluster and yield them"""
        for rechitIdx in self.rechits():
            yield RecHit(self._tree, rechitIdx, prefix="rechit")

    def __repr__(self):
        return "PFClusterFromMultiCl position: ({x}, {y}, {z}) eta: {eta}, phi: {phi}, energy: {energy}".format(
                        x=self.pos().x(), y=self.pos().y(), z=self.pos().z(),
                        eta=self.eta(), phi=self.phi(),
                        energy=self.energy())

class PFClustersFromMultiCl(_Collection):
    """Class presenting a colletion of  PFClusterFromMultiCl. """

    def __init__(self, tree, prefix):
        """Constructor.

        Arguments:
        tree -- TTree object
        prefix -- TBranch prefix
        """
        super(PFClustersFromMultiCl, self).__init__(tree, prefix + "_pt", PFClusterFromMultiCl, prefix)


##########
class SimCluster(_Object):
    """Class representing a SimCluster."""

    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree    -- TTree object
        index   -- Index of the SimCluster
        prefix -- TBranch prefix
        """
        super(SimCluster, self).__init__(tree, index, prefix)


class SimClusters(_Collection):
    """Class presenting a collection of SimClusters."""

    def __init__(self, tree, prefix):
        """Constructor.

        Arguments:
        tree -- TTree object
        prefix -- TBranch prefix
        """
        super(SimClusters, self).__init__(tree, prefix + "_pt", SimCluster, prefix)


##########
class CaloParticle(_Object):
    """Class representing a CaloParticle."""

    def __init__(self, tree, index, prefix):
        """Constructor.

        Arguments:
        tree    -- TTree object
        index   -- Index of the CaloParticle
        prefix -- TBranch prefix
        """
        super(CaloParticle, self).__init__(tree, index, prefix)


class CaloParticles(_Collection):
    """Class presenting a collection of CaloParticles."""

    def __init__(self, tree, prefix):
        """Constructor.

        Arguments:
        tree -- TTree object
        prefix -- TBranch prefix
        """
        super(CaloParticles, self).__init__(tree, prefix + "_pt", CaloParticle, prefix)

##########
class Electron(_Object):
        """Class representing an Electron. """

        def __init__(self, tree, index, prefix):
                """Constructor.

                Arguments:
                tree    -- TTree object
                index   -- Index of the Electron
                prefix  -- TBranch prefix
                """
                super(Electron, self).__init__(tree, index, prefix)

        def clustersFromMultiCl(self):
            """Loop over all PFClusters associated to the SC and yield them"""
            for pfclusterIdx in self.pfClusterIndex():
                yield PFClusterFromMultiCl(self._tree, pfclusterIdx, prefix="pfclusterFromMultiCl")

class Electrons(_Collection):
        """Class representing a collection of Electrons. """

        def __init__(self, tree, prefix):
                """Constructor.

                Arguments:
                tree   -- TTree object
                prefix -- TBranch prefix
                """
                super(Electrons, self).__init__(tree, prefix + "_pt", Electron, prefix)

