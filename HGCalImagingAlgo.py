##############################################################################
# Implementation of (stand-alone) functionalities of HGCalImagingAlgo,
# HGCal3DClustering, and HGCalDepthPreClusterer based on
# their CMSSW implementations mainly in RecoLocalCalo/HGCalRecAlgos
##############################################################################
from __future__ import print_function
# needed for ROOT funcs/types
import ROOT
import math
# needed for KDTree indexing & searches
import numpy as np
from scipy import spatial
# needed to extend the maximum recursion limit, for large data sets
import sys


sys.setrecursionlimit(100000)
# noise thresholds and MIPs
from RecHitCalibration import RecHitCalibration

# definition of Hexel element
class Hexel:
    def __init__(self, rHit = None, sigmaNoise = None):
        self.eta = 0
        self.phi = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.time = -1
        self.isHalfCell = False
        self.weight = 0
        self.fraction = 1
        self.detid = None
        self.rho = 0
        self.delta = 0
        self.nearestHigher = -1
        self.isBorder = False
        self.isHalo = False
        self.clusterIndex = -1
        self.clusterRECOIndex = -1
        self.sigmaNoise = 0.
        self.thickness = 0.
        if rHit is not None:
            self.eta = rHit.eta()
            self.phi = rHit.phi()
            self.x = rHit.x()
            self.y = rHit.y()
            self.z = rHit.z()
            self.weight = rHit.energy()
            self.detid = rHit.detid()
            self.layer = rHit.layer()
            self.isHalfCell = rHit.isHalf()
            self.thickness = rHit.thickness()
            self.time = rHit.time()
            self.clusterRECOIndex = rHit.cluster2d()
        if sigmaNoise is not None:
            self.sigmaNoise = sigmaNoise
    def __gt__(self, other_rho):
        return self.rho > other_rho

# definition of basic cluster (based on a set of sub-clusters or set of hexels)
class BasicCluster:
    def __init__(self, energy = None, position = None, thisCluster = None, algoId = None, caloId = None):
        self.eta = 0
        self.phi = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.energy = 0
        self.thisCluster = None
        self.algoId = None
        self.caloId = None
        if energy is not None:
            self.energy = energy
        if position is not None:
            self.eta = position.eta()
            self.phi = position.phi()
            self.x = position.x()
            self.y = position.y()
            self.z = position.z()
        if algoId is not None:
            self.algoId = algoId
        if caloId is not None:
            self.caloId = caloId
        if thisCluster is not None:
            self.thisCluster = thisCluster
    _usedIn3DClust = 0 # internal


# definition of the HGCalImagingAlgo class's methods & variables
class HGCalImagingAlgo:
    
    # depth of the KDTree before brute force is applied
    leafsize=100000
    # detector layers to consider
    lastLayerEE = 28 # last layer of EE
    lastLayerFH = 40 # last layer of FH
    maxlayer = 52 # last layer of BH
    
    def __init__(self, ecut = None, deltac = None, multiclusterRadii = None, minClusters = None, dependSensor = None, verbosityLevel = None):
        # sensor dependance or not
        self.dependSensor = False
        if dependSensor is not None: self.dependSensor = dependSensor
        
        # (multi)clustering parameters
        if not dependSensor: # (no sensor dependence, eta/phi coordinates for multi-clustering)
            # 2D clustering
            self.deltac = [2.,2.,2.]
            self.kappa = 10.
            self.ecut = 0.060 # in absolute units
            # multi-clustering
            self.realSpaceCone = False
            self.multiclusterRadii = [0.015,0.015,0.015] # it's in eta/phi coordinates, per detector
            self.minClusters = 3
        else: # (with sensor dependence, cartesian coordinates for multi-clustering)
            # 2D clustering
            self.deltac = [2.,2.,2.]
            self.kappa = 9.
            self.ecut = 3 # relative to the noise
            # multi-clustering
            self.realSpaceCone = True
            self.multiclusterRadii = [2.,2.,2.] # it's in cartesian coordiantes, per detector
            self.minClusters = 3

        # adjust params according to inputs, if necessary
        if ecut is not None: self.ecut = ecut
        if deltac is not None: self.deltac = deltac
        if minClusters is not None: self.minClusters = minClusters
        if multiclusterRadii is not None: self.multiclusterRadii = multiclusterRadii
        
        # others
        self.verbosityLevel = 0 # 0 - only basic info (default); 1 - additional info; 2 - detailed info printed
        if verbosityLevel is not None: self.verbosityLevel = verbosityLevel
        
        # print out the setup
        if (self.verbosityLevel>=1):
            print( "HGCalImagingAlgo setup: ")
            print( "   dependSensor: ", self.dependSensor)
            print( "   deltac: ", self.deltac)
            print( "   kappa: ", self.kappa)
            print( "   ecut: ", self.ecut)
            print( "   realSpaceCone: ", self.realSpaceCone)
            print( "   multiclusterRadii: ", self.multiclusterRadii)
            print( "   minClusters: ", self.minClusters)
            print( "   verbosityLevel: ", self.verbosityLevel)

    # calculate max local density in a 2D plane of hexels
    def calculateLocalDensity(self, nd, lp, layer):
        maxdensity = 0
        if(layer<=self.lastLayerEE): delta_c = self.deltac[0]
        elif(layer<=self.lastLayerFH): delta_c = self.deltac[1]
        else: delta_c = self.deltac[2]
        for iNode in nd:
            # search in a circle of radius delta_c or delta_c*sqrt(2) (not identical to search in the box delta_c)
            found = lp.query_ball_point([iNode.x,iNode.y],delta_c)
            for j in found:
                if(distanceReal2(iNode,nd[j]) < delta_c*delta_c):
                    iNode.rho += nd[j].weight
                    if(iNode.rho > maxdensity):
                        maxdensity = iNode.rho
        return maxdensity

    # calculate distance to the nearest hit with higher density (still does not use KDTree)
    def calculateDistanceToHigher(self, nd):
        #sort vector of Hexels by decreasing local density
        rs = sorted(range(len(nd)), key=lambda k: nd[k].rho, reverse=True)

        # intial values, and check if there are any hits
        maxdensity = 0.0
        nearestHigher = -1
        if(len(nd)>0):
            maxdensity = nd[rs[0]].rho
        else:
            return maxdensity # there are no hits

        #   start by setting delta for the highest density hit to the most distant hit - this is a convention
        dist2 = 0.
        for jNode in nd:
            tmp = distanceReal2(nd[rs[0]], jNode)
            if(tmp > dist2):
                dist2 = tmp
        nd[rs[0]].delta = pow(dist2,0.5)
        nd[rs[0]].nearestHigher = nearestHigher
                
        # now we save the largest distance as a starting point
        max_dist2 = dist2
        # calculate all remaining distances to the nearest higher density
        for oi in range(1,len(nd)): # start from second-highest density
            dist2 = max_dist2
            # we only need to check up to oi since hits are ordered by decreasing density
            # and all points coming BEFORE oi are guaranteed to have higher rho and the ones AFTER to have lower rho
            for oj in range(0,oi):
                tmp = distanceReal2(nd[rs[oi]], nd[rs[oj]])
                if(tmp <= dist2): #this "<=" instead of "<" addresses the (rare) case when there are only two hits
                    dist2 = tmp
                    nearestHigher = rs[oj]
            nd[rs[oi]].delta = pow(dist2,0.5)
            nd[rs[oi]].nearestHigher = nearestHigher #this uses the original unsorted hitlist

        return maxdensity

    # find cluster centers that satisfy delta & maxdensity/kappa criteria, and assign coresponding hexels
    def findAndAssignClusters(self, nd, points_0, points_1, lp, maxdensity, layer, verbosityLevel = None):  

        # adjust verbosityLevel if necessary
        if verbosityLevel is None: verbosityLevel = self.verbosityLevel
        clusterIndex = 0
        #sort Hexels by decreasing local density and by decreasing distance to higher
        rs = sorted(range(len(nd)), key=lambda k: nd[k].rho, reverse=True) # indices sorted by decreasing rho
        ds = sorted(range(len(nd)), key=lambda k: nd[k].delta, reverse=True) # sort in decreasing distance to higher

        if(layer<=self.lastLayerEE): delta_c = self.deltac[0]
        elif(layer<=self.lastLayerFH): delta_c = self.deltac[1]
        else: delta_c = self.deltac[2]

        for i in range(0,len(nd)):
            if(nd[ds[i]].delta < delta_c): break # no more cluster centers to be looked at
            # skip this as a potential cluster center because it fails the density cut
            if(self.dependSensor):
                if(nd[ds[i]].rho < self.kappa*nd[ds[i]].sigmaNoise): continue # set equal to kappa times noise threshold
            else:
                if(nd[ds[i]].rho < maxdensity/self.kappa): continue
            # store cluster index
            nd[ds[i]].clusterIndex = clusterIndex
            if (verbosityLevel>=2):
                print( "Adding new cluster with index ", clusterIndex)
                print( "Cluster center is hit ", ds[i], " with density rho: ", nd[ds[i]].rho, "and delta: ", nd[ds[i]].delta, "\n")
            clusterIndex += 1
            
        # at this point clusterIndex is equal to the number of cluster centers - if it is zero we are done
        if(clusterIndex==0):
            return []
        current_clusters = [[] for i in range(0,clusterIndex)]

        # assign to clusters, using the nearestHigher set from previous step (always set except for top density hit that is skipped)...
        for oi in range(1,len(nd)):
            ci = nd[rs[oi]].clusterIndex
            if(ci == -1):
                nd[rs[oi]].clusterIndex =  nd[nd[rs[oi]].nearestHigher].clusterIndex

        # assign points closer than dc to other clusters to border region and find critical border density
        rho_b = [0. for i in range(0,clusterIndex)]
        lp = spatial.KDTree(list(zip(points_0, points_1)), leafsize=self.leafsize) # new KDTree
        # now loop on all hits again :( and check: if there are hits from another cluster within d_c -> flag as border hit
        for iNode in nd:
            ci = iNode.clusterIndex
            flag_isolated = True
            if(ci != -1):
                # search in a circle of radius delta_c or delta_c*sqrt(2) (not identical to search in the box delta_c)
                found = lp.query_ball_point([iNode.x,iNode.y],delta_c)
                # found = lp.query_ball_point([iNode.x,iNode.y],delta_c*pow(2,0.5))
                for j in found:
                    # check if the hit is not within d_c of another cluster
                    if(nd[j].clusterIndex!=-1):
                        dist2 = distanceReal2(nd[j],iNode)
                        if(dist2 < delta_c*delta_c and nd[j].clusterIndex!=ci):
                            # in which case we assign it to the border
                            iNode.isBorder = True
                            break
                        # because we are using two different containers, we have to make sure that we don't unflag the
                        # hit when it finds *itself* closer than delta_c
                        if(dist2 < delta_c*delta_c and dist2 != 0. and nd[j].clusterIndex==ci):
                        # this is not an isolated hit
                            flag_isolated = False
                if(flag_isolated):
                    iNode.isBorder = True # the hit is more than delta_c from any of its brethren
            # check if this border hit has density larger than the current rho_b and update
            if(iNode.isBorder and rho_b[ci] < iNode.rho):
                rho_b[ci] = iNode.rho

        # flag points in cluster with density < rho_b as halo points, then fill the cluster vector
        for iNode in nd:
            ci = iNode.clusterIndex
            if(ci!=-1 and iNode.rho <= rho_b[ci]):
                pass
                iNode.isHalo = True # some issues to be debugged?
            if(ci!=-1):
                current_clusters[ci].append(iNode)
                if (verbosityLevel>=2):
                    print( "Pushing hit ", iNode, " into cluster with index ", ci)
                    print( "   rho_b[ci]: ", rho_b[ci], ", iNode.rho: ", iNode.rho, " iNode.isHalo: ", iNode.isHalo)

        return current_clusters

    # make list of Hexels out of rechits
    def populate(self, rHitsCollection, ecut = None):
        # adjust ecut if necessary
        if ecut is None: ecut = self.ecut
        # init 2D hexels
        points = [[] for i in range(0,2*(self.maxlayer+1))] # initialise list of per-layer-lists of hexels
        
        # loop over all hits and create the Hexel structure, skip energies below ecut
        for rHit in rHitsCollection:
            if (rHit.layer() > self.maxlayer): continue # current protection
            # energy treshold dependent on sensor
            sigmaNoise, aboveTreshold = recHitAboveTreshold(rHit, ecut = ecut, dependSensor = self.dependSensor)
            if not aboveTreshold : continue
            # organise layers accoring to the sgn(z)
            layerID = rHit.layer() + (rHit.z()>0)*(self.maxlayer+1) # +1 - yes or no?
            points[layerID].append(Hexel(rHit, sigmaNoise))

        return points

    # make 2D clusters out of rechists (need to introduce class with input params: delta_c, kappa, ecut, ...)
    def makeClusters(self, rHitsCollection, ecut = None):
        # adjust ecut if necessary
        if ecut is None: ecut = self.ecut
        # init 2D cluster lists
        clusters = [[] for i in range(0,2*(self.maxlayer+1))] # initialise list of per-layer-clusters

        # get the list of Hexels out of raw rechits
        points = self.populate(rHitsCollection, ecut = ecut)

        # loop over all layers, and for each layer create a list of clusters. layers are organised according to the sgn(z)
        for layerID in range(0, 2*(self.maxlayer+1)):
            if (len(points[layerID]) == 0): continue # protection
            layer = layerID - (points[layerID][0].z>0)*(self.maxlayer+1) # map back to actual layer
            points_0 = [hex.x for hex in points[layerID]] # list of hexels'coordinate 0 for current layer
            points_1 = [hex.y for hex in points[layerID]] # list of hexels'coordinate 1 for current layer
            hit_kdtree = spatial.KDTree(list(zip(points_0, points_1)), leafsize=self.leafsize) # create KDTree
            maxdensity = self.calculateLocalDensity(points[layerID], hit_kdtree, layer) # get the max density
            #print "layer: ", layer, ", max density: ", maxdensity, ", total hits: ", len(points[layer])
            self.calculateDistanceToHigher(points[layerID]) # get distances to the nearest higher density
            clusters[layerID] = self.findAndAssignClusters(points[layerID], points_0, points_1, hit_kdtree, maxdensity, layer) # get clusters per layer

        # return the clusters list
        return clusters

    # get basic clusters from the list of 2D clusters
    def getClusters(self, clusters, verbosityLevel = None):
        # adjust verbosityLevel if necessary
        if verbosityLevel is None: verbosityLevel = self.verbosityLevel
        # init the list
        clusters_v = []
        # loop over all layers and all clusters in each layer
        layer = 0
        for clist_per_layer in clusters:
            index = 0
            for cluster in clist_per_layer:
                position = calculatePosition(cluster)
                if (position == ROOT.Math.XYZPoint()): continue # skip the clusters where position could not be computed (either all weights are 0, or all hexels are tagged as Halo)
                energy = 0
                for iNode in cluster:
                    if (not iNode.isHalo):
                        energy += iNode.weight
                if (verbosityLevel>=1):
                    layerActual = layer - (cluster[0].z>0)*(self.maxlayer+1)
                    print( "LayerID: ", layer, "Actual layer: ", layerActual, "| 2D-cluster index: ", index, ", No. of cells = ", len(cluster), ", Energy  = ", energy, ", Phi = ", position.phi(), ", Eta = ", position.eta(), ", z = ", position.z())
                    for iNode in cluster:
                        if (not iNode.isHalo):
                            pass
                clusters_v.append(BasicCluster(energy = energy, position = position, thisCluster = cluster))
                index += 1
            layer += 1
            clusters_v.sort(key=getEnergy,reverse=True)
        return clusters_v

    # make multi-clusters starting from the 2D clusters, without KDTree
    def makePreClusters(self, clusters, multiclusterRadii = None, minClusters = None, verbosityLevel = None):
        # adjust multiclusterRadii, minClusters and/or verbosityLevel if necessary
        if multiclusterRadii is None: multiclusterRadii = self.multiclusterRadii
        if minClusters is None: minClusters = self.minClusters
        if verbosityLevel is None: verbosityLevel = self.verbosityLevel
        # get clusters in one list (just following original approach)
        thecls = self.getClusters(clusters)

        # init lists and vars
        thePreClusters = []
        vused = [0.]*len(thecls)
        used = 0
        # indices sorted by decreasing energy
        es = sorted(range(len(thecls)), key=lambda k: thecls[k].energy, reverse=True)
        # loop over all clusters
        index = 0
        for i in range(0,len(thecls)):
            if(vused[i]==0):
                temp = [thecls[es[i]]]
                if (thecls[es[i]].z>0): vused[i] = 1
                else: vused[i] = -1
                used += 1
                for j in range(i+1,len(thecls)):
                    if(vused[j]==0):
                        distanceCheck = 9999.
                        if(self.realSpaceCone):
                            distanceCheck = distanceReal2(thecls[es[i]],thecls[es[j]])
                        else:
                            distanceCheck = distanceDR2(thecls[es[i]],thecls[es[j]])
                        layer = thecls[es[j]].thisCluster[0].layer
                        multiclusterRadius = 9999.
                        multiclusterRadius = multiclusterRadii[0]
                        if(layer>self.lastLayerEE and layer<=self.lastLayerFH): multiclusterRadius = multiclusterRadii[1]
                        else: multiclusterRadius = multiclusterRadii[2]
                        if( distanceCheck < multiclusterRadius*multiclusterRadius and int(thecls[es[i]].z*vused[i])>0 ):
                            temp.append(thecls[es[j]])
                            vused[j] = vused[i]
                            used += 1
                if(len(temp) > minClusters):
                    position = getMultiClusterPosition(temp)
                    energy = getMultiClusterEnergy(temp)
                    thePreClusters.append(BasicCluster(energy = energy, position = position, thisCluster = temp))
                    if (verbosityLevel>=1): print( "Multi-cluster index: ", index, ", No. of 2D-clusters = ", len(temp), ", Energy  = ", energy, ", Phi = ", position.phi(), ", Eta = ", position.eta(), ", z = ", position.z())
                    index += 1
        return thePreClusters

    # make multi-clusters starting from the 2D clusters, with KDTree
    def make3DClusters(self, clusters, multiclusterRadii = None, minClusters = None, verbosityLevel = None):
        # adjust multiclusterRadii, minClusters and/or verbosityLevel if necessary
        if multiclusterRadii is None: multiclusterRadii = self.multiclusterRadii
        if minClusters is None: minClusters = self.minClusters
        if verbosityLevel is None: verbosityLevel = self.verbosityLevel
        # get clusters in one list (just following original approach)
        thecls = self.getClusters(clusters)
        
        # init "points" of 2D clusters for KDTree serach and zees of layers (check if it is really needed)
        points = [[] for i in range(0,2*(self.maxlayer+1))] # initialise list of per-layer-lists of clusters
        zees = [0. for layer in range(0,2*(self.maxlayer+1))]
        for cls in thecls: # organise layers accoring to the sgn(z)
            layerID = cls.thisCluster[0].layer
            layerID += (cls.z>0)*(self.maxlayer+1) # +1 - yes or no?
            points[layerID].append(cls)
            zees[layerID] = cls.z

        # init lists and vars
        thePreClusters = []
        vused = [0.]*len(thecls)
        used = 0
        
        # indices sorted by decreasing energy
        es = sorted(range(len(thecls)), key=lambda k: thecls[k].energy, reverse=True)
        # loop over all clusters
        index = 0
        for i in range(0,len(thecls)):
            #if(vused[i]==0):
            if (thecls[es[i]]._usedIn3DClust ==0):
                temp = [thecls[es[i]]]
                if (thecls[es[i]].z>0): thecls[es[i]]._usedIn3DClust = 1
                else: thecls[es[i]]._usedIn3DClust = -1
                used += 1
                from_ = [thecls[es[i]].x, thecls[es[i]].y, thecls[es[i]].z]
                firstlayer = (thecls[es[i]].z>0)*(self.maxlayer+1)
                lastlayer = firstlayer+self.maxlayer+1
                for j in range(firstlayer,lastlayer):
                    if(zees[j]==0.): continue
                    to_ = [0., 0., zees[j]]
                    to_[0]=(from_[0]/from_[2])*to_[2]
                    to_[1]=(from_[1]/from_[2])*to_[2]
                    layer = j-(zees[j]>0)*(self.maxlayer+1)  #maps back from index used for KD trees to actual layer
                    multiclusterRadius = 9999.
                    if(layer <= self.lastLayerEE): multiclusterRadius = multiclusterRadii[0]
                    elif(layer <= self.lastLayerFH): multiclusterRadius = multiclusterRadii[1]
                    elif(layer <= self.maxlayer): multiclusterRadius = multiclusterRadii[2]
                    else: print( "ERROR: Nonsense layer value - cannot assign multicluster radius")
                    # KD-tree search in layer j
                    points_0 = [cls.x for cls in points[j]] # list of cls' coordinate 0 for layer j
                    points_1 = [cls.y for cls in points[j]] # list of cls' coordinate 1 for layer j
                    hit_kdtree = spatial.KDTree(list(zip(points_0, points_1)), leafsize=self.leafsize) # create KDTree
                    found = hit_kdtree.query_ball_point([to_[0],to_[1]],multiclusterRadius)
                    for k in found:
                        h_to = Hexel(); h_to.x = to_[0]; h_to.y = to_[1] # dummy object
                        if((points[j][k]._usedIn3DClust==0) and (distanceReal2(points[j][k],h_to) < multiclusterRadius**2)):
                            temp.append(points[j][k])
                            points[j][k]._usedIn3DClust = thecls[es[i]]._usedIn3DClust
                            used += 1
                if(len(temp) > minClusters):
                    position = getMultiClusterPosition(temp)
                    energy = getMultiClusterEnergy(temp)
                    thePreClusters.append(BasicCluster(energy = energy, position = position, thisCluster = temp))
                    if (verbosityLevel>=1): print ("Multi-cluster index: ", index, ", No. of 2D-clusters = ", len(temp), ", Energy  = ", energy, ", Phi = ", position.phi(), ", Eta = ", position.eta(), ", z = ", position.z())
                    index += 1
        return thePreClusters


# distance squared (in eta/phi) between the two objects (hexels, clusters)
def distanceDR2(Hex1, Hex2):
    return (pow(Hex2.eta - Hex1.eta,2) + pow(Hex2.phi - Hex1.phi,2))

# distance squared (in x/y) between the two objects (hexels, clusters)
def distanceReal2(clust1, clust2):
    return (pow(clust2.x - clust1.x,2) + pow(clust2.y - clust1.y,2))

# position of the cluster, based on hexels positions weighted by the energy
def calculatePosition(cluster):
    total_weight = 0.
    x = 0.
    y = 0.
    z = 0.
    haloOnlyCluster = True

    # check if haloOnlyCluster
    for iNode in cluster:
        if (not iNode.isHalo):
            haloOnlyCluster = False

    if (not haloOnlyCluster):
        for iNode in cluster:
            if(not iNode.isHalo):
                total_weight += iNode.weight
                x += iNode.x*iNode.weight
                y += iNode.y*iNode.weight
                z += iNode.z*iNode.weight
        if (total_weight != 0.): 
            return ROOT.Math.XYZPoint( x/total_weight, y/total_weight, z/total_weight ) # return as ROOT.Math.XYZPoint
        else:
            return ROOT.Math.XYZPoint()
    if (haloOnlyCluster):
        maxenergy = - 1.0
        maxenergy_x,maxenergy_y,maxenergy_z = 0.,0.,0.
        for iNode in cluster:
            if (iNode.weight > maxenergy): 
                maxenergy   = iNode.weight
                maxenergy_x = iNode.x
                maxenergy_y = iNode.y
                maxenergy_z = iNode.z
        return ROOT.Math.XYZPoint(maxenergy_x,maxenergy_y,maxenergy_z)


# get position of the multi-cluster, based on the positions of its 2D clusters weighted by the energy
def getMultiClusterPosition(multi_clu):
    if(len(multi_clu) == 0): return ROOT.Math.XYZPoint()
    mcenergy = getMultiClusterEnergy(multi_clu)
    if (mcenergy == 0): return ROOT.Math.XYZPoint()

    # compute weighted mean x/y/z position
    acc_x = 0.0
    acc_y = 0.0
    acc_z = 0.0
    totweight = 0.0
    for layer_clu in multi_clu:
        if(layer_clu.energy<0.01*mcenergy): continue # cutoff < 1% layer energy contribution
        weight = layer_clu.energy # weight each corrdinate only by the total energy of the layer cluster
        acc_x += layer_clu.x * weight
        acc_y += layer_clu.y * weight
        acc_z += layer_clu.z * weight
        totweight += weight
    if (totweight != 0):
        acc_x /= totweight
        acc_y /= totweight
        acc_z /= totweight

    return ROOT.Math.XYZPoint(acc_x,acc_y,acc_z) # return x/y/z in absolute coordinates

# get energy of the multi-cluster, based on its 2D clusters
def getMultiClusterEnergy(multi_clu):
    acc = 0.
    for layer_clu in multi_clu:
        acc += layer_clu.energy
    return acc

# determine if the rechit energy is above the desired treshold
def recHitAboveTreshold(rHit, ecut, dependSensor = True):
    sigmaNoise = 1.
    if(dependSensor):
        thickIndex = -1
        if( rHit.layer() <= HGCalImagingAlgo.lastLayerFH ): # EE + FH
            thickness = rHit.thickness()
            if(thickness>99. and thickness<101.): thickIndex=0
            elif(thickness>199. and thickness<201.): thickIndex=1
            elif(thickness>299. and thickness<301.): thickIndex=2
            else:
                print( "ERROR - silicon thickness has a nonsensical value")
        # determine noise for each sensor/subdetector using RecHitCalibration library
        RecHitCalib = RecHitCalibration()
        sigmaNoise = 0.001 * RecHitCalib.sigmaNoiseMeV(rHit.layer(), thickIndex) # returns threshold for EE, FH, BH (in case of BH thickIndex does not play a role)
    aboveTreshold = rHit.energy() >= ecut*sigmaNoise  #this checks if rechit energy is above the threshold of ecut (times the sigma noise for the sensor, if that option is set)
    return sigmaNoise, aboveTreshold

def getEnergy(item):
    return item.energy
