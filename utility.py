# investigate shower development based on RecHits and SimClusters
import ROOT
import os
import numpy as np
from HGCalImagingAlgo import *
from HGCalNtupleUtility import *
from NtupleDataFormat import HGCalNtuple


### Basic setup for testing (sensor dependence, treshold w.r.t noise, cartesian coordinates for multi-clustering
dependSensor = True
# 2D clustering parameters
deltac = [2.,2.,2.] # in cartesian coordiantes in cm, per detector
ecut = 3 # relative to the noise
# multi-clustering parameters
multiclusterRadii = [2.,2.,2.] # in cartesian coordiantes in cm, per detector
minClusters = 3 # request at least minClusters+1 2D clusters
# verbosity, allowed events/layers for testing/histograming, etc.
allowedRangeLayers = [] # layer considered for histograming e.g. [10, 20], empty for none
allowedRangeEvents = list(range(0,3,1)) # event numbers considered for histograming, e.g. [0,1,2], empty for none
verbosityLevel = 0 # 0 - only basic info (default); 1 - additional info; 2 - detailed info printed, histograms produced




def getRecHitDetIds(rechits):
    recHitsList = []
    for rHit in rechits:
        recHitsList.append(rHit.detid())
    # print "RecHits -"*10
    # print recHitsList
    recHits = np.array(recHitsList)
    return recHits


def getHitList(simClus, recHitDetIds):
    sClusHitsList = []
    for DetId in simClus.hits():
        sClusHitsList.append(DetId)
    sClusHits = np.array(sClusHitsList)
    # thanks to http://stackoverflow.com/questions/11483863/python-intersection-indices-numpy-array
    recHitIndices = np.nonzero(np.in1d(recHitDetIds, sClusHits))
    return recHitIndices


# get list of rechist associated to sim-cluster hits
def getRecHitsSimAssoc(rechits_raw, simcluster):
    # get sim-cluster associations
    nSimClus = 0
    simClusHitAssoc = []
    recHitDetIds = getRecHitDetIds(rechits_raw)
    for simClusIndex, simClus in enumerate(simcluster):
        simClusHitAssoc.append(getHitList(simClus, recHitDetIds))
        nSimClus += 1
    # get list of rechist associated to simhits
    rHitsSimAssoc = [[] for k in range(0, nSimClus)]
    for simClusIndex, simCl in enumerate(simcluster):
        if (
            verbosityLevel >= 2): print("Sim-cluster index: ", simClusIndex, ", pT: ", simCl.pt(), ", E: ", simCl.energy(), ", phi: ", simCl.phi(), ", eta: ", simCl.eta())
        # loop over sim clusters and then rechits
        rHitsSimAssocTemp = []
        for hitIndexArray in simClusHitAssoc[simClusIndex]:
            for hitIndex in hitIndexArray:
                thisHit = rechits_raw[hitIndex]
                if (not recHitAboveTreshold(thisHit, ecut, dependSensor)[1]): continue
                # independent of sim cluster, after cleaning
                rHitsSimAssocTemp.append(thisHit)
        rHitsSimAssoc[simClusIndex] = rHitsSimAssocTemp
    return rHitsSimAssoc


# 3D histograming of rechist associated to sim-cluster hits ("event displays")
def histRecHitsSimAssoc(rHitsSimAssoc, currentEvent, histDict, tag="rHitsAssoc_", zoomed=False):
    # sanity check
    if (histDict == None): return

    # define event-level hists
    if (zoomed):  # zoomed for testing/convenience around the eta/phi of most energetic hit
        rs = sorted(range(len(rHitsSimAssoc[0])), key=lambda k: rHitsSimAssoc[0][k].energy(),
                    reverse=True)  # indices sorted by decreasing rho
        c_phi = rHitsSimAssoc[0][rs[0]].phi()
        c_eta = rHitsSimAssoc[0][rs[0]].eta()
        histDict[tag + "map_lay_phi_eta_evt{}".format(currentEvent)] = ROOT.TH3F(
            tag + "map_lay_phi_eta_evt{}".format(currentEvent),
            tag + "map_lay_phi_eta_evt{};z(cm);#phi;#eta".format(currentEvent), 200, 320, 520, 50, c_phi - 0.5,
            c_phi + 0.5, 50, c_eta - 0.5, c_eta + 0.5)  # 3D rechists associated to sim-cluster (with ecut cleaning)
    else:
        histDict[tag + "map_lay_phi_eta_evt{}".format(currentEvent)] = ROOT.TH3F(
            tag + "map_lay_phi_eta_evt{}".format(currentEvent),
            tag + "map_lay_phi_eta_evt{};z(cm);#phi;#eta".format(currentEvent), 200, 320, 520, 314, -3.14, +3.14, 320,
            -3.2, 3.2)  # 3D rechists associated to sim-cluster (with ecut cleaning)

    for simClusIndex in range(0, len(rHitsSimAssoc)):
        # define sim-cluster-level hists
        histDict[tag + "map_lay_phi_eta_evt{}_sim{}".format(currentEvent, simClusIndex)] = ROOT.TH3F(
            tag + "map_lay_phi_eta_evt{}_sim{}".format(currentEvent, simClusIndex),
            tag + "map_lay_phi_eta_evt{}_sim{};z(cm);#phi;#eta".format(currentEvent, simClusIndex), 200, 320, 520, 314,
            -3.14, +3.14, 320, -3.2, 3.2)
        # loop over assoc. rec hits
        for thisHit in rHitsSimAssoc[simClusIndex]:
            histDict[tag + "map_lay_phi_eta_evt{}".format(currentEvent)].Fill(abs(thisHit.z()), thisHit.phi(),
                                                                              thisHit.eta())  # for each sim cluster, after cleaning
            if (thisHit.energy < ecut): continue
            histDict[tag + "map_lay_phi_eta_evt{}_sim{}".format(currentEvent, simClusIndex)].Fill(abs(thisHit.z()),
                                                                                                  thisHit.phi(),
                                                                                                  thisHit.eta())  # independent of sim cluster, before cleaning

    return histDict


# 2D histograming of rechists in the chosen layerts, given by allowedRangeLayers
def histRecHits(rHits, currentEvent, histDict, tag="rHits_", zoomed=False):
    # sanity check
    if (histDict == None): return

    # define hists per layer
    for layer in range(1, 41):
        if (layer in allowedRangeLayers):  # testing limitation
            if (zoomed):  # zoomed for testing/convenience around the eta/phi of most energetic hit
                rs = sorted(range(len(rHits)), key=lambda k: rHits[k].energy(),
                            reverse=True)  # indices sorted by decreasing rho
                c_phi = rHits[rs[0]].phi()
                c_eta = rHits[rs[0]].eta()
                histDict[tag + "eng_eta_phi_evt{}_lay{}".format(currentEvent, layer)] = ROOT.TH2F(
                    tag + "eng_eta_phi_evt{}_lay{}".format(currentEvent, layer),
                    tag + "eng_eta_phi_evt{}_lay{};#eta;#phi".format(currentEvent, layer), 40, c_eta - 0.1, c_eta + 0.1,
                    40, c_phi - 0.1, c_phi + 0.1)  # 2D energy-weighted-map of raw rechists (with ecut cleaning)
            else:
                histDict[tag + "eng_eta_phi_evt{}_lay{}".format(currentEvent, layer)] = ROOT.TH2F(
                    tag + "eng_eta_phi_evt{}_lay{}".format(currentEvent, layer),
                    tag + "eng_eta_phi_evt{}_lay{};#eta;#phi".format(currentEvent, layer), 320, -3.2, 3.2, 314, -3.14,+3.14)  # 2D energy-weighted-map of raw rechists (with ecut cleaning)

    # loop over all raw rechits and fill per layer
    for rHit in rHits:
        if (rHit.layer() in allowedRangeLayers):  # testing limitation
            if (rHit.energy() < ecut): continue
            histDict[tag + "eng_eta_phi_evt{}_lay{}".format(currentEvent, rHit.layer())].Fill(rHit.eta(), rHit.phi(),
                                                                                              rHit.energy())

    return histDict


# 2D histograming of the clustered rechist with stand-alone algo, weighted by energy
def histHexelsClustered(hexelsClustered, currentEvent, histDict, tag="clustHex_", zoomed=False):
    # sanity check
    if (histDict == None): return

    # define event-level hists
    if (zoomed):  # zoomed for testing/convenience around the eta/phi of most energetic hit
        rs = sorted(range(len(hexelsClustered)), key=lambda k: hexelsClustered[k].weight,
                    reverse=True)  # indices sorted by decreasing rho
        c_phi = hexelsClustered[rs[0]].phi
        c_eta = hexelsClustered[rs[0]].eta
        histDict[tag + "eng_phi_eta_evt{}".format(currentEvent)] = ROOT.TH3F(
            tag + "eng_phi_eta_evt{}".format(currentEvent),
            tag + "eng_phi_eta_evt{};z(cm);#phi;#eta".format(currentEvent), 200, 320, 520, 80, c_phi - 0.8, c_phi - 0.8,
            80, c_eta - 0.8, c_eta - 0.8)  # 3D rechists clustered with algo (with ecut cleaning)
    else:
        histDict[tag + "eng_phi_eta_evt{}".format(currentEvent)] = ROOT.TH3F(
            tag + "eng_phi_eta_evt{}".format(currentEvent),
            tag + "eng_phi_eta_evt{};z(cm);#phi;#eta".format(currentEvent), 200, 320, 520, 314, -3.14, +3.14, 320, -3.2,
            3.2)  # 3D rechists clustered with algo (with ecut cleaning)

    # loop over all clustered rechist
    for iNode in hexelsClustered:
        histDict[tag + "eng_phi_eta_evt{}".format(currentEvent)].Fill(abs(iNode.z), iNode.phi, iNode.eta, iNode.weight)

    return histDict


# 1D histograming of given list of values
def histValue1D(fValues, histDict, tag="hist1D_", title="hist 1D", axunit="a.u.", binsRangeList=[10, -1, 1],
                ayunit="a.u."):
    # sanity check
    if (histDict == None): return

    # define event-level hists
    histDict[tag] = ROOT.TH1F(tag, title + ";" + axunit + ";" + ayunit, binsRangeList[0], binsRangeList[1],
                              binsRangeList[2])
    histDict[tag].GetYaxis().SetTitleOffset(histDict[tag].GetYaxis().GetTitleOffset() * 1.5)
    # loop over all values
    if (verbosityLevel >= 3): print( "tag: ", tag, ", fValues: ", fValues)
    for value in fValues:
        histDict[tag].Fill(value)
    return histDict


# print/save all histograms
def histPrintSaveAll(histDict, outDir):
    imgType = "pdf"
    canvas = ROOT.TCanvas(outDir, outDir, 500, 500)
    if (verbosityLevel >= 3): print( "histDict.items(): ", histDict.items())
    for key, item in histDict.items():
        # do not save empty histograms
        if (type(item) == ROOT.TH1F) or (type(item) == ROOT.TH2F) or (type(item) == ROOT.TH3F):
            if item.GetEntries() == 0:
                continue
        ROOT.gStyle.SetPalette(ROOT.kBird)
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPadTopMargin(0.05)
        ROOT.gStyle.SetPadBottomMargin(0.12)
        ROOT.gStyle.SetPadLeftMargin(0.15)
        ROOT.gStyle.SetPadRightMargin(0.02)
        if type(item) == ROOT.TH1F:
            item.Draw("hist0")
            canvas.SaveAs("{}/{}.{}".format(outDir, key, imgType))
        if type(item) == ROOT.TH2F:
            item.Draw("colz")
            canvas.SaveAs("{}/{}.{}".format(outDir, key, imgType))
        elif type(item) == ROOT.TH3F:
            item.Draw("box")
            canvas.SaveAs("{}/{}.{}".format(outDir, key, imgType))
        else:
            continue
    return





#######################################################################################################
#######################################################################################################
#######################################################################################################

labels = ['remulticlus_energy', 'remulticlus_x', 'remulticlus_y', 'remulticlus_z', 'remulticlus_nclus',
          'reclusters2D_energy', 'reclusters2D_x', 'reclusters2D_y', 'reclusters2D_z', 'reclusters2D_nhexel',
          'rechit_energy','rechit_x','rechit_y','rechit_z'
          ]

mydtype = [(label, 'object') for label in labels]

def AppendEventData(FlatNtuple, multiClustersList_rerun, clusters2DList_rerun, rHitsCleaned):
    remulticlus_energy, remulticlus_x, remulticlus_y, remulticlus_z, remulticlus_nclus = [], [], [], [], []
    for i in range(len(multiClustersList_rerun)):
        remulticlus_energy.append(multiClustersList_rerun[i].energy)
        remulticlus_x.append(multiClustersList_rerun[i].x)
        remulticlus_y.append(multiClustersList_rerun[i].y)
        remulticlus_z.append(multiClustersList_rerun[i].z)
        remulticlus_nclus.append(len(multiClustersList_rerun[i].thisCluster))
    remulticlus_energy = array(remulticlus_energy)
    remulticlus_x = array(remulticlus_x)
    remulticlus_y = array(remulticlus_y)
    remulticlus_z = array(remulticlus_z)
    remulticlus_nclus = array(remulticlus_nclus)

    reclusters2D_energy, reclusters2D_x, reclusters2D_y, reclusters2D_z, reclusters2D_nhexel = [], [], [], [], []
    for i in range(len(clusters2DList_rerun)):
        reclusters2D_energy.append(clusters2DList_rerun[i].energy)
        reclusters2D_x.append(clusters2DList_rerun[i].x)
        reclusters2D_y.append(clusters2DList_rerun[i].y)
        reclusters2D_z.append(clusters2DList_rerun[i].z)
        reclusters2D_nhexel.append(len(clusters2DList_rerun[i].thisCluster))
    reclusters2D_energy = array(reclusters2D_energy)
    reclusters2D_x = array(reclusters2D_x)
    reclusters2D_y = array(reclusters2D_y)
    reclusters2D_z = array(reclusters2D_z)
    reclusters2D_nhexel = array(reclusters2D_nhexel)
    
    rerechit_energy, rerechit_x, rerechit_y, rerechit_z = [], [], [], []
    for i in range(len(rHitsCleaned)):
        rerechit_energy.append(rHitsCleaned[i].energy())
        rerechit_x.append(rHitsCleaned[i].x())
        rerechit_y.append(rHitsCleaned[i].y())
        rerechit_z.append(rHitsCleaned[i].z())   
    rerechit_energy = array(rerechit_energy)
    rerechit_x = array(rerechit_x)
    rerechit_y = array(rerechit_y)
    rerechit_z = array(rerechit_z)

    row = [remulticlus_energy, remulticlus_x, remulticlus_y, remulticlus_z, remulticlus_nclus,
           reclusters2D_energy, reclusters2D_x, reclusters2D_y, reclusters2D_z, reclusters2D_nhexel,
           rerechit_energy,rerechit_x,rerechit_y,rerechit_z
           ]

    FlatNtuple.append(row)


def RerunReco(ntuple):
    FlatNtuple = []
    #i=0
    # start event loop
    for event in ntuple:
        
        if (not event.entry() in allowedRangeEvents): continue  # checking external condition
        if (verbosityLevel >= 1): print( "\nCurrent event: ", event.entry())

        # get collections of raw rechits, sim clusters, 2D clusters, multi clusters, etc.
        # ntuple = HGCalNtuple(DatasetFolder+DatasetFile+".root")
        # event = [i for i in ntuple][0]
        recHitsRaw = event.recHits()
        simClusters = event.simClusters()
        layerClusters = event.layerClusters()
        multiClusters = event.multiClusters()

        # get flat list of rechist associated to sim-cluster hits
        # rHitsSimAssoc = getRecHitsSimAssoc(recHitsRaw, simClusters)
        # get flat list of raw rechits which satisfy treshold condition
        rHitsCleaned = [rechit for rechit in recHitsRaw if recHitAboveTreshold(rechit, ecut, dependSensor)[1]]

        ### Imaging algo run as stand-alone (python)
        # instantiate the stand-alone clustering implemented in HGCalImagingAlgo
        HGCalAlgo = HGCalImagingAlgo(ecut=ecut, deltac=deltac, multiclusterRadii=multiclusterRadii,
                                     minClusters=minClusters, dependSensor=dependSensor, verbosityLevel=0)
        # produce 2D clusters with stand-alone algo, out of all raw rechits
        clusters2D_rerun = HGCalAlgo.makeClusters(recHitsRaw)  # nested list of "hexels", per layer, per 2D cluster
        # produce multi-clusters with stand-alone algo, out of all 2D clusters
        #multiClustersList_rerun = HGCalAlgo.makePreClusters(clusters2D_rerun) 
        multiClustersList_rerun = HGCalAlgo.make3DClusters(clusters2D_rerun)  # flat list of multi-clusters (as basic clusters)
        # get for testing: flat list of 2D clustered, and flat list of clustered non-halo "hexeles" (from stand-alone algo)
        clusters2DList_rerun = HGCalAlgo.getClusters(clusters2D_rerun,verbosityLevel=0)  # flat list of 2D clusters (as basic clusters)
        # hexelsClustered_rerun = [iNode for bClust in clusters2DList_rerun for iNode in bClust.thisCluster if not iNode.isHalo]  
        # flat list of clustered "hexeles", without the "halo" hexels
        #print(i,len(clusters2DList_rerun))
        #i = i+1

        AppendEventData(FlatNtuple,multiClustersList_rerun, clusters2DList_rerun,rHitsCleaned)

    FlatNtuple = pd.DataFrame.from_records(FlatNtuple, columns=labels)
    return FlatNtuple





def MakeMultArray(df, df_rerun, n):
    multreco = np.empty((0, 5))
    multrerun = np.empty((0, 5))
    for event in range(0, n, 1):
        gen, rech, mult, simclus, pfclus = LoadEvent(event, df)
        multreco = np.append(multreco,
                             np.c_[array(mult.e).T,
                                   array(mult.x).T,
                                   array(mult.y).T,
                                   array(mult.z).T,
                                   event * ones(mult.z.size)],
                             axis=0)
        multrerun = np.append(multrerun,
                              np.c_[df_rerun.remulticlus_energy[event].T,
                                    df_rerun.remulticlus_x[event].T,
                                    df_rerun.remulticlus_y[event].T,
                                    df_rerun.remulticlus_z[event].T,
                                    event * ones(df_rerun.remulticlus_z[event].size)],
                              axis=0)
    return multreco, multrerun


def CalMultEnergy(multreco, multrerun, n):
    Energy_reco_rerun = []
    Energy_reco = []
    Energy_rerun = []
    for event in range(0, n, 1):
        reco1 = sum(multreco[(multreco[:, 4] == event) & (multreco[:, 3] > 0)][:, 0])
        reco2 = sum(multreco[(multreco[:, 4] == event) & (multreco[:, 3] < 0)][:, 0])
        rerun1 = sum(multrerun[(multrerun[:, 4] == event) & (multrerun[:, 3] > 0)][:, 0])
        rerun2 = sum(multrerun[(multrerun[:, 4] == event) & (multrerun[:, 3] < 0)][:, 0])

        Energy_reco_rerun.append(reco1 - rerun1)
        Energy_reco_rerun.append(reco2 - rerun2)
        Energy_reco.append(reco1)
        Energy_reco.append(reco2)
        Energy_rerun.append(rerun1)
        Energy_rerun.append(rerun2)

    return array(Energy_reco_rerun), array(Energy_reco), array(Energy_rerun)


def plotEElayer():
    EElayerz = np.array([320.75,321.50,322.73,323.48,324.71,325.46,326.69,327.44,328.67,\
    329.42,330.73,331.60,332.91,333.78,335.09,335.96,337.27,338.14,339.45,340.32,\
    341.77, 342.84,344.29,345.36,346.81,347.88,349.33,350.40])
    for i in range(EElayerz.size):
        plt.axvline(x=EElayerz[i],color='k',linewidth=0.5,alpha=0.1)
        
def plotFHlayer(): 
    BHlayerz = np.array([356.33,361.01,365.69,370.37,375.05,379.73,384.41,389.09,393.77,398.45,403.13,407.81])
    for i in range(BHlayerz.size):
            plt.axvline(x=BHlayerz[i],color='k',linewidth=1,alpha=0.1) 



def rechitcleaned_CMSSW(dfcmsswrhit,df,eventnumber,layernumber=None):
    if layernumber is None:
        select2rch = (df_cmsswrhit.evn==df.event[eventnumber]) &  (df_cmsswrhit.z >0)
    else:
        select2rch = (df_cmsswrhit.evn==df.event[eventnumber]) & (df_cmsswrhit.layer==layernumber) & (df_cmsswrhit.z >0)
    
    HitCMSSW_x = array(df_cmsswrhit[select2rch].x)
    HitCMSSW_y = array(df_cmsswrhit[select2rch].y)
    HitCMSSW_z = array(df_cmsswrhit[select2rch].z)
    HitCMSSW_e = array(df_cmsswrhit[select2rch].energy)
    
    return HitCMSSW_x,HitCMSSW_y,HitCMSSW_z,HitCMSSW_e

#######################################################################################################
#######################################################################################################
#######################################################################################################

