# ntuple-tools

The python scripts in this repository should help you get started analysing the [HGCal ntuple](https://github.com/CMS-HGCAL/reco-ntuples).

## General usage

[NtupleDataFormat.py](NtupleDataFormat.py) provides a wrapper to the ntuple such that it can be used as if it contained classes. An example implementation can be found in [hgcalNtupleExample.py](hgcalNtupleExample.py). You need to provide an ntuple ROOT file to it:
```
python hgcalNtupleExample.py inputFile.root
```
In case your input file resides on EOS, you need to prepend `root://eoscms.cern.ch/` to the path, e.g. `python hgcalNtupleExample.py  "root://eoscms.cern.ch//eos/cms/store/cmst3/group/hgcal/CMG_studies/Production/FlatRandomPtGunProducer_SinglePion_Pt20To100GeV_Eta2p3To2p5_20170605/NTUP/partGun_PDGid211_x100_Pt20.0To100.0_NTUP_1.root"`

## HGCal imaging algorithm

[HGCalImagingAlgo.py](HGCalImagingAlgo.py) provides the HGCAl clustering code ported from CMSSW/C++ into a stand-alone python version that can run on the HGCAL ntuples. Parameterisation of the stand-alone clustering is identical to the CMSSW/C++ version. E.g. if parameter dependSensor is set to true, setting of E_c will be interpreted in terms of the local noise, and distance metric for both 2D clustering and multi-clustering is provided in terms of cm.

An example script for running different scenarios of stand-alone clustering, and comparison of sim-clusters and multi-clusters, is implemented in [hgcalReClusteringExample.py](hgcalReClusteringExample.py), while rechit calibration is implemented in [RecHitCalibration.py](RecHitCalibration.py). In order to run thre example:
```
python hgcalReClusteringExample.py

```
