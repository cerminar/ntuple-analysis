import ROOT
from math import cos, cosh, sqrt, ceil
from NtupleDataFormat import *
from glob import glob
from sys import maxint as MAXINT

USEPOLY = False
output_file = ROOT.TFile("zee_output.root", "RECREATE")
h_mass_category = {}
h_mass_category[2] = ROOT.TH1F("mass_bb", "mass_bb", 200, 0., 200.)
h_mass_category[1] = ROOT.TH1F("mass_be", "mass_be", 200, 0., 200.)
h_mass_category[0] = ROOT.TH1F("mass_ee", "mass_ee", 200, 0., 200.)
h_mass = ROOT.TH1F("mass", "mass", 200, 0., 200.)
h_mustache = ROOT.TH2F("mustache", "mustache", 100, -0.6, 0.6, 100, -0.1, 0.1)

def createPoly(dest, label, x_start, y_start, hex_size, cells_x, cells_y):
  dest[i] = ROOT.TH2Poly()
  dest[i].SetName('{label}_Layer{i}'.format(label=label,i=i))
  dest[i].SetTitle('{label}_Layer{i}'.format(label=label,i=i))
  dest[i].SetOption('colz l0')
  dest[i].Honeycomb(x_start, y_start, hex_size, cells_x, cells_y)

def createTH2F(dest, label, hex_size, full_x, full_y):
  dest[i] = ROOT.TH2F('{label}_Layer{i}'.format(label=label, i=i),
                      '{label}_Layer{i}'.format(label=label, i=i),
                      int(ceil(full_x/hex_size)), -full_x/2., full_x,
                      int(ceil(full_y/hex_size)), -full_y/2., full_y)

h_rechits = {}
h_sc = {}
h_seed = {}
extension_x = extension_y = 165
hex_size = 0.68 # 0.68 for 1.2 cm^2 cells, 0.45 for 0.53 cm^2 cells
cells_x = int(ceil(2*extension_x/(sqrt(3)*hex_size)))
cells_y = int(ceil(2*2*extension_y/(3*hex_size))) # Every 2 cells it grows by 3.5 side
# Only for the EE detector
for i in range(1,29):
  print("Booking histogram for layer {layer}".format(layer=i))
  if(USEPOLY):
    createPoly(h_rechits, 'RecHits', -extension_x,
               -extension_y, hex_size, cells_x, cells_y)
    createPoly(h_sc, 'SuperCluster', -extension_x,
               -extension_y, hex_size, cells_x, cells_y)
    createPoly(h_seed, 'Seed', -extension_x,
               -extension_y, hex_size, cells_x, cells_y)
  else:
    createTH2F(h_rechits, 'RecHits', hex_size, 2.*extension_x, 2*extension_y)
    createTH2F(h_sc, 'SuperCluster', hex_size, 2.*extension_x, 2*extension_y)
    createTH2F(h_seed, 'Seed', hex_size, 2.*extension_x, 2*extension_y)


#files = glob("/data/rovere/HGCAL/testNtupla/CMSSW_9_3_0_pre5/src/reco-prodtools/_RelValZEE_14_CMSSW_9_3_0_pre4-93X_upgrade2023_realistic_v0_2023D17noPU-v1_GEN-SIM-RECO/cfg/*.root")
#files = glob("/data/rovere/HGCAL/testNtupla/CMSSW_9_3_0_pre5/src/reco-prodtools/_RelValZEE_14_CMSSW_9_3_0_pre4-PU25ns_93X_upgrade2023_realistic_v0_D17PU200-v1_GEN-SIM-RECO/cfg/*.root")
files = glob("/data/rovere/HGCAL/testNtupla/CMSSW_9_3_0_pre5/src/reco-prodtools/_RelValSingleElectronPt35Extended_CMSSW_9_3_0_pre5-93X_upgrade2023_realistic_v1_2023D17noPU-v1_GEN-SIM-RECO/cfg/*.root")

ZMASS = 91.1876

def looseSelection(ele):
  return ele.pt() > 10. and abs(ele.track_simdz()) < 0.5

def tightSelection(ele):
  if ele.isEB() != 1 :
    return looseSelection(ele) and ele.hoe() < 0.01
  return looseSelection(ele)

def invMass(ele1, ele2):
  return sqrt(2.*ele1.pt()*ele2.pt()*(cosh(ele1.eta()-ele2.eta()) - cos(ele1.phi() - ele2.phi())))

def bestCandidate(electrons):
  best_mass = MAXINT
  min_distance = MAXINT
  best_candidates = (None, None)
  category = -1
  for ele1 in range(len(electrons)):
    e1 = electrons[ele1]
    for ele2 in range(ele1, len(electrons)):
      e2 = electrons[ele2]
      if e1.charge()*e2.charge() > 0:
        continue
      mass = invMass(e1, e2)
      if abs(ZMASS-mass) < min_distance:
        best_mass = mass
        min_distance = abs(ZMASS-mass)
        best_candidates = (e1, e2)
        category = e1.isEB()+e2.isEB()
  return (best_mass != MAXINT, best_mass, best_candidates, int(category))

def analysePFClustersFromMultiCl(ntuple):
  for ev in ntuple:
    for mc in ev.pfclustersFromMultiCl():
      print mc
      return

def analyseElectronsRecHits(ntuple):
#    ev = ntuple.getEvent(1)
  for ev in ntuple:
    for e in ev.electrons():
      if e.isEB(): continue
      print("SC Position: ({x}, {y}, {z}) eta: {eta}, phi: {phi}, energy: {energy}".format(
              x=e.scpos().x(), y=e.scpos().y(), z=e.scpos().z(),
              eta=e.scpos().eta(), phi=e.scpos().phi(),
              energy=e.energy())
           )
      for c in e.clustersFromMultiCl():
        print c
        h_mustache.Fill(e.seedphi()-c.phi(), e.seedeta()-c.eta())
        for i, rh in enumerate(c.hits()):
          if rh.layer() <= 28:
            h_rechits[rh.layer()].Fill(rh.x(), rh.y(), rh.energy()*c.fractions()[i])
        if e.seedlayer() <=28:
          h_sc[e.seedlayer()].Fill(e.scpos().x(), e.scpos().y(), e.energy())
          h_seed[e.seedlayer()].Fill(e.seedpos().x(), e.seedpos().y(), e.seedenergy())
      return

def ZeeAnalyses(ntuple):
  for ev in ntuple:
    good_candidate = [e for e in ev.electrons() if tightSelection(e)]
    (found, mass, (e1, e2), category) = bestCandidate(good_candidate)
    if found:
      print ev.event(), len(good_candidate), mass, category
      h_mass.Fill(mass)
      h_mass_category[category].Fill(mass)

def saveOutput():
  output_file.cd('')
  h_mass.Write()
  h_mustache.Write()
  for l, h in h_rechits.iteritems():
    h.Write()
  for l, h in h_sc.iteritems():
    h.Write()
  for l, h in h_seed.iteritems():
    h.Write()
  for k, h in h_mass_category.iteritems():
    h.Write()
  output_file.Close()


for f in files[0:1]:
  ntuple = HGCalNtuple(f)

#  ZeeAnalyses(ntuple)
  analyseElectronsRecHits(ntuple)
#  analysePFClustersFromMultiCl(ntuple)

  saveOutput()
