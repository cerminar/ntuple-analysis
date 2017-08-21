"""HGCalRecHit calibration -
   class for obtaining details from the RecHit calibration used in CMSSW
   reproduces number listed at https://twiki.cern.ch/twiki/pub/CMS/HGCALSimulationAndPerformance/rechit.txt"""


class RecHitCalibration:

    def __init__(self):
        """set variables used in the functions"""
        # https://github.com/cms-sw/cmssw/blob/CMSSW_9_3_X/RecoLocalCalo/HGCalRecProducers/python/HGCalRecHit_cfi.py#L5
        self.dEdX_weights = (0.0,   # there is no layer zero
                             8.603,  # Mev
                             8.0675,
                             8.0675,
                             8.0675,
                             8.0675,
                             8.0675,
                             8.0675,
                             8.0675,
                             8.0675,
                             8.9515,
                             10.135,
                             10.135,
                             10.135,
                             10.135,
                             10.135,
                             10.135,
                             10.135,
                             10.135,
                             10.135,
                             11.682,
                             13.654,
                             13.654,
                             13.654,
                             13.654,
                             13.654,
                             13.654,
                             13.654,
                             38.2005,
                             55.0265,
                             49.871,
                             49.871,
                             49.871,
                             49.871,
                             49.871,
                             49.871,
                             49.871,
                             49.871,
                             49.871,
                             49.871,
                             62.005,
                             83.1675,
                             92.196,
                             92.196,
                             92.196,
                             92.196,
                             92.196,
                             92.196,
                             92.196,
                             92.196,
                             92.196,
                             92.196,
                             46.098)

        # https://github.com/cms-sw/cmssw/blob/CMSSW_9_3_X/RecoLocalCalo/HGCalRecProducers/python/HGCalRecHit_cfi.py#L86
        self.thicknessCorrection = (1.132,1.092,1.084)  # 100, 200, 300 um

        # Base configurations for HGCal digitizers
        # https://github.com/cms-sw/cmssw/blob/CMSSW_9_3_X/SimCalorimetry/HGCalSimProducers/python/hgcalDigitizer_cfi.py#L5
        # self.eV_per_eh_pair = 3.62
        self.fC_per_ele = 1.6020506e-4
        self.nonAgedNoises = (2100.0, 2100.0, 1600.0)  # 100,200,300 um (in electrons)

        # https://github.com/cms-sw/cmssw/blob/CMSSW_9_3_X/RecoLocalCalo/HGCalRecProducers/python/HGCalUncalibRecHit_cfi.py#L25
        self.fCPerMIP = (1.25, 2.57, 3.88)  # 100um, 200um, 300um

        # https://github.com/cms-sw/cmssw/blob/CMSSW_9_3_X/SimCalorimetry/HGCalSimProducers/python/hgcalDigitizer_cfi.py#L127
        self.noise_MIP = 1.0/7.0 #expectation based on latest SiPM performance

    def MeVperMIP(self, layer, thicknessIndex):
        if layer > 40:
            # no thickness correction for BH
            return self.dEdX_weights[layer]
        else:
            return self.dEdX_weights[layer]/self.thicknessCorrection[thicknessIndex]

    def MIPperGeV(self, layer, thicknessIndex):
        return 1000./MeVperMIP(layer, thicknessIndex)

    def sigmaNoiseMIP(self, layer, thicknessIndex):
        if layer > 40:
            # for BH, sigmaNoiseMIP = noise_MIP
            return self.noise_MIP
        else:
            return self.fC_per_ele * self.nonAgedNoises[thicknessIndex] / self.fCPerMIP[thicknessIndex]

    def sigmaNoiseMeV(self, layer, thicknessIndex):
        return self.sigmaNoiseMIP(layer, thicknessIndex) * self.MeVperMIP(layer, thicknessIndex)
