import math

class PFRegionizer(object):
    def __init__(self):
        # setup the regions
        self.eta_boundaries_fiducial_ = [-5, -4, -3, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3, 4, 5]
        eta_overlap = 0.25
        phi_overlap = 0.25
        phiSlices = 9

        self.eta_boundaries = []
        self.phi_boundaries_fiducial_ = []
        self.phi_boundaries = []
        self.eta_centers = []
        self.phi_centers = []
        self.eta_boundaries_fiducial = []
        self.phi_boundaries_fiducial = []

        phiWidth = 2*math.pi/phiSlices

        for ieta, eta_low_fiducial in enumerate(self.eta_boundaries_fiducial_):
            if ieta >= len(self.eta_boundaries_fiducial_)-1:
                break
            eta_high_fiducial = self.eta_boundaries_fiducial_[ieta+1]
            # print 'fiducial boundaries: {}, {}'.format(eta_low_fiducial, eta_high_fiducial)
            eta_low = eta_low_fiducial - eta_overlap
            eta_high = eta_high_fiducial + eta_overlap
            self.eta_boundaries.append((eta_low, eta_high))
            self.eta_centers.append(eta_low_fiducial+(eta_high_fiducial-eta_low_fiducial)/2.)
            self.eta_boundaries_fiducial.append((eta_low_fiducial, eta_high_fiducial))

        for iphi in range(0, 9):
            phiCenter = (iphi+0.5)*phiWidth-math.pi
            self.phi_centers.append(phiCenter)
            # print iphi,phiCenter
            phi_low_fiducial = phiCenter - phiWidth/2.
            phi_high_fiducial = phiCenter+phiWidth/2.

            # print 'fiducial boundaries: {}, {}'.format(phi_low_fiducial, phi_high_fiducial)
            self.phi_boundaries_fiducial_.append(phi_low_fiducial)

            phi_low = phi_low_fiducial - phi_overlap
            if phi_low < -1*math.pi:
                phi_low = math.pi-phi_overlap
            phi_high = phi_high_fiducial + phi_overlap
            if phi_high > math.pi:
                phi_high = -1*math.pi+phi_overlap

            self.phi_boundaries.append((phi_low, phi_high))
            self.phi_boundaries_fiducial.append((phi_low_fiducial, phi_high_fiducial))
            # print 'boundaries: {} {}'.format(phi_low, phi_high)

        self.phi_boundaries_fiducial_.append(math.pi)

    def n_eta_regions(self):
        return len(self.eta_boundaries_fiducial_) - 1

    def n_phi_regions(self):
        return len(self.phi_boundaries_fiducial_) - 1

    def get_eta_boundaries(self, fiducial=False):
        if fiducial:
            return self.eta_boundaries_fiducial
        else:
            return self.eta_boundaries

    def get_phi_boundaries(self, fiducial=False):
        if fiducial:
            return self.phi_boundaries_fiducial
        else:
            return self.phi_boundaries


regionizer = PFRegionizer()

regions = {
'all': range(0, regionizer.n_eta_regions()),
'BRL': [4,5,6],
'HGC': [3, 7],
'HGCNoTk': [2, 8],
'HF': [0, 1, 9, 10]
}
