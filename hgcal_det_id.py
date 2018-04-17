import numpy as np

kHGCalCellOffset = 0
kHGCalCellMask = 0xFF
kHGCalWaferOffset = 8
kHGCalWaferMask = 0x3FF
kHGCalWaferTypeOffset = 18
kHGCalWaferTypeMask = 0x1
kHGCalLayerOffset = 19
kHGCalLayerMask = 0x1F
kHGCalZsideOffset = 24
kHGCalZsideMask = 0x1
kHGCalMaskCell = 0xFFFFFF00

kThird_offset = 4
kThird_mask = 0x3


def wafer(raw_id):
    return (raw_id >> kHGCalWaferOffset) & kHGCalWaferMask


def waferType(raw_id):
    type = (raw_id >> kHGCalWaferTypeOffset) & kHGCalWaferTypeMask
    if type:
        return 1
    else:
        return -1
    # result = 1 if ((raw_id >> kHGCalWaferTypeOffset) & kHGCalWaferTypeMask) else -1
    # return result


def cell(raw_id):
    return raw_id & kHGCalCellMask


def module_sector(raw_id):
    # if waferType(raw_id) != -1:
    #     return -1
    return (cell(raw_id) >> kThird_offset) & kThird_mask


def hgcroc(raw_id):
    # if waferType(raw_id) != -1:
    #     return -1
    return (cell(raw_id) >> kThird_offset) & kThird_mask


def hgcroc_small(raw_id):
    sector = module_sector(raw_id)
    subcell = cell(raw_id) - sector*16
    subsector = 0
    if subcell >= 8:
        subsector = 1
    return (2*sector)+subsector


def hgcroc_big(raw_id):
    return module_sector(raw_id)


v_wafer = np.vectorize(pyfunc=wafer)
v_waferType = np.vectorize(pyfunc=waferType)
v_cell = np.vectorize(pyfunc=cell)
v_module_sector = np.vectorize(pyfunc=module_sector)
v_hgcroc = np.vectorize(pyfunc=hgcroc)
v_hgcroc_small = np.vectorize(pyfunc=hgcroc_small)
v_hgcroc_big = np.vectorize(pyfunc=hgcroc_big)


# FIXME: in the current implementation of the TriggerGeometry all TCs have wafertype = 1
# here we need to know which modules have 120um thinckness to map them to the 6 HGCROC layouts
# we crudely assume that this is true for EE and FH modules with radius < 70cm
def settype_on_radius(radius):
    if radius < 70:
        return -1
    return 1


v_settype_on_radius = np.vectorize(pyfunc=settype_on_radius)


class HGCalDetId:
    def __init__(self, raw_id):
        self.raw_id = int(raw_id)
        self.kHGCalCellOffset = 0
        self.kHGCalCellMask = 0xFF
        self.kHGCalWaferOffset     = 8
        self.kHGCalWaferMask       = 0x3FF
        self.kHGCalWaferTypeOffset = 18
        self.kHGCalWaferTypeMask   = 0x1
        self.kHGCalLayerOffset     = 19
        self.kHGCalLayerMask       = 0x1F
        self.kHGCalZsideOffset     = 24
        self.kHGCalZsideMask       = 0x1
        self.kHGCalMaskCell        = 0xFFFFFF00

        self.kThird_offset_ = 4
        self.kThird_mask_ = 0x3

    def wafer(self):
        return (self.raw_id>>self.kHGCalWaferOffset) & self.kHGCalWaferMask

    def waferType(self):
        result = 1 if ((self.raw_id>>self.kHGCalWaferTypeOffset)&self.kHGCalWaferTypeMask) else -1
        return result

    def cell(self):
        return self.raw_id & self.kHGCalCellMask;

    def hgcroc(self):
        if self.waferType() != -1:
            return 1
        return (self.cell() >> self.kThird_offset_) & self.kThird_mask_
