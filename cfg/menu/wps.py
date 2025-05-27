from python.selections import *

menu_thresh_pt = [
    Selection('PtSta51', 'p_{T}^{TOBJ}#geq51GeV', lambda ar: ar.pt_off >= 51),
    Selection('PtSta37', 'p_{T}^{TOBJ}#geq37GeV', lambda ar: ar.pt_off >= 37),
    Selection('PtSta24', 'p_{T}^{TOBJ}#geq24GeV', lambda ar: ar.pt_off >= 24),
    Selection('PtSta12', 'p_{T}^{TOBJ}#geq12GeV', lambda ar: ar.pt_off >= 12),
    Selection('PtEle36', 'p_{T}^{TOBJ}#geq36GeV', lambda ar: ar.pt_off >= 36),
    Selection('PtEle25', 'p_{T}^{TOBJ}#geq25GeV', lambda ar: ar.pt_off >= 25),
    Selection('PtEle12', 'p_{T}^{TOBJ}#geq12GeV', lambda ar: ar.pt_off >= 12),
    Selection('PtIsoEle28', 'p_{T}^{TOBJ}#geq28GeV', lambda ar: ar.pt_off_iso >= 28),
    Selection('PtIsoEle22', 'p_{T}^{TOBJ}#geq22GeV', lambda ar: ar.pt_off_iso >= 22),
    Selection('PtIsoPho36', 'p_{T}^{TOBJ}#geq36GeV', lambda ar: ar.pt_off_iso >= 36),
    Selection('PtIsoPho22', 'p_{T}^{TOBJ}#geq22GeV', lambda ar: ar.pt_off_iso >= 22),
    Selection('PtIsoPho12', 'p_{T}^{TOBJ}#geq12GeV', lambda ar: ar.pt_off_iso >= 12),
]

Selector.selection_primitives = sm.selections.copy()
