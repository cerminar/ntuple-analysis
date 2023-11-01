"""Provides MVA classifiers."""

import ROOT
import array
import xgboost

def book_MVA_classifier(model, weight_file, variables):
    mva_classifier = ROOT.TMVA.Reader()

    for variable in variables:
        mva_classifier.AddVariable(variable, array.array('f', [0.]))
    mva_classifier.BookMVA(model, weight_file)
    return mva_classifier


# setup the EGID classifies
mva_pu_classifier = None


def mva_pu_classifier_builder():
    global mva_pu_classifier
    if mva_pu_classifier is None:
        mva_pu_classifier = book_MVA_classifier(model='BDT',
                                                weight_file='data/Photon_Pion_vs_Neutrino_BDTweights_1116.xml',
                                                variables=['eMax',
                                                           'eMaxOverE',
                                                           'sigmaPhiPhiTot',
                                                           'sigmaRRTot',
                                                           'triggerCells90percent'])
    return mva_pu_classifier



mva_pi_classifier = None


def mva_pi_classifier_builder():
    global mva_pi_classifier
    if mva_pi_classifier is None:
        mva_pi_classifier = book_MVA_classifier(model='BDT',
                                                weight_file='data/MVAnalysis_Bkg_BDTvsPions.weights.xml',
                                                variables=['pt_cl',
                                                           'eta_cl',
                                                           'maxLayer_cl',
                                                           'hOverE_cl',
                                                           'eMaxOverE_cl',
                                                           'sigmaZZ_cl'])
    return mva_pi_classifier



eg_hgc_model_xgb = xgboost.Booster()
eg_hgc_model_xgb.load_model('data/hgcegid_3151_loweta_xgboost.json')

pu_veto_model_xgb = xgboost.Booster()
pu_veto_model_xgb.load_model('data/puid_model_xgboost.json')
# loaded_model = XGBClassifier()
# loaded_model.load_model('xgb_model.json')
