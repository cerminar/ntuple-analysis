"""Provides MVA classifiers."""

import ROOT
import array


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
                                                weight_file='data/MVAnalysis_Bkg_BDTvsPU.weights.xml',
                                                variables=['pt_cl',
                                                           'eta_cl',
                                                           'maxLayer_cl',
                                                           'hOverE_cl',
                                                           'eMaxOverE_cl',
                                                           'sigmaZZ_cl'])
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
