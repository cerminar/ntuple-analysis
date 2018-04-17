import math as m
import numpy as np
from scipy.spatial import cKDTree


def match_etaphi(ref_etaphi, trigger_etaphi, trigger_pt, deltaR=0.2):
    '''Match object with the highest pT within a given DeltaR'''
    kdtree = cKDTree(trigger_etaphi)
    matched_indices = {}
    allmatches = {}
    # for iref,(eta,phi) in enumerate(ref_etaphi):
    for index, row in ref_etaphi.iterrows():
        # print (index)
        # print (row)
        matched = kdtree.query_ball_point([row.eta, row.phi], deltaR)

        # Handle the -pi pi transition
        matched_sym = kdtree.query_ball_point([row.eta, row.phi-np.sign(row.phi)*2.*m.pi], deltaR)
        matched = np.unique(np.concatenate((matched, matched_sym))).astype(int)
        # print matched
        # print type(matched)
        # print trigger_pt[matched]
        # print trigger_etaphi.iloc[matched]
        # Choose the match with highest pT
        if (len(matched) != 0):
            best_match = np.argmax(trigger_pt[matched])
            # print best_match
            matched_indices[index] = best_match
            allmatches[index] = matched
    return matched_indices, allmatches


def debugPrintOut(level, name, toCount, toPrint):
    if level == 0:
        return
    if level >= 2:
        print('# {}: {}'.format(name, len(toCount)))
    if level >= 3:
        print(toPrint)
