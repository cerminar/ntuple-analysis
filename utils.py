import math as m
import numpy as np
from scipy.spatial import cKDTree


def match_etaphi(ref_etaphi, trigger_etaphi, trigger_pt, deltaR=0.2):
    '''Match object with the highest pT within a given DeltaR'''
    kdtree = cKDTree(trigger_etaphi)
    matched_indices = {}
    for iref,(eta,phi) in enumerate(ref_etaphi):
        matched = kdtree.query_ball_point([eta,phi], deltaR)
        # Handle the -pi pi transition
        matched_sym = kdtree.query_ball_point([eta,phi-np.sign(phi)*2.*m.pi], deltaR)
        matched = np.unique(np.concatenate((matched, matched_sym))).astype(int)
        # Choose the match with highest pT
        best_match  = np.argmax(trigger_pt[matched])
        matched_indices[iref] = matched[best_match]
    return matched_indices
