from __future__ import print_function
import math as m
import numpy as np
from scipy.spatial import cKDTree


def match_etaphi(ref_etaphi, trigger_etaphi, trigger_pt, deltaR=0.2, return_positional=False):
    '''Match objects within a given DeltaR.
    
    If return_positional = False
     Returns the panda index of the best match (highest-pt)
       and of all the matches
    If return_positional = True 
     Returns the position of the best match (highest-pt)
       and of all the matches in the input trigger_etaphi and trigger_pt arrays.
       '''
    # print ("INPUT ref_etaphi")
    # print (ref_etaphi)
    # print ("INPUT trigger_etaphi")
    # print (trigger_etaphi)
    # print ("INPUT trigger_pt")
    # print (trigger_pt)
    kdtree = cKDTree(trigger_etaphi)
    best_match_indices = {}
    all_matches_indices = {}
    
    # for iref,(eta,phi) in enumerate(ref_etaphi):
    for index, row in ref_etaphi.iterrows():
        gen_eta, gen_phi = row.values
        matched = kdtree.query_ball_point([gen_eta, gen_phi], deltaR)
        # not this in an integer of the index of the array not the index in the pandas meaning: hence to beused with iloc
        # Handle the -pi pi transition
        matched_sym = kdtree.query_ball_point([gen_eta, gen_phi-np.sign(gen_phi)*2.*m.pi], deltaR)
        matched = np.unique(np.concatenate((matched, matched_sym))).astype(int)
        # print ('matched iloc:')
        # print (matched)
        # print type(matched)
        # print trigger_pt[matched]
        # print trigger_etaphi.iloc[matched]
        # Choose the match with highest pT
        if (len(matched) != 0):
            # print (trigger_pt.iloc[matched])
            # print (trigger_pt.iloc[matched].idxmax())
            # print (np.argmax(trigger_pt.iloc[matched]))

            if return_positional:
                best_match_indices[index] = matched[np.argmax(trigger_pt.iloc[matched])]
                all_matches_indices[index] = matched
            else:
                best_match = trigger_pt.iloc[matched].idxmax()
                best_match_indices[index] = best_match
                all_matches_indices[index] = trigger_pt.iloc[matched].index.values
                
            # print ('best match:')
            # print (best_match)
            # best_match_indices[index] = best_match
            # all_matches_indices[index] = trigger_pt.iloc[matched].index.values
            # print (trigger_pt.iloc[matched].index.values)

    # print best_match_indices
    # print all_matches_indices
    return best_match_indices, all_matches_indices


def debugPrintOut(level, name, toCount, toPrint):
    if level == 0:
        return
    if level >= 3:
        print(('# {}: {}'.format(name, len(toCount))))
    if level >= 4 and not toPrint.empty:
        print(toPrint)
