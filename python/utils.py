import math as m

import numpy as np
import pandas as pd
import awkward as ak
from scipy.spatial import cKDTree

def match_etaphi(ref_etaphi, trigger_etaphi, trigger_pt, deltaR=0.2, return_positional=False):
    """
    Match objects within a given DeltaR.

    If return_positional = False
     Returns the panda index of the best match (highest-pt)
       and of all the matches
    If return_positional = True
     Returns the position of the best match (highest-pt)
       and of all the matches in the input trigger_etaphi and trigger_pt arrays.
    """
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


def debugPrintOut(level, name, toCount, toPrint, max_lines=-1):
    if level == 0:
        return
    if level >= 3:
        print(f'# {name}: {len(toCount)}')
    if level >= 4 and not toPrint.empty:
        print(max_lines)
        if max_lines != -1:
            with pd.option_context('display.max_rows', max_lines, 'display.max_columns', None,):
                print(toPrint)
        else:
            print(toPrint)


def angle_range(angle):
    """
        returns awkward arrays of angles between -pi and pi
    """
    angle = ak.where(angle>np.pi, angle-2*np.pi, angle)
    angle = np.where(angle<-np.pi, angle+2*np.pi, angle)
    return angle


def gen_match(gen, objects, gen_eta_phi=('eta', 'phi'), dr=0.1):
    # perform the matching and returns pairs of indexes (gen_index, object_index)
    match_eta = ak.cartesian([objects.eta, gen[gen_eta_phi[0]]])
    match_phi = ak.cartesian([objects.phi, gen[gen_eta_phi[1]]])
    match_pt = ak.cartesian([objects.pt, gen.pt])
    match_idx = ak.argcartesian([objects.eta, gen.eta])

    obj_eta, gen_eta = ak.unzip(match_eta)
    obj_phi, gen_phi = ak.unzip(match_phi)
    obj_pt, gen_pt = ak.unzip(match_pt)
    obj_idx, gen_idx = ak.unzip(match_idx)
    dpt = np.abs(obj_pt - gen_pt)
    dphi = obj_phi-gen_phi
    dphi = angle_range(dphi)
    dr2 = (obj_eta-gen_eta)**2+(dphi)**2
    match = ak.Array(data={'ele_idx': obj_idx, 'gen_idx': gen_idx, 'dpt': dpt, 'dr2': dr2})
    dr_match=match[match.dr2<dr*dr]
    ret = []
    for genid in np.unique(ak.flatten(dr_match.gen_idx)):
        gen_match_id = dr_match[dr_match.gen_idx == genid]
        dpt_min_index = ak.argmin(gen_match_id.dpt, axis=1, keepdims=True)
        best_match_id = gen_match_id[dpt_min_index]
        # matched_obj = objects[best_match_id.ele_idx]
        # matched_gen = gen[best_match_id.gen_idx]
        ret.append((best_match_id.gen_idx, best_match_id.ele_idx))
    return ret

