import awkward as ak
import numpy as np
import math

from python.collections import DFCollection
from python import pf_regions

def mc_fixtures(particles):
    particles['abseta'] = np.abs(particles.eta)
    return particles

def ele_mc_fixtures(particles):
    if 'pdgid' not in particles.fields:
        particles['pdgid'] = particles.charge*11
    return mc_fixtures(particles)

def pho_mc_fixtures(particles):
    if 'pdgid' not in particles.fields:
        particles['pdgid'] = 22
    return mc_fixtures(particles)

def pi_mc_fixtures(particles):
    if 'pdgid' not in particles.fields:
        particles['pdgid'] = particles.charge*211
    return mc_fixtures(particles)

def highest_pt(objs, num=2):
    sel_objs = objs[objs.prompt >= 2]
    index = ak.argsort(sel_objs.pt)
    array = sel_objs[index]
    # print (ak.local_index(array))
    return array[ak.local_index(array.pt, axis=1)<num]


def cl3d_fixtures(clusters):
    # print(clusters.show())

    # print(clusters)
    # print(clusters.type.show())
    # print(clusters.energy)

    mask_loose = 0b0010
    mask_tight = 0b0001
    clusters['IDTightEm'] = np.bitwise_and(clusters.hwQual, mask_tight) > 0
    clusters['IDLooseEm'] = np.bitwise_and(clusters.hwQual, mask_loose) > 0
    clusters['eMax'] = clusters.emaxe*clusters.energy
    # clusters['meanz_scaled'] = clusters.meanz-320.
    # clusters['abseta'] =  np.abs(clusters.eta)

    # if False:
    #     input_array = ak.flatten(
    #         clusters[[
    #             'coreshowerlength',
    #             'showerlength',
    #             'firstlayer',
    #             'maxlayer',
    #             'szz',
    #             'srrmean',
    #             'srrtot',
    #             'seetot',
    #             'spptot']],
    #         axis=1)
    #     input_data = ak.concatenate(ak.unzip(input_array[:, np.newaxis]), axis=1)
    #     input_matrix = xgboost.DMatrix(np.asarray(input_data))
    #     score =  classifiers.eg_hgc_model_xgb.predict(input_matrix)

    # pu_input_array = ak.flatten(
    #     clusters[[
    #         'eMax',
    #         'emaxe',
    #         'spptot',
    #         'srrtot',
    #         'ntc90']],
    #     axis=1)
    # pu_input_data = ak.concatenate(ak.unzip(pu_input_array[:, np.newaxis]), axis=1)
    # pu_input_matrix = xgboost.DMatrix(np.asarray(pu_input_data))
    # pu_score =  classifiers.pu_veto_model_xgb.predict(pu_input_matrix)

    # counts = ak.num(clusters)
    # clusters_flat = ak.flatten(clusters)
    # clusters_flat['egbdtscore'] = score
    # clusters_flat['pubdtscore'] = pu_score

    # clusters_flat['egbdtscoreproba'] = -np.log(1.0/score - 1.0)
    # clusters_flat['pubdtscoreproba'] = -np.log(1.0/pu_score - 1.0)


    # clusters = ak.unflatten(clusters_flat, counts)
    # print(clusters.type.show())

    return clusters

def quality_flags(objs):
    # print(objs.hwQual)
    objs['hwQual'] = ak.values_astype(objs.hwQual, np.int32)
    mask_tight_sta = 0b0001
    mask_tight_ele = 0b0010
    mask_tight_pho = 0b0100
    mask_no_brem = 0b1000
    objs['IDTightSTA'] = np.bitwise_and(objs.hwQual, mask_tight_sta) > 0
    objs['IDTightEle'] = np.bitwise_and(objs.hwQual, mask_tight_ele) > 0
    objs['IDTightPho'] = np.bitwise_and(objs.hwQual, mask_tight_pho) > 0
    objs['IDNoBrem'] = np.bitwise_and(objs.hwQual, mask_no_brem) > 0
    objs['IDBrem'] = np.bitwise_and(objs.hwQual, mask_no_brem) == 0
    return objs

def quality_ele_fixtures(objs):
    # print(objs)
    objs['dpt'] = objs.tkPt - objs.pt
    return quality_flags(objs)

def decodedTk_fixtures(objects):
    objects['deltaZ0'] = objects.z0 - objects.simz0
    objects['deltaPt'] = objects.pt - objects.simpt
    objects['deltaEta'] = objects.eta - objects.simeta
    objects['deltaCaloEta'] = objects.caloeta - objects.simcaloeta
    # have dphi between -pi and pi
    comp_remainder = np.vectorize(math.remainder)
    objects['deltaCaloPhi'] = comp_remainder(objects.calophi - objects.simcalophi, 2*np.pi)

    objects['abseta'] = np.abs(objects.eta)
    objects['simabseta'] = np.abs(objects.simeta)
    return objects

def build_double_obj(obj):
    ret = ak.combinations(
        array=obj,
        n=2,
        axis=1,
        fields=['leg0', 'leg1'])
    # ret.show()
    return ret

def double_obj_fixtures(obj):
    # for the rate computation we assign the low-pt leg pt as pt of the pair
    obj['pt'] = obj.leg1.pt
    return obj

def double_electron_fixtures(obj):
    obj = double_obj_fixtures(obj)
    obj['dz'] = np.abs(obj.leg0.vz - obj.leg1.vz)
    return obj

def map2pfregions(objects, eta_var, phi_var, fiducial=False):
    for ieta, eta_range in enumerate(pf_regions.regionizer.get_eta_boundaries(fiducial)):
        # print(f'eta_reg_{ieta}')
        objects[f'eta_reg_{ieta}'] = (objects[eta_var] > eta_range[0]) & (objects[eta_var] <= eta_range[1])
        # print(objects[['eta', 'phi', f'eta_reg_{ieta}']].show())

    for iphi, phi_range in enumerate(pf_regions.regionizer.get_phi_boundaries(fiducial)):
        objects[f'phi_reg_{iphi}'] = (objects[phi_var] > phi_range[0]) & (objects[phi_var] <= phi_range[1])

    return objects


def maptk2pfregions_in(objects):
    return map2pfregions(objects, 'caloeta', 'calophi', fiducial=False)


def mapcalo2pfregions_in(objects):
    return map2pfregions(objects, 'eta', 'phi', fiducial=False)


def mapcalo2pfregions_out(objects):
    return map2pfregions(objects, 'eta', 'phi', fiducial=True)


gen_ele = DFCollection(
    name='GEN', label='GEN particles (ele)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='GenEl', entry_block=entry_block),
    fixture_function=ele_mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    max_print_lines=None,
    debug=0)
# gen_ele.activate()


gen_highestpt_ele = DFCollection(
    name='GEN', label='GEN particles (ele highest-pT)',
    filler_function=lambda event, entry_block: highest_pt(gen_ele.df),
    # fixture_function=mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    max_print_lines=None,
    depends_on=[gen_ele],
    debug=0)
# gen_highestpt_ele.activate()

gen_pho = DFCollection(
    name='GEN', label='GEN particles (pho)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='GenPh', entry_block=entry_block),
    fixture_function=pho_mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    max_print_lines=None,
    debug=0)

gen_pi = DFCollection(
    name='GEN', label='GEN particles (pi)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='GenPi', entry_block=entry_block),
    fixture_function=pi_mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    max_print_lines=None,
    debug=0)


gen = DFCollection(
    name='GEN', label='GEN particles',
    filler_function=lambda event, entry_block: ak.concatenate([gen_ele.df, gen_pho.df], axis=1),
    # fixture_function=mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    depends_on=[gen_ele, gen_pho],
    max_print_lines=None,
    debug=0)
# gen.activate()


gen_jet = DFCollection(
    name='GEN', label='GEN jets',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='GenJets', entry_block=entry_block),
    fixture_function=mc_fixtures,
    # print_function=lambda df: df[['pdgid', 'pt', 'eta', 'phi']],
    # print_function=lambda df: df[(df.pdgid==23 | (abs(df.pdgid)==15))],
    max_print_lines=None,
    debug=0)


hgc_cl3d = DFCollection(
    name='HGCCl3d', label='HGC Cl3d',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='HGCal3DCl', entry_block=entry_block, fallback='HMvDR'),
    fixture_function=lambda clusters: cl3d_fixtures(clusters),
    # read_entry_block=500,
    debug=0,
    # print_function=lambda df: df[['rho', 'eta', 'phi', 'hwQual', 'ptEm', 'egbdtscore', 'pubdtscore', 'egbdtscoreproba', 'pubdtscoreproba', 'pfPuIdScore', 'egEmIdScore']].sort_values(by='rho', ascending=False)
    print_function=lambda df: df.columns
    )

tracks = DFCollection(
    name='L1Trk', label='L1Track',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='l1Trk', entry_block=entry_block),
    print_function=lambda df: df.sort_values(by='pt', ascending=False)[:10],
    debug=0)


TkEleEE = DFCollection(
    name='TkEleEE', label='TkEle EE',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEleEE', entry_block=entry_block),
    fixture_function=quality_ele_fixtures,
    print_function=lambda df:df.columns,
    debug=0)


TkEleEB = DFCollection(
    name='TkEleEB', label='TkEle EB',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEleEB', entry_block=entry_block),
    fixture_function=quality_ele_fixtures,
    debug=0)

TkEleEllEE = DFCollection(
    name='TkEleEllEE', label='TkEle EE (Ell.)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEleEllEE', entry_block=entry_block),
    fixture_function=quality_ele_fixtures,
    debug=0)

TkEmEE = DFCollection(
    name='TkEmEE', label='TkEm EE',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEmEE', entry_block=entry_block),
    print_function=lambda df: df.loc[(abs(df.eta) > 2.4), ['energy', 'pt', 'eta', 'phi','hwQual']].sort_values(by='pt', ascending=False)[:10],
    fixture_function=quality_flags,
    debug=0)

TkEmEB = DFCollection(
    name='TkEmEB', label='TkEm EB',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEmEB', entry_block=entry_block),
    fixture_function=quality_flags,
    # read_entry_block=200,
    debug=0)

TkEmL2 = DFCollection(
    name='TkEmL2', label='TkEm L2',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='TkEmL2', entry_block=entry_block),
    fixture_function=quality_flags,
    debug=0)

# -- FP
TkEleL2 = DFCollection(
    name='TkEleL2', label='TkEle L2',
    filler_function=lambda event, entry_block : event.getDataFrame(
        prefix='TkEleL2', entry_block=entry_block, fallback='L2TkEle'),
    fixture_function=quality_ele_fixtures,
    debug=0)

TkEmL2Ell = DFCollection(
    name='TkEmL2Ell', label='TkEm L2 (ell.)',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='L2TkEmEll', entry_block=entry_block),
    fixture_function=quality_flags,
    debug=0)

TkEleL2Ell = DFCollection(
    name='TkEleL2Ell', label='TkEle L2 (ell.)',
    filler_function=lambda event, entry_block : event.getDataFrame(
        prefix='L2TkEleEll', entry_block=entry_block, fallback='TkEleL2Ell'),
    fixture_function=quality_ele_fixtures,
    debug=0)

DoubleTkEleL2 = DFCollection(
    name='DoubleTkEleL2', label='DoubleTkEle L2',
    filler_function=lambda event, entry_block: build_double_obj(obj=TkEleL2.df),
    fixture_function=double_electron_fixtures,
    depends_on=[TkEleL2],
    debug=0)

DoubleTkEmL2 = DFCollection(
    name='DoubleTkEmL2', label='DoubleTkEm L2',
    filler_function=lambda event, entry_block: build_double_obj(obj=TkEmL2.df),
    fixture_function=double_obj_fixtures,
    depends_on=[TkEmL2],
    debug=0)

EGStaEE = DFCollection(
    name='EGStaEE', label='EG EE',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='EGStaEE', entry_block=entry_block),
    print_function=lambda df: df.loc[(abs(df.eta) > 2.4), ['energy', 'pt', 'eta', 'phi','hwQual']].sort_values(by='pt', ascending=False)[:10],
    # fixture_function=mapcalo2pfregions,
    fixture_function=quality_flags,
    debug=0)


EGStaEB = DFCollection(
    name='EGStaEB', label='EG EB',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='EGStaEB', entry_block=entry_block),
    # print_function=lambda df: df[['energy', 'pt', 'eta', 'hwQual']].sort_values(by='hwQual', ascending=False)[:10],
    fixture_function=quality_flags,
    # read_entry_block=200,
    debug=0)

decTk = DFCollection(
    name='PFDecTk', label='decoded Tk',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='pfdtk', entry_block=entry_block),
    fixture_function=decodedTk_fixtures,
    debug=0)


tkCl3DMatch = DFCollection(
    name='TkCl3DMatch', label='TkCl3DMatch',
    filler_function=lambda event, entry_block: get_trackmatched_egs(egs=hgc_cl3d, tracks=tracks),
    # fixture_function=mapcalo2pfregions_in,
    depends_on=[hgc_cl3d, tracks],
    debug=0)


hgc_cl3d_pfinputs = DFCollection(
    name='HGCCl3dPfIN', label='HGC Cl3d L1TC IN',
    filler_function=lambda event, entry_block: hgc_cl3d.df,
    fixture_function=mapcalo2pfregions_in,
    depends_on=[hgc_cl3d],
    debug=0)

EGStaEB_pfinputs = DFCollection(
    name='EGStaEBPFin', label='EG EB  L1TC IN',
    filler_function=lambda event, entry_block: EGStaEB.df,
    fixture_function=mapcalo2pfregions_in,
    depends_on=[EGStaEB],
    print_function=lambda df: df.loc[~(df.eta_reg_4 | df.eta_reg_5 | df.eta_reg_6 | df.eta_reg_7 | df.eta_reg_8 | df.eta_reg_9), ['eta', 'phi', 'eta_reg_0', 'eta_reg_1', 'eta_reg_2', 'eta_reg_3', 'eta_reg_4', 'eta_reg_5', 'eta_reg_6', 'eta_reg_7', 'eta_reg_8', 'eta_reg_9', 'eta_reg_10', 'eta_reg_11', 'eta_reg_12', 'eta_reg_13']].sort_values(by='eta', ascending=False),
    debug=0)

TkEleEB_pf_reg = DFCollection(
    name='PFOuttkEleEB', label='TkEle EB (old EMU)',
    filler_function=lambda event, entry_block: tkeles_EB_pf.df,
    fixture_function=mapcalo2pfregions_out,
    depends_on=[TkEleEB],
    debug=0)

tk_pfinputs = DFCollection(
    name='L1TrkPfIn', label='L1Track Input',
    filler_function=lambda event, entry_block: tracks.df,
    fixture_function=maptk2pfregions_in,
    depends_on=[tracks],
    debug=0)

pfjets = DFCollection(
    name='PFJets', label='Ak4 PFJets',
    filler_function=lambda event, entry_block: event.getDataFrame(
        prefix='L1PFJets', entry_block=entry_block),
    print_function=lambda df: df.sort_values(by='pt', ascending=False)[:10],
    debug=0)


