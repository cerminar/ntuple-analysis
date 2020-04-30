"""
Manager of the calibration data.

This module provides the code managing the calibration data used by the collections.

Classes:
    EventManager
    DFCollection
    TPSet

Objects:
    all collections (DFCollection and TPSet instances) that
    can be used by plotters.
"""
import pandas as pd
import numpy as np


class CalibManager(object):
    """
    CalibManager.

    Manages the calibration data ensuring they are read only once
    and  coherently served to the collections when/if needed.

    It is a singleton.
    """

    class __TheManager:
        def __init__(self):
            self.calib_version = None
            self.calib_table = {
                'calib-v96bis': {
                      'HMvDRCalib': {
                           'layer_calibs': [1., 0.98, 1.03, 1.07, 0.94, 0.96, 1.09, 1.03, 0.81, 1.0, 0.9, 1.08, 1.5, 1.81]
                           },
                      'HMvDRcylind10Calib': {
                           'layer_calibs': [1., 0.99, 1.03, 1.07, 0.94, 0.96, 1.09, 1.03, 0.8, 1.02, 0.9, 1.08, 1.5, 1.83]
                           },
                      'HMvDRcylind5Calib': {
                           'layer_calibs': [1., 0.98, 1.05, 1.09, 0.96, 0.97, 1.11, 1.06, 0.83, 1.04, 0.92, 1.08, 1.5, 1.89]
                           },
                      'HMvDRcylind2p5Calib': {
                           'layer_calibs': [1., 0.87, 1.16, 1.21, 1.06, 1.04, 1.25, 1.15, 0.97, 1.11, 1.01, 1.15, 1.64, 2.15]
                           },
                      'HMvDRshapeCalib': {
                           'layer_calibs': [1., 1.38, 1.05, 1.07, 0.96, 0.97, 1.11, 1.04, 0.81, 1.02, 0.9, 1.07, 1.52, 1.84]
                           },
                      'HMvDRshapeDrCalib': {
                           'layer_calibs': [1., 0.98, 1.05, 1.09, 0.96, 0.97, 1.11, 1.05, 0.83, 1.03, 0.91, 1.07, 1.51, 1.89]
                           },
                          },
                'calib-v120': {
                      'HMvDRCalib': {
                           'layer_calibs': [1., 0.98, 1.03, 1.07, 0.94, 0.96, 1.09, 1.03, 0.81, 1.0, 0.9, 1.08, 1.5, 1.81]
                           },
                      'HMvDRcylind10Calib': {
                           'layer_calibs': [1., 0.99, 1.03, 1.07, 0.94, 0.96, 1.09, 1.03, 0.8, 1.02, 0.9, 1.08, 1.5, 1.83]
                           },
                      'HMvDRcylind5Calib': {
                           'layer_calibs': [1., 0.98, 1.05, 1.09, 0.96, 0.97, 1.11, 1.06, 0.83, 1.04, 0.92, 1.08, 1.5, 1.89]
                           },
                      'HMvDRcylind2p5Calib': {
                           'layer_calibs': [1., 0.87, 1.16, 1.21, 1.06, 1.04, 1.25, 1.15, 0.97, 1.11, 1.01, 1.15, 1.64, 2.15]
                           },
                      'HMvDRshapeCalib': {
                           'layer_calibs': [1., 1.38, 1.05, 1.07, 0.96, 0.97, 1.11, 1.04, 0.81, 1.02, 0.9, 1.07, 1.52, 1.84]
                           },
                      'HMvDRshapeDrCalib': {
                           'layer_calibs': [0.0, 1.23, 0.87, 1.2, 0.97, 0.95, 1.05, 1.05, 1.01, 0.93, 1.04, 0.96, 1.35, 1.83],
                           'eta_calibs': (-17.6839, 39.2417)
                           },
                      'HMvDRshapeDtDuCalib': {
                           'layer_calibs': [0.0, 1.53, 0.83, 1.26, 1.05, 0.98, 1.19, 1.07, 1.04, 0.89, 1.27, 1.07, 1.34, 1.8],
                           'eta_calibs': (-14.5587, 34.5388)
                           },
                         }
                }

        def set_calibration_version(self, version):
            if version not in self.calib_table.keys():
                print("calibration version: {} not among availble sets: {}".format(version,
                                                                                   self.calib_table.keys()))
                raise KeyError('Unknown calibration version: {}, check configuration file!'.format(version))
            self.calib_version = version

        def get_calibration(self, collection_name, calib_key):
            return self.calib_table[self.calib_version][collection_name][calib_key]

    instance = None

    def __new__(cls):
        if not CalibManager.instance:
            CalibManager.instance = CalibManager.__TheManager()
        return CalibManager.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


tpg_layer_calib_v8 = [0.0,
                      0.0183664,
                      0.,
                      0.0305622,
                      0.,
                      0.0162589,
                      0.,
                      0.0143918,
                      0.,
                      0.0134475,
                      0.,
                      0.0185754,
                      0.,
                      0.0204934,
                      0.,
                      0.016901,
                      0.,
                      0.0207958,
                      0.,
                      0.0167985,
                      0.,
                      0.0238385,
                      0.,
                      0.0301146,
                      0.,
                      0.0274622,
                      0.,
                      0.0468671,
                      0.,
                      0.078819,
                      0.0555831,
                      0.0609312,
                      0.0610768,
                      0.0657626,
                      0.0465653,
                      0.0629383,
                      0.0610061,
                      0.0517326,
                      0.0492882,
                      0.0699336,
                      0.0673457,
                      0.119896,
                      0.125327,
                      0.143235,
                      0.153295,
                      0.104777,
                      0.109345,
                      0.161386,
                      0.174656,
                      0.108053,
                      0.121674,
                      0.1171,
                      0.328053]

dEdX_weights_v8 = [0.0,   # there is no layer zero
                   8.603,  # Mev
                   8.0675,
                   8.0675,
                   8.0675,
                   8.0675,
                   8.0675,
                   8.0675,
                   8.0675,
                   8.0675,
                   8.9515,
                   10.135,
                   10.135,
                   10.135,
                   10.135,
                   10.135,
                   10.135,
                   10.135,
                   10.135,
                   10.135,
                   11.682,
                   13.654,
                   13.654,
                   13.654,
                   13.654,
                   13.654,
                   13.654,
                   13.654,
                   38.2005,
                   55.0265,
                   49.871,
                   49.871,
                   49.871,
                   49.871,
                   49.871,
                   49.871,
                   49.871,
                   49.871,
                   49.871,
                   49.871,
                   62.005,
                   83.1675,
                   92.196,
                   92.196,
                   92.196,
                   92.196,
                   92.196,
                   92.196,
                   92.196,
                   92.196,
                   92.196,
                   92.196,
                   46.098]


dEdX_weights_v9 = [0.0,      # there is no layer zero
                   8.366557,  # Mev
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   10.425456,
                   31.497849,
                   51.205434,
                   52.030486,
                   52.030486,
                   52.030486,
                   52.030486,
                   52.030486,
                   52.030486,
                   52.030486,
                   52.030486,
                   52.030486,
                   52.030486,
                   71.265149,
                   90.499812,
                   90.894274,
                   90.537470,
                   89.786205,
                   89.786205,
                   89.786205,
                   89.786205,
                   89.786205,
                   89.786205,
                   89.786205,
                   89.786205,
                   89.786205]


def compute_tpg_weights(weights):
    tpg_weights = []
    for lid, lw in enumerate(weights):
        if lid > 29:
            tpg_weights.append(lw)
        else:
            if (lid % 2) == 1:
                tpg_weights.append(lw+weights[lid-1])
            else:
                tpg_weights.append(0)
    return tpg_weights


tpg_dEdx_weights_v8 = compute_tpg_weights(dEdX_weights_v8)
tpg_dEdx_weights_v9 = compute_tpg_weights(dEdX_weights_v9)

thickness_s200_v8 = 1.092
thickness_s200_v9 = 0.76


def compute_kfact(layer_calib, dedx_weight, thick_corr):
    ret = []
    for lid, calib in enumerate(layer_calib):
        if dedx_weight[lid] != 0:
            ret.append(calib/(dedx_weight[lid]*0.001/thick_corr))
        else:
            ret.append(0.)
    return ret


kfact_v8 = compute_kfact(tpg_layer_calib_v8, tpg_dEdx_weights_v8, thickness_s200_v8)


def get_layer_pt(cl2d, clweights=[1.]*53):
    return cl2d.pt*clweights[cl2d.layer]


def get_layer_pt_lcl(cl2d, clweights=tpg_layer_calib_v8):
    return cl2d.mipPt*clweights[cl2d.layer]


def get_layer_pt_dedx(cl2d, clweights=tpg_dEdx_weights_v8):
    return cl2d.mipPt*clweights[cl2d.layer]*0.001/1.092


def get_layer_pt_calibv9(cl2d, clweights=tpg_dEdx_weights_v9):
    return cl2d.mipPt*clweights[cl2d.layer]*0.001*kfact_v8[cl2d.layer]/thickness_s200_v9


def get_component_pt(cl3d, cl2ds, clweights=[1.]*53):
    # print cl3d
    # print cl3d.clusters
    # print cl2ds.id.isin(cl3d.clusters)
    # print cl2ds
    components = cl2ds[cl2ds.id.isin(cl3d.clusters)]
    components['ptcalib'] = components.apply(lambda x: get_layer_pt(x, clweights), axis=1)

    return components.ptcalib.sum()


def get_component_pt_lcl(cl3d, cl2ds):
    # print cl3d
    # print cl3d.clusters
    # print cl2ds.id.isin(cl3d.clusters)
    # print cl2ds
    components = cl2ds[cl2ds.id.isin(cl3d.clusters)]
    components['ptcalib_lc'] = components.apply(lambda x: get_layer_pt_lcl(x), axis=1)

    return components.ptcalib_lc.sum()


def get_component_pt_dedx(cl3d, cl2ds):
    # print cl3d
    # print cl3d.clusters
    # print cl2ds.id.isin(cl3d.clusters)
    # print cl2ds
    components = cl2ds[cl2ds.id.isin(cl3d.clusters)]
    components['ptcalib_dedx'] = components.apply(lambda x: get_layer_pt_dedx(x), axis=1)

    return components.ptcalib_dedx.sum()


def get_component_pt_kfact(cl3d, cl2ds):
    # print cl3d
    # print cl3d.clusters
    # print cl2ds.id.isin(cl3d.clusters)
    # print cl2ds
    components = cl2ds[cl2ds.id.isin(cl3d.clusters)]
    components['ptcalib_kfact'] = components.apply(lambda x: get_layer_pt(x, clweights=kfact_v8), axis=1)

    return components.ptcalib_kfact.sum()


def get_component_pt_v9calib(cl3d, cl2ds):
    # print cl3d
    # print cl3d.clusters
    # print cl2ds.id.isin(cl3d.clusters)
    # print cl2ds
    components = cl2ds[cl2ds.id.isin(cl3d.clusters)]
    components['ptcalib_v9calib'] = components.apply(lambda x: get_layer_pt_calibv9(x), axis=1)

    return components.ptcalib_v9calib.sum()



calib_factors = None


def get_calib_factors():

    global calib_factors
    if calib_factors is None:
        calibration_file_name = 'data/calib_v2.json'
        calib_factors = pd.read_json(calibration_file_name, dtype={'calib': np.float64,
                                                                   'eta_h': np.float64,
                                                                   'eta_l': np.float64,
                                                                   'pt_h': np.float64,
                                                                   'pt_l': np.float64})
    return calib_factors
# get_layer_pt_calibv9
