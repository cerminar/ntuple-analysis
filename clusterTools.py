import pandas as pd
import numpy as np
import json
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import math


def buildDBSCANClustersUnpack(arg):
    return buildDBSCANClusters(sel_layer=arg[0], sel_zside=arg[1], tcs=arg[2])


def buildDBSCANClusters(sel_layer, sel_zside, tcs):
    tcs_layer = tcs[(tcs.layer == sel_layer) & (tcs.zside == sel_zside)]
    new2Dcls = pd.DataFrame()

    if len(tcs_layer['x']) == 0:
        return new2Dcls

    X = tcs_layer[['x', 'y']]
    densities = [1, 2, 4, 6, 10, 14, 18, 18, 18, 18, 20, 20, 20, 16, 10, 6, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    db = DBSCAN(eps=3.5,
                min_samples=densities[sel_layer-1],
                algorithm='kd_tree',
                n_jobs=3).fit(X, sample_weight=tcs_layer['energy']*10)
    labels = db.labels_
    unique_labels = set(labels)
    tcs_layer['dbs_label'] = labels

    for label in unique_labels:
        if label == -1:
            continue
        components = tcs_layer[tcs_layer.dbs_label == label]
        cl = pd.DataFrame()
        cl['energy'] = [components.energy.sum()]
        cl['x'] = [np.sum(components.x*components.energy)/components.energy.sum()]
        cl['y'] = [np.sum(components.y*components.energy)/components.energy.sum()]
        cl['z'] = [components.z.iloc[0]]
        cl['layer'] = [int(sel_layer)]
        cl['zside'] = [int(sel_zside)]
        cl['eta'] = [math.asinh(cl.z/math.sqrt(cl.x**2+cl.y**2))]
        if cl.x.item() > 0:
            cl['phi'] = [math.atan(cl.y/cl.x)]
        elif cl.x.item() < 0:
            cl['phi'] = [math.pi - math.asin(cl.y/math.sqrt(cl.x**2+cl.y**2))]
        else:
            cl['phi'] = [0]
        cl['cells'] = [np.array(components.index)]
        # FIXME: broken
        cl['ncells'] = [components.shape[0]]
        new2Dcls = new2Dcls.append(cl.copy(), ignore_index=True)
    return new2Dcls


def build3DClusters(cl2D):
    X = cl2D[['eta', 'phi']]
    db = DBSCAN(eps=0.015,
                algorithm='kd_tree',
                min_samples=10,
                n_jobs=3).fit(X, sample_weight=cl2D['energy'])
    labels = db.labels_
    unique_labels = set(labels)
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print '# of 3D clusters: {}'.format(n_clusters_ )
    cl2D['dbs_labels'] = labels
    new3DCls = pd.DataFrame()
    for label in unique_labels:
        if label == -1:
            continue
        # print tcs_layer[tcs_layer.dbs_label == label].indexsi s
        cl3D = pd.DataFrame()
        calib_factor = 1.084
        components = cl2D[cl2D.dbs_labels == label]
        cl3D['energy'] = [components.energy.sum()*calib_factor]
        # print components
        cl3D['eta'] = [np.sum(components.eta*components.energy)/components.energy.sum()]
        cl3D['phi'] = [np.sum(components.phi*components.energy)/components.energy.sum()]
        #print cl3D.energy/np.cosh(cl3D.eta)
        #print type(cl3D.energy/np.cosh(cl3D.eta))
        cl3D['pt'] = [(cl3D.energy/np.cosh(cl3D.eta)).values[0]]

        cl3D['layers'] = [components.layer.values]
        cl3D['clusters'] = [np.array(components.index)]
        # FIXME: broken
        cl3D['nclu'] = [components.shape[0]]
        cl3D['firstlayer'] = [np.min(components.layer.values)]
        # FIXME: placeholder
        cl3D['showerlength'] = [1]
        cl3D['seetot'] = [1]
        cl3D['seemax'] = [1]
        cl3D['spptot'] = [1]
        cl3D['sppmax'] = [1]
        cl3D['szz'] = [1]
        cl3D['emaxe'] = [1]
        # cl3D['color'] = [np.random.rand(3,)]

        # print cl3D
        new3DCls = new3DCls.append(cl3D, ignore_index=True)
    return new3DCls
