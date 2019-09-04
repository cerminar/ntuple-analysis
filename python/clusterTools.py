import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import math


def buildTriggerTowerCluster(allTowers, seedTower, debug):
    eta_seed = seedTower.eta.values[0]
    iEta_seed = seedTower.iEta.values[0]
    iPhi_seed = seedTower.iPhi.values[0]
    clusterTowers = allTowers[(allTowers.eta*eta_seed > 0)
                              & (allTowers.iEta <= (iEta_seed + 1))
                              & (allTowers.iEta >= (iEta_seed - 1))
                              & (allTowers.iPhi <= (iPhi_seed + 1))
                              & (allTowers.iPhi >= (iPhi_seed - 1))]
    clusterTowers.loc[clusterTowers.index, 'logEnergy'] = np.log(clusterTowers.energy)
    if debug >= 5:
        print '---- SEED:'
        print seedTower
        print 'Cluster components:'
        print clusterTowers
    ret = pd.DataFrame(columns=['energy', 'eta', 'phi', 'pt'])
    ret['energy'] = [clusterTowers.energy.sum()]
    ret['logEnergy'] = np.log(ret.energy)
    ret['eta'] = [np.sum(clusterTowers.eta*clusterTowers.energy)/ret.energy.values[0]]
    ret['phi'] = [np.sum(clusterTowers.phi*clusterTowers.energy)/ret.energy.values[0]]
    ret['etalw'] = [np.sum(clusterTowers.eta*clusterTowers.logEnergy)/np.sum(clusterTowers.logEnergy)]
    ret['philw'] = [np.sum(clusterTowers.phi*clusterTowers.logEnergy)/np.sum(clusterTowers.logEnergy)]
    ret['pt'] = [(ret.energy / np.cosh(ret.eta)).values[0]]
    return ret


def buildDBSCANClustersUnpack(arg):
    # print arg[2].loc[:10]
    return buildDBSCANClusters(sel_layer=arg[0], sel_zside=arg[1], tcs=arg[2])


def buildDBSCANClusters(sel_layer, sel_zside, tcs):
    new2Dcls = pd.DataFrame()

    tcs_layer = tcs[(tcs.layer == sel_layer) & (tcs.zside == sel_zside)].copy(deep=True)
    if tcs_layer.empty:
        return new2Dcls

    X = tcs_layer[['x', 'y']]
    # tuned on 25GeV e-
    densities = [0.05, 0.05, 0.1, 0.25, 0.3, 0.3, 0.5, 0.45, 0.4, 0.35, 0.4, 0.25, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15]

    # densities = [0.1, 0.2, 0.5, 0.5, 1.1, 1.3, 1.7, 1.8, 2.0, 2.2, 2.6, 2.0, 1.8, 1.4, 1.2, 0.8, 0.6, 0.4, 0.2, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    densities.extend([0.05 for i in range(0, 40)])
    # print 'ECCOCI: {}'.format(len(densities))
    # photon Pt35 tunes (no selection on unconverted)
    # densities = [0.05, 0.05, 0.05, 0.1, 0.25, 0.45, 1.1, 1.6, 2.5, 3.55, 4.85, 4.6, 4.35, 3.55, 3.15, 2.25, 1.8, 1.05, 1.0, 0.65, 0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.2]
    db = DBSCAN(eps=2.5,
                min_samples=densities[sel_layer-1]*100,
                algorithm='kd_tree',
                n_jobs=3).fit(X, sample_weight=tcs_layer['energy']*100)
    labels = db.labels_
    unique_labels = set(labels)
    tcs_layer['dbs_label'] = labels
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    tcs_layer['core'] = core_samples_mask
    # print db.core_sample_indices_
    # print tcs_layer

    for label in unique_labels:
        if label == -1:
            continue
        components = tcs_layer[tcs_layer.dbs_label == label]
        cl = build2D(components)
        new2Dcls = new2Dcls.append(cl.copy(), ignore_index=True)
    return new2Dcls


def buildHDBSCANClustersUnpack(arg):
    # print arg[2].loc[:10]
    return buildHDBSCANClusters(sel_layer=arg[0], sel_zside=arg[1], tcs=arg[2])


def buildHDBSCANClusters(sel_layer, sel_zside, tcs):
    new2Dcls = pd.DataFrame()

    tcs_layer = tcs[(tcs.layer == sel_layer) & (tcs.zside == sel_zside)].copy(deep=True)
    if tcs_layer.empty:
        return new2Dcls

    X = tcs_layer[['x', 'y']]
    # tuned on 25GeV e-
    densities = [0.05, 0.05, 0.1, 0.25, 0.3, 0.3, 0.5, 0.45, 0.4, 0.35, 0.4, 0.25, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.15]

    # densities = [0.1, 0.2, 0.5, 0.5, 1.1, 1.3, 1.7, 1.8, 2.0, 2.2, 2.6, 2.0, 1.8, 1.4, 1.2, 0.8, 0.6, 0.4, 0.2, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    # photon Pt35 tunes (no selection on unconverted)
    # densities = [0.05, 0.05, 0.05, 0.1, 0.25, 0.45, 1.1, 1.6, 2.5, 3.55, 4.85, 4.6, 4.35, 3.55, 3.15, 2.25, 1.8, 1.05, 1.0, 0.65, 0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.05, 0.2]
    db = DBSCAN(eps=2.5,
                min_samples=densities[sel_layer-1]*100,
                algorithm='kd_tree',
                n_jobs=3).fit(X, sample_weight=tcs_layer['energy']*100)
    labels = db.labels_
    unique_labels = set(labels)
    tcs_layer['dbs_label'] = labels
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    tcs_layer['core'] = core_samples_mask
    # print db.core_sample_indices_
    # print tcs_layer

    for label in unique_labels:
        if label == -1:
            continue
        components = tcs_layer[tcs_layer.dbs_label == label]
        cl = build2D(components)
        new2Dcls = new2Dcls.append(cl.copy(), ignore_index=True, sort=False)
    return new2Dcls


def build3DClustersEtaPhi(cl2D):
    X = cl2D[['eta', 'phi']]
    db = DBSCAN(eps=0.03,  # 0.03
                algorithm='kd_tree',
                min_samples=3,
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
        components = cl2D[cl2D.dbs_labels == label]
        cl3D = build3D(components)
        new3DCls = new3DCls.append(cl3D, ignore_index=True, sort=False)
    return new3DCls


def build3DClustersEtaPhi2(cl2D):
    new3DCls = pd.DataFrame()

    if cl2D.empty:
        return new3DCls
    X = cl2D[['eta', 'phi']]
    db = DBSCAN(eps=0.015,  # 0.03
                algorithm='kd_tree',
                min_samples=3,
                n_jobs=3).fit(X)
    labels = db.labels_
    unique_labels = set(labels)
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print '# of 3D clusters: {}'.format(n_clusters_ )
    cl2D['dbs_labels'] = labels
    for label in unique_labels:
        if label == -1:
            continue
        # print tcs_layer[tcs_layer.dbs_label == label].indexsi s
        components = cl2D[cl2D.dbs_labels == label]

        cl3D = build3D(components)
        new3DCls = new3DCls.append(cl3D, ignore_index=True, sort=False)
    return new3DCls


def build3DClustersProj(cl2D):
    cl2D['projx'] = cl2D.x/cl2D.z
    cl2D['projy'] = cl2D.y/cl2D.z

    X = cl2D[['projx', 'projy']]
    db = DBSCAN(eps=0.005,  # 0.03
                algorithm='kd_tree',
                min_samples=3,
                n_jobs=3).fit(X)
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
        components = cl2D[cl2D.dbs_labels == label]

        cl3D = build3D(components)
        new3DCls = new3DCls.append(cl3D, ignore_index=True, sort=False)
    return new3DCls


def build3DClustersProjTowers(cl2D):
    new3Dclusters = pd.DataFrame()
    for zside in [-1, 1]:
        seeds = getClusterSeeds(cl2D[(cl2D.eta*zside > 0)])
        for seed in seeds:
            # print seed
            components = getClusterComponents(seed, cl2D[(cl2D.eta*zside > 0)])
            new3Dclusters = new3Dclusters.append(build3D(components), ignore_index=True, sort=False)
    return new3Dclusters


def getClusterSeeds(triggerClusters):
    rod_bin_seeds = []
    rod_sums = triggerClusters.groupby('rod_bin_max').sum().sort_values(by='energy', ascending=False)
    all_cluster_rod_bins = []
    filtered_rod_sums = rod_sums[~rod_sums.index.isin(all_cluster_rod_bins)]
    while(not filtered_rod_sums.empty):

        rod_bin_seed = filtered_rod_sums.iloc[0].name
        rod_bin_seeds.append(rod_bin_seed)
        cluster_rod_bins = [(x, y)
                            for x in range(rod_bin_seed[0]-1, rod_bin_seed[0]+2)
                            for y in range(rod_bin_seed[1]-1, rod_bin_seed[1]+2)]
        all_cluster_rod_bins.extend(cluster_rod_bins)
        filtered_rod_sums = rod_sums[~rod_sums.index.isin(all_cluster_rod_bins)]

    return rod_bin_seeds


def getClusterComponents(rod_bin_seed, triggerClusters):
    cluster_rod_bins = [(x, y)
                        for x in range(rod_bin_seed[0]-1, rod_bin_seed[0]+2)
                        for y in range(rod_bin_seed[1]-1, rod_bin_seed[1]+2)]
    # print rod_bin_seed
    # print cluster_rod_bins

    return triggerClusters[(triggerClusters.rod_bin_max.isin(cluster_rod_bins))]


def build2D(components):
    cl = pd.DataFrame()
    cl['energy'] = [components.energy.sum()]

    # cl['energyCore'] = [components[components.core].energy.sum()]
    cl['x'] = [np.sum(components.x*components.energy)/components.energy.sum()]
    cl['y'] = [np.sum(components.y*components.energy)/components.energy.sum()]
    cl['z'] = [components.z.iloc[0]]
    cl['layer'] = [components.layer.iloc[0]]
    cl['zside'] = [components.zside.iloc[0]]
    cl['eta'] = [math.asinh(cl.z/math.sqrt(cl.x**2+cl.y**2))]
    cl['pt'] = [(cl.energy/np.cosh(cl.eta)).values[0]]
    cl['subdet'] = components.iloc[0].subdet
    # if cl.x.item() > 0:
    cl['phi'] = [math.atan2(cl.y, cl.x)]
    # elif cl.x.item() < 0:
    #     cl['phi'] = [math.pi - math.asin(cl.y/math.sqrt(cl.x**2+cl.y**2))]
    # else:
    #     cl['phi'] = [0]
    cl['cells'] = [np.array(components.id)]
    cl['ncells'] = [components.shape[0]]
    # cl['nCoreCells'] = [components[components.core].shape[0]]
    cl['id'] = components.iloc[0].id
    return cl


def build3D(components, calib_factor=1.084):
    cl3D = pd.DataFrame(columns=['energy', 'eta', 'phi', 'pt',
                                 'layers', 'clusters', 'nclu', 'firstlayer',
                                 'showerlength', 'seetot', 'seemax', 'spptot',
                                 'sppmax', 'szz', 'emaxe', 'id', 'eem', 'ehad',
                                 'hoe', 'ptem'])
    tot_energy = np.sum(components.energy)
    cl3D['energy'] = [tot_energy*calib_factor]
#     cl3D['energyCore'] = [components.energyCore.sum()*calib_factor]
#     cl3D['energyCentral'] = [components[(components.layer > 9) & (components.layer < 21)].energy.sum()*calib_factor]

    # print components
    # cl3D['PT'] = components.momentum.sum().Pt()
    #
    # cl3D['E'] = components.momentum.sum().E()
    cl3D['eta'] = [np.sum(components.eta*components.energy)/tot_energy]
    cl3D['phi'] = [np.sum(components.phi*components.energy)/tot_energy]
    # print cl3D.energy/np.cosh(cl3D.eta)
    # print type(cl3D.energy/np.cosh(cl3D.eta))
    cl3D['pt'] = [(cl3D.energy/np.cosh(cl3D.eta)).values[0]]
    # cl3D['ptCore'] = [(cl3D.energyCore/np.cosh(cl3D.eta)).values[0]]
    cl3D['layers'] = [components.layer.unique()]
    cl3D['clusters'] = [components.id.values]
    cl3D['nclu'] = [components.shape[0]]
    cl3D['firstlayer'] = [np.min(components.layer.values)]
    cl3D['eem'] = [np.sum(components[components.layer <= 28].energy)*calib_factor]
    cl3D['ehad'] = [np.sum(components[components.layer > 28].energy)*calib_factor]
    cl3D['hoe'] = cl3D.ehad/cl3D.eem
    cl3D['ptem'] = cl3D.pt/(1+cl3D.hoe)
    # FIXME: placeholder
    cl3D['showerlength'] = [1]
    cl3D['seetot'] = [1]
    cl3D['seemax'] = [1]
    cl3D['spptot'] = [1]
    cl3D['sppmax'] = [1]
    cl3D['szz'] = [1]
    cl3D['emaxe'] = [1]
    cl3D['id'] = components.iloc[0].id
    # print cl3D
    return cl3D


def computeClusterRodSharing(cl2ds, tcs):
    cl2ds['rod_bin_max'] = pd.Series(index=cl2ds.index, dtype=object)
    cl2ds['rod_bin_shares'] = pd.Series(index=cl2ds.index, dtype=object)
    cl2ds['rod_bins'] = pd.Series(index=cl2ds.index, dtype=object)

    for index, cl2d in cl2ds.iterrows():
        # print "---------------------"
        # print cl2d
        matchedTriggerCells = tcs[tcs.id.isin(cl2d.cells)]
        # print matchedTriggerCells
        energy_sums_byRod = matchedTriggerCells.groupby(by='rod_bin', axis=0).sum()
        bin_max = energy_sums_byRod[['energy']].idxmax()[0]
        cl2ds.set_value(index, 'rod_bin_max', bin_max)
        cl2ds.set_value(index, 'rod_bins', energy_sums_byRod.index.values)

        shares = []
        for iy in range(bin_max[1]-1, bin_max[1]+2):
            for ix in range(bin_max[0]-1, bin_max[0]+2):
                bin = (ix, iy)
                energy = 0.
                if bin in energy_sums_byRod.index:
                    energy = energy_sums_byRod.loc[[bin]].energy[0]
                shares.append(energy)
        cl2ds.set_value(index, 'rod_bin_shares', shares)


def merge3DClustersEtaPhi(cl3ds):
    new3DCls = pd.DataFrame()
    if cl3ds.empty:
        return new3DCls
    X = cl3ds[['eta', 'phi']]
    db = DBSCAN(eps=0.15,  # 0.03
                algorithm='kd_tree',
                min_samples=1,
                n_jobs=3).fit(X)
    labels = db.labels_
    unique_labels = set(labels)
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print '# of 3D clusters: {}'.format(n_clusters_ )
    cl3ds['dbs_labels'] = labels
    # print '======== CULSTER'
    # print cl3ds[['id', 'energy', 'dbs_labels']]
    for label in unique_labels:
        if label == -1:
            continue
        # print tcs_layer[tcs_layer.dbs_label == label].indexsi s
        components = cl3ds[cl3ds.dbs_labels == label]
        if components.shape[0] == 1:
            new3DCls = new3DCls.append(components.iloc[0], ignore_index=True, sort=False)
        else:
            refeta, refphi = components.sort_values(by='pt', ascending=False).iloc[0][['eta', 'phi']]
            # print '--- cluster components:'
            # print components.sort_values(by='pt', ascending=False)
            components_etaphi_ok = components[(np.fabs(components.eta - refeta) < 0.02) & (np.fabs(components.phi - refphi) < 0.1)]
            # print '--- selected components:'
            # print components_etaphi_ok
            cl3D = sum3DClusters(components_etaphi_ok)
            # print '--- summed:'
            # print cl3D
            new3DCls = new3DCls.append(cl3D, ignore_index=True, sort=False)
            components_etaphi_no = components[~((np.fabs(components.eta - refeta) < 0.02) & (np.fabs(components.phi - refphi) < 0.1))]
            if not components_etaphi_no.empty:
                new3DCls = new3DCls.append(components_etaphi_no, ignore_index=True, sort=False)
    return new3DCls


def sum3DClusters(components):
    ret = pd.DataFrame(columns=components.columns)

    tot_energy = np.sum(components.energy)
    ret['energy'] = [tot_energy]
    ret['eem'] = [np.sum(components.eem)]

    # ret['energyCore'] = [components.energy.sum()]
    # ret['energyCentral'] = [components.energy.sum()]

    ret['eta'] = [np.sum(components.eta*components.energy)/tot_energy]
    ret['phi'] = [np.sum(components.phi*components.energy)/tot_energy]
    ret['pt'] = [(ret.energy/np.cosh(ret.eta)).values[0]]
    ret['ptem'] = [(ret.eem/np.cosh(ret.eta)).values[0]]

    # ret['ptCore'] = [(ret.energyCore/np.cosh(ret.eta)).values[0]]
    # ret['layers'] = [np.unique(np.concatenate(components.layers.values))]
    ret['clusters'] = [np.concatenate(components.clusters.values)]
    ret['nclu'] = [np.sum(components.nclu)]
    ret['firstlayer'] = [np.min(components.firstlayer.values)]
    # FIXME: placeholder
    ret['showerlength'] = [1]
    ret['seetot'] = [1]
    ret['seemax'] = [1]
    ret['spptot'] = [1]
    ret['sppmax'] = [1]
    ret['szz'] = [1]
    ret['emaxe'] = [1]
    ret['id'] = [components.id.values[0]]
    ret['n010'] = len(components[components.pt > 0.1])
    ret['n025'] = len(components[components.pt > 0.25])
    ret['quality'] = [np.max(components.quality)]
    ret['bdt_out'] = [0]
    # print '-------- merged:'
    # print components.sort_values(by='pt', ascending=False)
    # print '   - merged sum:'
    # print ret
    return ret


def get_cylind_clusters(cl3ds, tcs, cylind_size=[3]*28):
    ret = cl3ds.copy(deep=True)
    em_layers = range(1, 29, 2)

    if not ret.empty:
        ret.loc[ret.index, 'sinh_eta'] = np.sinh(ret.eta)
        ret.loc[ret.index, 'cos_phi'] = np.cos(ret.phi)
        ret.loc[ret.index, 'sin_phi'] = np.sin(ret.phi)

    def compute_dr_cluster(cluster, tcs, cylind_size):
        components = tcs[tcs.id.isin(cluster.clusters)].copy()
        components['R_cl'] = components.z/cluster.sinh_eta
        components['x_cl'] = components.R_cl*cluster.cos_phi
        components['y_cl'] = components.R_cl*cluster.sin_phi
        components['dist2'] = (components.x_cl-components.x)**2+(components.y_cl-components.y)**2
        # components['dist'] = np.sqrt(components.dist2)
        # components['dr'] = components.dist/np.abs(components.z)

        def assign_size(comp):
            comp['clsize'] = cylind_size[comp.layer - 1]
            return comp

        components = components.apply(assign_size, axis=1)
        selected_components = components[components.dist2 < components.clsize**2]

        new_cluster = build3D(selected_components, calib_factor=1.).iloc[0]
        cluster['energy'] = new_cluster.energy
        cluster['eta'] = new_cluster.eta
        cluster['phi'] = new_cluster.phi
        cluster['pt'] = new_cluster.pt
        cluster['layers'] = new_cluster.layers
        cluster['clusters'] = new_cluster.clusters
        cluster['nclu'] = new_cluster.nclu
        cluster['firstlayer'] = new_cluster.firstlayer
        cluster['eem'] = new_cluster.eem
        cluster['ehad'] = new_cluster.ehad
        cluster['hoe'] = new_cluster.hoe
        cluster['ptem'] = new_cluster.ptem

        cluster['layer_energy'] = [[np.sum(selected_components[selected_components.layer == layer].energy) for layer in em_layers]]
        return cluster



    ret = ret.apply(lambda cl3d : compute_dr_cluster(cl3d, tcs, cylind_size), axis=1)

    return ret


def get_cylind_clusters_unpack(clusters_tcs_cylsize):
    # print clusters_tcs_cylsize
    return get_cylind_clusters(clusters_tcs_cylsize[0], clusters_tcs_cylsize[1], clusters_tcs_cylsize[2])


def get_dr_clusters2(cl3ds, tcs, cylind_size=[3]*28):
    # def fill_momentum(cluster):
    #     vector = ROOT.TLorentzVector()
    #     vector.SetPtEtaPhiE(cluster.pt, cluster.eta, cluster.phi, cluster.energy)
    #     cluster.momentum = vector
    #     # print tower.pt, tower.momentum.Pt()
    #     return cluster

    # columns = cl3ds.columns.values
    # columns = np.append(columns, ['E', 'PT'])
    # ret = pd.DataFrame(columns=cl3ds.columns)
    ret = cl3ds.copy(deep=True)
    em_layers = range(1, 29, 2)

    if not ret.empty:
        ret.loc[ret.index, 'sinh_eta'] = np.sinh(ret.eta)
        ret.loc[ret.index, 'cos_phi'] = np.cos(ret.phi)
        ret.loc[ret.index, 'sin_phi'] = np.sin(ret.phi)

    def compute_dr_cluster(cluster, tcs, cylind_size):
        components = tcs[tcs.id.isin(cluster.clusters)].copy()
        components['R_cl'] = components.z/cluster.sinh_eta
        components['x_cl'] = components.R_cl*cluster.cos_phi
        components['y_cl'] = components.R_cl*cluster.sin_phi
        components['dist2'] = (components.x_cl-components.x)**2+(components.y_cl-components.y)**2
        components['dist'] = np.sqrt(components.dist2)
        components['dr'] = components.dist/np.abs(components.z)

        def assign_size(comp):
            comp['clsize'] = cylind_size[comp.layer - 1]
            return comp

        components = components.apply(assign_size, axis=1)
        selected_components = components[components.dr < components.clsize]

        new_cluster = build3D(selected_components, calib_factor=1.).iloc[0]
        cluster['energy'] = new_cluster.energy
        cluster['eta'] = new_cluster.eta
        cluster['phi'] = new_cluster.phi
        cluster['pt'] = new_cluster.pt
        cluster['layers'] = new_cluster.layers
        cluster['clusters'] = new_cluster.clusters
        cluster['nclu'] = new_cluster.nclu
        cluster['firstlayer'] = new_cluster.firstlayer
        cluster['eem'] = new_cluster.eem
        cluster['ehad'] = new_cluster.ehad
        cluster['hoe'] = new_cluster.hoe
        cluster['ptem'] = new_cluster.ptem

        cluster['layer_energy'] = [[np.sum(selected_components[selected_components.layer == layer].energy) for layer in em_layers]]
        return cluster



    ret = ret.apply(lambda cl3d : compute_dr_cluster(cl3d, tcs, cylind_size), axis=1)

    return ret



def get_dr_clusters(cl3ds, tcs, cylind_size=[3]*28):
    # def fill_momentum(cluster):
    #     vector = ROOT.TLorentzVector()
    #     vector.SetPtEtaPhiE(cluster.pt, cluster.eta, cluster.phi, cluster.energy)
    #     cluster.momentum = vector
    #     # print tower.pt, tower.momentum.Pt()
    #     return cluster

    # columns = cl3ds.columns.values
    # columns = np.append(columns, ['E', 'PT'])
    ret = pd.DataFrame(columns=cl3ds.columns)
    em_layers = range(1, 29, 2)
    if not cl3ds.empty:
        cl3ds.loc[cl3ds.index, 'sinh_eta'] = np.sinh(cl3ds.eta)
        cl3ds.loc[cl3ds.index, 'cos_phi'] = np.cos(cl3ds.phi)
        cl3ds.loc[cl3ds.index, 'sin_phi'] = np.sin(cl3ds.phi)
        for index, cluster in cl3ds.iterrows():
            components = tcs[tcs.id.isin(cluster.clusters)].copy()
            components['R_cl'] = components.z/cluster.sinh_eta
            components['x_cl'] = components.R_cl*cluster.cos_phi
            components['y_cl'] = components.R_cl*cluster.sin_phi
            components['dist2'] = (components.x_cl-components.x)**2+(components.y_cl-components.y)**2
            components['dist'] = np.sqrt(components.dist2)
            components['dr'] = components.dist/np.abs(components.z)

            def assign_size(comp):
                comp['clsize'] = cylind_size[comp.layer - 1]
                return comp

            components = components.apply(assign_size, axis=1)

            # components['dist'] = np.sqrt(components.dist2)
            # components['momentum'] = ROOT.TLorentzVector()
            # components = components.apply(fill_momentum, axis=1)

            # components['dist_bool'] = components[((components.x1-components.x)**2+(components.y1-components.y)**2)<cylind_size**2]
            # print components[['id', 'layer', 'eta', 'phi', 'energy', 'x', 'y', 'R_cl', 'z', 'x_cl', 'y_cl', 'dist']].sort_values(by='layer', ascending=True)
            # print components[['id', 'layer', 'eta', 'phi', 'energy', 'dist', 'z', 'dr', 'clsize']].sort_values(by='layer', ascending=True)

            selected_components = components[components.dr < components.clsize]
            if selected_components.empty:
                continue
            new_cluster = build3D(selected_components, calib_factor=1.)
            # this is used for layer calibrations
            # FIXME: sume np.sum(...value) and measure speedup
            new_cluster['layer_energy'] = [[np.sum(selected_components[selected_components.layer == layer].energy) for layer in em_layers]]
            new_cluster['showerlength'] = cluster.showerlength
            new_cluster['seetot'] = cluster.seetot
            new_cluster['seemax'] = cluster.seemax
            new_cluster['spptot'] = cluster.spptot
            new_cluster['sppmax'] = cluster.sppmax
            new_cluster['szz'] = cluster.szz
            new_cluster['emaxe'] = cluster.emaxe
            new_cluster['quality'] = cluster.quality
            if 'hwQual' in cl3ds.columns:
                new_cluster['hwQual'] = cluster.hwQual
            ret = ret.append(new_cluster, ignore_index=True, sort=False)
            # ret['showerlength'] = cluster.showerlength
    # print cl3ds[['id', 'energy', 'pt', 'phi', 'eta', 'showerlength', 'nclu']]
    # print ret[['id', 'energy', 'pt', 'phi', 'eta', 'showerlength', 'nclu']]
    return ret


def get_dr_clusters_unpack2(clusters_tcs_cylsize):
    # print clusters_tcs_cylsize
    return get_dr_clusters2(clusters_tcs_cylsize[0], clusters_tcs_cylsize[1], clusters_tcs_cylsize[2])



def get_dr_clusters_unpack(clusters_tcs_cylsize):
    # print clusters_tcs_cylsize
    return get_dr_clusters(clusters_tcs_cylsize[0], clusters_tcs_cylsize[1], clusters_tcs_cylsize[2])
