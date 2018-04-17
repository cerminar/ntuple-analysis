from NtupleDataFormat import HGCalNtuple, Event
import pandas as pd
import numpy as np
import sys
from analyzeHgcalL1Tntuple import convertGeomTreeToDF, Parameters, listFiles, Particle, PID, build3DClusters, main
from multiprocessing import Pool
from utils import debugPrintOut


import ROOT
import os
import math
import copy
import socket
import datetime

import l1THistos as histos
import utils as utils
import clusterTools as clAlgo
import traceback
import optparse
import ConfigParser
import hgcal_det_id as hgcdetid
import hgcal_display as display


def analyze(params, batch_idx=0):
    print (params)
    doAlternative = False

    debug = int(params.debug)
    pool = Pool(5)

    tc_geom_df = pd.DataFrame()
    cell_geom_df = pd.DataFrame()
    geom_file = params.input_base_dir+'/geom/test_triggergeom_v1.root'
    print 'Loading the geometry...'
    tc_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeTriggerCells')
    tc_geom_df = convertGeomTreeToDF(tc_geom_tree._tree)
    tc_geom_df['radius'] = np.sqrt(tc_geom_df['x']**2+tc_geom_df['y']**2)
    tc_geom_df['eta'] = np.arcsinh(tc_geom_df.z/tc_geom_df.radius)
    tc_geom_df['phi'] = np.arctan2(tc_geom_df.y, tc_geom_df.x)


    cell_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeCells')
    cell_geom_df = convertGeomTreeToDF(cell_geom_tree._tree)

    bhcell_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeCellsBH')
    bhcell_geom_df = convertGeomTreeToDF(bhcell_geom_tree._tree)
    print '...done'

    display_mgr = display.EventDisplayManager(cell_geom=cell_geom_df,
                                              trigger_cell_geom=tc_geom_tree)


    phi_bins = np.linspace(-1*math.pi, math.pi, 72+1)
    eta_bins = np.linspace(1.479, 3, 18+1)




    # tc_geom_df['cell'] = hgcdetid.v_cell(tc_geom_df.id)
    # FIXME: in the current implementation of the TriggerGeometry all TCs have wafertype = 1
    # here we need to know which modules have 120um thinckness to map them to the 6 HGCROC layouts
    # we crudely assume that this is true for EE and FH modules with radius < 70cm
    tc_geom_df['wafertype'] = 1
    tc_geom_df.loc[tc_geom_df.subdet != 5, 'wafertype'] = hgcdetid.v_settype_on_radius(tc_geom_df.loc[tc_geom_df.subdet != 5, 'radius'])

    # now we actually correct the wafertype for the rest of the module based on majority logic
    for subdet in tc_geom_df[tc_geom_df.wafertype == -1].subdet.unique():
        tc_subdet = tc_geom_df[(tc_geom_df.wafertype == -1) & (tc_geom_df.subdet == subdet)]
        for zside in tc_subdet.zside.unique():
            tc_zside = tc_subdet[tc_subdet.zside == zside]
            for layer in tc_zside.layer.unique():
                tc_layer = tc_zside[tc_zside.layer == layer]
                for wafer in tc_layer.wafer.unique():
                    # we want them all despite being of wafertype -1 or +1 hence we redo the selection
                    tc_wafer = tc_geom_df[(tc_geom_df.subdet == subdet) &
                                          (tc_geom_df.zside == zside) &
                                          (tc_geom_df.layer == layer) &
                                          (tc_geom_df.wafer == wafer)]
                    # we count the values and take the one with highest score
                    wafertype = tc_wafer['wafertype'].value_counts().index[0]
                    # we assign it to all the other TCs in the wafer
                    tc_geom_df.loc[tc_wafer.index, ('wafertype')] = wafertype

    # we now assign the hgcroc value
    # tc_geom_df['sector'] = hgcdetid.v_module_sector(tc_geom_df.id)
    # tc_geom_df['subcell'] = tc_geom_df.cell - tc_geom_df.sector*16
    tc_geom_df['hgcroc'] = hgcdetid.v_hgcroc_big(tc_geom_df.id)
    tc_geom_df.loc[tc_geom_df.wafertype == -1, ('hgcroc')] = hgcdetid.v_hgcroc_small(tc_geom_df.loc[tc_geom_df.wafertype == -1, ('id')])

    tc_geom_df['eta_bin'] = np.digitize(np.fabs(tc_geom_df.eta), eta_bins)-1
    # tc_geom_df['eta_bin_c'] = np.digitize(np.fabs(tc_geom_df.eta), eta_bins)-1
    tc_geom_df['phi_bin'] = np.digitize(tc_geom_df.phi, phi_bins)-1
    tc_geom_df['tt_bin'] = tc_geom_df.apply(func=lambda cell: (int(cell.eta_bin), int(cell.phi_bin)), axis=1)

    temp_bins = pd.Series()

    if False:
        for subdet in [3, 4]:
            tc_subdet = tc_geom_df[(tc_geom_df.subdet == subdet)]
            for zside in tc_subdet.zside.unique():
                tc_zside = tc_subdet[tc_subdet.zside == zside]
                if subdet == 3:
                    layerrange = range(1, 29, 2)
                else:
                    layerrange = range(1, 13)
                for layer in layerrange:
                    print 'subdet: {} zside: {} layer: {}'.format(subdet, zside, layer)
                    tc_layer = tc_zside[tc_zside.layer == layer]
                    for wafer in tc_layer.wafer.unique():
                        tc_wafer = tc_layer[tc_layer.wafer == wafer]
                        for hgcroc in tc_wafer.hgcroc.unique():
                            tc_hgcroc = tc_wafer[tc_wafer.hgcroc == hgcroc]
                            # print tc_hgcroc[['id', 'zside', 'subdet', 'layer', 'wafer', 'wafertype', 'triggercell', 'hgcroc', 'eta', 'phi', 'eta_bin', 'phi_bin', 'tt_bin']]
                            # print tc_hgcroc['tt_bin'].value_counts()
                            counts = tc_hgcroc['tt_bin'].value_counts()
                            if len(counts) > 1:
                                tt_bin = counts.index[0]
                            # for idx in tc_hgcroc.index:
                            #     temp_bins.set_value(idx, tt_bin)
                            #temp_bins.loc[tc_hgcroc.index, ('hgcroc_bin_x', 'hgcroc_bin_y')] = tt_bin
                                tc_geom_df.loc[tc_hgcroc.index, ('tt_bin')] = pd.Series([tt_bin for x in tc_hgcroc.index],
                                                                                    index=tc_hgcroc.index)
                            # print tc_geom_df.loc[tc_hgcroc.index][['id', 'zside', 'subdet', 'layer', 'wafer', 'wafertype', 'triggercell', 'hgcroc', 'eta', 'phi', 'eta_bin', 'phi_bin', 'tt_bin']]


    # print 'assigning'
    # tc_geom_df = tc_geom_df.assign(hgcroc_tt_bin=temp_bins)
    # print '...done'

    sel_cells = tc_geom_df[~((tc_geom_df.subdet == 3) & (tc_geom_df.layer % 2 == 0))]

    print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
    print sel_cells[(sel_cells.subdet == 3) & (sel_cells.layer % 2 != 0)]
    print '#################################'
    print sel_cells[(sel_cells.subdet == 3) & (sel_cells.layer % 2 == 0)]

    tower_tc_mapping = pd.DataFrame(columns=['id', 'towerbin_x', 'towerbin_y'], dtype=np.int64)

    tower_tc_mapping.id = sel_cells.id
    tower_tc_mapping.towerbin_x = sel_cells.apply(func=(lambda x: x.tt_bin[0]), axis=1)
    tower_tc_mapping.towerbin_y = sel_cells.apply(func=(lambda x: x.tt_bin[1]), axis=1)

    tower_tc_mapping.to_csv('TCmapping_hgcroc_eta-phi_v0.txt', sep=' ', float_format='%.0f', header=False, index=False)

    missing = np.array(['1780744193', '1780744194', '1780744195', '1780744196',
       '1780744197', '1780744198', '1780744199', '1780744200',
       '1780744201', '1780744202', '1780744203', '1780744204',
       '1780744205', '1780744206', '1780744207', '1780744208',
       '1780744209', '1780744210', '1780744211', '1780744212',
       '1780744213', '1780744214', '1780744215', '1780744216',
       '1780744217', '1780744218', '1780744219', '1780744220',
       '1780744221', '1780744222', '1780744223', '1780744224',
       '1780744225', '1780744226', '1780744227', '1780744228',
       '1780744229', '1780744230', '1780744231', '1780744232',
       '1780744233', '1780744234', '1780744235', '1780744236',
       '1780744237', '1780744238', '1780744239', '1780744240',
       '1780744241', '1780744242', '1780744243', '1780744244',
       '1780744245', '1780744246', '1780744247', '1780744248',
       '1780744249', '1780744250', '1780744251', '1780744252',
       '1780744253', '1780744254', '1780744255', '1780744256',
       '1780744257', '1780744258', '1780744259', '1780744260',
       '1780744261', '1780744262', '1780744263', '1780744264',
       '1797521409', '1797521410', '1797521411', '1797521412',
       '1797521413', '1797521414', '1797521415', '1797521416',
       '1797521417', '1797521418', '1797521419', '1797521420',
       '1797521421', '1797521422', '1797521423', '1797521424',
       '1797521425', '1797521426', '1797521427', '1797521428',
       '1797521429', '1797521430', '1797521431', '1797521432',
       '1797521433', '1797521434', '1797521435', '1797521436',
       '1797521437', '1797521438', '1797521439', '1797521440',
       '1797521441', '1797521442', '1797521443', '1797521444',
       '1797521445', '1797521446', '1797521447', '1797521448',
       '1797521449', '1797521450', '1797521451', '1797521452',
       '1797521453', '1797521454', '1797521455', '1797521456',
       '1797521457', '1797521458', '1797521459', '1797521460',
       '1797521461', '1797521462', '1797521463', '1797521464',
       '1797521465', '1797521466', '1797521467', '1797521468',
       '1797521469', '1797521470', '1797521471', '1797521472',
       '1797521473', '1797521474', '1797521475', '1797521476',
       '1797521477', '1797521478', '1797521479', '1797521480'],
      dtype='|S10')

    print tc_geom_df[tc_geom_df.id.isin(missing)]
    sys.exit(0)

    # tc_ids_all = pd.DataFrame(columns=['wf', 'wtf', 'hgcroc'])
    # results = []
    #
    #
    #
    #
    # for index, tc in tc_geom_df.iterrows():
    #     if index % 1000 == 0:
    #         print 'TC: {}'.format(index)
    #     detid = HGCalDetId(tc.id)
    #
    #     tc_ids = pd.DataFrame(columns=['wf', 'wtf', 'hgcroc'])
    #     tc_ids['wf'] = detid.wafer()
    #     tc_ids['wft'] = detid.waferType()
    #     tc_ids['hgcroc'] = detid.hgcroc()
    #     results.append(tc_ids)
    # tc_ids_all = pd.concatenate(results)
    #
    # for index, cell in cell_geom_df.iterrows():
    #     if index % 1000 == 0:
    #         print 'Cell: {}'.format(index)
    #     detid = HGCalDetId(cell.tc_id)
    #     cell['wf'] = detid.wafer()
    #     cell['wft'] = detid.waferType()
    #     cell['hgcroc'] = detid.hgcroc()

    cell_sel_type_p1 = cell_geom_df[(cell_geom_df.wafertype == 1) &
                                    (cell_geom_df.layer == 1) &
                                    (cell_geom_df.wafer == 180) &
                                    (cell_geom_df.zside == -1)]
    cell_sel_type_m1 = cell_geom_df[(cell_geom_df.wafertype == -1) &
                                    (cell_geom_df.layer == 1) &
                                    (cell_geom_df.wafer == 101) &
                                    (cell_geom_df.zside == -1)]
    tc_sel_p1 = tc_geom_df[(tc_geom_df.subdet == 3) &
                           (tc_geom_df.layer == 1) &
                           # (tc_geom_df.wafer == 101) &
                           (tc_geom_df.wafertype == -1) &
                           (tc_geom_df.zside == -1)]


    print '---------------------------------------------------------------------'
    print cell_sel_type_p1[['id', 'layer', 'subdet', 'zside', 'wafer', 'cell']]
    print '---------------------------------------------------------------------'
    print cell_sel_type_m1
    #
    #
    # tc_geom_df['wf'] = tc_geom_df.apply(compute_wafer, axis=1)
    # cell_geom_df['wf'] = cell_geom_df.apply(compute_wafer, axis=1)
    # tc_geom_df['wft'] = tc_geom_df.apply(compute_waferType, axis=1)
    # cell_geom_df['wft'] = cell_geom_df.apply(compute_waferType, axis=1)
    #
    print '---------------------------------------------------------------------'
    debugPrintOut(debug, 'Cell geometry',
                  toCount=cell_geom_df,
                  toPrint=cell_geom_df.iloc[:3])
    print '---------------------------------------------------------------------'
    debugPrintOut(debug, 'BH geometry',
                  toCount=bhcell_geom_df,
                  toPrint=bhcell_geom_df.iloc[:3])
    print '---------------------------------------------------------------------'
    debugPrintOut(debug, 'TC geometry',
                  toCount=tc_geom_df,
                  toPrint=tc_geom_df.iloc[:3])

    cell_sel_type_p1['color'] = cell_sel_type_p1.cell
    cell_sel_type_m1['color'] = cell_sel_type_m1.cell
    tc_sel_p1['energy'] = tc_sel_p1.hgcroc

    print '---------------------------------------------------------------------'
    print tc_sel_p1

    # display_mgr.displayCells(event=1, cells=cell_sel_type_p1)
    # display_mgr.displayCells(event=1, cells=cell_sel_type_m1)
    display_mgr.displayTriggerCells(event=1, tcs=tc_sel_p1)
    display_mgr.show(event=1)

    sys.exit(0)


if __name__ == "__main__":
    try:
        main(analyze=analyze)
    except Exception as inst:
        print (str(inst))
        print ("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc()
        sys.exit(100)
