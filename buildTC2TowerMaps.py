from NtupleDataFormat import HGCalNtuple
import pandas as pd
import numpy as np
import sys
from analyzeHgcalL1Tntuple import convertGeomTreeToDF
from multiprocessing import Pool
from python.utils import debugPrintOut


import ROOT
import os
import math
import copy
import socket
import datetime

import optparse
import yaml
import traceback

# import l1THistos as histos
# import utils as utils
# import clusterTools as clAlgo
# import traceback
# import optparse
# import ConfigParser
import hgcal_det_id as hgcdetid
# import hgcal_display as display


def analyze(params, batch_idx=0):
    print str(params)

    debug = int(params.debug)
    # pool = Pool(5)
    n_phi_bins = 72
    n_eta_bins = 18

    phi_bins = np.linspace(-1*math.pi, math.pi, n_phi_bins+1)
    eta_bins = np.linspace(1.479, 3.0, n_eta_bins+1)
    # eta_bins = np.linspace(1.41, 3.1, n_eta_bins+1)

    eta_bin_size = eta_bins[1] - eta_bins[0]
    eta_bin_first = eta_bins[0] + eta_bin_size/2
    phi_bin_size = phi_bins[1] - phi_bins[0]
    phi_bin_first = phi_bins[0] + phi_bin_size/2

    print '-- Eta bin size: {}, first bin center: {}, # bins: {}'.format(eta_bin_size, eta_bin_first, len(eta_bins)-1)
    print '    {}'.format(eta_bins)

    print '-- Phi bin size: {}, first bin center: {}, # bins: {}'.format(phi_bin_size, phi_bin_first, len(phi_bins)-1)
    print '    {}'.format(phi_bins)

    tc_geom_df = pd.DataFrame()
    cell_geom_df = pd.DataFrame()
    geom_file = os.path.join(params.input_base_dir, 'geom/test_triggergeom.root')

    # geom_file = params.input_base_dir+'/geom/test_triggergeom_v1.root'
    print 'Loading the geometry...'
    tc_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeTriggerCells')
    tc_geom_tree.setCache(learn_events=100)

    tc_geom_df = convertGeomTreeToDF(tc_geom_tree._tree)
    tc_geom_df['radius'] = np.sqrt(tc_geom_df['x']**2+tc_geom_df['y']**2)
    tc_geom_df['eta'] = np.arcsinh(tc_geom_df.z/tc_geom_df.radius)
    tc_geom_df['phi'] = np.arctan2(tc_geom_df.y, tc_geom_df.x)

    # cell_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeCells')
    # cell_geom_tree.setCache(learn_events=100)
    # cell_geom_df = convertGeomTreeToDF(cell_geom_tree._tree)
    #
    # bhcell_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeCellsBH')
    # bhcell_geom_tree.setCache(learn_events=100)
    # bhcell_geom_df = convertGeomTreeToDF(bhcell_geom_tree._tree)
    print '...done'

    # display_mgr = display.EventDisplayManager(cell_geom=cell_geom_df,
    #                                           trigger_cell_geom=tc_geom_tree)

    # tc_geom_df['cell'] = hgcdetid.v_cell(tc_geom_df.id)

    sel_cells = tc_geom_df[tc_geom_df.subdet.isin([3, 4]) & ~((tc_geom_df.subdet == 3) & (tc_geom_df.layer % 2 == 0))].copy()

    # some checks on the selection:
    #
    # print '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'
    # print sel_cells[(sel_cells.subdet == 3) & (sel_cells.layer % 2 != 0)]
    # print '#################################'
    # print sel_cells[(sel_cells.subdet == 3) & (sel_cells.layer % 2 == 0)]
    #
    print sel_cells.subdet.unique()
    print sel_cells[sel_cells.subdet == 3].layer.unique()

    sel_cells['wafertype'] = 1
    # FIXME: in the current implementation of the TriggerGeometry all TCs have wafertype = 1
    # here we need to know which modules have 120um thinckness to map them to the 6 HGCROC layouts
    # we crudely assume that this is true for EE and FH modules with radius < 70cm
    sel_cells['wafertype'] = hgcdetid.v_settype_on_radius(sel_cells['radius'])

    def map_wafertype_majority(data):
        # print data
        # counts = data['wafertype'].value_counts()
        # # if len(counts) > 1:
        # wafertype = counts.index[0]
        # data.loc[data.index, 'wafertype'] = wafertype
        data['wafertype'] = data['wafertype'].value_counts().index[0]
        return data

    # now we actually correct the wafertype for the rest of the module based on majority logic
    print 'Starting wafertype mapping on majority logic: {}'.format(datetime.datetime.now())
    sel_cells = sel_cells.groupby(['subdet', 'zside', 'layer', 'wafer']).apply(map_wafertype_majority)
    print '...done: {}'.format(datetime.datetime.now())

    # we now assign the hgcroc value
    # tc_geom_df['sector'] = hgcdetid.v_module_sector(tc_geom_df.id)
    # tc_geom_df['subcell'] = tc_geom_df.cell - tc_geom_df.sector*16
    sel_cells['hgcroc'] = hgcdetid.v_hgcroc_big(sel_cells.id)
    sel_cells.loc[sel_cells.wafertype == -1, ('hgcroc')] = hgcdetid.v_hgcroc_small(sel_cells.loc[sel_cells.wafertype == -1, ('id')])

    sel_cells['eta_bin'] = np.digitize(np.fabs(sel_cells.eta), eta_bins)-1
    # tc_geom_df['eta_bin_c'] = np.digitize(np.fabs(tc_geom_df.eta), eta_bins)-1
    sel_cells['phi_bin'] = np.digitize(sel_cells.phi, phi_bins)-1

    # deal with rounding effects on pi
    sel_cells.loc[sel_cells.phi_bin == n_phi_bins, ('phi_bin')] = n_phi_bins-1
    sel_cells.loc[sel_cells.phi_bin == -1, ('phi_bin')] = 0
    # deal with the fact that some of the cells hactually have eta outside the bin range
    sel_cells.loc[sel_cells.eta_bin == n_eta_bins, ('eta_bin')] = n_eta_bins-1
    sel_cells.loc[sel_cells.eta_bin == -1, ('eta_bin')] = 0

    tc_overflow = sel_cells[(sel_cells.eta_bin < 0) |
                             (sel_cells.eta_bin > n_eta_bins-1) |
                             (sel_cells.phi_bin < 0) |
                             (sel_cells.phi_bin > n_phi_bins-1)][['id', 'eta', 'phi', 'eta_bin', 'phi_bin']]
    if not tc_overflow.empty:
        print 'ERROR: some of the TCs have a bin outside the allowed range'
        print tc_overflow

    # This needs to be fixed after all the rounding has been take care of
    sel_cells['tt_bin'] = sel_cells.apply(func=lambda cell: (int(cell.eta_bin), int(cell.phi_bin)), axis=1)
    sel_cells['hgcroc_tt_bin'] = sel_cells['tt_bin']
    sel_cells['wafer_tt_bin'] = sel_cells['tt_bin']

    # temp_bins = pd.Series()

    # now we assign all hgcrocs or modules to the same tower on a majority logic on the TCs belonging to them
    def map_hgcroctt_majority(data):
        tt_bin = data['tt_bin'].value_counts().index[0]
        data['hgcroc_tt_bin'] = pd.Series([tt_bin for x in data.index],
                                          index=data.index)

        return data
        # counts = data['tt_bin'].value_counts()
        # if len(counts) > 1:
        #     tt_bin = counts.index[0]
        #     data.loc[data.index, 'hgcroc_tt_bin'] = pd.Series([tt_bin for x in data.index],
        #                                                       index=data.index)

    print 'Starting hgcroc mapping to TT on majority logic: {}'.format(datetime.datetime.now())
    sel_cells = sel_cells.groupby(['subdet', 'zside', 'layer', 'wafer', 'hgcroc']).apply(map_hgcroctt_majority)
    print '...done: {}'.format(datetime.datetime.now())

    def map_wafertt_majority(data):
        tt_bin = data['tt_bin'].value_counts().index[0]
        data['wafer_tt_bin'] = pd.Series([tt_bin for x in data.index],
                                         index=data.index)
        return data
        # counts = data['tt_bin'].value_counts()
        # if len(counts) > 1:
        #     tt_bin = counts.index[0]
        #     data.loc[data.index, 'hgcroc_tt_bin'] = pd.Series([tt_bin for x in data.index],
        #                                                       index=data.index)

    print 'Starting wafer mapping to TT on majority logic: {}'.format(datetime.datetime.now())
    sel_cells = sel_cells.groupby(['subdet', 'zside', 'layer', 'wafer']).apply(map_wafertt_majority)
    print '...done: {}'.format(datetime.datetime.now())

    def dump_mapping(tc_map, field, file_name):
        tower_tc_mapping = pd.DataFrame(columns=['id', 'towerbin_x', 'towerbin_y'], dtype=np.int64)

        tower_tc_mapping.id = tc_map.id
        tower_tc_mapping.towerbin_x = tc_map.apply(func=(lambda x: x[field][0]), axis=1)
        tower_tc_mapping.towerbin_y = tc_map.apply(func=(lambda x: x[field][1]), axis=1)

        tower_tc_mapping.to_csv(file_name, sep=' ', float_format='%.0f', header=False, index=False)
        return tower_tc_mapping

    hgcroc_sel_cells = dump_mapping(sel_cells, field='hgcroc_tt_bin', file_name='TCmapping_hgcroc_eta-phi_v3.txt')
    wafer_sel_cells = dump_mapping(sel_cells, field='wafer_tt_bin', file_name='TCmapping_wafer_eta-phi_v3.txt')

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

    print sel_cells[sel_cells.id.isin(missing)]
    print "# of TCs = {}".format(len(tc_geom_df.id.unique()))
    print "# of TCs mapped to TT (hgcroc) = {}".format(len(hgcroc_sel_cells.id.unique()))
    print "# of bins (hgcroc) = {}".format(len(sel_cells.hgcroc_tt_bin.unique()))
    print "# of TCs mapped to TT (wafer) = {}".format(len(wafer_sel_cells.id.unique()))
    print "# of bins (wafer) = {}".format(len(sel_cells.wafer_tt_bin.unique()))

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

    print '--------------------------------------------------------------------'
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

    # display_mgr.displayTriggerCells(event=1, tcs=tc_sel_p1)
    # display_mgr.show(event=1)

    sys.exit(0)


def main(analyze):

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)
    parser.add_option('-f', '--file', dest='CONFIGFILE', help='specify the ini configuration file')
    parser.add_option('-d', '--debug', dest='DEBUG', help='debug level (default is 0)', default=0)
    # parser.add_option('-n', '--nevents', dest='NEVENTS', help='# of events to process per sample (default is 10)', default=10)
    # parser.add_option("-b", "--batch", action="store_true", dest="BATCH", default=False, help="submit the jobs via CONDOR")
    # parser.add_option("-r", "--run", dest="RUN", default=None, help="the batch_id to run (need to be used with the option -b)")
    # parser.add_option("-o", "--outdir", dest="OUTDIR", default=None, help="override the output directory for the files")
    # parser.add_option("-i", "--inputJson", dest="INPUT", default='input.json', help="list of input files and properties in JSON format")

    global opt, args
    (opt, args) = parser.parse_args()

    # read the config file
    cfgfile = None
    with open(opt.CONFIGFILE, 'r') as stream:
        cfgfile = yaml.load(stream)

    class Params:
        def __init__(self):
            self.input_base_dir = None
            self.debug = None

        def __str__(self):
            return 'Params: \n \
                    input dir: {}\n \
                    debug: {}'.format(self.input_base_dir,
                                      self.debug)


    params = Params()
    params.input_base_dir = cfgfile['common']['input_dir']
    params.debug = opt.DEBUG
    analyze(params)


if __name__ == "__main__":
    try:
        main(analyze=analyze)
    except Exception as inst:
        print (str(inst))
        print ("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc()
        sys.exit(100)
