from NtupleDataFormat import HGCalNtuple, Event
import pandas as pd
import numpy as np
import sys
from analyzeHgcalL1Tntuple import convertGeomTreeToDF,Parameters,listFiles,Particle,PID,debugPrintOut,build3DClusters,main
from multiprocessing import Pool


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



from bokeh.io import show
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    LogColorMapper,
    LinearColorMapper,
    LogTicker,
    ColorBar
)

#from bokeh.palettes import Viridis6 as palette
from bokeh.palettes import RdYlBu11 as palette
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
from bokeh.palettes import PiYG11
from bokeh.palettes import Viridis256
from bokeh.palettes import Plasma256
import math


class EventDisplayManager:
    def __init__(self, cell_geom, trigger_cell_geom):
        self.cell_geom = cell_geom
        self.cell_geom['width'] = self.cell_geom.x2-self.cell_geom.x3
        k = math.tan(math.radians(30))
        self.cell_geom['delta'] = 0.5*self.cell_geom.width*k
        # self.cell_geom['corners_x'] = self.cell_geom.apply(func=self.cell_geom.apply(func = lambda cell: [cell.x1, cell.x2, cell.x2-cell.width*0.5,  cell.x3, cell.x4, cell.x4+cell.width*0.5],
        #                                                    axis=1))
        # self.cell_geom['corners_y'] = self.cell_geom.apply(func=self.cell_geom.apply(func = lambda cell: [cell.y1, cell.y2, cell.y2+cell.delta,  cell.y3, cell.y4, cell.y4-cell.delta],
        #                                                    axis=1))
        self.color_mapper = LinearColorMapper(palette=palette)#, low=0.1, high=30)

        self.trigger_cell_geom = trigger_cell_geom
        self.figures = {}
        self.ranges = {}

    def displayTowers(self, event, zside, subdet, layer, grid):
        all_corners_x = []
        all_corners_y = []
        figure = self.getFigure(event, zside, subdet, layer)
        print grid.nbins_x
        print grid.nbins_y
        for idx_x in range(0, grid.nbins_x):
            for idx_y in range(0, grid.nbins_y):
                corners = grid.getCorners(idx_x, idx_y)
                all_corners_x.append([corner.x for corner in corners])
                all_corners_y.append([corner.y for corner in corners])

        source = ColumnDataSource(data=dict(x=all_corners_x,
                                            y=all_corners_y,
                                            ))
        figure.patches('x', 'y', source=source,
                       fill_color=None,
                       fill_alpha=0.7,
                       line_color="black",
                       line_width=0.1)
        return


    def displayGenParticle(self, event, genParts):
        print genParts
        for pt_idx in genParts.index:
            genPart = genParts.loc[pt_idx]
            zside = -1
            if genPart.eta > 0:
                zside = 1
                continue
            for lid in range(0, len(genPart.posx), 2):
                if lid < 28:
                    subdet = 3
                else:
                    continue
                index = (event, zside, subdet, lid+1)
                figure = self.getFigure(event, zside, subdet, lid+1)
                figure.cross(x=genPart.posx[lid],
                             y=genPart.posy[lid],
                             size=20,
                             color="#E6550D",
                             line_width=1)
        # for
        # plot.cross(x=[1, 2, 3], y=[1, 2, 3], size=20,
        #    color="#E6550D", line_width=2)


    def displayCells(self, event, cells):
        debugPrintOut(4, 'cells', toCount=cells, toPrint=cells.loc[:3])
        for zside in cells.zside.unique():
            if zside > 0:
                continue
            cells_zside = cells[cells.zside == zside]
            for subdet in cells_zside.subdet.unique():
                if subdet != 3:
                    continue
                cells_subdet = cells_zside[cells_zside.subdet == subdet]
                for layer in cells_subdet.layer.unique():
                    cells_layer = cells_subdet[cells_subdet.layer == layer]
                    figure = self.getFigure(event, zside, subdet, layer)
                    all_corners_x = []
                    all_corners_y = []
                    all_cells_ids = []
                    for cell_idx in cells_layer.index:
                        cell = cells.loc[cell_idx]
                        all_corners_x.append([cell.x1, cell.x2, cell.x2-cell.width*0.5,  cell.x3, cell.x4, cell.x4+cell.width*0.5])
                        all_corners_y.append([cell.y1, cell.y2, cell.y2+cell.delta,  cell.y3, cell.y4, cell.y4-cell.delta])
                        all_cells_ids.append(cell.id)
                    print len(all_corners_x)
                    print len(all_corners_y)
                    print len(all_cells_ids)

                    source = ColumnDataSource(data=dict(x=all_corners_x,
                                                        y=all_corners_y,
                                                        id=all_cells_ids,
                                                        ))

                    figure.patches('x', 'y', source=source,
                                   fill_color={'field': 'id', 'transform': self.color_mapper},
                                   fill_alpha=0.7,
                                   line_color="black",
                                   line_width=0.1)



    def displayTriggerCells(self, event, tcs):
        for zside in tcs.zside.unique():
            if zside > 0:
                continue
            zside_tcs = tcs[tcs.zside == zside]
            for subdet in zside_tcs.subdet.unique():
                if subdet != 3:
                    continue
                subdet_tcs = zside_tcs[zside_tcs.subdet == subdet]
                for layer in subdet_tcs.layer.unique():
                    layer_tcs = subdet_tcs[subdet_tcs.layer == layer]
                    figure = self.getFigure(event, zside, subdet, layer)
                    all_corners_x = []
                    all_corners_y = []
                    all_energies = []
                    all_tc_ids = []
                    for index in layer_tcs.index:
                        tc = layer_tcs.loc[index]

                        cells = self.cell_geom[self.cell_geom.tc_id == tc.id]
                        corners_x = []
                        corners_y = []
                        for cell_idx in cells.index:
                            cell = cells.loc[cell_idx]
                            corners_x.append([cell.x1, cell.x2, cell.x2-cell.width*0.5,  cell.x3, cell.x4, cell.x4+cell.width*0.5])
                            corners_y.append([cell.y1, cell.y2, cell.y2+cell.delta,  cell.y3, cell.y4, cell.y4-cell.delta])
                        cell_energies = [tc.energy] * len(cells.index)
                        cell_tcids = [tc.id] * len(cells.index)
                        #print cell_energies
                        #print cell_tcids
                        all_corners_x.extend(corners_x)
                        all_corners_y.extend(corners_y)
                        all_energies.extend(cell_energies)
                        all_tc_ids.extend(cell_tcids)
                    source = ColumnDataSource(data=dict(x=all_corners_x,
                                                        y=all_corners_y,
                                                        id=all_tc_ids,
                                                        energy=all_energies,
                                                        ))

                    figure.patches('x', 'y', source=source,
                                   fill_color={'field': 'energy', 'transform': self.color_mapper},
                                   fill_alpha=0.7,
                                   line_color="black",
                                   line_width=0.1)

        #for tc in tcs:

        return

    def displayClusters(self, event, cl2ds, tcs):
        # print cl2ds
        for zside in [-1, 1]:
            if zside > 0:
                continue
            zside_cl2ds = cl2ds[cl2ds.eta*zside > 0]
            for subdet in zside_cl2ds.subdet.unique():
                if subdet != 3:
                    continue
                subdet_cl2ds = zside_cl2ds[zside_cl2ds.subdet == subdet]
                for layer in subdet_cl2ds.layer.unique():
                    layer_cl2ds = subdet_cl2ds[subdet_cl2ds.layer == layer]
                    figure = self.getFigure(event, zside, subdet, layer)
                    clus_ids = [str(clid) for clid in layer_cl2ds.sort_values(by=['pt'], ascending=False).id.values]

                    all_corners_x = []
                    all_corners_y = []
                    all_energies = []
                    all_tc_ids = []
                    all_clus_ids = []
                    print '======= layer: {}, # of 2D clusters: {}'.format(layer, len(layer_cl2ds.index))
                    for idx in layer_cl2ds.sort_values(by=['pt'], ascending=False).index:
                        # print '-------- CL2D ------------'
                        cl2d = layer_cl2ds.loc[idx]
                        print cl2d.pt
                        components = tcs[tcs.id.isin(cl2d.cells)]
                        # print '# of TCS: {}'.format(len(components.index))
                        for tc_idx in components.index:
                            tc = components.loc[tc_idx]
                            cells = self.cell_geom[self.cell_geom.tc_id == tc.id]
                            # print '# of cells: {}'.format(len(cells.index))
                            # corners_x = []
                            # corners_y = []
                            # cell_energies = []
                            # cell_tcids = []
                            #clus_id = cl2d.id*len(components.index)
                            for cell_idx in cells.index:
                                cell = cells.loc[cell_idx]
                                all_corners_x.append([cell.x1, cell.x2, cell.x2-cell.width*0.5,  cell.x3, cell.x4, cell.x4+cell.width*0.5])
                                all_corners_y.append([cell.y1, cell.y2, cell.y2+cell.delta,  cell.y3, cell.y4, cell.y4-cell.delta])
                                all_energies.append(tc.energy)
                                all_tc_ids.append(tc.id)
                                all_clus_ids.append(str(cl2d.id))
                                #print cell_energies
                                #print cell_tcids
                                # all_corners_x.extend(corners_x)
                                # all_corners_y.extend(corners_y)
                                # all_energies.extend(cell_energies)
                                # all_tc_ids.extend(cell_tcids)
                    # print len(all_corners_x)
                    # print len(all_corners_y)
                    # print len(all_energies)
                    # print len(all_tc_ids)

                    print '# of 2D cluster in layer {}'.format(len(all_clus_ids))
                    # print all_clus_ids
                    source = ColumnDataSource(data=dict(x=all_corners_x,
                                                        y=all_corners_y,
                                                        energy=all_energies,
                                                        id=all_tc_ids,
                                                        cl_id=all_clus_ids
                                                        ))
                    # print clus_ids

                    figure.patches('x', 'y', source=source,
                                   fill_color=None,
                                   fill_alpha=0.7,
                                   # line_color={'field': 'energy', 'transform': self.color_mapper},
                                   line_color=factor_cmap('cl_id', palette=Plasma256, factors=clus_ids),
                                   line_width=3.,
                                   # legend=clus_ids
                                   )

        return

    def createFigure(self, event, zside, subdet, layer):
        layer_range = range(1, 29, 2)
        # if subdet == 3:
        #     layer_range = range(1,29)
        #
        range_x = None
        range_y = None
        for layer in layer_range:
            index = (event, zside, subdet, layer)
            title = 'Event: {}, SubDet: {}, zside: {}, layer: {}'.format(event, subdet, zside, layer)
            TOOLS = "pan,wheel_zoom,reset,box_zoom,hover,save"
            if range_x is None:
                self.figures[index] = figure(title=title,
                                             tools=TOOLS,
                                             toolbar_location='right',
                                             x_axis_location='below',
                                             y_axis_location='left',
                                             x_range=[-170, +170],
                                             y_range=[-170, 170])
                range_x = self.figures[index].x_range
                range_y = self.figures[index].y_range
            else:
                self.figures[index] = figure(title=title,
                                             tools=TOOLS,
                                             toolbar_location='right',
                                             x_axis_location='below',
                                             y_axis_location='left',
                                             x_range=range_x,
                                             y_range=range_y)
            self.figures[index].grid.grid_line_color = None
            color_bar = ColorBar(color_mapper=self.color_mapper, ticker=LogTicker(),
                                 label_standoff=12, border_line_color=None, location=(0, 0))
            self.figures[index].add_layout(color_bar, 'right')

            hover = self.figures[index].select_one(HoverTool)
            hover.point_policy = "follow_mouse"
            hover.tooltips = [
                ("TC ID", "@id"),
                 #("energy", "@energy GeV"),
                 ("(x, y)", "($x, $y)"),
                 #("CL ID", "@cl_id")
             ]

    def getFigure(self, event, zside, subdet, layer):
        index = (event, zside, subdet, layer)
        if index not in self.figures.keys():
            self.createFigure(event, zside, subdet, layer)
        return self.figures[index]

    def show(self, event):
        plots_ee_m = []
        plots_ee_p = []
        zside = -1
        subdet = 3
        for layer in range(0,29):
            idx = (event, zside, subdet, layer)
            if idx in self.figures.keys():
                plots_ee_m.append(self.figures[idx])
        show(column(plots_ee_m))





class GridPoint:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.phi = math.atan2(y, x)
        self.r = math.sqrt(x**2+y**2)
        if self.r == 0:
            self.eta = 0
        else:
            self.eta = math.sinh(z/self.r)

    def extrapolateXY(self, z_new):
        r_new = 0
        if self.eta != 0:
            r_new = z_new/math.asinh(self.eta)
        x_new = r_new*math.sin(self.phi)
        y_new = r_new*math.cos(self.phi)
        return GridPoint(x_new, y_new, z_new)

    def __str__(self):
        return '(x={}, y={}, z={}, phi={}, eta={}, r={})'.format(self.x, self.y, self.z,self.phi,self.eta, self.r)

    def __repr__(self):
        # return '(x={}, y={}, z={}, phi={}, eta={}, r={})'.format(self.x, self.y, self.z,self.phi,self.eta, self.r)
        return '(x={}, y={}, z={})'.format(self.x, self.y, self.z)

class Grid:
    def __init__(self,
                 x_nbins, x_min, x_max,
                 y_nbins, y_min, y_max,
                 z):
        self.x_bins = np.linspace(x_min, x_max, x_nbins+1)
        self.y_bins = np.linspace(y_min, y_max, y_nbins+1)
        self.x_pitch = self.x_bins[1] - self.x_bins[0]
        self.y_pitch = self.y_bins[1] - self.y_bins[0]

        self.z = z
        print '--- Creating new Grid a@ z: {} with nbins_x: {} nbins_y: {}'.format(self.z, self.nbins_x, self.nbins_y)
        self.grid_points = np.ndarray(shape=(x_nbins+1, y_nbins+1), dtype=object)
        for idx_x in range(0, x_nbins+1):
            for idx_y in range(0, y_nbins+1):
                self.grid_points[idx_x][idx_y] = GridPoint(self.x_bins[idx_x],
                                                           self.y_bins[idx_y],
                                                           z)

    @property
    def nbins_x(self):
        return len(self.x_bins)-1

    @property
    def nbins_y(self):
        return len(self.y_bins)-1

    def getCorners(self, bin_x, bin_y):
        return [self.grid_points[bin_x][bin_y],
                self.grid_points[bin_x+1][bin_y],
                self.grid_points[bin_x+1][bin_y+1],
                self.grid_points[bin_x][bin_y+1]]

    def extrapolateXY(self, z):
        point_1 = self.grid_points[0][0].extrapolateXY(z)
        point_2 = self.grid_points[-1][-1].extrapolateXY(z)
        return Grid(len(self.x_bins), point_1.x, point_2.x, len(self.y_bins), point_1.y, point_2.y, z)
        # extrapolator = np.vectorize(lambda x: x.extrapolateXY(z))
        # return extrapolator(self.grid_points)

    def getCorners(self, bin_x, bin_y):
        return [self.grid_points[bin_x][bin_y],
                self.grid_points[bin_x+1][bin_y],
                self.grid_points[bin_x+1][bin_y+1],
                self.grid_points[bin_x][bin_y+1]]

    def getBinCenter(self, bin_x, bin_y):
        corner = self.grid_points[bin_x][bin_y]
        x = corner.x+self.x_pitch/2.
        y = corner.y+self.y_pitch/2.
        r = math.sqrt(x**2+y**2)
        eta = math.asinh(self.z/r)
        phi = math.atan2(y, x)
        print "Bin center x: {}, y: {}, eta: {}, phi: {}".format(x,y,eta,phi)

class TowerMaps:
    def __init__(self, refGridPlus, refGridMinus):
        self.refGrid_plus = refGridPlus
        self.refGrid_minus = refGridMinus
        self.grid_at_z = {}
        self.grid_at_z[self.refGrid_plus.z] = self.refGrid_plus
        self.grid_at_z[self.refGrid_minus.z] = self.refGrid_minus

    def extrapolateXY(self, z):
        if z not in self.grid_at_z.keys():
            newgrid = None
            if z < 0:
                newgrid = self.refGrid_minus.extrapolateXY(z)
            else:
                newgrid = self.refGrid_plus.extrapolateXY(z)
            self.grid_at_z[newgrid.z] = newgrid
        return self.grid_at_z[z]




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

    cell_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeCells')
    cell_geom_df = convertGeomTreeToDF(cell_geom_tree._tree)

    bhcell_geom_tree = HGCalNtuple([geom_file], tree='hgcaltriggergeomtester/TreeCellsBH')
    bhcell_geom_df = convertGeomTreeToDF(bhcell_geom_tree._tree)

    debugPrintOut(debug, 'Cell geometry',
                  toCount=cell_geom_df,
                  toPrint=cell_geom_df.iloc[:3])
    debugPrintOut(debug, 'TC geometry',
                  toCount=tc_geom_df,
                  toPrint=tc_geom_df.iloc[:3])



    display = EventDisplayManager(cell_geom=cell_geom_df, trigger_cell_geom=tc_geom_tree)
# for index, tc_geom in tc_geom_df.iterrows():
#     tc_geom.max_dist_neigh = np.max(tc_geom.neighbor_distance)

    algos = ['DEF', 'DBS']
    particles = [Particle('ele', PID.electron),
                 Particle('photon', PID.photon),
                 Particle('pion', PID.pion),
                 Particle('pizero', PID.pizero)]


    print 'build the grid for layer 1'

    gridM = Grid(x_nbins=68, x_min=-170., x_max=170.,
                y_nbins=68, y_min=-170., y_max=170.,
                z=-320.755005)

    gridP = Grid(x_nbins=68, x_min=-170., x_max=170.,
                y_nbins=68, y_min=-170., y_max=170.,
                z=320.755005)


    towerMap = TowerMaps(refGridPlus=gridP, refGridMinus=gridM)

    tower_tc_mapping = pd.DataFrame(columns=['tc', 'towerbin_x', 'towerbin_y'], dtype=np.int64)

    for subdet in [2, 4, 3]:
        layerrange = []
        if subdet == 3:
            layerrange = range(1, 29, 2)
        else:
            layerrange = range(1, 13)
        for zside in [-1, 1]:
            for layer in layerrange:
                # if subdet != 2:
                #     continue
                # if layer != 1:
                #     continue
                # if zside != -1:
                #     continue
                sel_cells = None
                if subdet != 2:
                    sel_cells = cell_geom_df[(cell_geom_df.zside == zside) &
                                             (cell_geom_df.subdet == subdet) &
                                             (cell_geom_df.layer == layer)]
                else:
                    sel_cells = bhcell_geom_df[(bhcell_geom_df.zside == zside) &
                                               (bhcell_geom_df.subdet == subdet) &
                                               (bhcell_geom_df.layer == layer)]
                # print 'Reference z: {}'.format(sel_cells.iloc[0].z)
                grid = towerMap.extrapolateXY(sel_cells.iloc[0].z)
                # print grid.x_bins
                # print grid.y_bins

                sel_cells['tt_bin_x'] = np.digitize(sel_cells.x, grid.x_bins)
                sel_cells['tt_bin_y'] = np.digitize(sel_cells.y, grid.y_bins)
                sel_cells['tt_bin'] = sel_cells.apply(func=lambda cell: (int(cell.tt_bin_x), int(cell.tt_bin_y)), axis=1)

                # do some _checks
                print '-------- a cell in subdet: {} layer: {} zside: {}'.format(subdet, layer, zside)
                acell = sel_cells.iloc[10]
                print acell[['x', 'y', 'tt_bin_x', 'tt_bin_y']]
                print towerMap.extrapolateXY(acell.z).getCorners(int(acell.tt_bin_x-1), int(acell.tt_bin_y-1))

                for tcid in sel_cells.tc_id.unique():
                    bins = sel_cells[sel_cells.tc_id == tcid]['tt_bin'].value_counts().index[0]
                    data = [tcid, int(bins[0])-1, int(bins[1])-1]
                    # print 'TC: {} data: {}'.format(tcid, data)
                    thiscell = pd.DataFrame([data], columns=['tc', 'towerbin_x', 'towerbin_y'], dtype=np.int64)
                    tower_tc_mapping = tower_tc_mapping.append(thiscell, ignore_index=True)
                    # print tower_tc_mapping

    print tower_tc_mapping.shape



    tower_tc_mapping.to_csv('TCmapping_v3.txt', sep=' ', float_format='%.0f', header=False, index=False)

    if False:
        for tcid in sel_cells.tc_id.unique():
            towers = sel_cells[sel_cells.tc_id == tcid][['tt_bin_x', 'tt_bin_y', 'tt_bin']]
            #towers['tt_count'] = towers[['tt_bin_x', 'tt_bin_y']].count()
            print 'TC: {}'.format(tcid)
            print towers
            print towers['tt_bin'].value_counts()

    print 'display all cells in 1 layer'

#    display.displayCells(event=0, cells=sel_cells)
#    display.displayTowers(event=0, zside=-1, subdet=3, layer=sel_cells.iloc[0].layer, grid=towerMap.extrapolateXY(sel_cells.iloc[0].z))
#    display.show(0)




if __name__ == "__main__":
    try:
        main(analyze=analyze)
    except Exception as inst:
        print (str(inst))
        print ("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc()
        sys.exit(100)
