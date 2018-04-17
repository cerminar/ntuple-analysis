
from utils import debugPrintOut

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
        debugPrintOut(4, 'cells', toCount=cells, toPrint=cells[['id', 'cell']])
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
                    all_cells_colors = []
                    debugPrintOut(4, 'cells_layer', toCount=cells_layer, toPrint=cells_layer[['id', 'cell']])

                    count = 0
                    for index, cell in cells_layer.iterrows():
                        count += 1
                        print '{} {} {}'.format(index, cell.cell, long(cell.id)) # WTF
                        all_corners_x.append([cell.x1, cell.x2, cell.x2-cell.width*0.5,  cell.x3, cell.x4, cell.x4+cell.width*0.5])
                        all_corners_y.append([cell.y1, cell.y2, cell.y2+cell.delta,  cell.y3, cell.y4, cell.y4-cell.delta])
                        all_cells_ids.append(int(cell.id))
                        all_cells_colors.append(cell.color)
                    print count

                    source = ColumnDataSource(data=dict(x=all_corners_x,
                                                        y=all_corners_y,
                                                        id=all_cells_ids,
                                                        color=all_cells_colors
                                                        ))

                    figure.patches('x', 'y', source=source,
                                   fill_color={'field': 'color', 'transform': self.color_mapper},
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
                        for cell_idx, cell in cells.iterrows():
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
                            for cell_idx, cell in cells.iterrows():
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
                ("ID", "@id"),
                #("X", "@x"),
                 ("energy", "@energy GeV"),
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
