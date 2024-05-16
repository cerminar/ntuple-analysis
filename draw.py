import typer
import sys
import ROOT
import pandas as pd
import os
import pathlib
import importlib


import python.collections as collections
import python.selections as selections
import python.histos as histos
import python.calibrations as calib
from python.histos import RateHistos

import python.draw.webpager as wp
from python.draw.drawingTools import *
from python.draw.rate_utilities import convertRateToGraph

# this is needed to get all the labels defined in cfg modules
# import python.plotters_config 
# from cfg import eg_genmatch


# ROOT.enableJSVis()

# some globals for ROOT
normalized_histos = list()

ROOT.gROOT.SetBatch(True)

def extract_samples(type_prexif, input_files, samples):
    if input_files == '': 
        return
    for ief,efile in enumerate(input_files.split(',')):
        ftype = f'{type_prexif}-{ief}'
        file_name = efile.split(':')[0]
        path = os.path.dirname(file_name)
        filestr = os.path.split(file_name)[-1]
        pu = 'PU0'
        if 'PU200' in filestr or 'pu200' in filestr:
            pu = 'PU200'
        smp = HistoFile(name='null', pu=pu, type=ftype, label=efile.split(':')[1], path=path)
        smp.histo_filename = file_name
        samples.append(smp)


def draw(
    input_files: str = typer.Option("", '-i', '--input-files', help='specify the input files containing the plots.\n Syntax: filename1.root:label1,filename2.root:label2,....'),
    module: str = typer.Option(..., '-m', '--module', help='specify the draw module'),
    what: str = typer.Option(..., '-w', '--what', help='specify the draw function within the module'),
    target_dir: str = typer.Option(..., '-t', '--target-dir', help='specify the directory where the plot images will be saved'),
    
    do_pt_rate_wps: bool = False
):
    
    # we load the python module with the same name as the yaml file
    pymoudule_path = pathlib.Path(module)
    formatted_path = '.'.join(pymoudule_path.with_suffix('').parts)
    draw_module = importlib.import_module(formatted_path)


    samples = []
    smps = []

    extract_samples('ele', input_files, samples)
    smps = [s.type for s in samples]

    for smp in samples:
        smp.print_primitives()
    # return
    labels_dict = {}

    evm = collections.EventManager()
    labels_dict.update(evm.get_labels())
    selm = selections.SelectionManager()
    labels_dict.update(selm.get_labels())

    hplot = HPlot(samples, labels_dict)
    hplot.create_histo_proxies(draw_module.histo_class)


    if do_pt_rate_wps:
        calib_mgr = calib.CalibManager()
        calib_mgr.set_pt_wps_version('data/rate_pt_wps_v160A.91G.json')
        rate_pt_wps = calib_mgr.get_pt_wps()
        print(rate_pt_wps)
        data_selections = calib.rate_pt_wps_selections(
        rate_pt_wps, 'TkEleEE')
        # data_selections = calib.rate_pt_wps_selections(
        # rate_pt_wps, 'TkEleL2')
        data_selections = calib.rate_pt_wps_selections(
        rate_pt_wps, 'TkEmL2')
        data_selections = calib.rate_pt_wps_selections(
        rate_pt_wps, 'TkEleL2')

        for sel in data_selections:
            print (sel.selection)

        print(data_selections)
        selm = selections.SelectionManager()
        # labels_dict.update(selm.get_labels())
        hplot.labels_dict.update(selm.get_labels())


    web_path = pathlib.PurePath(target_dir)
    project_dir = web_path.name

    base_dir = target_dir.split(project_dir)[0]
    wc = wp.WebPageCreator(
        topic_dir=draw_module.wc_label, 
        project_dir=project_dir, 
        base_dir=base_dir, 
        tmp_dir=os.environ['TMPDIR'],
        samples=samples)
    
    draw_func = getattr(draw_module, f'{what}_draw')
    draw_func(hplot, smps, wc)
    wc.publish()



if __name__ == '__main__':
    typer.run(draw)
