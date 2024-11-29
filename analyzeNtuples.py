import sys

import typer
import yaml
import importlib
import pathlib

from rich import print as pprint

from python.parameters import Parameters, get_collection_parameters
from python.analyzer import analyze
from python.submission import to_HTCondor
from python.timecounter import print_stats
import ROOT

description = """
Main script for L1 TP analysis.

The script reads the configuration,
opens the input and output files for the given sample,
runs the event loop and saves histograms to disk.
All the analysis logic is anyhow elsewhere:

Data:
    which data are potentially read is handled in the `collections` module.
    How to select the data is handled in the `selections` module.
Plotters:
    what to do with the data is handled in the `plotters` module
Histograms:
    which histograms are produced is handled in the
    `histos` module (and the plotters).
"""



@print_stats
def analyzeNtuples(  # noqa: PLR0913
    configfile: str = typer.Option(..., '-f', '--file', help='specify the yaml configuration file'),
    datasetfile: str = typer.Option(
        ..., '-i', '--input-dataset', help='specify the yaml file defining the input dataset'
    ),
    collection: str = typer.Option(..., '-p', '--plotters', help='specify the plotters to be run'),
    sample: str = typer.Option(
        ...,
        '-s',
        '--sample',
        help='specify the sample to be processed',
    ),
    debug: int = typer.Option(0, '-d', '--debug', help='debug level'),
    nevents: int = typer.Option(10, '-n', '--nevents', help='# of events to process per sample'),
    batch: bool = typer.Option(None, '-b', '--batch', help='submit the jobs via CONDOR'),
    run: str = typer.Option(None, '-r', '--run', help='the batch_id to run (need to be used with the option -b)'),
    outdir: str = typer.Option(None, '-o', '--outdir', help='override the output directory for the files'),
    local: bool = typer.Option(False, '-l', '--local', help='run the batch on local resources'),
    workers: int = typer.Option(2, '-j', '--jobworkers', help='# of local workers'),
    workdir: str = typer.Option(None, '-w', '--workdir', help='local work directory'),
    submit: bool = typer.Option(False, '-S', '--submit', help='submit the jobs via CONDOR'),
):
    if submit and local and not workdir:
        raise ValueError('The --workdir option is required when submitting jobs locally')

    def parse_yaml(filename):
        with open(filename) as stream:
            return yaml.load(stream, Loader=yaml.FullLoader)

    cfgfile = {}

    # we load the python module with the same name as the yaml file
    pymoudule_path = pathlib.Path(configfile.split('.yaml')[0])
    formatted_path = '.'.join(pymoudule_path.with_suffix('').parts)
    sys.modules[formatted_path] = importlib.import_module(formatted_path)

    cfgfile.update(parse_yaml(configfile))
    cfgfile.update(parse_yaml(datasetfile))

    opt = Parameters(
        {
            'COLLECTION': collection,
            'SAMPLE': sample,
            'DEBUG': debug,
            'NEVENTS': nevents,
            'BATCH': batch,
            'RUN': run,
            'OUTDIR': outdir,
            'LOCAL': local,
            'WORKERS': workers,
            'WORKDIR': workdir,
            'SUBMIT': submit,
        }
    )
    collection_params = get_collection_parameters(opt, cfgfile)

    samples_to_process = []

    if not opt.COLLECTION:
        print(f'\nAvailable plotter collections: {collection_params.keys()}')
        sys.exit(0)
    if opt.COLLECTION not in collection_params:
        print(f'ERROR: plotter collection {opt.COLLECTION} not in the cfg file')
        sys.exit(10)
    if not opt.SAMPLE:
        print(f'Plotter collection: {opt.COLLECTION}, available samples: {collection_params[opt.COLLECTION]}')
        sys.exit(0)

    if opt.SAMPLE == 'all':
        samples_to_process.extend(collection_params[opt.COLLECTION])
    else:
        sel_sample = [sample for sample in collection_params[opt.COLLECTION] if sample.name == opt.SAMPLE]
        samples_to_process.append(sel_sample[0])

    pprint(f'About to process samples: {samples_to_process}')

    plot_version = f"{cfgfile['common']['plot_version']}.{cfgfile['dataset']['version']}"

    to_HTCondor(
        analyze=analyze,
        opt=opt,
        submit_mode=submit,
        plot_version=plot_version,
        samples_to_process=samples_to_process,
    )

    batch_idx = -1
    if opt.BATCH and opt.RUN:
        batch_idx = int(opt.RUN)

    ret_nevents = 0
    for idx, sample in enumerate(samples_to_process):
        pprint(
            f'\n\n========================== #{idx+1}/{len(samples_to_process)}: {sample.name} ==========================\n'
        )
        ret_nevents += analyze(sample, batch_idx=batch_idx)
    return ret_nevents


if __name__ == '__main__':
    typer.run(analyzeNtuples)
