from __future__ import print_function
import sys
# The purpose of this file is to demonstrate mainly the objects
# that are in the HGCalNtuple
import os
import socket
import optparse
import yaml
import subprocess32
import platform
from shutil import copyfile

import python.file_manager as fm
import python.plotters_config
import python.calibrations as calib


class Parameters(dict):

    def __getattr__(self, name):
        return self[name]

    def __str__(self):
        return f'Name: {self.name},\n                 clusterize: {self.clusterize}\n                 compute density: {self.computeDensity}\n                 maxEvents: {self.maxEvents}\n                 output file: {self.output_filename}\n                 events per job: {self.events_per_job}\n                 debug: {self.debug}'

    def __repr__(self):
        return self.name


def get_collection_parameters(opt, cfgfile):
    outdir = cfgfile['common']['output_dir']['default']
    hostname = socket.gethostname()
    for machine, odir in cfgfile['common']['output_dir'].items():
        if machine in hostname:
            outdir = odir
    plot_version = f"{cfgfile['common']['plot_version']}.{cfgfile['dataset']['version']}"

    collection_params = {}
    for collection, collection_data in cfgfile['collections'].items():
        samples = cfgfile['samples'].keys()
        print(f'--- Collection: {collection} with samples: {samples}')
        sample_params = []

        plotters = []
        for plotter in collection_data['plotters']:
            plotters.extend(plotter)

        for sample in samples:
            events_per_job = -1
            output_filename_base = f"histos_{sample}_{collection_data['file_label']}_{plot_version}"
            out_file_name = f'{output_filename_base}i.root'
            if opt.BATCH:
                events_per_job = cfgfile['samples'][sample]['events_per_job']
                if 'events_per_job' in collection_data.keys():
                    if sample in collection_data['events_per_job']:
                        events_per_job = collection_data['events_per_job'][sample]

                if opt.RUN:
                    out_file_name = f'{output_filename_base}_{opt.RUN}.root'

            if opt.OUTDIR:
                outdir = opt.OUTDIR

            out_file = os.path.join(outdir, out_file_name)

            weight_file = None
            if 'weights' in collection_data.keys():
                if sample in collection_data['weights'].keys():
                    weight_file = collection_data['weights'][sample]

            rate_pt_wps = None
            if 'rate_pt_wps' in cfgfile['dataset']:
                rate_pt_wps = cfgfile['dataset']['rate_pt_wps']

            priority = 2
            if 'priorities' in collection_data and sample in collection_data['priorities']:
                priority = collection_data['priorities'][sample]

            params = Parameters({'input_base_dir': cfgfile['dataset']['input_dir'],
                                 'input_sample_dir': cfgfile['samples'][sample]['input_sample_dir'],
                                 'tree_name': cfgfile['dataset']['tree_name'],
                                 'output_filename_base': output_filename_base,
                                 'output_filename': out_file,
                                 'output_dir': outdir,
                                 'clusterize': cfgfile['common']['run_clustering'],
                                 'eventsToDump': [],
                                 'version': plot_version,
                                 'calib_version':  cfgfile['dataset']['calib_version'],
                                 'rate_pt_wps': rate_pt_wps,
                                 'maxEvents': int(opt.NEVENTS),
                                 'events_per_job': events_per_job,
                                 'computeDensity': cfgfile['common']['run_density_computation'],
                                 'plotters': plotters,
                                 'htc_jobflavor': collection_data['htc_jobflavor'],
                                 'htc_priority': priority,
                                 'weight_file': weight_file,
                                 'debug': opt.DEBUG,
                                 'name': sample})
            sample_params.append(params)
        collection_params[collection] = sample_params
    return collection_params


def editTemplate(infile, outfile, params):
    template_file = open(infile)
    template = template_file.read()
    template_file.close()

    for param in params.keys():
        template = template.replace(param, params[param])

    out_file = open(outfile, 'w')
    out_file.write(template)
    out_file.close()


def parse_yaml(filename):
    with open(filename, 'r') as stream:
        cfgfile = yaml.load(stream, Loader=yaml.FullLoader)
    return cfgfile


def main(analyze, submit_mode=False):
    # ============================================
    # configuration bit

    usage = ('usage: %prog [options]\n'
             + '%prog -h for help')
    parser = optparse.OptionParser(usage)
    parser.add_option(
        '-f', '--file',
        dest='CONFIGFILE',
        help='specify the yaml configuration file')
    parser.add_option(
        '-i', '--input-dataset',
        default=None,
        dest='DATASETFILE',
        help='specify the yaml file defining the input dataset')
    parser.add_option('-c', '--collection', dest='COLLECTION',
                      help='specify the collection to be processed')
    parser.add_option('-s', '--sample', dest='SAMPLE',
                      help='specify the sample (within the collection) to be processed ("all" to run the full collection)')
    parser.add_option('-d', '--debug', dest='DEBUG', help='debug level (default is 0)', default=0)
    parser.add_option('-n', '--nevents', dest='NEVENTS',
                      help='# of events to process per sample (default is 10)', default=10)
    parser.add_option("-b", "--batch", action="store_true", dest="BATCH",
                      default=False, help="submit the jobs via CONDOR")
    parser.add_option("-r", "--run", dest="RUN", default=None,
                      help="the batch_id to run (need to be used with the option -b)")
    parser.add_option("-o", "--outdir", dest="OUTDIR", default=None,
                      help="override the output directory for the files")
    if submit_mode:
        parser.add_option("-l", "--local", action="store_true", dest="LOCAL", default=False,
                          help="run the batch on local resources")
        parser.add_option("-j", "--jobworkers", dest="WORKERS", default=2,
                          help="# of local workers")
        parser.add_option("-w", "--workdir", dest="WORKDIR",
                          help="local work directory")

    # parser.add_option("-i", "--inputJson", dest="INPUT", default='input.json', help="list of input files and properties in JSON format")

    global opt, args
    (opt, args) = parser.parse_args()

    if submit_mode:
        if opt.LOCAL:
            if not opt.WORKDIR:
                parser.error('workdir not provided')

    # read the config file
    cfgfile = {}
    if opt.DATASETFILE is not None:
        cfgfile.update(parse_yaml(opt.DATASETFILE))
    cfgfile.update(parse_yaml(opt.CONFIGFILE))

    collection_params = get_collection_parameters(opt, cfgfile)

    samples_to_process = list()
    if opt.COLLECTION:
        if opt.COLLECTION in collection_params.keys():
            if opt.SAMPLE:
                if opt.SAMPLE == 'all':
                    samples_to_process.extend(collection_params[opt.COLLECTION])
                else:
                    sel_sample = [sample for sample in collection_params[opt.COLLECTION]
                                  if sample.name == opt.SAMPLE]
                    samples_to_process.append(sel_sample[0])
            else:
                print(f'Collection: {opt.COLLECTION}, available samples: {collection_params[opt.COLLECTION]}')
                sys.exit(0)
        else:
            print(f'ERROR: collection {opt.COLLECTION} not in the cfg file')
            sys.exit(10)
    else:
        print(f'\nAvailable collections: {collection_params.keys()}')
        sys.exit(0)

    print(f'About to process samples: {samples_to_process}')

    plot_version = f"{cfgfile['common']['plot_version']}.{cfgfile['dataset']['version']}"

    if opt.BATCH and not opt.RUN:
        batch_dir = f'batch_{opt.COLLECTION}_{plot_version}'
        if not os.path.exists(batch_dir):
            os.mkdir(batch_dir)
            os.mkdir(batch_dir+'/conf/')
            os.mkdir(batch_dir+'/logs/')

        dagman_sub = ''
        dagman_dep = ''
        dagman_ret = ''
        for sample in samples_to_process:
            dagman_spl = ''
            dagman_spl_retry = ''
            sample_batch_dir = os.path.join(batch_dir, sample.name)
            sample_batch_dir_logs = os.path.join(sample_batch_dir, 'logs')
            os.mkdir(sample_batch_dir)
            os.mkdir(sample_batch_dir_logs)
            print(sample)
            nevents = int(opt.NEVENTS)
            n_jobs = fm.get_number_of_jobs_for_batchprocessing(input_dir=os.path.join(sample.input_base_dir, sample.input_sample_dir),
                                                               tree=sample.tree_name,
                                                               nev_toprocess=nevents,
                                                               nev_perjob=sample.events_per_job,
                                                               debug=int(opt.DEBUG))
            print(f'Total # of events to be processed: {nevents}')
            print(f'# of events per job: {sample.events_per_job}')
            if n_jobs == 0:
                n_jobs = 1
            print(f'# of jobs to be submitted: {n_jobs}')
            sample['nbatch_jobs'] = n_jobs

            params = {}
            params['TEMPL_TASKDIR'] = sample_batch_dir
            params['TEMPL_NJOBS'] = str(n_jobs)
            params['TEMPL_WORKDIR'] = os.environ["PWD"]
            params['TEMPL_CFG'] = opt.CONFIGFILE
            params['TEMPL_INPUT'] = opt.DATASETFILE
            params['TEMPL_COLL'] = opt.COLLECTION
            params['TEMPL_SAMPLE'] = sample.name
            params['TEMPL_OUTFILE'] = f'{sample.output_filename_base}.root'
            params['TEMPL_EOSPROTOCOL'] = fm.get_eos_protocol(dirname=sample.output_dir)
            params['TEMPL_INFILE'] = f'{sample.output_filename_base}_*.root'
            params['TEMPL_FILEBASE'] = sample.output_filename_base
            params['TEMPL_OUTDIR'] = sample.output_dir
            params['TEMPL_VIRTUALENV'] = os.path.basename(os.environ['VIRTUAL_ENV'])
            params['TEMPL_VERSION'] = sample.version
            params['TEMPL_JOBFLAVOR'] = sample.htc_jobflavor

            editTemplate(infile='templates/batch.sub',
                         outfile=os.path.join(sample_batch_dir, 'batch.sub'),
                         params=params)

            editTemplate(infile='templates/run_batch.sh',
                         outfile=os.path.join(sample_batch_dir, 'run_batch.sh'),
                         params=params)

            editTemplate(infile='templates/copy_files.sh',
                         outfile=os.path.join(sample_batch_dir, 'copy_files.sh'),
                         params=params)
            os.chmod(os.path.join(sample_batch_dir, 'copy_files.sh'),  0o754)

            editTemplate(infile='templates/batch_hadd.sub',
                         outfile=os.path.join(sample_batch_dir, 'batch_hadd.sub'),
                         params=params)

            editTemplate(infile='templates/run_batch_hadd.sh',
                         outfile=os.path.join(sample_batch_dir, 'run_batch_hadd.sh'),
                         params=params)

            editTemplate(infile='templates/batch_cleanup.sub',
                         outfile=os.path.join(sample_batch_dir, 'batch_cleanup.sub'),
                         params=params)

            editTemplate(infile='templates/run_batch_cleanup.sh',
                         outfile=os.path.join(sample_batch_dir, 'run_batch_cleanup.sh'),
                         params=params)

            editTemplate(infile='templates/hadd_dagman.dag',
                         outfile=os.path.join(batch_dir, f'hadd_{sample.name}.dag'),
                         params=params)

            editTemplate(infile='templates/run_harvest.sh',
                         outfile=os.path.join(sample_batch_dir, 'run_harvest.sh'),
                         params=params)

            editTemplate(infile='templates/batch_harvest.sub',
                         outfile=os.path.join(sample_batch_dir, 'batch_harvest.sub'),
                         params=params)

            if submit_mode:
                if opt.LOCAL:
                    editTemplate(infile='templates/run_local.sh',
                                 outfile=os.path.join(sample_batch_dir, 'run_local.sh'),
                                 params=params)

            for jid in range(0, n_jobs):
                dagman_spl += f'JOB Job_{jid} batch.sub\n'
                dagman_spl += f'VARS Job_{jid} JOB_ID="{jid}\"\n'
                dagman_spl_retry += f'Retry Job_{jid} 3\n'
                dagman_spl_retry += f'PRIORITY Job_{jid} {sample.htc_priority}\n'

            dagman_sub += f'SPLICE {sample.name} {sample.name}.spl DIR {sample_batch_dir}\n'
            dagman_sub += f"JOB {sample.name + '_hadd'} {sample_batch_dir}/batch_hadd.sub\n"
            dagman_sub += f"JOB {sample.name + '_cleanup'} {sample_batch_dir}/batch_cleanup.sub\n"

            dagman_dep += f"PARENT {sample.name} CHILD {sample.name + '_hadd'}\n"
            dagman_dep += f"PARENT {sample.name + '_hadd'} CHILD {sample.name + '_cleanup'}\n"

            # dagman_ret += 'Retry {} 3\n'.format(sample.name)
            dagman_ret += f"Retry {sample.name + '_hadd'} 3\n"
            dagman_ret += f"PRIORITY {sample.name + '_hadd'} {sample.htc_priority}\n"

            dagman_splice = open(os.path.join(sample_batch_dir, f'{sample.name}.spl'), 'w')
            dagman_splice.write(dagman_spl)
            dagman_splice.write(dagman_spl_retry)
            dagman_splice.close()

            # copy the config file in the batch directory
            # copyfile(opt.CONFIGFILE, os.path.join(sample_batch_dir, opt.CONFIGFILE))

        dagman_file_name = os.path.join(batch_dir, 'dagman.dag')
        dagman_file = open(dagman_file_name, 'w')
        dagman_file.write(dagman_sub)
        dagman_file.write(dagman_dep)
        dagman_file.write(dagman_ret)
        dagman_file.close()

        # create targz file of the code from git
        git_proc = subprocess32.Popen(['git', 'archive', '--format=tar.gz', 'HEAD', '-o',
                                       os.path.join(batch_dir, 'ntuple-tools.tar.gz')], stdout=subprocess32.PIPE)
        git_proc.wait()
        # cp TEMPL_TASKDIR/TEMPL_CFG
        print('Ready for HT-Condor submission please run the following commands:')
        # print('condor_submit {}'.format(condor_file_path))
        print(f'condor_submit_dag {dagman_file_name}')

        if submit_mode:
            if opt.LOCAL:
                print('will now run jobs on local resources')
                analyze(batch_dir,
                        samples_to_process,
                        opt.WORKERS,
                        opt.WORKDIR)

        sys.exit(0)

    if submit_mode:
        sys.exit(0)

    batch_idx = -1
    if opt.BATCH and opt.RUN:
        batch_idx = int(opt.RUN)

    # samples = test_sample
    ret_nevents = 0
    for sample in samples_to_process:
        ret_nevents += analyze(sample, batch_idx=batch_idx)
    return ret_nevents
