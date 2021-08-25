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


class Parameters(dict):

    def __getattr__(self, name):
        return self[name]

    def __str__(self):
        return 'Name: {},\n \
                clusterize: {}\n \
                compute density: {}\n \
                maxEvents: {}\n \
                output file: {}\n \
                events per job: {}\n \
                debug: {}'.format(self.name,
                                  self.clusterize,
                                  self.computeDensity,
                                  self.maxEvents,
                                  self.output_filename,
                                  self.events_per_job,
                                  self.debug)

    def __repr__(self):
        return self.name


def get_collection_parameters(opt, cfgfile):
    outdir = cfgfile['common']['output_dir']['default']
    hostname = socket.gethostname()
    for machine, odir in cfgfile['common']['output_dir'].items():
        if machine in hostname:
            outdir = odir
    plot_version = '{}.{}'.format(
        cfgfile['common']['plot_version'], 
        cfgfile['samples']['version'])

    collection_params = {}
    for collection, collection_data in cfgfile['collections'].items():
        samples = collection_data['samples']
        print('--- Collection: {} with samples: {}'.format(collection, samples))
        sample_params = []

        plotters = []
        for plotter in collection_data['plotters']:
            plotters.extend(plotter)

        for sample in samples:
            events_per_job = -1
            output_filename_base = 'histos_{}_{}_{}'.format(sample, collection_data['file_label'], plot_version)
            out_file_name = '{}i.root'.format(output_filename_base)
            if opt.BATCH:
                events_per_job = cfgfile['samples'][sample]['events_per_job']
                if 'events_per_job' in collection_data.keys():
                    if sample in collection_data['events_per_job']:
                        events_per_job = collection_data['events_per_job'][sample]

                if opt.RUN:
                    out_file_name = '{}_{}.root'.format(output_filename_base, opt.RUN)

            if opt.OUTDIR:
                outdir = opt.OUTDIR

            out_file = os.path.join(outdir, out_file_name)

            weight_file = None
            if 'weights' in collection_data.keys():
                if sample in collection_data['weights'].keys():
                    weight_file = collection_data['weights'][sample]

            params = Parameters({'input_base_dir': cfgfile['samples']['input_dir'],
                                 'input_sample_dir': cfgfile['samples'][sample]['input_sample_dir'],
                                 'tree_name': cfgfile['samples']['tree_name'],
                                 'output_filename_base': output_filename_base,
                                 'output_filename': out_file,
                                 'output_dir': outdir,
                                 'clusterize': cfgfile['common']['run_clustering'],
                                 'eventsToDump': [],
                                 'version': plot_version,
                                 'calib_version':  cfgfile['samples']['calib_version'],
                                 'maxEvents': int(opt.NEVENTS),
                                 'events_per_job': events_per_job,
                                 'computeDensity': cfgfile['common']['run_density_computation'],
                                 'plotters': plotters,
                                 'htc_jobflavor': collection_data['htc_jobflavor'],
                                 'htc_priority': collection_data['priorities'][sample],
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
        if '3.8' in platform.python_version():
            cfgfile = yaml.load(stream, Loader=yaml.FullLoader)
        else:
            cfgfile = yaml.load(stream)
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
    cfgfile.update(parse_yaml(opt.CONFIGFILE))
    if opt.DATASETFILE is not None:
        cfgfile.update(parse_yaml(opt.DATASETFILE))

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
                print(('Collection: {}, available samples: {}'.format(
                    opt.COLLECTION, collection_params[opt.COLLECTION])))
                sys.exit(0)
        else:
            print('ERROR: collection {} not in the cfg file'.format(opt.COLLECTION))
            sys.exit(10)
    else:
        print('\nAvailable collections: {}'.format(collection_params.keys()))
        sys.exit(0)

    print('About to process samples: {}'.format(samples_to_process))

    plot_version = '{}.{}'.format(
        cfgfile['common']['plot_version'], 
        cfgfile['samples']['version'])

    if opt.BATCH and not opt.RUN:
        batch_dir = 'batch_{}_{}'.format(opt.COLLECTION, plot_version)
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
            print('Total # of events to be processed: {}'.format(nevents))
            print('# of events per job: {}'.format(sample.events_per_job))
            if n_jobs == 0:
                n_jobs = 1
            print('# of jobs to be submitted: {}'.format(n_jobs))
            sample['nbatch_jobs'] = n_jobs

            params = {}
            params['TEMPL_TASKDIR'] = sample_batch_dir
            params['TEMPL_NJOBS'] = str(n_jobs)
            params['TEMPL_WORKDIR'] = os.environ["PWD"]
            params['TEMPL_CFG'] = opt.CONFIGFILE
            params['TEMPL_INPUT'] = opt.DATASETFILE
            params['TEMPL_COLL'] = opt.COLLECTION
            params['TEMPL_SAMPLE'] = sample.name
            params['TEMPL_OUTFILE'] = '{}.root'.format(sample.output_filename_base)
            params['TEMPL_EOSPROTOCOL'] = fm.get_eos_protocol(dirname=sample.output_dir)
            params['TEMPL_INFILE'] = '{}_*.root'.format(sample.output_filename_base)
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
                         outfile=os.path.join(batch_dir, 'hadd_{}.dag'.format(sample.name)),
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
                dagman_spl += 'JOB Job_{} batch.sub\n'.format(jid)
                dagman_spl += 'VARS Job_{} JOB_ID="{}"\n'.format(jid, jid)
                dagman_spl_retry += 'Retry Job_{} 3\n'.format(jid)
                dagman_spl_retry += 'PRIORITY Job_{} {}\n'.format(jid, sample.htc_priority)

            dagman_sub += 'SPLICE {} {}.spl DIR {}\n'.format(
                sample.name, sample.name, sample_batch_dir)
            dagman_sub += 'JOB {} {}/batch_hadd.sub\n'.format(sample.name+'_hadd', sample_batch_dir)
            dagman_sub += 'JOB {} {}/batch_cleanup.sub\n'.format(
                sample.name+'_cleanup', sample_batch_dir)

            dagman_dep += 'PARENT {} CHILD {}\n'.format(sample.name, sample.name+'_hadd')
            dagman_dep += 'PARENT {} CHILD {}\n'.format(sample.name+'_hadd', sample.name+'_cleanup')

            # dagman_ret += 'Retry {} 3\n'.format(sample.name)
            dagman_ret += 'Retry {} 3\n'.format(sample.name+'_hadd')
            dagman_ret += 'PRIORITY {} {}\n'.format(sample.name+'_hadd', sample.htc_priority)

            dagman_splice = open(os.path.join(sample_batch_dir, '{}.spl'.format(sample.name)), 'w')
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
        print('condor_submit_dag {}'.format(dagman_file_name))

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
