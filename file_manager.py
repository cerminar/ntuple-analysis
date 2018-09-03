import os
import subprocess32
from NtupleDataFormat import HGCalNtuple
import json
import uuid


def copy_from_eos(input_dir, file_name, target_file_name):
    protocol = ''
    if '/eos/user/' in input_dir:
        protocol = 'root://eosuser.cern.ch/'
    elif '/eos/cms/' in input_dir:
        protocol = 'root://eoscms.cern.ch/'
    eos_proc = subprocess32.Popen(['eos', protocol, 'cp', os.path.join(input_dir, file_name), target_file_name], stdout=subprocess32.PIPE)
    print eos_proc.stdout.readlines()
    return eos_proc.returncode


def copy_to_eos(file_name, target_dir, target_file_name):
    protocol = ''
    if '/eos/user/' in target_dir:
        protocol = 'root://eosuser.cern.ch/'
    elif '/eos/cms/' in target_dir:
        protocol = 'root://eoscms.cern.ch/'
    eos_proc = subprocess32.Popen(['eos', protocol, 'cp', file_name, os.path.join(target_dir, target_file_name)], stdout=subprocess32.PIPE)
    print eos_proc.stdout.readlines()
    # eos_proc.wait()
    return eos_proc.returncode


def listFiles(input_dir, match='.root'):
    onlyfiles = []
    if not input_dir.startswith('/eos'):
        onlyfiles = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    else:
        # we read the input files via EOS
        protocol = ''
        if '/eos/user/' in input_dir:
            protocol = 'root://eosuser.cern.ch/'
        elif '/eos/cms/' in input_dir:
            protocol = 'root://eoscms.cern.ch/'
        eos_proc = subprocess32.Popen(['eos', protocol, 'ls', input_dir], stdout=subprocess32.PIPE)
        onlyfiles = [os.path.join(input_dir, f.rstrip()) for f in eos_proc.stdout.readlines() if match in f]

    return sorted(onlyfiles)


def stage_files(files_to_stage):
    ret_files = []
    for file_name in files_to_stage:
        copy_from_eos(os.path.dirname(file_name), os.path.basename(file_name), os.path.basename(file_name))
        # FIXME: this is a very loose check...
        if os.path.isfile(os.path.basename(file_name)):
            ret_files.append(os.path.basename(file_name))
        else:
            ret_files.append(file_name)
    # print ret_files
    return ret_files


def get_files_for_processing(input_dir, tree, nev_toprocess, debug=0):
    metadata = get_metadata(input_dir, tree, debug)
    return get_files_to_process(nev_toprocess, metadata, debug)


def get_files_and_events_for_batchprocessing(input_dir, tree, nev_toprocess, nev_perjob, batch_id, debug=0):
    metadata = get_metadata(input_dir, tree, debug)
    file_list, event_range = get_njobs(nev_toprocess, nev_perjob, metadata, debug)[batch_id]
    # return stage_files(files_to_stage=file_list), event_range
    return file_list, event_range


def get_number_of_jobs_for_batchprocessing(input_dir, tree, nev_toprocess, nev_perjob, debug=0):
    metadata = get_metadata(input_dir, tree, debug)
    return len(get_njobs(nev_toprocess, nev_perjob, metadata, debug).keys())


def get_metadata(input_dir, tree, debug=0):
    json_name = 'metadata.json'
    file_metadata = {}
    json_files = listFiles(input_dir, match=json_name)
    if len(json_files) == 0:
        print 'no metadata file {} in input dir: {}'.format(json_name, input_dir)
        print 'Will now index files...'
        files = listFiles(input_dir)
        print '# of files: {}'.format(len(files))

        for idx, file_name in enumerate(files):
            ntuple = HGCalNtuple([file_name], tree)
            nevents = ntuple.nevents()
            file_metadata[file_name] = nevents
            if debug > 2:
                print ' [{}] file: {} # events: {}'.format(idx, file_name, nevents)

        with open(json_name, 'w') as fp:
            json.dump(file_metadata, fp)
        copy_to_eos(file_name=json_name,
                    target_dir=input_dir,
                    target_file_name=json_name)
    else:
        print 'dir already indexed, will read metadata...'
        unique_filename = '{}.json'.format(uuid.uuid4())
        copy_from_eos(input_dir=input_dir,
                      file_name=json_name,
                      target_file_name=unique_filename)
        with open(unique_filename, 'r') as fp:
            file_metadata = json.load(fp)
        os.remove(unique_filename)

    return file_metadata



def get_files_to_process(nev_toprocess, metadata, debug=0):
    nevents_tot = 0
    for key, value in metadata.iteritems():
        # FIXME: if value is 0 maybe one should check again and rewrite the json?
        nevents_tot += int(value)
    if debug > 2:
        print 'Tot.# events: {}'.format(nevents_tot)

    if nev_toprocess == -1:
        return metadata.keys()

    nev_sofar = 0
    files_sofar = []
    for file_n in sorted(metadata.keys()):
        nev_sofar += int(metadata[file_n])
        files_sofar.append(file_n)
        if nev_sofar >= nev_toprocess:
            break

    if debug > 3:
        print files_sofar
        print '# of files: {}'.format(len(files_sofar))
    return files_sofar


def get_njobs(nev_toprocess, nev_perjob, metadata, debug=0):

    needed_files = sorted(get_files_to_process(nev_toprocess, metadata))
    nevents_tot = 0
    comulative = {}
    for file_name in needed_files:
        comulative[file_name] = nevents_tot
        nevents_tot += int(metadata[file_name])

    if debug > 3:
        print 'Tot.# events: {}'.format(nevents_tot)
    if nev_toprocess == -1:
        nev_toprocess = nevents_tot

    njobs = int(nev_toprocess/nev_perjob)
    print '# of jobs: {}'.format(njobs)
    ret = {}
    for job_id in range(0, njobs):
        files_perjob = []
        eventrange = (-1, -1)
        events_injob = range(job_id*nev_perjob, (job_id+1)*nev_perjob)
        first_ev_injob = events_injob[0]
        last_ev_injob = events_injob[-1]
        if debug > 3:
            print ' jobid: {}, i: {} e: {}'.format(job_id, first_ev_injob, last_ev_injob)
        for file_name in needed_files:
            first_ev = comulative[file_name]
            last_ev = first_ev + metadata[file_name]
            if(first_ev_injob >= first_ev and first_ev_injob < last_ev):
                files_perjob.append(file_name)
                eventrange = (first_ev_injob - first_ev, first_ev_injob - first_ev+nev_perjob-1)
            elif(first_ev_injob < first_ev and last_ev_injob >= last_ev):
                files_perjob.append(file_name)
            elif(last_ev_injob > first_ev and last_ev_injob < last_ev):
                files_perjob.append(file_name)
        if debug > 3:
            print '   files: {}, range: {}'.format(files_perjob, eventrange)
        totv = 0
        for file_n in files_perjob:
            if debug > 3:
                print '    file: {} ({})'.format(file_n, metadata[file_n])
            totv += metadata[file_n]
        if debug > 3:
            print '   # ev in files: {}'.format(totv)
        ret[job_id] = (files_perjob, eventrange)
    return ret


if __name__ == "__main__":

    file_metadata = get_metadata(input_dir='/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1015/SingleE_FlatPt-2to100/SingleE_FlatPt-2to100_PU0_v11/180814_140939/0000/',
                                 tree='hgcalTriggerNtuplizer/HGCalTriggerNtuple')

    get_files_to_process(nev_toprocess=20, metadata=file_metadata)

    get_files_to_process(nev_toprocess=1000, metadata=file_metadata)

    get_files_to_process(nev_toprocess=3000, metadata=file_metadata)

    get_files_to_process(nev_toprocess=-1, metadata=file_metadata)

    get_njobs(nev_toprocess=3000, nev_perjob=500, metadata=file_metadata)

    jobs = get_njobs(nev_toprocess=8000, nev_perjob=2000, metadata=file_metadata)

    # copy_from_eos(input_dir='/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1015/SingleE_FlatPt-2to100/SingleE_FlatPt-2to100_PU0_v11/180814_140939/0000/',
    #               file_name='ntuple_23.root', target_file_name='pippo.json')

    # copy_to_eos(file_name='data.json',
    #             target_dir='/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1015/SingleE_FlatPt-2to100/SingleE_FlatPt-2to100_PU0_v11/180814_140939/0000/',
    #             target_file_name='metadata.json')

    print jobs
# get_njobs(nev_toprocess=-1, nev_perjob=500, metadata=file_metadata)
