# from __future__ import absolute_import
from __future__ import print_function
from importlib.resources import path
import os
import time
from unittest import result
import subprocess as subproc
import uproot as up
import json
import uuid
from io import open


class FileEntry(object):
    def __init__(self, name, date, attributes, owner, group, size) -> None:
        self.name = name
        self.attributes = attributes
        self.owner = owner
        self.group = group
        self.size = size
        self.date = date
    
    def is_dir(self):
        return self.attributes[0] == 'd'

    def __str__(self) -> str:
        return f'{self.attributes} {self.name}'
    
    def basename(self):
        return os.path.basename(self.name)
    
    def dirname(self):
        return os.path.dirname(self.name)


class FileSystem(object):
    def __init__(self, protocol) -> None:
        self.protocol = protocol
        self.protocol_host = protocol.lstrip('root://')
        self.cmd_base_ = []

        return

    def list_dir(self, path, recursive=False):
        ls_cmd = self.list_dir_cmd(path, recursive)
        ok, entries = self.exec(ls_cmd)
        if ok:
            return self.list_dir_parse(entries, path)
        else:
            raise RuntimeError(f'Failed to list path: {path}')
        return []

    def parse_file_list(self):
        pass

    def list_dir_cmd(self,  path, recursive=False):
        pass
    
    def list_dir_parse(self,  lines, path):
        pass

    def checksum_cmd(self, filename):
        pass
  
    def checksum_parse(self, results):
        pass

    def copy_cmd(self, source, target, options=[]):
        pass

    def checksum(self, filename):
        cmd = self.checksum_cmd(filename)
        ok, result = self.exec(cmd)
        if ok:
            return self.checksum_parse(result)
        else:
            # FIXME: not sure this is the desired behaviour...
            raise RuntimeError(f'Failed to checksum file: {filename}')
        return 'dummy'

    def exec(self, cmd, timeout=15, throw_on_timeout=False, debug=False):
        proc = subproc.Popen(cmd, stdout=subproc.PIPE)
        lines = []
        try:
            outs, errs = proc.communicate(timeout=timeout)
            lines = outs.splitlines()
            # print(lines)
        except subproc.TimeoutExpired as exc:
            print(f'[exec] cmd: {cmd} Time-out exceeded!')
            proc.kill()
            outs, errs = proc.communicate()
            print(outs)
            print(errs)
            if throw_on_timeout:
                raise exc
        if debug:
            print(f'cmd: {cmd} return code: {proc.returncode}')
        return (proc.returncode == 0),lines

    def copy(self, source, target, silent=False):
        cmd = self.copy_cmd(source, target)
        ok, result = False, []
        try:
            ok, result = self.exec(cmd, 60, True)
        except subproc.TimeoutExpired:
            # in this case we retry after some waiting time to randomize acces and adding --continue to the copy command
            # when possible
            time.sleep(5)
            cmd = self.copy_cmd(source, target, ['--continue'])
            ok, result = self.exec(cmd, 60)
        if not silent:
            print(result)
        if not ok:
            source_cks = filesystem(source).checksum(source)
            target_cks = filesystem(target).checksum(target)
            print(f'ckecksums source: {source_cks}, target {target_cks}')
            return source_cks == target_cks
        
        return ok




class XrdFileSystem(FileSystem):
    def __init__(self, protocol) -> None:
        super().__init__(protocol)
        self.cmd_base_ = ['xrdfs', self.protocol]

    def list_dir_cmd(self,  path, recursive=False):
        ls_cmd = []
        ls_cmd.extend(self.cmd_base_)
        ls_cmd.extend(['ls', '-l'])
        if recursive:
            ls_cmd.append('-R')
        ls_cmd.append(path)
        return ls_cmd
    
    def list_dir_parse(self, lines, path):
        ret = []
        for line in lines:
            line = line.decode('utf-8')
            parts = line.split()
            if len(parts) == 7:
                ret.append(
                    FileEntry(
                        name=parts[6], 
                        date=f'{parts[3]} {parts[4]}', 
                        attributes=parts[0], 
                        owner=parts[1], 
                        group=parts[2], 
                        size=parts[3]))
            else:
                ret.append(
                    FileEntry(
                        name=parts[4], 
                        date=f'{parts[1]} {parts[2]}', 
                        attributes=parts[0], 
                        owner='', 
                        group='', 
                        size=parts[2]))
        return ret

    def checksum_cmd(self, filename):
        cmd= []
        cmd.extend(self.cmd_base_)
        cmd.extend(['query', 'checksum', filename])
        return cmd
  
    def checksum_parse(self, results):
        return results[0].split()[1]

    def copy_cmd(self, source, target, options=[]):
        ret = ['xrdcp']
        ret.extend(options)
        ret.extend([file_name_wprotocol(source), file_name_wprotocol(target)])
        return ret

class LocalFileSystem(FileSystem):
    def __init__(self, protocol) -> None:
        super().__init__(protocol)
    
    def list_dir_cmd(self,  path, recursive=False):
        ls_cmd = []
        ls_cmd.extend(self.cmd_base_)
        ls_cmd.extend(['ls', '-l'])
        if recursive:
            ls_cmd.append('-R')
        ls_cmd.append(path)
        return ls_cmd
        
    
    def list_dir_parse(self, lines, path):
        ret = []
        for line in lines:
            # move to string for all the rest of the operations
            line = line.decode('utf-8')
            parts = line.split()
            if len(parts) == 0:
                continue
            elif len(parts) == 1:
                if parts[0][-1] == ':':
                    path = (parts[0].rstrip(':'))
                else:
                    print(f'ERROR [LocalFileSystem::list_dir_parse] parsing: {parts}')
                continue
            elif len(parts) == 2:
                if parts[0] == 'total':
                    continue
                else:
                    print(f'ERROR [LocalFileSystem::list_dir_parse] parsing: {parts}')
            ret.append(FileEntry(os.path.join(path, parts[8]), f'{parts[4]} {parts[5]} {parts[6]} {parts[7]}', parts[0], parts[2], parts[3], parts[4]))
        return ret

    def checksum_cmd(self, filename):
        cmd = self.cmd_base_
        cmd.extend(['xrdadler32', filename])
        return cmd

    def checksum_parse(self, results):
        return results[0].split()[0]
    
    def copy_cmd(self, source, target, options=[]):
        if '--continue' in options:
            options.remove('continue')
        ret = ['cp']
        ret.extend(options)
        ret.extend([source, target])
        return ret



def filesystem(filename):
    protocol = get_eos_protocol(filename)
    fs = XrdFileSystem(protocol)
    if protocol == '':
        fs = LocalFileSystem(protocol)
    return fs


def get_checksum(filename):
    fs = filesystem(filename)
    return fs.checksum(filename)
#     xrdfs root://eosuser.cern.ch/  query checksum /eos/user/c/cerminar/hgcal/CMSSW1015/plots/histos_ele_flat2to100_PU200_v55_93.root
#     xrdadler32 plots1/histos_ele_flat2to100_PU200_v55_93.root


def get_eos_protocol(dirname):
    protocol = ''
    if '/eos/user/' in dirname:
        protocol = 'root://eosuser.cern.ch/'
    elif '/eos/cms/' in dirname:
        protocol = 'root://eoscms.cern.ch/'
    return protocol


def file_name_wprotocol(filename):
    protocol = get_eos_protocol(filename)
    return f'{protocol}{filename}'


def copy_from_eos(input_dir, file_name, target_file_name, dowait=False, silent=False):
    fs = XrdFileSystem(get_eos_protocol(input_dir))
    return fs.copy(os.path.join(input_dir, file_name), target_file_name, silent)
    

def copy_to_eos(file_name, target_dir, target_file_name):
    fs = XrdFileSystem(get_eos_protocol(target_dir))
    return fs.copy(file_name, os.path.join(target_dir, target_file_name))


def listFiles(input_dir, match='.root', recursive=True, debug=0):
    fs = filesystem(input_dir)
    allfiles = fs.list_dir(input_dir, recursive)
    matchedfiles = [f.name for f in allfiles if match in f.name]
    return sorted(matchedfiles)


def stage_files(files_to_stage):
    ret_files = []
    for file_name in files_to_stage:
        copy_ok = copy_from_eos(os.path.dirname(file_name), os.path.basename(file_name), os.path.basename(file_name))
        print(f'copy of file {file_name}, returned: {copy_ok}')
        if not copy_ok:
            print('  copy of file {file_name} failed, skipping')
            continue
        # FIXME: this is a very loose check...
        if os.path.isfile(os.path.basename(file_name)):
            ret_files.append(os.path.basename(file_name))
        else:
            ret_files.append(file_name)
    # print ret_files
    return ret_files


def get_files_for_processing(input_dir, tree, nev_toprocess, debug=0):
    metadata = get_metadata(input_dir, tree, debug)
    # return ['./ntuple_1417.root']
    return get_files_to_process(nev_toprocess, metadata, debug)
    # return stage_files(files_to_stage=get_files_to_process(nev_toprocess, metadata, debug))


def get_files_and_events_for_batchprocessing(input_dir, tree, nev_toprocess, nev_perjob, batch_id, debug=0):
    metadata = get_metadata(input_dir, tree, debug)
    file_list, event_range = get_njobs(nev_toprocess, nev_perjob, metadata, debug)[batch_id]
    return stage_files(files_to_stage=file_list), event_range
    # return file_list, event_range


def get_number_of_jobs_for_batchprocessing(input_dir, tree, nev_toprocess, nev_perjob, debug=0):
    metadata = get_metadata(input_dir, tree, debug)
    return len(get_njobs(nev_toprocess, nev_perjob, metadata, debug).keys())


def get_metadata(input_dir, tree, debug=0):
    json_name = 'metadata.json'
    file_metadata = {}
    json_files = listFiles(input_dir, match=json_name)
    if len(json_files) == 0:
        print(f'no metadata file {json_name} in input dir: {input_dir}')
        print('Will now index files...')
        files = listFiles(input_dir)
        print(f'# of files: {len(files)}')

        for idx, file_name in enumerate(files):
            nevents = 0
            try:
                tree_file = up.open(file_name_wprotocol(file_name))
                nevents = tree_file[tree].num_entries
                tree_file.close()
            except OSError as error:
                print(error.strerror)
                print(f'WARNING: file {file_name} can not be indexed, skipping!')
                continue 
            
            file_metadata[file_name] = nevents
            if debug > 2:
                print(f' [{idx}] file: {file_name} # events: {nevents}')

        with open(json_name, 'w', encoding='utf-8') as fp:
            json.dump(file_metadata, fp)
        retc = copy_to_eos(
            file_name=json_name,
            target_dir=input_dir,
            target_file_name=json_name)
        print(f'COPY: {json_name} to {input_dir} w name: {json_name} return: {retc}')
    else:
        print('dir already indexed, will read metadata...')
        unique_filename = f'{uuid.uuid4()}.json'
        ret = copy_from_eos(input_dir=input_dir,
                            file_name=json_name,
                            target_file_name=unique_filename)
        print(f'copy file: {unique_filename} ret: {ret}')
        with open(unique_filename, 'r', encoding='utf-8') as fp:
            file_metadata = json.load(fp)
        os.remove(unique_filename)

    return file_metadata


def get_files_to_process(nev_toprocess, metadata, debug=0):
    nevents_tot = 0
    for key, value in metadata.items():
        if debug > 4:
            print(key, value)
        # FIXME: if value is 0 maybe one should check again and rewrite the json?
        nevents_tot += int(value)
    if debug > 2:
        print(f'Tot.# events: {nevents_tot}')

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
        print(files_sofar)
        print(f'# of files: {len(files_sofar)}')
    return files_sofar


def get_njobs(nev_toprocess, nev_perjob, metadata, debug=0):

    needed_files = sorted(get_files_to_process(nev_toprocess, metadata, debug))
    nevents_tot = 0
    comulative = {}
    for file_name in needed_files:
        comulative[file_name] = nevents_tot
        nevents_tot += int(metadata[file_name])

    if debug > 3:
        print(f'Tot.# events: {nevents_tot}')
    if nev_toprocess == -1:
        nev_toprocess = nevents_tot

    njobs = int(nev_toprocess/nev_perjob)
    print(f'# of jobs: {njobs}')
    ret = {}
    for job_id in range(0, njobs):
        files_perjob = []
        eventrange = (-1, -1)
        events_injob = range(job_id*nev_perjob, (job_id+1)*nev_perjob)
        first_ev_injob = events_injob[0]
        last_ev_injob = events_injob[-1]
        if debug > 3:
            print(f' jobid: {job_id}, i: {first_ev_injob} e: {last_ev_injob}')
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
            print(f'   files: {files_perjob}, range: {eventrange}')
        totv = 0
        for file_n in files_perjob:
            if debug > 3:
                print(f'    file: {file_n} ({metadata[file_n]})')
            totv += metadata[file_n]
        if debug > 3:
            print(f'   # ev in files: {totv}')
        ret[job_id] = (files_perjob, eventrange)
    return ret


if __name__ == "__main__":
    """
    Meant to test the module functionality

    Run from main directory using:
    python -m python.file_manager

    """

    #
    # file_metadata = get_metadata(input_dir='/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1015/SingleE_FlatPt-2to100/SingleE_FlatPt-2to100_PU0_v11/180814_140939/0000/',
    #                              tree='hgcalTriggerNtuplizer/HGCalTriggerNtuple')
    #
    # get_files_to_process(nev_toprocess=20, metadata=file_metadata)
    #
    # get_files_to_process(nev_toprocess=1000, metadata=file_metadata)
    #
    # get_files_to_process(nev_toprocess=3000, metadata=file_metadata)
    #
    # get_files_to_process(nev_toprocess=-1, metadata=file_metadata)
    #
    # get_njobs(nev_toprocess=3000, nev_perjob=500, metadata=file_metadata)
    #
    # jobs = get_njobs(nev_toprocess=8000, nev_perjob=2000, metadata=file_metadata)
    #
    # # copy_from_eos(input_dir='/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1015/SingleE_FlatPt-2to100/SingleE_FlatPt-2to100_PU0_v11/180814_140939/0000/',
    # #               file_name='ntuple_23.root', target_file_name='pippo.json')
    #
    # # copy_to_eos(file_name='data.json',
    # #             target_dir='/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1015/SingleE_FlatPt-2to100/SingleE_FlatPt-2to100_PU0_v11/180814_140939/0000/',
    # #             target_file_name='metadata.json')
    #
    # print jobs

    print('Local fs:')
    local_fs = LocalFileSystem(protocol='root://localhost')
    dir = u'/Users/cerminar/cernbox/hgcal/CMSSW1015/'
    print(f'list dir: :{dir}')
    for f in local_fs.list_dir(path=dir):
        print(f)
    
    dir = u'/Users/cerminar/CERNbox/hgcal/CMSSW1015/plots/'
    print(f'list dir: :{dir}')
    rfiles = [f.name for f in local_fs.list_dir(path=dir) if '.root' in f.name]
    print (f'# files: {len(rfiles)}')
    

    print(f'Checksum file: {rfiles[0]}: {local_fs.checksum(rfiles[0])}')

    print ('List eos dir: /eos/cms/store/cmst3/group/l1tr/cerminar/l1teg/ntuples/TT_TuneCP5_14TeV-powheg-pythia8/TT_PU200_v81C/')
    xrd_fs = XrdFileSystem(protocol='root://eoscms.cern.ch')
    res =  xrd_fs.list_dir(path=u'/eos/cms/store/cmst3/group/l1tr/cerminar/l1teg/ntuples/TT_TuneCP5_14TeV-powheg-pythia8/TT_PU200_v81C/')
    print([f.name for f in res])

    for f in xrd_fs.list_dir(path=u'/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1110pre6/NeutrinoGun_E_10GeV/NuGunAllEta_PU200_v53/'):
        print(f)

    dir = u'/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1110pre6/NeutrinoGun_E_10GeV/NuGunAllEta_PU200_v53/'
    rfiles = [f.name for f in xrd_fs.list_dir(path=dir, recursive=True) if '.root' in f.name]
    print(f'Checksum file: {rfiles[0]}: {xrd_fs.checksum(rfiles[0])}')



    local_dir = u'/Users/cerminar/cernbox/hgcal/CMSSW1015/'
    local_files = listFiles(local_dir, match='.root')
    print(len(local_files))

    input_dir = u'/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1110pre6/NeutrinoGun_E_10GeV/NuGunAllEta_PU200_v53/'
    # input_dir = '/Users/cerminar/Workspace/hgcal-analysis/ntuple-tools/'
    found_files = listFiles(input_dir, match='.root')
    print(found_files)
    print(f'# of files: {len(found_files)}')

    # # input_dir='/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1061p2/NeutrinoGun_E_10GeV/NuGunAllEta_PU200_v29/190902_144701/0000/'
    # tree_name = 'hgcalTriggerNtuplizer/HGCalTriggerNtuple'
    # input_files, range_ev = get_files_and_events_for_batchprocessing(
    #     input_dir=input_dir,
    #     tree=tree_name,
    #     nev_toprocess=-1,
    #     nev_perjob=200,
    #     batch_id=121,
    #     debug=True)


    input_dir = '/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1110pre6//DoubleElectron_FlatPt-1To100/DoubleElectron_FlatPt-1To100_PU0_v63A/'
    tree_name = 'l1CaloTriggerNtuplizer_egOnly/HGCalTriggerNtuple'
    nev_toprocess = 100
    print(f'Input dir: {input_dir}, tree_name: {tree_name}, nev: {nev_toprocess}')
    files = get_files_for_processing(input_dir, tree_name, nev_toprocess, debug=4)
    print(f'   files for processing: {files}')

    input_dir = '/eos/cms/store/cmst3/group/l1tr/cerminar/hgcal/CMSSW1110pre6//DoubleElectron_FlatPt-1To100/DoubleElectron_FlatPt-1To100_PU200_v63B/'
    tree_name = 'l1CaloTriggerNtuplizer_egOnly/HGCalTriggerNtuple'
    nev_toprocess = 100
    print(f'Input dir: {input_dir}, tree_name: {tree_name}, nev: {nev_toprocess}')
    #files = get_files_for_processing(input_dir, tree_name, nev_toprocess, debug=4)
    #print(f'   files for processing: {files}')

    json_name = 'metadata.json'
    file_metadata = {}
    json_files = listFiles(input_dir, match=json_name)


    # get_checksum(filename=plots1/histos_nugun_alleta_pu200_v55.root)

# get_njobs(nev_toprocess=-1, nev_perjob=500, metadata=file_metadata)
