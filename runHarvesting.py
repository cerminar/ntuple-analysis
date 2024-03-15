import logging
import multiprocessing
import optparse
import os
import sys
import time
import traceback
from shutil import copyfile

import ROOT
import subprocess32

import python.file_manager as fm

logger = multiprocessing.log_to_stderr()
logger.setLevel(logging.DEBUG)

sentinel = -1


def data_creator(input_dir, sample_name, version, q):
    """
    Creates data to be consumed and waits for the consumer
    to finish processing
    """
    logger.info('Creating data and putting it on the queue')

    ncopied = 0
    while True:
        data = fm.listFiles(input_dir)

        for id, item in enumerate(data):
            file_name = os.path.split(item)[1]
            # print file_name
            # print id
            if sample_name in item and f'{version}_' in item:
                # or not os.path.isfile('{}.checked'.format(os.path.splitext(file)[0])):
                if os.path.isfile(file_name):
                    if not os.path.isfile(f'{os.path.splitext(file_name)[0]}.checked'):
                        # logger.debug ('file {} exists but check failed...'.format(file_name))
                        remote_checksum = fm.get_checksum(item)
                        local_checksum = fm.get_checksum(file_name)
                        if remote_checksum == local_checksum:
                            logger.debug(f'   remote checksum for file: {file_name} did not change...skipping for now')
                            continue
                        else:
                            logger.info(f'   remote checksum for file: {file_name} changed: will copy it again')
                    else:
                        continue
                copy_ret = fm.copy_from_eos(input_dir=input_dir,
                                            file_name=file_name,
                                            target_file_name=file_name,
                                            dowait=True,
                                            silent=True)
                logger.debug(f'copy returned: {copy_ret}')
                if copy_ret == 0:
                    q.put(file_name)
                    ncopied += 1

            if ncopied > 999:
                q.put(sentinel)
                break
        if ncopied > 999:
            break
        time.sleep(20)


def data_checker(queue_all, queue_ready):
    """
    Consumes some data and works on it
    """
    logger.info('Checking files and putting it on the queue "queue_ready"')

    while True:
        data = queue_all.get()
        if data is sentinel:
            queue_ready.put(sentinel)
            break

        # print('data found to be processed: {}'.format(data))
        file = ROOT.TFile(os.path.join(fm.get_eos_protocol(data), data))
        if len(file.GetListOfKeys()) == 0:
            logger.info(f'file: {data} is not OK')
        else:
            fname = f'{os.path.splitext(data)[0]}.checked'
            open(fname, 'a').close()
            if not os.path.isfile(f'{os.path.splitext(data)[0]}.hadded'):
                queue_ready.put(data)
            else:
                logger.debug(f'file: {data} has already been hadded...skipping it')
        file.Close()


def data_consumer(sample_name, version, queue_ready, queue_tomove):
    logger.info('Starting data consumer')
    out_file_name = f'{sample_name}_{version}_temp.root'
    new_data = []
    index = 0
    while True:
        data = queue_ready.get()

        if data is sentinel:
            queue_tomove.put(sentinel)
            break
        new_data.append(data)
        if(len(new_data) >= 20):
            logger.info(f'Launch hadd on {len(new_data)} files: ')
            hadd_proc = subprocess32.Popen(['hadd', '-a', '-j', '2', '-k', out_file_name]+new_data, stdout=subprocess32.PIPE, stderr=subprocess32.STDOUT)
            hadd_proc.wait()
            if hadd_proc.returncode == 0:
                logger.info(f'   hadd succeeded with exit code: {hadd_proc.returncode}')
                logger.debug(f'   hadd output follows: {hadd_proc.stdout.readlines()}')
                index += 1
                for file in new_data:
                    fname = f'{os.path.splitext(file)[0]}.hadded'
                    open(fname, 'a').close()
                out_file_name_copy = f'{sample_name}_tocopy_{index}.root'
                copyfile(out_file_name, out_file_name_copy)
                queue_tomove.put(out_file_name_copy)
                del new_data[:]
                logger.debug(f'  resetting file list for hadd operation to {len(new_data)}')
            else:
                logger.info(f'   hadd failed with exit code: {hadd_proc.returncode}')
                logger.debug(f'   hadd output follows: {hadd_proc.stdout.readlines()}')
                file = ROOT.TFile(out_file_name)
                if len(file.GetListOfKeys()) == 0:
                    logger.info(f'file: {out_file_name} is not OK')
                else:
                    logger.info(f'file: {out_file_name} is OK, will retry hadding!')
                file.Close()


def data_mover(sample_name, version, out_dir, queue_tomove):
    logger.info('Starting data mover')
    while True:
        data = queue_tomove.get()
        if data is sentinel:
            break
        out_file_name = f'{sample_name}t.root'
        fm.copy_to_eos(data, out_dir, out_file_name)


def main():

    usage = 'usage: %prog [options]\n%prog -h for help'
    parser = optparse.OptionParser(usage)
    parser.add_option('-i', '--input-dir',
                      dest='INPUTDIR',
                      help='input directory (can be an EOS path)')
    parser.add_option('-s', '--sample-name',
                      dest='FILENAMEBASE',
                      help='name of the sample file base (part of the file-name)')
    parser.add_option('-v', '--version',
                      dest='VERSION',
                      help='version of the processing (part of the filename)')
    parser.add_option('-o', '--output-dir',
                      dest='OUTPUTDIR',
                      help='output directory (can be an EOS path)')

    global opt, args
    (opt, args) = parser.parse_args()

    input_dir = opt.INPUTDIR
    version = opt.VERSION
    sample_name = opt.FILENAMEBASE
    out_dir = opt.OUTPUTDIR

    logger.info('Starting...')

    q = multiprocessing.Queue()
    queue_ready = multiprocessing.Queue()
    queue_tomove = multiprocessing.Queue()
    # r1 = pool.apply_async(func=data_creator, args=(input_dir, sample_name, version, q))
    # r2 = pool.apply_async(func=data_checker, args=(q, queue_ready))
    processes = []
    processes.append(multiprocessing.Process(target=data_creator,
                                             args=(input_dir, sample_name, version, q)))
    processes.append(multiprocessing.Process(target=data_checker,
                                             args=(q, queue_ready)))
    processes.append(multiprocessing.Process(target=data_consumer,
                                             args=(sample_name, version, queue_ready, queue_tomove)))
    processes.append(multiprocessing.Process(target=data_mover,
                                             args=(sample_name, version, out_dir, queue_tomove)))

    for proc in processes:
        proc.start()

    q.close()
    q.join_thread()
    queue_ready.close()
    queue_ready.join_thread()
    queue_tomove.close()
    queue_tomove.join_thread()

    for proc in processes:
        proc.join()
    return 0


if __name__ == '__main__':
    try:
        status = main()
        sys.exit(status)
    except Exception as inst:
        print(str(inst))
        print('Unexpected error:', sys.exc_info()[0])
        traceback.print_exc()
        sys.exit(100)
