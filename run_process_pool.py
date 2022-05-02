import subprocess32 as subprocess
import os
import shutil

import concurrent.futures as futures
from python.main import main


def work(job_id, cfg_dir, work_main_dir, params):
    print(f'running job {job_id}/{params.nbatch_jobs}, cfg dir: {cfg_dir}, work dir: {work_main_dir}')
    job_name = f'job_{job_id}'
    work_dir_path = os.path.join(work_main_dir, job_name)

    stdoutput_file = open(f'{work_main_dir}/{job_name}.out', "w")
    stderr_file = open(f'{work_main_dir}/{job_name}.err', "w")

    os.mkdir(work_dir_path)
    shutil.copyfile(os.path.join(cfg_dir, 'ntuple-tools.tar.gz'), os.path.join(work_dir_path, 'ntuple-tools.tar.gz'))

    run_p = subprocess.Popen(
        ["sh", os.path.join(cfg_dir, params.name, 'run_local.sh'),  work_dir_path, str(job_id)],
        stdout=stdoutput_file, stderr=stderr_file)
    run_p.wait()
    if run_p.returncode:
        print(f'*** Error: running the job {job_id} for sample: {params.name}')
        return (params.name, job_id, 1)

    return (params.name, job_id, 0)


def submit(batch_cfg_dir,
           sample_params,
           n_workers,
           work_dir):
    print(f'running local submission on {n_workers} workers')
    print(f'batch configuration dir: {batch_cfg_dir}')

    batch_work_dir = os.path.join(work_dir, batch_cfg_dir)
    mkdir_p = subprocess.run(
        ["mkdir", '-p', batch_work_dir],
        capture_output=True)
    if mkdir_p.returncode:
        print(mkdir_p.stdout)
        print(f'*** Error: mkdir {batch_work_dir} failed')
        return 1

    results = []
    hadd_files = {}
    with futures.ThreadPoolExecutor(max_workers=int(n_workers)) as executor:

        for sample in sample_params:
            sample_work_dir = os.path.join(batch_work_dir, sample.name)
            os.mkdir(sample_work_dir)
            print(f'submitting {sample.nbatch_jobs} jobs for sample {sample.name}')
            hadd_files[sample.name] = []
            # inputs = [(job_id, batch_cfg_dir, batch_work_dir) for job_id in range(0, sample.nbatch_jobs)]
            # executor.map(work_unpack, inputs)
            for job_id in range(0, sample.nbatch_jobs):
                results.append(executor.submit(work, job_id, batch_cfg_dir, sample_work_dir, sample))
            #
        for future in futures.as_completed(results):
            res = future.result()
            print(res)
            if res[2] == 0:
                out_file = os.path.join(batch_work_dir, sample.name, f'job_{res[1]}', sample.output_filename_base+f'_{res[1]}.root')
                hadd_files[sample.name].append(out_file)

    # results.append(('doublephoton_flat1to100_PU200', 0, 0))
    # results.append(('doublephoton_flat1to100_PU200', 1, 0))
    # results.append(('doublephoton_flat1to100_PU200', 2, 0))
    #
    # for sample in sample_params:
    #     hadd_files[sample.name] = []
    #     for res in results:
    #         if res[2] == 0:
    #             out_file = os.path.join(batch_work_dir, sample.name, f'job_{res[1]}', sample.output_filename_base+f'_{res[1]}.root')
    #             hadd_files[sample.name].append(out_file)

    for sample in sample_params:
        print(f'will now hadd files for sample: {sample.name}')
        hadded_file = os.path.join(batch_work_dir, sample.name, f'{sample.output_filename_base}l.root')
        print(f' target file: {hadded_file}')
        cmd = ['hadd', '-k', hadded_file] + hadd_files[sample.name]
        print(cmd)
        hadd_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        hadd_proc.wait()
        out_file_name = os.path.join(sample.output_dir, f'{sample.output_filename_base}l.root')
        if hadd_proc.returncode == 0:
            print(f'   hadd succeded: copy file to : {out_file_name}')
            print(hadd_proc.stdout.readlines())
            shutil.copyfile(hadded_file, out_file_name)

    #     index += 1


if __name__ == "__main__":
    main(analyze=submit, submit_mode=True)
