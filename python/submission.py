#ruff: noqa
import os
import sys

import subprocess32

import python.file_manager as fm


def editTemplate(infile, outfile, params):
    template_file = open(infile)
    template = template_file.read()
    template_file.close()

    for param in params:
        template = template.replace(param, params[param])

    out_file = open(outfile, "w")
    out_file.write(template)
    out_file.close()


def to_HTCondor(*, analyze, opt, submit_mode, plot_version, samples_to_process):
    if opt.BATCH and not opt.RUN:
        batch_dir = f"batch_{opt.COLLECTION}_{plot_version}"
        if not os.path.exists(batch_dir):
            os.mkdir(batch_dir)
            os.mkdir(batch_dir + "/conf/")
            os.mkdir(batch_dir + "/logs/")

        dagman_sub = ""
        dagman_dep = ""
        dagman_ret = ""
        for sample in samples_to_process:
            dagman_spl = ""
            dagman_spl_retry = ""
            sample_batch_dir = os.path.join(batch_dir, sample.name)
            sample_batch_dir_logs = os.path.join(sample_batch_dir, "logs")
            os.mkdir(sample_batch_dir)
            os.mkdir(sample_batch_dir_logs)
            print(sample)
            nevents = int(opt.NEVENTS)
            n_jobs = fm.get_number_of_jobs_for_batchprocessing(
                input_dir=os.path.join(sample.input_base_dir, sample.input_sample_dir),
                tree=sample.tree_name,
                nev_toprocess=nevents,
                nev_perjob=sample.events_per_job,
                debug=int(opt.DEBUG),
            )
            print(f"Total # of events to be processed: {nevents}")
            print(f"# of events per job: {sample.events_per_job}")
            if n_jobs == 0:
                n_jobs = 1
            print(f"# of jobs to be submitted: {n_jobs}")
            sample["nbatch_jobs"] = n_jobs

            params = {}
            params["TEMPL_TASKDIR"] = sample_batch_dir
            params["TEMPL_NJOBS"] = str(n_jobs)
            params["TEMPL_WORKDIR"] = os.environ["PWD"]
            params["TEMPL_CFG"] = opt.CONFIGFILE
            params["TEMPL_INPUT"] = opt.DATASETFILE
            params["TEMPL_COLL"] = opt.COLLECTION
            params["TEMPL_SAMPLE"] = sample.name
            params["TEMPL_OUTFILE"] = f"{sample.output_filename_base}.root"
            params["TEMPL_EOSPROTOCOL"] = fm.get_eos_protocol(dirname=sample.output_dir)
            params["TEMPL_INFILE"] = f"{sample.output_filename_base}_*.root"
            params["TEMPL_FILEBASE"] = sample.output_filename_base
            params["TEMPL_OUTDIR"] = sample.output_dir
            params["TEMPL_VIRTUALENV"] = os.path.basename(os.environ["VIRTUAL_ENV"])
            params["TEMPL_VERSION"] = sample.version
            params["TEMPL_JOBFLAVOR"] = sample.htc_jobflavor

            editTemplate(
                infile="templates/batch.sub", outfile=os.path.join(sample_batch_dir, "batch.sub"), params=params
            )

            editTemplate(
                infile="templates/run_batch.sh", outfile=os.path.join(sample_batch_dir, "run_batch.sh"), params=params
            )

            editTemplate(
                infile="templates/copy_files.sh", outfile=os.path.join(sample_batch_dir, "copy_files.sh"), params=params
            )
            os.chmod(os.path.join(sample_batch_dir, "copy_files.sh"), 0o754)

            editTemplate(
                infile="templates/batch_hadd.sub",
                outfile=os.path.join(sample_batch_dir, "batch_hadd.sub"),
                params=params,
            )

            editTemplate(
                infile="templates/run_batch_hadd.sh",
                outfile=os.path.join(sample_batch_dir, "run_batch_hadd.sh"),
                params=params,
            )

            editTemplate(
                infile="templates/batch_cleanup.sub",
                outfile=os.path.join(sample_batch_dir, "batch_cleanup.sub"),
                params=params,
            )

            editTemplate(
                infile="templates/run_batch_cleanup.sh",
                outfile=os.path.join(sample_batch_dir, "run_batch_cleanup.sh"),
                params=params,
            )

            editTemplate(
                infile="templates/hadd_dagman.dag",
                outfile=os.path.join(batch_dir, f"hadd_{sample.name}.dag"),
                params=params,
            )

            editTemplate(
                infile="templates/run_harvest.sh",
                outfile=os.path.join(sample_batch_dir, "run_harvest.sh"),
                params=params,
            )

            editTemplate(
                infile="templates/batch_harvest.sub",
                outfile=os.path.join(sample_batch_dir, "batch_harvest.sub"),
                params=params,
            )

            if submit_mode and opt.LOCAL:
                editTemplate(
                    infile="templates/run_local.sh",
                    outfile=os.path.join(sample_batch_dir, "run_local.sh"),
                    params=params,
                )

            for jid in range(n_jobs):
                dagman_spl += f"JOB Job_{jid} batch.sub\n"
                dagman_spl += f'VARS Job_{jid} JOB_ID="{jid}"\n'
                dagman_spl_retry += f"Retry Job_{jid} 3\n"
                dagman_spl_retry += f"PRIORITY Job_{jid} {sample.htc_priority}\n"

            dagman_sub += f"SPLICE {sample.name} {sample.name}.spl DIR {sample_batch_dir}\n"
            dagman_sub += f"JOB {sample.name + '_hadd'} {sample_batch_dir}/batch_hadd.sub\n"
            dagman_sub += f"JOB {sample.name + '_cleanup'} {sample_batch_dir}/batch_cleanup.sub\n"

            dagman_dep += f"PARENT {sample.name} CHILD {sample.name + '_hadd'}\n"
            dagman_dep += f"PARENT {sample.name + '_hadd'} CHILD {sample.name + '_cleanup'}\n"

            dagman_ret += f"Retry {sample.name + '_hadd'} 3\n"
            dagman_ret += f"PRIORITY {sample.name + '_hadd'} {sample.htc_priority}\n"

            dagman_splice = open(os.path.join(sample_batch_dir, f"{sample.name}.spl"), "w")
            dagman_splice.write(dagman_spl)
            dagman_splice.write(dagman_spl_retry)
            dagman_splice.close()

        dagman_file_name = os.path.join(batch_dir, "dagman.dag")
        dagman_file = open(dagman_file_name, "w")
        dagman_file.write(dagman_sub)
        dagman_file.write(dagman_dep)
        dagman_file.write(dagman_ret)
        dagman_file.close()

        # create targz file of the code from git
        git_proc = subprocess32.Popen(
            ["git", "archive", "--format=tar.gz", "HEAD", "-o", os.path.join(batch_dir, "ntuple-tools.tar.gz")],
            stdout=subprocess32.PIPE,
        )
        git_proc.wait()
        # cp TEMPL_TASKDIR/TEMPL_CFG
        print("Ready for HT-Condor submission please run the following commands:")
        print(f"condor_submit_dag {dagman_file_name}")

        if submit_mode and opt.LOCAL:
            print("will now run jobs on local resources")
            analyze(batch_dir, samples_to_process, opt.WORKERS, opt.WORKDIR)

        sys.exit(0)

    if submit_mode:
        sys.exit(0)
