import os
import socket


class Parameters(dict):
    def __getattr__(self, name):
        return self[name]

    def __str__(self):
        return (
            f"Name: {self.name},\n"
            f"clusterize: {self.clusterize}\n"
            f"compute density: {self.computeDensity}\n"
            f"maxEvents: {self.maxEvents}\n"
            f"output file: {self.output_filename}\n"
            f"events per job: {self.events_per_job}\n"
            f"debug: {self.debug}"
        )

    def __repr__(self):
        return self.name


def get_collection_parameters(opt, cfgfile):
    outdir = cfgfile["common"]["output_dir"]["default"]
    hostname = socket.gethostname()
    for machine, odir in cfgfile["common"]["output_dir"].items():
        if machine in hostname:
            outdir = odir
    plot_version = f"{cfgfile['common']['plot_version']}.{cfgfile['dataset']['version']}"

    collection_params = {}
    for collection, collection_data in cfgfile["collections"].items():
        samples = cfgfile["samples"].keys()
        print(f"--- Collection: {collection} with samples: {samples}")
        sample_params = []

        plotters = []
        for plotter in collection_data["plotters"]:
            plotters.extend(plotter)

        for sample in samples:
            events_per_job = -1
            output_filename_base = f"histos_{sample}_{collection_data['file_label']}_{plot_version}"
            out_file_name = f"{output_filename_base}i.root"
            if opt.BATCH:
                events_per_job = cfgfile["samples"][sample]["events_per_job"]
                if "events_per_job" in collection_data and sample in collection_data["events_per_job"]:
                        events_per_job = collection_data["events_per_job"][sample]

                if opt.RUN:
                    out_file_name = f"{output_filename_base}_{opt.RUN}.root"

            if opt.OUTDIR:
                outdir = opt.OUTDIR

            out_file = os.path.join(outdir, out_file_name)

            weight_file = None
            if "weights" in collection_data and sample in collection_data["weights"]:
                    weight_file = collection_data["weights"][sample]

            rate_pt_wps = None
            if "rate_pt_wps" in cfgfile["dataset"]:
                rate_pt_wps = cfgfile["dataset"]["rate_pt_wps"]

            priority = 2
            if "priorities" in collection_data and sample in collection_data["priorities"]:
                priority = collection_data["priorities"][sample]

            params = Parameters(
                {
                    "input_base_dir": cfgfile["dataset"]["input_dir"],
                    "input_sample_dir": cfgfile["samples"][sample]["input_sample_dir"],
                    "tree_name": cfgfile["dataset"]["tree_name"],
                    "output_filename_base": output_filename_base,
                    "output_filename": out_file,
                    "output_dir": outdir,
                    "clusterize": cfgfile["common"]["run_clustering"],
                    "eventsToDump": [],
                    "version": plot_version,
                    "calib_version": cfgfile["dataset"]["calib_version"],
                    "rate_pt_wps": rate_pt_wps,
                    "maxEvents": int(opt.NEVENTS),
                    "events_per_job": events_per_job,
                    "computeDensity": cfgfile["common"]["run_density_computation"],
                    "plotters": plotters,
                    "htc_jobflavor": collection_data["htc_jobflavor"],
                    "htc_priority": priority,
                    "weight_file": weight_file,
                    "debug": opt.DEBUG,
                    "name": sample,
                }
            )
            sample_params.append(params)
        collection_params[collection] = sample_params
    return collection_params
