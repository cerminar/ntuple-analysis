import os
import socket

from rich import print as pprint
from rich.console import Console
from rich.table import Table


class Parameters(dict):
    def __getattr__(self, name):
        return self[name]

    def __str__(self):
        return (
            f"\n--------------------Parameters--------------------\n"
            f"{'Name':<16}: {self.name:<12}\n"
            f"{'clusterize':<16}: {self.clusterize:<12}\n"
            f"{'compute density':<16}: {self.computeDensity:<12}\n"
            f"{'maxEvents':<16}: {self.maxEvents:<12}\n"
            f"{'output file':<16}: {self.output_filename:<12}\n"
            f"{'events per job':<16}: {self.events_per_job:<12}\n"
            f"{'debug':<16}: {self.debug:<12}"
            "\n"
        )

    def __repr__(self):
        return self.name

    def print(self):
        table = Table(title="Parameters")
        table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Name", self.name)
        table.add_row("clusterize", str(self.clusterize))
        table.add_row("compute density", str(self.computeDensity))
        table.add_row("maxEvents", str(self.maxEvents))
        table.add_row("output file", self.output_filename)
        table.add_row("events per job", str(self.events_per_job))
        table.add_row("debug", str(self.debug))
        console = Console()
        console.print(table)


def get_collection_parameters(opt, cfgfile):
    outdir = cfgfile["common"]["output_dir"]["default"]
    hostname = socket.gethostname()
    for machine, odir in cfgfile["common"]["output_dir"].items():
        if machine in hostname:
            outdir = odir
    plot_version = f"{cfgfile['common']['plot_version']}.{cfgfile['dataset']['version']}"

    collection_params = {}
    print("")
    for collection, collection_data in cfgfile["collections"].items():
        samples = cfgfile["samples"].keys()
        pprint(f"--- Collection: {collection} with samples: {samples}")
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
