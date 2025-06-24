import sys
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys_env = os.environ.copy()
librarypath_cmd = ""
# We've used os.environ.copy() so we can make mods
# to the subprocess environment if needed without
# affecting the parent process.

if sys.platform == 'darwin':
    if "LD_LIBRARY_PATH" in sys_env:
        librarypath_cmd = f"export LD_LIBRARY_PATH={sys_env['LD_LIBRARY_PATH']} && {librarypath_cmd}"
    if "DYLD_LIBRARY_PATH" in sys_env:
        librarypath_cmd = f"export DYLD_LIBRARY_PATH={sys_env['DYLD_LIBRARY_PATH']} && {librarypath_cmd}"

print(librarypath_cmd)
# Define all tasks and their shell commands
TASKS = {
    "eg_genmatch_menu": (
        librarypath_cmd +
        "python analyzeNtuples.py -f cfg/eg_genmatch.yaml "
        "-i cfg/datasets/ntpfp_{ver}.yaml -p egmenu -s doubleele_flat1to100_PU200 -n -1 -d 0 && "
        "cp -v {file_dir}/histos_doubleele_flat1to100_PU200_egmenu_v200D.{ver}i.root "
        "{file_dir}/histos_doubleele_flat1to100_PU200_egmenu_v200D.{ver}.root"
    ),
    "eg_genmatch_ctl2": (
        librarypath_cmd +
        "python analyzeNtuples.py -f cfg/eg_genmatch.yaml "
        "-i cfg/datasets/ntpfp_{ver}.yaml -p ctl2_tkeg -s doubleele_flat1to100_PU200 -n -1 -d 0 && "
        "cp -v {file_dir}/histos_doubleele_flat1to100_PU200_eg_v200D.{ver}i.root "
        "{file_dir}/histos_doubleele_flat1to100_PU200_eg_v200D.{ver}.root"
    ),
    "eg_unmatched": (
        librarypath_cmd +
        "python analyzeNtuples.py -f cfg/egplots.yaml "
        "-i cfg/datasets/ntpfp_{ver}.yaml -p tkeg_plots -s doubleele_flat1to100_PU200 -n -1 -d 0 && "
        "cp -v {file_dir}/histos_doubleele_flat1to100_PU200_egplots_v160A.{ver}i.root "
        "{file_dir}/histos_doubleele_flat1to100_PU200_egplots_v160A.{ver}.root"
    ),
    "eg_rate_menu": (
        librarypath_cmd +
        "python analyzeNtuples.py -f cfg/eg_rate.yaml "
        "-i cfg/datasets/ntpfp_{ver}.yaml -p rate_menu -s nugun_alleta_pu200 -n 500000 -d 0 && "
        "cp -v {file_dir}/histos_nugun_alleta_pu200_egratemenu_v200C.{ver}i.root "
        "{file_dir}/histos_nugun_alleta_pu200_egratemenu_v200C.{ver}.root"
    ),
    "eg_rate_ctl2": (
        librarypath_cmd +
        "python analyzeNtuples.py -f cfg/eg_rate.yaml "
        "-i cfg/datasets/ntpfp_{ver}.yaml -p rate_ctl2 -s nugun_alleta_pu200 -n 500000 -d 0 && "
        "cp -v {file_dir}/histos_nugun_alleta_pu200_egrate_v200C.{ver}i.root "
        "{file_dir}/histos_nugun_alleta_pu200_egrate_v200C.{ver}.root"
    ),
    "eg_rate_counter": (
        librarypath_cmd +
        "python analyzeNtuples.py -f cfg/eg_rate.yaml "
        "-i cfg/datasets/ntpfp_{ver}.yaml -p rate_counter_menu -s nugun_alleta_pu200 -n 500000 -d 0 && "
        "cp -v {file_dir}/histos_nugun_alleta_pu200_egratecount_v200C.{ver}i.root "
        "{file_dir}/histos_nugun_alleta_pu200_egratecount_v200C.{ver}.root"
    ),
    "met_rate": (
        librarypath_cmd +
        "python analyzeNtuples.py -f cfg/jetmet_rate.yaml "
        "-i cfg/datasets/ntpfp_{ver}.yaml -p met -s nugun_alleta_pu200 -n 500000 -d 0 && "
        "cp -v {file_dir}/histos_nugun_alleta_pu200_jmratemet_v200A.{ver}i.root "
        "{file_dir}/histos_nugun_alleta_pu200_jmratemet_v200A.{ver}.root"
    ),
    "jet_rate": (
        librarypath_cmd +
        "python analyzeNtuples.py -f cfg/jetmet_rate.yaml "
        "-i cfg/datasets/ntpfp_{ver}.yaml -p jets -s nugun_alleta_pu200 -n 500000 -d 0 && "
        "cp -v {file_dir}/histos_nugun_alleta_pu200_jmratejets_v200A.{ver}i.root "
        "{file_dir}/histos_nugun_alleta_pu200_jmratejets_v200A.{ver}.root"
    ),
    "jet_genmatch": (
        librarypath_cmd +
        "python analyzeNtuples.py -f cfg/jetmet_genmatch.yaml "
        "-i cfg/datasets/ntpfp_{ver}.yaml -p jets -s ttbar_PU200 -n -1 -d 0 && "
        "cp -v {file_dir}/histos_ttbar_PU200_jets_v200C.{ver}i.root "
        "{file_dir}/histos_ttbar_PU200_jets_v200C.{ver}.root"
    ),
}

# Define groups
GROUPS = {
    "eg_all": ["eg_genmatch_menu", "eg_genmatch_ctl2", "eg_unmatched"],
    "eg_rate": ["eg_rate_menu", "eg_rate_ctl2", "eg_rate_counter"],
    "jetmet": ["met_rate", "jet_genmatch", "jet_rate"],
}


def expand_tasks(what):
    """Expand group names to their tasks, remove duplicates, preserve order."""
    result = []
    seen = set()
    for item in what:
        if item in GROUPS:
            for t in GROUPS[item]:
                if t not in seen:
                    result.append(t)
                    seen.add(t)
        elif item in TASKS:
            if item not in seen:
                result.append(item)
                seen.add(item)
        elif item == "all":
            for t in TASKS:
                if t not in seen:
                    result.append(t)
                    seen.add(t)
    return result

def run_task(taskname, ver, file_dir):
    cmd = TASKS[taskname].format(ver=ver, file_dir=file_dir)
    print(f"Running: {taskname}")
    result = subprocess.run(cmd, shell=True)
    return (taskname, result.returncode)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run validation tasks in parallel.")
    parser.add_argument("test_version", help="Test version string (e.g. 140Xv0B10)")
    parser.add_argument("what", nargs="?", default="all", help="Comma-separated list of tasks or groups (default: all)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel workers (default: 2)")
    parser.add_argument("--file-dir", default='./plots/', help="Directory for input files (default: ./plots/)")

    args = parser.parse_args()

    what_list = [w.strip() for w in args.what.split(",")]
    tasks_to_run = expand_tasks(what_list)

    if not tasks_to_run:
        print("No valid tasks to run.")
        sys.exit(1)

    print(f"Tasks to run: {tasks_to_run}")
    success, failed = [], []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, t, args.test_version, args.file_dir): t for t in tasks_to_run}
        for future in as_completed(futures):
            taskname, rc = future.result()
            if rc == 0:
                success.append(taskname)
            else:
                failed.append(taskname)

    print("\n========== SUMMARY ==========")
    if success:
        print("Successful tasks:")
        for t in success:
            print(f"  {t}")
    else:
        print("No successful tasks.")

    if failed:
        print("Failed tasks:")
        for t in failed:
            print(f"  {t}")
    else:
        print("No failed tasks.")
    print("=============================")

if __name__ == "__main__":
    main()