import sys
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from rich import print as rich_print
import yaml


def _out(text, style=None):
    if not style:
        print(text)
        return
    rich_print(f"[{style}]{text}[/{style}]")


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
DEFAULT_CONFIG_FILE = "cfg/validation_tasks.yaml"


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    tasks = config.get("tasks", {})
    groups = config.get("groups", {})

    if not isinstance(tasks, dict):
        raise ValueError("Config key 'tasks' must be a dictionary.")
    if not isinstance(groups, dict):
        raise ValueError("Config key 'groups' must be a dictionary.")

    for task_name, command in tasks.items():
        if not isinstance(command, str):
            raise ValueError(f"Task '{task_name}' must map to a command string.")

    for group_name, group_tasks in groups.items():
        if not isinstance(group_tasks, list) or not all(isinstance(t, str) for t in group_tasks):
            raise ValueError(f"Group '{group_name}' must map to a list of task names.")

    return tasks, groups


def expand_tasks(what, tasks, groups):
    """Expand group names to their tasks, remove duplicates, preserve order."""
    result = []
    seen = set()
    for item in what:
        if item in groups:
            for t in groups[item]:
                if t not in seen:
                    result.append(t)
                    seen.add(t)
        elif item in tasks:
            if item not in seen:
                result.append(item)
                seen.add(item)
        elif item == "all":
            for t in tasks:
                if t not in seen:
                    result.append(t)
                    seen.add(t)
    return result

def run_task(taskname, ver, file_dir, tasks):
    cmd = librarypath_cmd + tasks[taskname].format(ver=ver, file_dir=file_dir)
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
    parser.add_argument("--config", default=DEFAULT_CONFIG_FILE, help=f"Path to YAML config file (default: {DEFAULT_CONFIG_FILE})")

    args = parser.parse_args()

    try:
        tasks, groups = load_config(args.config)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        print(f"Failed to load config '{args.config}': {exc}")
        sys.exit(1)

    what_list = [w.strip() for w in args.what.split(",")]
    tasks_to_run = expand_tasks(what_list, tasks, groups)

    if not tasks_to_run:
        print("No valid tasks to run.")
        sys.exit(1)

    print(f"Tasks to run: {tasks_to_run}")
    success, failed = [], []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, t, args.test_version, args.file_dir, tasks): t for t in tasks_to_run}
        for future in as_completed(futures):
            taskname, rc = future.result()
            if rc == 0:
                success.append(taskname)
            else:
                failed.append(taskname)

    _out("\n========== SUMMARY ==========" , "bold")
    if success:
        _out("Successful tasks:", "green")
        for t in success:
            _out(f"  {t}", "green")
    else:
        _out("No successful tasks.", "yellow")

    if failed:
        _out("Failed tasks:", "red")
        for t in failed:
            _out(f"  {t}", "red")
    else:
        _out("No failed tasks.", "green")
    _out("=============================", "bold")

if __name__ == "__main__":
    main()