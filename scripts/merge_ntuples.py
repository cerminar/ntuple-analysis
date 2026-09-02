#!/usr/bin/env python
"""merge_ntuples.py

Merge ROOT ntuple files N-by-N (default 5) per sample using hadd.

The script reads the same YAML dataset file used by analyzeNtuples so no
paths have to be duplicated.

Optional YAML key
-----------------
  dataset:
    output_dir: /eos/cms/store/.../merged   # EOS destination for merged files

Usage
-----
  python scripts/merge_ntuples.py -i cfg/datasets/ntpfp_140Xv1F5.yaml \\
                                   -o /tmp/merged_local               \\
                                   [--chunk-size 5]                    \\
                                   [--samples ttbar_PU200 nugun_alleta_pu200] \\
                                   [--eos-copy-to /eos/cms/...]        \\
                                   [--dry-run]

EOS destination priority: --eos-copy-to CLI flag > dataset.output_dir in YAML.
If an EOS destination is set, each merged file is xrdcp-ed there after a
successful hadd and then deleted from the local output directory.

The output keeps the same sample sub-directory structure as the input.
Each merged file is named <original_first_file_stem>_merged_<index>.root.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_eos_protocol(path: str) -> str:
    if '/eos/user/' in path:
        return 'root://eosuser.cern.ch/'
    if '/eos/cms/' in path:
        return 'root://eoscms.cern.ch/'
    return ''


def with_protocol(path: str) -> str:
    proto = get_eos_protocol(path)
    return f'{proto}{path}' if proto else path


def list_root_files(directory: str) -> list[str]:
    """Return sorted list of .root file paths under *directory*."""
    proto = get_eos_protocol(directory)
    if proto:
        # Use xrdfs ls to list the remote directory
        host = proto.rstrip('/')  # e.g. root://eoscms.cern.ch
        result = subprocess.run(
            ['xrdfs', host, 'ls', directory],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f'ERROR listing {directory}: {result.stderr.strip()}')
            return []
        files = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.strip().endswith('.root')
        ]
    else:
        if not os.path.isdir(directory):
            print(f'ERROR: directory not found: {directory}')
            return []
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith('.root')
        ]

    return sorted(files)


def chunked(iterable, size: int):
    """Yield successive chunks of *size* from *iterable*."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def ensure_dir(path: str, dry_run: bool) -> None:
    proto = get_eos_protocol(path)
    if proto:
        host = proto.rstrip('/')
        cmd = ['xrdfs', host, 'mkdir', '-p', path]
    else:
        cmd = ['mkdir', '-p', path]
    if dry_run:
        print(f'[dry-run] {" ".join(cmd)}')
    else:
        subprocess.run(cmd, check=True)


def copy_to_eos(local_file: str, eos_base: str, sample_subdir: str, dry_run: bool) -> bool:
    """Copy *local_file* to EOS, mirroring the sample sub-directory structure."""
    eos_dir = f'{eos_base.rstrip("/")}/{sample_subdir}'
    eos_target = with_protocol(f'{eos_dir}/{os.path.basename(local_file)}')
    cmd = ['xrdcp', '-f', '-s', local_file, eos_target]
    print(f'  xrdcp -> {eos_target}')
    if dry_run:
        print(f'    [dry-run] {" ".join(cmd)}')
        return True
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'  ERROR: xrdcp failed for {eos_target}')
        return False
    # Remove the local file now that it is safely on EOS
    try:
        os.remove(local_file)
        print(f'  removed local file: {local_file}')
    except OSError as exc:
        print(f'  WARNING: could not remove local file {local_file}: {exc}')
    return True


def run_hadd(output_file: str, input_files: list[str], dry_run: bool, hadd_workers: int = 1) -> bool:
    proto = get_eos_protocol(output_file)
    output_with_proto = with_protocol(output_file)
    inputs_with_proto = [with_protocol(f) for f in input_files]

    cmd = ['hadd', '-k', '-fk']
    if hadd_workers and hadd_workers > 1:
        cmd.extend(['-j', str(hadd_workers)])
    cmd.append(output_with_proto)
    cmd.extend(inputs_with_proto)
    print(f'  hadd -> {output_with_proto}  ({len(input_files)} files)')
    if dry_run:
        print(f'    [dry-run] {" ".join(cmd)}')
        return True

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'  ERROR: hadd failed for {output_with_proto}')
        return False
    return True


def process_chunk(job: dict) -> tuple[str, bool]:
    """Run one hadd job and optional EOS copy.

    Returns a tuple of (output_path, success).
    """
    out_path = job['out_path']
    ok = run_hadd(out_path, job['chunk'], job['dry_run'], job['hadd_workers'])
    if not ok:
        return out_path, False

    if job['eos_copy_to']:
        copy_ok = copy_to_eos(
            out_path,
            job['eos_copy_to'],
            job['output_sample_subdir'],
            job['dry_run'],
        )
        if not copy_ok:
            return out_path, False

    return out_path, True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Merge ROOT ntuple files N-by-N per sample using hadd.'
    )
    parser.add_argument(
        '-i', '--input-dataset', required=True,
        help='YAML dataset file (same one used by analyzeNtuples.py)'
    )
    parser.add_argument(
        '-o', '--output-dir', required=True,
        help='Base output directory; sample sub-directories are created automatically'
    )
    parser.add_argument(
        '-n', '--chunk-size', type=int, default=5,
        help='Number of files to merge per output file (default: 5)'
    )
    parser.add_argument(
        '-j', '--workers', type=int, default=1,
        help='Number of merge jobs to run in parallel (default: 1)'
    )
    parser.add_argument(
        '--hadd-workers', type=int, default=1,
        help='Number of worker processes used internally by each hadd command (default: 1)'
    )
    parser.add_argument(
        '-s', '--samples', nargs='+', default=None,
        help='Which samples to process (default: all samples in the YAML)'
    )
    parser.add_argument(
        '--eos-copy-to', default=None, metavar='EOS_DIR',
        help=(
            'After each successful hadd, xrdcp the merged file to this EOS base directory '
            'and delete the local copy. Overrides dataset.output_dir from the YAML.'
        )
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print commands without executing them'
    )
    args = parser.parse_args()

    # --- load YAML -----------------------------------------------------------
    with open(args.input_dataset) as fh:
        cfg = yaml.safe_load(fh)

    input_base = cfg['dataset']['input_dir'].rstrip('/')
    # EOS destination: CLI flag > YAML dataset.output_dir > input_base (same EOS dir,
    # output is separated by the _merged suffix on the sample sub-directory)
    eos_copy_to = args.eos_copy_to or cfg['dataset'].get('output_dir') or input_base
    eos_copy_to = eos_copy_to.rstrip('/')
    all_samples = cfg.get('samples', {})

    samples_to_process = args.samples if args.samples else list(all_samples.keys())

    unknown = [s for s in samples_to_process if s not in all_samples]
    if unknown:
        print(f'ERROR: unknown sample(s): {unknown}')
        print(f'Available: {list(all_samples.keys())}')
        sys.exit(1)

    print(f'Dataset      : {args.input_dataset}')
    print(f'Input base   : {input_base}')
    print(f'Output base  : {args.output_dir}')
    print(f'Chunk size   : {args.chunk_size}')
    print(f'Workers      : {args.workers}')
    print(f'Hadd workers : {args.hadd_workers}')
    print(f'Samples      : {samples_to_process}')
    if eos_copy_to:
        print(f'EOS copy to  : {eos_copy_to}  (local copy deleted after transfer)')
    if args.dry_run:
        print('[DRY RUN - no files will be written]')
    print()

    total_ok = total_fail = 0
    jobs = []
    eos_dirs_to_create = set()

    for sample_name in samples_to_process:
        sample_cfg = all_samples[sample_name]
        sample_subdir = sample_cfg['input_sample_dir'].strip('/')
        # Derive output sample subdir: replace last path level with <level>_merged
        sample_parts = sample_subdir.rsplit('/', 1)
        output_sample_subdir = sample_parts[0] + '/' + sample_parts[-1] + '_merged' if '/' in sample_subdir else sample_subdir + '_merged'
        input_dir = f'{input_base}/{sample_subdir}'
        output_dir = os.path.join(args.output_dir.rstrip('/'), output_sample_subdir)

        print(f'=== {sample_name} ===')
        print(f'  Input  : {input_dir}')
        print(f'  Output : {output_dir}')

        files = list_root_files(input_dir)
        if not files:
            print(f'  WARNING: no .root files found, skipping.')
            print()
            continue

        print(f'  Found {len(files)} file(s), will produce {-(-len(files)//args.chunk_size)} merged file(s)')
        ensure_dir(output_dir, args.dry_run)

        for idx, chunk in enumerate(chunked(files, args.chunk_size)):
            first_stem = os.path.splitext(os.path.basename(chunk[0]))[0]
            out_name = f'{first_stem}_merged_{idx:04d}.root'
            out_path = os.path.join(output_dir, out_name)
            jobs.append({
                'sample_name': sample_name,
                'output_sample_subdir': output_sample_subdir,
                'out_path': out_path,
                'chunk': chunk,
                'hadd_workers': args.hadd_workers,
                'eos_copy_to': eos_copy_to,
                'dry_run': args.dry_run,
            })
            if eos_copy_to:
                eos_dirs_to_create.add(f'{eos_copy_to.rstrip("/")}/{output_sample_subdir}')

        print()

    for eos_dir in sorted(eos_dirs_to_create):
        ensure_dir(eos_dir, args.dry_run)

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(process_chunk, job): job for job in jobs}
            for future in as_completed(futures):
                out_path, ok = future.result()
                if ok:
                    total_ok += 1
                else:
                    total_fail += 1

    print('========== SUMMARY ==========')
    print(f'  Merged files OK  : {total_ok}')
    print(f'  Failed           : {total_fail}')
    print('=============================')
    if total_fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
