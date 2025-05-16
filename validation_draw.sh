#!/usr/bin/env zsh
source ~/scripts/setPortPython.sh
source setVirtualEnvWrapper.sh
workon pandas2

# Input arguments
export TEST_VERSION="$1"
WHAT="${2:-all}"  # e.g., "menu_rate,met,eff"

if [ -z "$TEST_VERSION" ]; then
  echo "Usage: $0 <TEST_VERSION> [what_to_draw]"
  echo "  where what_to_draw can be: all or a comma-separated list: eg_reso,menu_eg_rate,eg_eff,ctl2_eg_rate,eg_unmatched,counter,met_rate"
  exit 1
fi

set -x

source scripts/conf_${TEST_VERSION}.sh

# Convert WHAT to array for matching
IFS=',' read -rA WHAT_LIST <<< "$WHAT"

# Check if a specific task should run
should_run() {
  if [[ "$WHAT" == "all" ]]; then
    [[ "$1" == "counter" ]] && return 1  # exclude 'counter' from 'all'
    return 0
  fi
  for item in "${WHAT_LIST[@]}"; do
    [[ "$item" == "$1" ]] && return 0
  done
  return 1
}

# Resolution plots
if should_run "eg_reso"; then
  python draw.py -m cfg/eg_genmatch_draw.py \
    --target-dir ${TARGET_DIR} \
    -w ctl2_tkeg_reso \
    --input-files $(IFS=,; echo "${gen_match_ele_files[*]}")
fi

# Menu rate plots
if should_run "menu_eg_rate"; then
  python draw.py -m cfg/eg_rate_draw.py \
    --target-dir ${TARGET_DIR} \
    -w menu_rate \
    --input-files $(IFS=,; echo "${nugun_ratemenu_files[*]}")
fi

# Efficiency plots
if should_run "eg_eff"; then
  python draw.py -m cfg/eg_genmatch_draw.py \
    --target-dir ${TARGET_DIR} \
    -w ctl2_tkeg \
    --input-files $(IFS=,; echo "${gen_match_ele_files[*]}")
fi

# CTL2 rate plots
if should_run "ctl2_eg_rate"; then
  python draw.py -m cfg/eg_rate_draw.py \
    --target-dir ${TARGET_DIR} \
    -w ctl2_rate \
    --input-files $(IFS=,; echo "${nugun_ratectl2_files[*]}")
fi

# Unmatched plots
if should_run "eg_unmatched"; then
  python draw.py -m cfg/egplots_draw.py \
    --target-dir ${TARGET_DIR} \
    -w tkeg_plots \
    --input-files $(IFS=,; echo "${nomatch_ele_files[*]}")
fi

# Menu rate counter plots
if should_run "counter"; then
  python draw.py -m cfg/eg_rate_draw.py \
    --target-dir ${TARGET_DIR} \
    -w menu_ratecounter \
    --input-files $(IFS=,; echo "${nugun_egratecount_files[*]}")
fi

# MET rate plots
if should_run "met_rate"; then
  python draw.py -m cfg/jetmet_rate_draw.py \
    --target-dir ${TARGET_DIR} \
    -w met \
    --input-files $(IFS=,; echo "${nugun_metrate_files[*]}")
fi

if should_run "jet_reso"; then
    python draw.py -m cfg/jetmet_genmatch_draw.py \
    --target-dir ${TARGET_DIR} \
    -w jet_reso \
    --input-files $(IFS=,; echo "${ttbar_jetreso_files[*]}")
fi
