#!/usr/bin/env zsh

# TARGET_DIR=/Users/cerminar/CERNbox/www/plots/142X_newmodels_v12/
source  ~/scripts/setPortPython.sh
source setVirtualEnvWrapper.sh
workon pandas2


# Input arguments
export TEST_VERSION="$1"
WHAT="${2:-all}"  # e.g., "menu_rate,met,eff"

if [ -z "$TEST_VERSION" ]; then
  echo "Usage: $0 <TEST_VERSION> [what_to_draw]"
  echo "  where what_to_draw can be: all or a comma-separated list: eg_genmatch,menu_eg_rate,eg_eff,ctl2_eg_rate,eg_unmatched,counter,met_rate"
  exit 1
fi

# Convert WHAT to array for matching
IFS=',' read -rA WHAT_LIST <<< "$WHAT"

# Check if a specific task should run
should_run() {
  if [[ "$WHAT" == "all" ]]; then
    # [[ "$1" == "counter" ]] && return 1  # exclude 'counter' from 'all'
    return 0
  fi
  for item in "${WHAT_LIST[@]}"; do
    [[ "$item" == "$1" ]] && return 0
  done
  return 1
}


set -x

# echo $LD_LIBRARY_PATH
# echo $PYTHONPATH

# TEST_VERSION=131Xv3M
FILE_DIR=/Users/cerminar/cernbox/hgcal/CMSSW1015/plots

if should_run "eg_genmatch"; then
    python  analyzeNtuples.py -f cfg/eg_genmatch.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml  \
        -p ctl2_tkeg  \
        -s doubleele_flat1to100_PU200 -n -1 -d 0

    cp ${FILE_DIR}/histos_doubleele_flat1to100_PU200_eg_v200C.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_doubleele_flat1to100_PU200_eg_v200C.${TEST_VERSION}.root
fi

if should_run "eg_unmatched"; then
    python  analyzeNtuples.py -f cfg/egplots.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p tkeg_plots  \
        -s doubleele_flat1to100_PU200 -n -1 -d 0
    cp ${FILE_DIR}/histos_doubleele_flat1to100_PU200_egplots_v160A.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_doubleele_flat1to100_PU200_egplots_v160A.${TEST_VERSION}.root
fi

if should_run "menu_eg_rate"; then
    python  analyzeNtuples.py -f cfg/eg_rate.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p rate_menu  \
        -s nugun_alleta_pu200 -n 500000 -d 0
    cp ${FILE_DIR}/histos_nugun_alleta_pu200_egratemenu_v200A.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_nugun_alleta_pu200_egratemenu_v200A.${TEST_VERSION}.root
fi

if should_run "ctl2_eg_rate"; then
    python  analyzeNtuples.py -f cfg/eg_rate.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p rate_ctl2  \
        -s nugun_alleta_pu200 -n 500000 -d 0
    cp ${FILE_DIR}/histos_nugun_alleta_pu200_egrate_v200A.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_nugun_alleta_pu200_egrate_v200A.${TEST_VERSION}.root
fi

if should_run "counter"; then
    python  analyzeNtuples.py -f cfg/eg_rate.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p rate_counter_menu  \
        -s nugun_alleta_pu200 -n 500000 -d 0
    cp ${FILE_DIR}/histos_nugun_alleta_pu200_egratecount_v200A.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_nugun_alleta_pu200_egratecount_v200A.${TEST_VERSION}.root
fi

if should_run "met_rate"; then
    python  analyzeNtuples.py -f cfg/jetmet_rate.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p met  \
        -s nugun_alleta_pu200 -n 500000 -d 0
    cp ${FILE_DIR}/histos_nugun_alleta_pu200_jetmet_rate_v200A.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_nugun_alleta_pu200_jetmet_rate_v200A.${TEST_VERSION}.root
fi


if should_run "jet_reso"; then
    python  analyzeNtuples.py -f cfg/jetmet_genmatch.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p jets  \
        -s ttbar_PU200 -n -1 -d 0

    cp ${FILE_DIR}/histos_ttbar_PU200_jets_v200C.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_ttbar_PU200_jets_v200C.${TEST_VERSION}.root
fi
