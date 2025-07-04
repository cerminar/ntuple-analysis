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
    echo "  where what_to_draw can be: all or a comma-separated list: eg_genmatch_ctl2,eg_genmatch_menu,eg_rate_menu,eg_eff,eg_rate_ctl2,eg_unmatched,eg_rate_counter,met_rate,jet_genmatch"
    exit 1
fi

# Convert WHAT to array for matching
IFS=',' read -rA WHAT_LIST <<< "$WHAT"

# Check if a specific task should run
should_run() {
    if [[ "$WHAT" == "all" ]]; then
        # [[ "$1" == "eg_rate_counter" ]] && return 1  # exclude 'eg_rate_counter' from 'all'
        return 0
    fi
    for item in "${WHAT_LIST[@]}"; do
        [[ "$item" == "$1" ]] && return 0
    done
    return 1
}

FILE_DIR=/Users/cerminar/cernbox/hgcal/CMSSW1015/plots

# Arrays to track results
SUCCESS_TASKS=()
FAILED_TASKS=()

run_and_track() {
    local taskname="$1"
    shift
    if should_run "$taskname"; then
        "$@"
        local exit_code=$?
        if [[ $exit_code -eq 0 ]]; then
            SUCCESS_TASKS+=("$taskname")
        else
            FAILED_TASKS+=("$taskname")
        fi
        return $exit_code
    else
        echo "Skipping task: $taskname"
        return 1  # return 1 so that cp is not executed
    fi
}

run_and_track "eg_genmatch_menu" python analyzeNtuples.py -f cfg/eg_genmatch.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml  \
        -p egmenu  \
        -s doubleele_flat1to100_PU200 -n -1 -d 0 && \
cp -v ${FILE_DIR}/histos_doubleele_flat1to100_PU200_egmenu_v200D.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_doubleele_flat1to100_PU200_egmenu_v200D.${TEST_VERSION}.root

run_and_track "eg_genmatch_ctl2" python analyzeNtuples.py -f cfg/eg_genmatch.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml  \
        -p ctl2_tkeg  \
        -s doubleele_flat1to100_PU200 -n -1 -d 0 && \
cp -v ${FILE_DIR}/histos_doubleele_flat1to100_PU200_eg_v200D.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_doubleele_flat1to100_PU200_eg_v200D.${TEST_VERSION}.root

run_and_track "eg_unmatched" python analyzeNtuples.py -f cfg/egplots.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p tkeg_plots  \
        -s doubleele_flat1to100_PU200 -n -1 -d 0 && \
cp -v ${FILE_DIR}/histos_doubleele_flat1to100_PU200_egplots_v160A.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_doubleele_flat1to100_PU200_egplots_v160A.${TEST_VERSION}.root

run_and_track "eg_rate_menu" python analyzeNtuples.py -f cfg/eg_rate.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p rate_menu  \
        -s nugun_alleta_pu200 -n 500000 -d 0 && \
cp -v ${FILE_DIR}/histos_nugun_alleta_pu200_egratemenu_v200C.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_nugun_alleta_pu200_egratemenu_v200C.${TEST_VERSION}.root

run_and_track "eg_rate_ctl2" python analyzeNtuples.py -f cfg/eg_rate.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p rate_ctl2  \
        -s nugun_alleta_pu200 -n 500000 -d 0 && \
cp -v ${FILE_DIR}/histos_nugun_alleta_pu200_egrate_v200C.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_nugun_alleta_pu200_egrate_v200C.${TEST_VERSION}.root

run_and_track "eg_rate_counter" python analyzeNtuples.py -f cfg/eg_rate.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p rate_counter_menu  \
        -s nugun_alleta_pu200 -n 500000 -d 0 && \
cp -v ${FILE_DIR}/histos_nugun_alleta_pu200_egratecount_v200C.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_nugun_alleta_pu200_egratecount_v200C.${TEST_VERSION}.root

run_and_track "met_rate" python analyzeNtuples.py -f cfg/jetmet_rate.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p met  \
        -s nugun_alleta_pu200 -n 500000 -d 0 && \
cp -v ${FILE_DIR}/histos_nugun_alleta_pu200_jetmet_rate_v200A.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_nugun_alleta_pu200_jetmet_rate_v200A.${TEST_VERSION}.root

run_and_track "jet_genmatch" python analyzeNtuples.py -f cfg/jetmet_genmatch.yaml \
        -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml \
        -p jets  \
        -s ttbar_PU200 -n -1 -d 0 && \
cp -v ${FILE_DIR}/histos_ttbar_PU200_jets_v200C.${TEST_VERSION}i.root \
        ${FILE_DIR}/histos_ttbar_PU200_jets_v200C.${TEST_VERSION}.root

echo
echo "========== SUMMARY =========="
if [[ ${#SUCCESS_TASKS[@]} -gt 0 ]]; then
    echo "Successful tasks:"
    for t in "${SUCCESS_TASKS[@]}"; do
        echo "  $t"
    done
else
    echo "No successful tasks."
fi

if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo "Failed tasks:"
    for t in "${FAILED_TASKS[@]}"; do
        echo "  $t"
    done
else
    echo "No failed tasks."
fi
echo "============================="
