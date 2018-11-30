#!/bin/bash

CLUSTERID=$1
PROCID=$2

echo "Runnin Cluster ${CLUSTERID} Job ${PROCID}"
BATCH_DIR=${PWD}
echo "Current dir: ${BATCH_DIR}"
# cd TEMPL_WORKDIR
# echo "Now in dir: ${PWD}"
#
hostname

tar xfz ntuple-tools.tar.gz

source ./setup_lxplus.sh
source ./setVirtualEnvWrapper.sh
workon TEMPL_VIRTUALENV
# cd ${BATCH_DIR}
date
python runHarvesting.py -i TEMPL_OUTDIR/tmp/ -s TEMPL_SAMPLE -v TEMPL_VERSION -o TEMPL_OUTDIR
