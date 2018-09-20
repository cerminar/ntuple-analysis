#!/bin/bash

CLUSTERID=$1
PROCID=$2

echo "Runnin Cluster ${CLUSTERID} Job ${PROCID}"
BATCH_DIR=${PWD}
echo "Current dir: ${BATCH_DIR}"
cd TEMPL_WORKDIR
echo "Now in dir: ${PWD}"

hostname

source ./setup_lxplus.sh
source ~/setVirtualEnvWrapper.sh
workon TEMPL_VIRTUALENV
cd ${BATCH_DIR}
date
python TEMPL_WORKDIR/analyzeHgcalL1Tntuple.py -f TEMPL_CFG -c TEMPL_COLL -s TEMPL_SAMPLE -n -1 -o ${BATCH_DIR} -r ${PROCID} -b
