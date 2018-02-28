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
source ~/scripts/setVirtualEnvWrapper.sh
workon TEMPL_VIRTUALENV

python analyzeHgcalL1Tntuple.py -f TEMPL_CFG -c TEMPL_COLL -s TEMPL_SAMPLE -n -1 -o ${BATCH_DIR} -r ${PROCID} -b
