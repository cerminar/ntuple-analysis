#!/bin/bash

CLUSTERID=$1
PROCID=$2
echo $PATH
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
python analyzeHgcalL1Tntuple.py -f TEMPL_CFG -i TEMPL_INPUT -c TEMPL_COLL -s TEMPL_SAMPLE -n -1 -o ${BATCH_DIR} -r ${PROCID} -b
