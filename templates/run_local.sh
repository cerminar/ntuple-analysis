#!/bin/bash

BATCH_DIR=$1
PROCID=$2
echo $PATH
echo "Runnin Job ${PROCID} in ${BATCH_DIR}"

cd ${BATCH_DIR}
echo "Current dir: ${PWD}"
# cd TEMPL_WORKDIR
# echo "Now in dir: ${PWD}"
#
hostname

tar xfz ntuple-tools.tar.gz

if [ -d "/cvmfs/cms.cern.ch" ]; then
  source ./setup_lxplus.sh
else
  source ~/scripts/setPortPython.sh
fi
source ./setVirtualEnvWrapper.sh
workon TEMPL_VIRTUALENV
# cd ${BATCH_DIR}
date
python analyzeHgcalL1Tntuple.py -f TEMPL_CFG -i TEMPL_INPUT -c TEMPL_COLL -s TEMPL_SAMPLE -n -1 -o ${BATCH_DIR} -r ${PROCID} -b
