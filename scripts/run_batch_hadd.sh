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
source ./setVirtualEnvWrapper.sh
workon TEMPL_VIRTUALENV

cd ${BATCH_DIR}
date
mkdir infiles/
for filename in TEMPL_OUTDIR/tmp/TEMPL_INFILE; do
    echo ${filename}
    xrdcp TEMPL_EOSPROTOCOL${filename} infiles/
done
hadd -j 10 -k TEMPL_OUTFILE `ls infiles/TEMPL_INFILE`
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "****** Error: hadd failed!"
    exit $retVal
fi
# rm TEMPL_INFILE
#mv TEMPL_OUTFILE TEMPL_OUTDIR
