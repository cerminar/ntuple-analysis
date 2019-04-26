#!/bin/bash

CLUSTERID=$1
PROCID=$2
ISTMP=$3
echo "Runnin Cluster ${CLUSTERID} Job ${PROCID}"
BATCH_DIR=${PWD}
echo "Current dir: ${BATCH_DIR}"
ls -l

OUTDIR=TEMPL_OUTDIR
if [ "${ISTMP}" == "true" ]; then
  OUTDIR=TEMPL_OUTDIR/tmp/
fi
cp histos_TEMPL_SAMPLE_*.root ${OUTDIR}
