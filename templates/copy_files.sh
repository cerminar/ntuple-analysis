#!/bin/bash

CLUSTERID=$1
PROCID=$2
echo "Runnin Cluster ${CLUSTERID} Job ${PROCID}"
BATCH_DIR=${PWD}
echo "Current dir: ${BATCH_DIR}"
ls -l

cp histos_TEMPL_SAMPLE_*_*.root TEMPL_OUTDIR
