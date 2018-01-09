
CLUSTERID=$1
PROCID=$2


source setup_lxplus.sh
source ~/scripts/setVirtualEnvWrapper.sh
workon HGCTPGPerformance-1
cd TEMPL_WORKDIR
python analyzeHgcalL1Tntuple.py -f TEMPL_CFG -c TEMPL_COLL -s ${PROCID} -n -1
