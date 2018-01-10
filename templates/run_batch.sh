
CLUSTERID=$1
PROCID=$2

echo "Runnin Cluster ${CLUSTERID} Job ${PROCID}"
echo "Current dir: ${PWD}"
cd TEMPL_WORKDIR
echo "Now in dir: ${PWD}"

source ./setup_lxplus.sh
source ~/scripts/setVirtualEnvWrapper.sh
workon HGCTPGPerformance-1

python analyzeHgcalL1Tntuple.py -f TEMPL_CFG -c TEMPL_COLL -s ${PROCID} -n -1
