# CVMFS_TOOLS=/cvmfs/cms.cern.ch/slc6_amd64_gcc630/
# source ${CVMFS_TOOLS}/external/bootstrap-bundle/1.7/etc/profile.d/init.sh
# source ${CVMFS_TOOLS}/external/python/2.7.11-fmblme/etc/profile.d/init.sh
# source ${CVMFS_TOOLS}/external/gcc/6.3.0/etc/profile.d/init.sh
# source ${CVMFS_TOOLS}/external/pcre/8.37/etc/profile.d/init.sh
# source ${CVMFS_TOOLS}/lcg/root/6.10.08/bin/thisroot.sh
# source ${CVMFS_TOOLS}/external/py2-pip/9.0.1-fmblme/etc/profile.d/init.sh
# source ${CVMFS_TOOLS}/external/tbb/2017_U6/etc/profile.d/init.sh
# source ${CVMFS_TOOLS}/external/xrootd/4.5.0/etc/profile.d/init.sh
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /cvmfs/cms.cern.ch/${SCRAM_ARCH}/cms/cmssw/CMSSW_12_5_0/src/
echo $?
eval `scram runtime -sh`
cd -
