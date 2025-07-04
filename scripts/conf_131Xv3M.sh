

TEST_VERSION=131Xv3M


TARGET_DIR=/Users/cerminar/CERNbox/www/plots/fp131Xv3-CMSSW14.0.X-GCTEmu/
# TARGET_DIR=/Users/cerminar/CERNbox/www/plots/142X_newmodels_test_reso/


# python  analyzeNtuples.py -f cfg/eg_genmatch.yaml -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml  -p ctl2_tkeg  -s doubleele_flat1to100_PU200 -n -1 -d 0

# python  analyzeNtuples.py -f cfg/egplots.yaml -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml  -p tkeg_plots  -s doubleele_flat1to100_PU200 -n -1 -d 0

# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml -p rate_menu  -s nugun_alleta_pu200 -n 500000 -d 0
# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml -p rate_ctl2  -s nugun_alleta_pu200 -n 500000 -d 0
# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml -p rate_counter_menu  -s nugun_alleta_pu200 -n 500000 -d 0

nomatch_ele_files=(
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.131Xv3.root:AR2024"
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.131Xv3M.root:AR2024+GCT"
)


gen_match_ele_files=(
    "plots/histos_doubleele_flat1to100_PU200_eg_v200C.131Xv3.root:AR2024"
    "plots/histos_doubleele_flat1to100_PU200_eg_v200C.131Xv3M.root:AR2024+GCT"
)

nugun_ratemenu_files=(
    "plots/histos_nugun_alleta_pu200_egratemenu_v200A.131Xv3.root:AR2024"
    "plots/histos_nugun_alleta_pu200_egratemenu_v200A.131Xv3M.root:AR2024+GCT"
)

nugun_egratecount_files=(
    "plots/histos_nugun_alleta_pu200_egratecount_v200A.131Xv3.root:AR2024"
    "plots/histos_nugun_alleta_pu200_egratecount_v200A.131Xv3M.root:AR2024+GCT"

)


nugun_ratectl2_files=(
    "plots/histos_nugun_alleta_pu200_egrate_v200A.131Xv3.root:AR2024"
    "plots/histos_nugun_alleta_pu200_egrate_v200A.131Xv3M.root:AR2024+GCT"

)

nugun_metrate_files=(
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.131Xv3.root:AR2024"
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.131Xv3M.root:AR2024+GCT"

)

