

TEST_VERSION=140Xv0B20


TARGET_DIR=/Users/cerminar/CERNbox/www/plots/142X_newmodels_140Xv0B20/
# TARGET_DIR=/Users/cerminar/CERNbox/www/plots/142X_newmodels_test_reso/


# python  analyzeNtuples.py -f cfg/eg_genmatch.yaml -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml  -p ctl2_tkeg  -s doubleele_flat1to100_PU200 -n -1 -d 0

# python  analyzeNtuples.py -f cfg/egplots.yaml -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml  -p tkeg_plots  -s doubleele_flat1to100_PU200 -n -1 -d 0

# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml -p rate_menu  -s nugun_alleta_pu200 -n 500000 -d 0
# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml -p rate_ctl2  -s nugun_alleta_pu200 -n 500000 -d 0
# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TEST_VERSION}.yaml -p rate_counter_menu  -s nugun_alleta_pu200 -n 500000 -d 0

nomatch_ele_files=(
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.131Xv3M.root:AR2024"
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.131Xv9A.root:DPS-Note"
#    "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.140Xv0B6.root:142X-int-ell"
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.140Xv0B12.root:142X-int"
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.140Xv0B20.root:142X-int-gct"
)


gen_match_ele_files=(
    "plots/histos_doubleele_flat1to100_PU200_eg_v200C.131Xv3M.root:AR2024"
    "plots/histos_doubleele_flat1to100_PU200_eg_v200C.131Xv9A.root:DPS-Note"
    # "plots/histos_doubleele_flat1to100_PU200_eg_v200C.140Xv0B6.root:142X-int-ell"
    "plots/histos_doubleele_flat1to100_PU200_eg_v200C.140Xv0B12.root:142X-int"
    "plots/histos_doubleele_flat1to100_PU200_eg_v200C.140Xv0B20.root:142X-int-gct"
)

nugun_ratemenu_files=(
    "plots/histos_nugun_alleta_pu200_egratemenu_v200A.131Xv3M.root:AR2024"
    "plots/histos_nugun_alleta_pu200_egratemenu_v200A.131Xv9A.root:DPS-Note"
    # "plots/histos_nugun_alleta_pu200_egratemenu_v200A.140Xv0B5.root:142X-int-ell"
    "plots/histos_nugun_alleta_pu200_egratemenu_v200A.140Xv0B12.root:142X-int"    
    "plots/histos_nugun_alleta_pu200_egratemenu_v200A.140Xv0B20.root:142X-int-gct"
)

nugun_egratecount_files=(
    "plots/histos_nugun_alleta_pu200_egratecount_v200A.131Xv3M.root:AR2024"
    "plots/histos_nugun_alleta_pu200_egratecount_v200A.131Xv9A.root:DPS-Note"
    # "plots/histos_nugun_alleta_pu200_egratemenu_v200A.140Xv0B5.root:142X-int-ell"
    # "plots/histos_nugun_alleta_pu200_egratecount_v200A.140Xv0B12.root:142X-int"    
    "plots/histos_nugun_alleta_pu200_egratecount_v200A.140Xv0B20.root:142X-int-gct"
)


nugun_ratectl2_files=(
    "plots/histos_nugun_alleta_pu200_egrate_v200A.131Xv3M.root:AR2024"
    "plots/histos_nugun_alleta_pu200_egrate_v200A.131Xv9A.root:DPS-Note"
    # "plots/histos_nugun_alleta_pu200_egrate_v200A.140Xv0B6.root:142X-int-ell"
    "plots/histos_nugun_alleta_pu200_egrate_v200A.140Xv0B12.root:142X-int"    
    "plots/histos_nugun_alleta_pu200_egrate_v200A.140Xv0B20.root:142X-int-gct"
)

nugun_metrate_files=(
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.131Xv3M.root:AR2024"
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.131Xv9A.root:DPS-Note"
    # "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.140Xv0B5.root:142X-int-ell"
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.140Xv0B12.root:142X-int"    
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.140Xv0B20.root:142X-int-gct"
)

