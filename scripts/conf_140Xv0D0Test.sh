

TARGER_VERSION=140Xv0D0


TARGET_DIR=/Users/cerminar/CERNbox/www/plots/152X_validation_${TARGER_VERSION}/
# TARGET_DIR=/Users/cerminar/CERNbox/www/plots/142X_newmodels_test_reso/


# python  analyzeNtuples.py -f cfg/eg_genmatch.yaml -i cfg/datasets/ntpfp_${TARGER_VERSION}.yaml  -p ctl2_tkeg  -s doubleele_flat1to100_PU200 -n -1 -d 0

# python  analyzeNtuples.py -f cfg/egplots.yaml -i cfg/datasets/ntpfp_${TARGER_VERSION}.yaml  -p tkeg_plots  -s doubleele_flat1to100_PU200 -n -1 -d 0

# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TARGER_VERSION}.yaml -p rate_menu  -s nugun_alleta_pu200 -n 500000 -d 0
# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TARGER_VERSION}.yaml -p rate_ctl2  -s nugun_alleta_pu200 -n 500000 -d 0
# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TARGER_VERSION}.yaml -p rate_counter_menu  -s nugun_alleta_pu200 -n 500000 -d 0

nomatch_ele_files=(
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.131Xv3M.root:AR2024"
#    "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.131Xv9A.root:DPS-Note"
#    "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.140Xv0B6.root:142X-int-ell"
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.140Xv0C1.root:151X+PR"
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.${TARGER_VERSION}.root:151X"
)


gen_match_ele_files=(
    "plots/histos_doubleele_flat1to100_PU200_eg_v200C.131Xv3M.root:AR2024"
    # "plots/histos_doubleele_flat1to100_PU200_eg_v200C.131Xv9A.root:DPS-Note"
    # "plots/histos_doubleele_flat1to100_PU200_eg_v200C.140Xv0B6.root:142X-int-ell"
    "plots/histos_doubleele_flat1to100_PU200_eg_v200C.140Xv0C1.root:152X+PR"
    "plots/histos_doubleele_flat1to100_PU200_eg_v200C.${TARGER_VERSION}.root:151X"
)

nugun_ratemenu_files=(
    "plots/histos_nugun_alleta_pu200_egratemenu_v200A.131Xv3M.root:AR2024"
    # "plots/histos_nugun_alleta_pu200_egratemenu_v200A.131Xv9A.root:DPS-Note"
    # "plots/histos_nugun_alleta_pu200_egratemenu_v200A.140Xv0B5.root:142X-int-ell"
    "plots/histos_nugun_alleta_pu200_egratemenu_v200A.140Xv0C1.root:151X+PR"    
    "plots/histos_nugun_alleta_pu200_egratemenu_v200A.${TARGER_VERSION}.root:151X"
)

nugun_egratecount_files=(
    # "plots/histos_nugun_alleta_pu200_egratecount_v200A.131Xv3M.root:AR2024"
    # "plots/histos_nugun_alleta_pu200_egratecount_v200A.131Xv9A.root:DPS-Note"
    # "plots/histos_nugun_alleta_pu200_egratemenu_v200A.140Xv0B5.root:142X-int-ell"
    # "plots/histos_nugun_alleta_pu200_egratecount_v200A.140Xv0C1.root:151X+PRold"    
    "plots/histos_nugun_alleta_pu200_egratecount_v200B.140Xv0C1.root:151X+PR"    
    "plots/histos_nugun_alleta_pu200_egratecount_v200B.${TARGER_VERSION}.root:151X"
)


nugun_ratectl2_files=(
    "plots/histos_nugun_alleta_pu200_egrate_v200A.131Xv3M.root:AR2024"
    # "plots/histos_nugun_alleta_pu200_egrate_v200A.131Xv9A.root:DPS-Note"
    # "plots/histos_nugun_alleta_pu200_egrate_v200A.140Xv0B6.root:142X-int-ell"
    "plots/histos_nugun_alleta_pu200_egrate_v200A.140Xv0C1.root:151X+PR"    
    "plots/histos_nugun_alleta_pu200_egrate_v200A.${TARGER_VERSION}.root:151X"
)

nugun_metrate_files=(
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.131Xv3M.root:AR2024"
    # "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.131Xv9A.root:DPS-Note"
    # "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.140Xv0B5.root:142X-int-ell"
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.140Xv0C1.root:151X+PR"    
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.${TARGER_VERSION}.root:151X"
)

ttbar_jetreso_files=(
    # "plots/histos_ttbar_PU200_jets_v200C.131Xv3M.root:AR2024"
    # "plots/histos_ttbar_PU200_jets_v200C.131Xv9A.root:DPS-Note"
    # "plots/histos_ttbar_jetmet_rate_v200A.140Xv0B5.root:142X-int-ell"
    "plots/histos_ttbar_PU200_jets_v200C.140Xv0C1.root:151X+PR"    
    "plots/histos_ttbar_PU200_jets_v200C.${TARGER_VERSION}.root:151X"
)