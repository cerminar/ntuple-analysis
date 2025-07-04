

TARGET_VERSION=140Xv0C4


TARGET_DIR=/Users/cerminar/CERNbox/www/plots/151X_newvalid_${TARGET_VERSION}/
# TARGET_DIR=/Users/cerminar/CERNbox/www/plots/142X_newmodels_test_reso/


# python  analyzeNtuples.py -f cfg/eg_genmatch.yaml -i cfg/datasets/ntpfp_${TARGET_VERSION}.yaml  -p ctl2_tkeg  -s doubleele_flat1to100_PU200 -n -1 -d 0

# python  analyzeNtuples.py -f cfg/egplots.yaml -i cfg/datasets/ntpfp_${TARGET_VERSION}.yaml  -p tkeg_plots  -s doubleele_flat1to100_PU200 -n -1 -d 0

# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TARGET_VERSION}.yaml -p rate_menu  -s nugun_alleta_pu200 -n 500000 -d 0
# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TARGET_VERSION}.yaml -p rate_ctl2  -s nugun_alleta_pu200 -n 500000 -d 0
# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TARGET_VERSION}.yaml -p rate_counter_menu  -s nugun_alleta_pu200 -n 500000 -d 0

nomatch_ele_files=(
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.140Xv0D0.root:151X"
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.140Xv0C1.root:151X+PR"
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.${TARGET_VERSION}.root:151X+PR+newWPs"
)


gen_match_ele_files=(
    "plots/histos_doubleele_flat1to100_PU200_eg_v200D.140Xv0D0.root:151X"
    "plots/histos_doubleele_flat1to100_PU200_eg_v200D.140Xv0C1.root:152X+PR"
    "plots/histos_doubleele_flat1to100_PU200_eg_v200D.${TARGET_VERSION}.root:151X+PR+newWPs"
)

gen_match_menu_ele_files=(
    "plots/histos_doubleele_flat1to100_PU200_egmenu_v200D.140Xv0D0.root:151X"
    "plots/histos_doubleele_flat1to100_PU200_egmenu_v200D.140Xv0C1.AR25.root:152X+PR"
    "plots/histos_doubleele_flat1to100_PU200_egmenu_v200D.${TARGET_VERSION}.root:151X+PR+newWPs"
)



nugun_ratemenu_files=(
    "plots/histos_nugun_alleta_pu200_egratemenu_v200C.140Xv0D0.root:151X"
    "plots/histos_nugun_alleta_pu200_egratemenu_v200C.140Xv0C1.AR25.root:151X+PR"        
    "plots/histos_nugun_alleta_pu200_egratemenu_v200C.${TARGET_VERSION}.root:151X+PR+newWPS"
)

nugun_egratecount_files=(
    "plots/histos_nugun_alleta_pu200_egratecount_v200C.140Xv0D0.root:151X"
    "plots/histos_nugun_alleta_pu200_egratecount_v200C.140Xv0C1.AR25.root:151X+PR"
    "plots/histos_nugun_alleta_pu200_egratecount_v200C.${TARGET_VERSION}.root:151X+PR+newWPs"
)


nugun_ratectl2_files=(
    "plots/histos_nugun_alleta_pu200_egrate_v200C.140Xv0D0.root:151X"
    "plots/histos_nugun_alleta_pu200_egrate_v200C.140Xv0C1.root:151X+PR"
    "plots/histos_nugun_alleta_pu200_egrate_v200C.${TARGET_VERSION}.root:151X+PR+newWPs"
)


nugun_metrate_files=(
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.140Xv0D0.root:151X"    
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.140Xv0C1.root:151X+PR"    
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.${TARGET_VERSION}.root:151X+PR+newWPs"
)

ttbar_jetreso_files=(
    "plots/histos_ttbar_PU200_jets_v200C.140Xv0D0.root:151X"
    "plots/histos_ttbar_PU200_jets_v200C.140Xv0C1.root:151X+PR"    
    "plots/histos_ttbar_PU200_jets_v200C.${TARGET_VERSION}.root:151X+PR+newWPs"
)