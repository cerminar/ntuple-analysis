

TARGER_VERSION=140Xv0D0


TARGET_DIR=/Users/cerminar/CERNbox/www/plots/151X_newwps_AR25/
# TARGET_DIR=/Users/cerminar/CERNbox/www/plots/142X_newmodels_test_reso/


# python  analyzeNtuples.py -f cfg/eg_genmatch.yaml -i cfg/datasets/ntpfp_${TARGER_VERSION}.yaml  -p ctl2_tkeg  -s doubleele_flat1to100_PU200 -n -1 -d 0

# python  analyzeNtuples.py -f cfg/egplots.yaml -i cfg/datasets/ntpfp_${TARGER_VERSION}.yaml  -p tkeg_plots  -s doubleele_flat1to100_PU200 -n -1 -d 0

# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TARGER_VERSION}.yaml -p rate_menu  -s nugun_alleta_pu200 -n 500000 -d 0
# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TARGER_VERSION}.yaml -p rate_ctl2  -s nugun_alleta_pu200 -n 500000 -d 0
# python  analyzeNtuples.py -f cfg/eg_rate.yaml -i cfg/datasets/ntpfp_${TARGER_VERSION}.yaml -p rate_counter_menu  -s nugun_alleta_pu200 -n 500000 -d 0

nomatch_ele_files=(
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.${TARGER_VERSION}.root:151X"
   "plots/histos_doubleele_flat1to100_PU200_egplots_v160A.140Xv0C1.root:151X+PR"
)


gen_match_ele_files=(
    "plots/histos_doubleele_flat1to100_PU200_eg_v200D.${TARGER_VERSION}.root:151X"
    "plots/histos_doubleele_flat1to100_PU200_eg_v200D.140Xv0C1.root:152X+PR"
    "plots/histos_doubleele_flat1to100_PU200_eg_v200D.140Xv0C1_AR25.root:152X+PR+newWPs"
)




nugun_ratemenu_files=(
    "plots/histos_nugun_alleta_pu200_egratemenu_v200C.${TARGER_VERSION}.root:151X"
    "plots/histos_nugun_alleta_pu200_egratemenu_v200C.140Xv0C1.root:151X+PR"
    "plots/histos_nugun_alleta_pu200_egratemenu_v200C.140Xv0C1.AR25.root:151X+PR+newWPs"        
)

nugun_egratecount_files=(
    "plots/histos_nugun_alleta_pu200_egratecount_v200C.${TARGER_VERSION}.root:151X"
    "plots/histos_nugun_alleta_pu200_egratecount_v200C.140Xv0C1.root:151X+PR"
    "plots/histos_nugun_alleta_pu200_egratecount_v200C.140Xv0C1.AR25.root:151X+PR+newWPs"
)


nugun_ratectl2_files=(
    "plots/histos_nugun_alleta_pu200_egrate_v200C.${TARGER_VERSION}.root:151X"
    "plots/histos_nugun_alleta_pu200_egrate_v200C.140Xv0C1.root:151X+PR"
)

nugun_metrate_files=(
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.${TARGER_VERSION}.root:151X"
    "plots/histos_nugun_alleta_pu200_jetmet_rate_v200A.140Xv0C1.root:151X+PR"    
)

ttbar_jetreso_files=(
    "plots/histos_ttbar_PU200_jets_v200C.${TARGER_VERSION}.root:151X"
    "plots/histos_ttbar_PU200_jets_v200C.140Xv0C1.root:151X+PR"    
)