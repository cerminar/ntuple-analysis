from plotters import Selection, PID, add_selections
# ---------------------------------------------------
# TP selections
tp_id_selections = [Selection('all', '', ''),
                    Selection('Em', 'EGId', 'quality >0'),
                    Selection('Emv1', 'EGId V1', 'bdt_out > 0')]


tp_pt_selections = [Selection('all', '', ''),
                    Selection('Pt10', 'p_{T}^{L1}>=10GeV', 'pt >= 10'),
                    Selection('Pt20', 'p_{T}^{L1}>=20GeV', 'pt >= 20'),
                    Selection('Pt25', 'p_{T}^{L1}>=25GeV', 'pt >= 25'),
                    Selection('Pt30', 'p_{T}^{L1}>=30GeV', 'pt >= 30')]


tp_eta_selections = [Selection('all', '', ''),
                     Selection('EtaA', '|#eta^{L1}| <= 1.52', 'abs(eta) <= 1.52'),
                     Selection('EtaB', '1.52 < |#eta^{L1}| <= 1.7', '1.52 < abs(eta) <= 1.7'),
                     Selection('EtaC', '1.7 < |#eta^{L1}| <= 2.4', '1.7 < abs(eta) <= 2.4'),
                     Selection('EtaD', '2.4 < |#eta^{L1}| <= 2.8', '2.4 < abs(eta) <= 2.8'),
                     Selection('EtaE', '|#eta^{L1}| > 2.8', 'abs(eta) > 2.8'),
                     Selection('EtaAB', '|#eta^{L1}| <= 1.7', 'abs(eta) <= 1.7'),
                     Selection('EtaABC', '|#eta^{L1}| <= 2.4', 'abs(eta) <= 2.4'),
                     Selection('EtaBC', '1.52 < |#eta^{L1}| <= 2.4', '1.52 < abs(eta) <= 2.4'),
                     Selection('EtaBCD', '1.52 < |#eta^{L1}| <= 2.8', '1.52 < abs(eta) <= 2.8'),
                     Selection('EtaBCDE', '1.52 < |#eta^{L1}|', '1.52 < abs(eta)')]


tp_rate_selections = add_selections(tp_id_selections, tp_eta_selections)

tp_match_selections = add_selections(tp_id_selections, tp_pt_selections)

gen_part_selections = [Selection('Ele', 'e^{#pm}', 'abs(pdgid) == {}'.format(PID.electron)),
                       # Selection('Phot', '#gamma', 'abs(pdgid) == {}'.format(PID.photon)),
                       # Selection('Pion', '#pi', 'abs(pdgid) == {}'.format(PID.pion))
                       ]

gen_ee_selections = [Selection('', '', 'reachedEE == 2')]

gen_eta_selections = [Selection('EtaA', '|#eta^{GEN}| <= 1.52', 'abs(eta) <= 1.52'),
                      Selection('EtaB', '1.52 < |#eta^{GEN}| <= 1.7', '1.52 < abs(eta) <= 1.7'),
                      Selection('EtaC', '1.7 < |#eta^{GEN}| <= 2.4', '1.7 < abs(eta) <= 2.4'),
                      Selection('EtaD', '2.4 < |#eta^{GEN}| <= 2.8', '2.4 < abs(eta) <= 2.8'),
                      Selection('EtaE', '|#eta^{GEN}| > 2.8', 'abs(eta) > 2.8'),
                      Selection('EtaAB', '|#eta^{GEN}| <= 1.7', 'abs(eta) <= 1.7'),
                      Selection('EtaABC', '|#eta^{GEN}| <= 2.4', 'abs(eta) <= 2.4'),
                      Selection('EtaBC', '1.52 < |#eta^{GEN}| <= 2.4', '1.52 < abs(eta) <= 2.4'),
                      Selection('EtaBCD', '1.52 < |#eta^{GEN}| <= 2.8', '1.52 < abs(eta) <= 2.8'),
                      Selection('EtaBCDE', '1.52 < |#eta^{GEN}|', '1.52 < abs(eta)')]


gen_pt_selections = [Selection('Pt10', 'p_{T}^{GEN}>=10GeV', 'pt >= 10'),
                     Selection('Pt20', 'p_{T}^{GEN}>=20GeV', 'pt >= 20'),
                     Selection('Pt30', 'p_{T}^{GEN}>=30GeV', 'pt >= 30'),
                     Selection('Pt40', 'p_{T}^{GEN}>=40GeV', 'pt >= 40')]


gen_part_ee_sel = add_selections(gen_part_selections, gen_ee_selections)
gen_part_ee_pt_sel = add_selections(gen_part_ee_sel, gen_pt_selections)
gen_part_ee_eta_sel = add_selections(gen_part_ee_sel, gen_eta_selections)

gen_part_selections = []
gen_part_selections += gen_part_ee_sel
gen_part_selections += gen_part_ee_pt_sel
gen_part_selections += gen_part_ee_eta_sel

gen_part_sel_genplotting = [Selection('all')]
gen_part_sel_genplotting += gen_part_ee_sel



if __name__ == "__main__":
    for sel in gen_part_selections:
        print sel
