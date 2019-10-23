# %load settings.py
# === samples =====================================================
import python.selections as selections
import python.collections as collections

import pprint


samples = []

# samples += samples_nugunrates
# samples += samples_nugunrates_V8
samples += samples_ele_V9

for smp in samples:
    smp.open_file()


sample = 'ele-V9'

do_rate = False
do_eff = False
do_reso = False
do_calib = True

# === TP ==========================================================
# tps = ['HMvDR',
#        'HMvDRshapeDr']

tps = [
       'EG',
       'TkEle',
       'TkEleEL',
       'TkEleBRL',
       'TkEleELBRL',
       'TkEleALL',
       'TkEleELALL'
]

# === Load the Histo Primitives ====================================
histo_primitives = pd.DataFrame()

if do_rate:
    from python.plotters import rate_plotters, eg_rate_plotters

    for plotter in rate_plotters:
        histo_primitives = histo_primitives.append(plotter.get_histo_primitives(), ignore_index=True)
    for plotter in eg_rate_plotters:
        histo_primitives = histo_primitives.append(plotter.get_histo_primitives(), ignore_index=True)
if do_eff or do_reso:
    from python.plotters import eg_genmatched_plotters, track_genmatched_plotters
    for plotter in eg_genmatched_plotters:
        histo_primitives = histo_primitives.append(plotter.get_histo_primitives(), ignore_index=True)
    for plotter in track_genmatched_plotters:
        histo_primitives = histo_primitives.append(plotter.get_histo_primitives(), ignore_index=True)
if do_calib:
    from python.plotters import tp_calib_plotters

    for plotter in tp_calib_plotters:
        histo_primitives = histo_primitives.append(plotter.get_histo_primitives(), ignore_index=True)


# print histo_primitives.data.unique()
# === TP selections ================================================
tp_select = {}

for tp in tps:
    tp_select[tp] = histo_primitives[histo_primitives.data == tp].data_sel.unique().tolist()

# ==== GEN selections ===============================================
gen_select ={}
for tp in tps:
    gen_select[tp] = histo_primitives[histo_primitives.data == tp].gen_sel.unique().tolist()

#  ==== labels ===============================================
tp_labels = histo_primitives[['data', 'data_label']].drop_duplicates().set_index('data').T.to_dict('records')[0]
tp_selection_labels = histo_primitives[['data_sel', 'data_sel_label']].drop_duplicates().set_index('data_sel').T.to_dict('records')[0]
gen_selection_labels = histo_primitives[['gen_sel', 'gen_sel_label']].drop_duplicates().set_index('gen_sel').T.to_dict('records')[0]



import pprint
pp = pprint.PrettyPrinter(indent=4)
print '--- TPs: '
pp.pprint(tps)
print '--- TP selections:'
pp.pprint(tp_select)
print '--- GEN selections:'
pp.pprint(gen_select)


# print selections.eg_rate_selections
