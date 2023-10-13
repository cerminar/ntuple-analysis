from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections

pfjet_selections = (selections.Selector('^Pt[34]0$|all'))()

genjet_selections = (selections.Selector('^GENJ$')*('^EtaE[EB]$|all')+selections.Selector('GENJ$')*('^Pt30'))()

jets_genmatched = [
    plotters.JetGenMatchPlotter(
        collections.pfjets, collections.gen_jet,
        pfjet_selections, genjet_selections),
]

if False:
    print('---- pfjet_selections ----------------')
    for sel in pfjet_selections:
        print(sel)

    print('---- genjet_selections ----------------')
    for sel in genjet_selections:
        print(sel)