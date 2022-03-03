from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections


# simple_selections = (selections.Selector('^EGq[4-5]$')*('^Pt[1-3][0]$|all'))()

simple_selections = [selections.Selection("all", '', ''),
                     selections.Selection('Pt10', 'p_{T}^{TOBJ}>=10GeV', 'pt >= 10'),]

print(f"simple_selections: {simple_selections}")

l1tc_simple_plotters = [
    # EE Tk-electrons
    plotters.TkElePlotter(collections.tkeles_EE_pfnf, simple_selections),
    ]
