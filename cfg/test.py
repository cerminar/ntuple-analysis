from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections

egid_ee_selections = (selections.Selector('^EGq[4-5]'))()
egid_ee_pfnf_selections = (selections.Selector('^EGq[1-2]$'))()

gen_pid_ee_selections = (selections.Selector('GEN$')*('Ee$'))()

plotters = [
    plotters.IsoTuplePlotter(
        collections.tkeles_EE,
        collections.gen_parts,
        egid_ee_selections,
        gen_pid_ee_selections
        ),
    plotters.IsoTuplePlotter(
        collections.tkem_EE,
        collections.gen_parts,
        egid_ee_selections,
        gen_pid_ee_selections
        ),
    plotters.IsoTuplePlotter(
        collections.tkeles_EE_pfnf,
        collections.gen_parts,
        egid_ee_pfnf_selections,
        gen_pid_ee_selections
        ),
    plotters.IsoTuplePlotter(
        collections.tkem_EE_pfnf,
        collections.gen_parts,
        egid_ee_pfnf_selections,
        gen_pid_ee_selections
        ),
]
