from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections

test_plotters = [
    plotters.IsoTuplePlotter(
        collections.tkeles_EE,
        collections.gen_parts,
        selections.egid_ee_selections,
        selections.gen_pid_ee_selections
        ),
    plotters.IsoTuplePlotter(
        collections.tkem_EE,
        collections.gen_parts,
        selections.egid_ee_selections,
        selections.gen_pid_ee_selections
        ),
    plotters.IsoTuplePlotter(
        collections.tkeles_EE_pfnf,
        collections.gen_parts,
        selections.egid_ee_pfnf_selections,
        selections.gen_pid_ee_selections
        ),
    plotters.IsoTuplePlotter(
        collections.tkem_EE_pfnf,
        collections.gen_parts,
        selections.egid_ee_pfnf_selections,
        selections.gen_pid_ee_selections
        ),
]
