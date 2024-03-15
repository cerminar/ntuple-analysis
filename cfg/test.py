from python import collections, plotters, selections

egid_ee_selections = (selections.Selector('^EGq[4-5]'))()
egid_ee_pfnf_selections = (selections.Selector('^EGq[1-2]$'))()

gen_pid_ee_selections = (selections.Selector('GEN$')*('Ee$'))()

plotters = [
    plotters.IsoTuplePlotter(
        collections.TkEleEE,
        collections.gen_parts,
        egid_ee_selections,
        gen_pid_ee_selections
        ),
    plotters.IsoTuplePlotter(
        collections.TkEmEE,
        collections.gen_parts,
        egid_ee_selections,
        gen_pid_ee_selections
        ),
]
