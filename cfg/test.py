from python import plotters, selections
import cfg.datasets.fastpuppi_collections as coll

egid_ee_selections = (selections.Selector('^EGq[4-5]'))()
egid_ee_pfnf_selections = (selections.Selector('^EGq[1-2]$'))()

gen_pid_ee_selections = (selections.Selector('GEN$')*('Ee$'))()

plotters = [
    # plotters.IsoTuplePlotter(
    #     coll.TkEleEE,
    #     coll.gen_parts,
    #     egid_ee_selections,
    #     gen_pid_ee_selections
    #     ),
    # plotters.IsoTuplePlotter(
    #     coll.TkEmEE,
    #     coll.gen_parts,
    #     egid_ee_selections,
    #     gen_pid_ee_selections
        # ),
]
