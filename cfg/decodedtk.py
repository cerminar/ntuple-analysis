from __future__ import absolute_import
import python.plotters as plotters
import python.collections as collections
import python.selections as selections

dectk_selections = (selections.Selector('^Eta[AB]*[BC]$|all')*('^Pt[1,2,5][0]$|all'))()
dectk_match_selections = (selections.Selector('^Pt5$|^Pt[1,2,5][0]$|all'))()
track_selections = (selections.Selector('^TkCTL1|all')&('^Pt5$|^Pt[1,2,5][0]$|all'))()

decTk_plotters = [
    plotters.DecTkPlotter(
        collections.decTk,
        dectk_selections
    ),
    plotters.DecTrackGenMatchPlotter(
        collections.decTk,
        collections.sim_parts,
        dectk_match_selections,
        selections.gen_ee_tk_selections
    )
]

tk_plotters = [
    plotters.TrackPlotter(
        collections.tracks,
        track_selections
    ),
    plotters.TrackGenMatchPlotter(
        collections.tracks,
        collections.sim_parts,
        track_selections,
        selections.gen_ee_tk_selections
    )
]
