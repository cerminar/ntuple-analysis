import python.histos as histos
from python.draw.drawingTools import *
import python.draw.utilities as draw_utils
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


# Turn-on fit utilities for extracting the 95% efficiency threshold (x95)
# and its uncertainty with multiple methods:
#   - slope      : local conversion from vertical CI to horizontal uncertainty
#   - covariance : first-order covariance propagation in parameter space
#   - toys       : toy-MC propagation by sampling fit parameters


_TURNON_PARAM_LIMITS = {
    0: (0.0, 1.0),
    1: (0.0, 50.0),
    2: (0.1, 1.0),
    3: (0.9, 1.0),
    4: (0.0, 0.0),
}


def _turnon_value(ROOT, x_value, params):
    """Evaluate the turn-on model used for the per-threshold fit."""
    return (ROOT.Math.normal_cdf(params[0] * (x_value - params[1]), params[0] * params[2], 0)
            - ROOT.TMath.Exp(-params[0] * (x_value - params[1]) + params[0] * params[0] * params[2] * params[2] / 2)
            * ROOT.Math.normal_cdf(params[0] * (x_value - params[1]), params[0] * params[2], params[0] * params[0] * params[2] * params[2])) * (params[3] - params[4]) + params[4]


def _solve_x95_from_params(ROOT, params, target=0.95, x_low=0.0, x_high=100.0, max_iter=80):
    """Solve f(x; params) = target with bisection in [x_low, x_high].

        Returns:
            - x value if a crossing is found
            - None if the function does not bracket the target in the range
    """
    f_low = _turnon_value(ROOT, x_low, params) - target
    f_high = _turnon_value(ROOT, x_high, params) - target
    if abs(f_low) < 1e-12:
        return x_low
    if abs(f_high) < 1e-12:
        return x_high
    if f_low * f_high > 0:
        return None

    low = x_low
    high = x_high
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = _turnon_value(ROOT, mid, params) - target
        if abs(f_mid) < 1e-10:
            return mid
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid
    return 0.5 * (low + high)


def _covariance_element(cov_matrix, row_idx, col_idx):
    """Read one covariance element with multiple PyROOT access conventions."""
    for getter in (
        lambda matrix, i, j: matrix[i][j],
        lambda matrix, i, j: matrix(i, j),
        lambda matrix, i, j: matrix[i, j],
    ):
        try:
            return float(getter(cov_matrix, row_idx, col_idx))
        except Exception:
            continue
    raise TypeError('Unable to access covariance matrix element')


def _covariance_to_numpy(cov_matrix, n_pars):
    """Convert ROOT covariance object to a dense numpy array."""
    cov_array = np.zeros((n_pars, n_pars), dtype=float)
    for row_idx in range(n_pars):
        for col_idx in range(n_pars):
            cov_array[row_idx, col_idx] = _covariance_element(cov_matrix, row_idx, col_idx)
    return cov_array


def _clip_params(params):
    """Clip parameter vector to fit limits before model evaluation."""
    clipped = list(params)
    for par_idx, (par_min, par_max) in _TURNON_PARAM_LIMITS.items():
        clipped[par_idx] = min(max(clipped[par_idx], par_min), par_max)
    return clipped


def _x95_error_from_slope(ROOT, tf_yc, pt095):
    """Uncertainty method 1: local slope conversion at x95.

        Steps:
            1. Get vertical confidence interval sigma_y at fixed x95.
            2. Convert to horizontal uncertainty with sigma_x ~= sigma_y / |f'(x95)|.
    """
    # ROOT returns the confidence interval on the fitted curve at a fixed x, i.e.
    # a vertical uncertainty sigma_y around y = f(x). Here we want the uncertainty
    # on the derived threshold x95 defined by f(x95) = 0.95, so we convert the
    # vertical band into a horizontal one using the local slope:
    #   sigma_x95 ~= sigma_y(x95) / |f'(x95)|.
    # This is a local linearization of the inverse function. It is fast and easy
    # to run inside the parallel worker, but it can inflate the error when the
    # turn-on is shallow near 95% efficiency because |f'(x95)| becomes small.
    grint = ROOT.TGraphErrors(1)
    grint.SetPoint(0, pt095, 0.0)
    ROOT.TVirtualFitter.GetFitter().GetConfidenceIntervals(grint)
    yerr95 = grint.GetErrorY(0)

    slope = abs(tf_yc.Derivative(pt095))
    return yerr95 / slope if slope > 1e-12 else 0.0


def _x95_error_from_covariance(ROOT, tf_yc, fit_result, pt095):
    """Uncertainty method 2: first-order covariance propagation.

        Steps:
            1. Read fit covariance matrix C.
            2. Compute numerical gradient J_i = d(x95)/d(p_i) with finite differences.
            3. Return sqrt(J C J^T).
    """
    params = [float(tf_yc.GetParameter(par_idx)) for par_idx in range(tf_yc.GetNpar())]
    cov_matrix = _covariance_to_numpy(fit_result.GetCovarianceMatrix(), len(params))
    gradients = np.zeros(len(params), dtype=float)

    for par_idx, param_value in enumerate(params):
        par_error = float(tf_yc.GetParError(par_idx))
        if par_error <= 0:
            continue

        # Adaptive finite-difference step based on fit uncertainty and scale.
        step = max(par_error * 0.1, 1e-4 * max(1.0, abs(param_value)))
        par_min, par_max = _TURNON_PARAM_LIMITS[par_idx]
        plus_value = min(param_value + step, par_max)
        minus_value = max(param_value - step, par_min)
        if plus_value == minus_value:
            continue

        plus_params = params.copy()
        minus_params = params.copy()
        plus_params[par_idx] = plus_value
        minus_params[par_idx] = minus_value

        x_plus = _solve_x95_from_params(ROOT, plus_params)
        x_minus = _solve_x95_from_params(ROOT, minus_params)
        if x_plus is None or x_minus is None:
            continue

        gradients[par_idx] = (x_plus - x_minus) / (plus_value - minus_value)

    variance = float(gradients @ cov_matrix @ gradients)
    return math.sqrt(max(variance, 0.0))


def _x95_error_from_toys(ROOT, tf_yc, fit_result, n_toys, seed):
    """Uncertainty method 3: toy-MC propagation from parameter covariance.

        Steps:
            1. Sample parameter vectors from N(best_fit, covariance).
            2. Solve x95 for each sampled parameter set.
            3. Use the sample standard deviation of valid x95 values.
    """
    params = np.array([float(tf_yc.GetParameter(par_idx)) for par_idx in range(tf_yc.GetNpar())], dtype=float)
    cov_matrix = _covariance_to_numpy(fit_result.GetCovarianceMatrix(), len(params))
    # Force covariance to positive-semidefinite for robust sampling.
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    cov_psd = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    rng = np.random.default_rng(seed)
    sampled_params = rng.multivariate_normal(params, cov_psd, size=max(2, int(n_toys)))

    x95_samples = []
    for sample_params in sampled_params:
        x95_value = _solve_x95_from_params(ROOT, _clip_params(sample_params.tolist()))
        if x95_value is not None:
            x95_samples.append(x95_value)

    if len(x95_samples) < 2:
        return 0.0
    return float(np.std(np.asarray(x95_samples, dtype=float), ddof=1))


def _compute_x95_error(ROOT, tf_yc, fit_result, pt095, error_method, n_toys, seed):
    """Dispatch x95 uncertainty computation based on selected method."""
    method = str(error_method).strip().lower() if error_method is not None else 'slope'
    if method in ('', 'default', 'legacy'):
        method = 'slope'
    elif method in ('cov', 'covar', 'covariance_matrix'):
        method = 'covariance'
    elif method in ('toy', 'toy_mc', 'toy-mc'):
        method = 'toys'

    if method == 'slope':
        return _x95_error_from_slope(ROOT, tf_yc, pt095)
    if method == 'covariance':
        return _x95_error_from_covariance(ROOT, tf_yc, fit_result, pt095)
    if method == 'toys':
        return _x95_error_from_toys(ROOT, tf_yc, fit_result, max(2, int(n_toys)), seed)
    raise ValueError(f'Unknown x95 error method: {error_method!r}')


def _fit_turnon_point(task):
    """Fit a single turn-on graph in a worker process.

    Returns (pt_threshold, pt95, pt95_err, fit_status).
    """
    # The task payload contains one threshold fit input and method options.
    pt, x_vals, y_vals, ey_low_vals, ey_high_vals, error_method, n_toys = task

    import ROOT

    def f_yc(x, par):
        return (ROOT.Math.normal_cdf(par[0]*(x[0]-par[1]), par[0]*par[2], 0)
            - ROOT.TMath.Exp(-par[0]*(x[0]-par[1]) + par[0]*par[0]*par[2]*par[2]/2)
            * ROOT.Math.normal_cdf(par[0]*(x[0]-par[1]), par[0]*par[2], par[0]*par[0]*par[2]*par[2])) * (par[3] - par[4]) + par[4]

    ROOT.TVirtualFitter.SetDefaultFitter("Fumili2")
    ROOT.TVirtualFitter.SetPrecision(1e-04)

    tf_yc = ROOT.TF1('tf_yc_worker', f_yc, 0, 100, 5)
    tf_yc.SetParNames("#lambda", "#mu", "#sigma", "Plateau", "Baseline")
    tf_yc.FixParameter(4, 0)
    tf_yc.SetParLimits(1, 0, 50)
    tf_yc.SetParLimits(3, 0.9, 1)
    tf_yc.SetParLimits(0, 0, 1)
    tf_yc.SetParLimits(2, 0.1, 1)

    tf_yc.SetParameters(0.6, pt, 0.1, 1.0, 0.0)


    # Rebuild graph in the worker process to avoid sharing ROOT objects
    # across processes.
    graph = ROOT.TGraphAsymmErrors(len(x_vals))
    for i, (xv, yv, eyl, eyh) in enumerate(zip(x_vals, y_vals, ey_low_vals, ey_high_vals)):
        graph.SetPoint(i, xv, yv)
        graph.SetPointError(i, 0.0, 0.0, eyl, eyh)

    # Fit options:
    # E: improve error estimation, M: improve fit strategy,
    # Q: quiet mode, R: use TF1 fit range, S: return fit result.
    result = graph.Fit(tf_yc, 'EMQRS')
    fit_status = int(result) if result else -1

    pt095 = tf_yc.GetX(0.95)
    pt095_err = _compute_x95_error(ROOT, tf_yc, result, pt095, error_method, n_toys, seed=1000 + int(pt))

    return pt, pt095, pt095_err, fit_status


def _extract_graph_points(graph):
    """Extract graph points/errors into plain Python lists for multiprocessing."""
    n_points = graph.GetN()
    x_vals = [float(graph.GetX()[i]) for i in range(n_points)]
    y_vals = [float(graph.GetY()[i]) for i in range(n_points)]

    ey_low_vals = []
    ey_high_vals = []
    if hasattr(graph, 'GetEYlow') and hasattr(graph, 'GetEYhigh'):
        ey_low_vals = [float(graph.GetEYlow()[i]) for i in range(n_points)]
        ey_high_vals = [float(graph.GetEYhigh()[i]) for i in range(n_points)]
    elif hasattr(graph, 'GetEY'):
        ey_vals = [float(graph.GetEY()[i]) for i in range(n_points)]
        ey_low_vals = ey_vals
        ey_high_vals = ey_vals
    else:
        ey_low_vals = [0.0] * n_points
        ey_high_vals = [0.0] * n_points

    return x_vals, y_vals, ey_low_vals, ey_high_vals


def _run_turnon_fits(fit_tasks, n_workers=None):
    """Run per-threshold fits serially or in a process pool."""
    if not fit_tasks:
        return []

    if n_workers is None:
        n_workers = min(len(fit_tasks), max(1, (os.cpu_count() or 1) - 1))
    n_workers = max(1, min(n_workers, len(fit_tasks)))

    if n_workers == 1:
        return [_fit_turnon_point(task) for task in fit_tasks]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        return list(executor.map(_fit_turnon_point, fit_tasks))



def draw_ton_scaling(hplot, smps, wc_eff, draw_style, configs):
    """Build turn-on scaling graphs and draw turn-on plus scaling summaries.

        Workflow:
            1. Ensure turn-on histograms exist for requested selections.
            2. Fit each threshold turn-on to extract x95 and its uncertainty.
            3. Store/update scaling graphs in hplot.data.
            4. Draw per-threshold turn-ons and final scaling plots.
    """

    scaling_histo_class = 'ScalingGraph'
    def f_yc(x, par):
        return (ROOT.Math.normal_cdf(par[0]*(x[0]-par[1]), par[0]*par[2], 0) - ROOT.TMath.Exp(-par[0]*(x[0]-par[1])+par[0]*par[0]*par[2]*par[2]/2)*ROOT.Math.normal_cdf(par[0]*(x[0]-par[1]), par[0]*par[2], par[0]*par[0]*par[2]*par[2])) * (par[3] - par[4]) + par[4]
    
    ROOT.TVirtualFitter.SetDefaultFitter("Fumili2")
    ROOT.TVirtualFitter.SetPrecision(1e-04)



    print('Computing Turn-on curves...')
    pt_points_try = [10, 15, 20, 25, 30, 35, 40, 50]

    for smp in smps:
        for objs, objs_sel, gen_sel, h_name, opts in configs:
            for obj in objs:
                for osel in objs_sel:
                    for gsel in gen_sel:
                        print(f' -> sample: {smp.type}, obj: {obj}, sel: {osel}, gen_sel: {gsel}')    
                        pt_points = []
                        pt_95 = []
                        pt_95_err = []
                        hsetden = hplot.get_histo(histos.HistoSetEff, smp.type, ['PU200'], obj, osel, gsel)
                        gset =  hplot.get_histo(scaling_histo_class, smp.type, ['PU200'], obj, osel, gsel)
                        print(f'     found histo set for {osel}, gen_sel: {gsel}: {hsetden}, gset: {gset}')
                        dotonfits = False
                        if gset[0] is None:
                            print(f' -> histo set for {osel}, gen_sel: {gsel} does not have scaling graph: we will compute it!')
                            dotonfits = True

                        fit_tasks = []
                        for pt in pt_points_try:
                            if osel == 'all':
                                osel_pt = f'Pt{pt}'
                            else:
                                osel_pt =  f'{osel}Pt{pt}'
                            hsets, labels, text = hplot.get_histo(
                                            histos.HistoSetEff, 
                                            smp.type, 
                                            ['PU200'], 
                                            obj, 
                                            osel_pt, 
                                            gsel, debug=False)
                            if not hsets:
                                continue
                            if len(hsets) > 1:
                                print(f' -> found histos for {osel_pt}, gen_sel: {gsel}, but more than 1 histo set found. Skip pt point.')
                                continue
                            hset = hsets[0]
                            if hset.h_ton is None:
                                print(f' -> histo set for {osel_pt}, gen_sel: {gsel} does not have turn-on histo: computing it!')
                                hset.computeTurnOn(hsetden[0][0].h_num)

                            if dotonfits:
                                # Collect only serializable data for worker processes.
                                h_eff_vs_pt = hset.h_ton.h_pt.CreateGraph()
                                fit_tasks.append((pt, *_extract_graph_points(h_eff_vs_pt)))

                        if dotonfits and fit_tasks:
                            # Per-config steering of uncertainty method and speed.
                            # x95_error_method: 'slope' (default), 'covariance', 'toys'
                            # x95_error_toys  : number of toys when method='toys'
                            # n_fit_workers   : number of worker processes
                            error_method = 'slope'
                            if isinstance(opts, dict):
                                configured_method = opts.get('x95_error_method', 'slope')
                                if configured_method not in (None, ''):
                                    error_method = configured_method

                            n_toys = 200
                            if isinstance(opts, dict):
                                try:
                                    n_toys = max(2, int(opts.get('x95_error_toys', 200)))
                                except (TypeError, ValueError):
                                    n_toys = 200
                            n_fit_workers = opts.get('n_fit_workers', None) if isinstance(opts, dict) else None
                            fit_tasks = [
                                (*task, error_method, n_toys)
                                for task in fit_tasks
                            ]
                            fit_results = _run_turnon_fits(fit_tasks, n_workers=n_fit_workers)
                            fit_results.sort(key=lambda row: row[0])

                            for pt, pt095, err95, fit_status in fit_results:
                                pt_points.append(pt)
                                pt_95.append(pt095)
                                pt_95_err.append(err95)
                                if fit_status != 0:
                                    print(f'       fit status={fit_status} for pt th: {pt}')
                                print(f'       pt th: {pt}, 95% eff: {pt095}, err: {err95}')
                                #                 points.append((tp, pu, pt, pt095))
                                # draw(h_eff_vs_pt, labels, text=text, min_y=0, max_y=1.1, y_axis_label='L1 matched to GEN and p_{T}>th./L1 matched to GEN', h_lines=[0.95, 1])
                                # ROOT.gStyle.SetOptFit(11111)
                                # grint.Draw('same')

                        if dotonfits:
                            # pt_points = [10, 20, 30, 40]
                            # pt_95 = [20, 30, 40, 50]
                            # pt_95_err = [5, 5, 5, 5]
                            print (len(pt_points))
                            print (len(pt_95))
                            print (len(pt_95_err))
                            print (pt_95_err)

                            graph = ROOT.TGraphErrors(len(pt_points), array.array('d', pt_points), 
                                                                    array.array('d', pt_95),
                                                                    array.array('d', [0.]*len(pt_points)),
                                                                    array.array('d', pt_95_err) )

                            graph.SetMarkerStyle(7)
                            graph.SetMarkerColor(2)
                            stuff.append(graph)
                            new_rows = []
                            new_rows.append({'sample': smp.type,
                                            'pu': 'PU200',
                                            'tp': obj,
                                            'tp_sel': osel,
                                            'gen_sel': gsel,
                                            'classtype': scaling_histo_class,
                                            'histo': HWrapper(graph),})
                            new_df = pd.DataFrame(new_rows)
                            # print(new_df)

                            hplot.data = pd.concat([hplot.data, new_df], ignore_index=True)


    print(hplot.data)

    # pt_points = ['Pt20', 'Pt30', 'Pt40']
    for objs, objs_sel, gen_sel, h_name_sfx, opts in configs:
        if len(smps) == 0:
            continue
        # print (smps)
        # print(f'obj: {objs}, sel: {objs_sel}, histo: {h_name}')
        objs_sel_base = objs_sel[0]
        for pt in pt_points_try:
            objs_sel = f'{objs_sel_base}Pt{pt}'

            dm = DrawMachine(draw_style)
            dm.config.legend_position = (0.6,0.05)

            hsets, labels, text = hplot.get_histo(
                histos.HistoSetEff, 
                [s.type for s in smps], 
                ['PU200'], 
                objs, 
                objs_sel, 
                gen_sel, debug=False)
            
            # print(f"# of hsets: {len(hsets)}")
            # for hset in hsets:
            #     hset.computeEff(rebin=2)
            if not hsets:
                continue
            dm.addHistos([his.h_ton.h_pt.CreateGraph() for his in hsets], labels=labels)

            for i in range(1,len(hsets)):
                # print(f'add ratio: {i} to 0')
                dm.addRatioHisto(i,0)
                # dm.addRatioHisto(2,0)
                # dm.addRatioHisto(3,0)
                # dm.addRatioHisto(4,0)

            dm.draw(
                text=text, 
                x_min=0, x_max=100, 
                y_min=0.0, y_max=1.1, 
                h_lines=[1.0, 0.95],
                do_ratio=True,
                y_min_ratio=0.9,
                y_max_ratio=1.1,
                h_lines_ratio=[0.95, 1., 1.05],
                v_lines=[pt],
                y_axis_label='Turn-on efficiency (w.r.t matching)'
        )
            # dm.write(name='eg_TDRvsSummer20_matchig_eff')
            h_name = f'hTonVsPt_{objs[0]}_{objs_sel}_{gen_sel[0]}'
            if h_name_sfx != '':
                h_name = f'{h_name}_{h_name_sfx}'
            dm.toWeb(name=h_name, page_creator=wc_eff)


    for objs, objs_sel, gen_sel, h_name_sfx, opts in configs:
        if len(smps) == 0:
            continue
        # print (smps)
        print(f'obj: {objs}, sel: {objs_sel}, histo: {h_name}')

        dm = DrawMachine(draw_style)
        dm.config.legend_position = (0.4,0.2)

        hsets, labels, text = hplot.get_histo(
            scaling_histo_class, 
            [s.type for s in smps], 
            ['PU200'], 
            objs, 
            objs_sel, 
            gen_sel, debug=True)
        
        if not hsets:
            continue
        for i, his in enumerate(hsets):

            result = his.Fit('pol1', 'MES+', '')
            params = result.GetParams()
            labels[i] += f' (y = {params[1]:.3f}x + {params[0]:.3f})'

        dm.addHistos([his for his in hsets], labels=labels)
        max_pt = 0
        for his in hsets:
            for i in range(his.GetN()):
                val = max(his.GetX()[i],his.GetY()[i])
                if val > max_pt:
                    max_pt = val


        # max_pt = 50

        dm.draw(
            text=text, 
            x_min=0, x_max=max_pt*1.2, 
            y_min=0.0, y_max=max_pt*1.2, 
            # h_lines=[1.0, 0.95],
            # do_ratio=True,
            # y_min_ratio=0.9,
            # y_max_ratio=1.1,
            # h_lines_ratio=[0.95, 1., 1.05],
            # v_lines=[pt],
            y_axis_label='95% efficiency point (GeV)',
            x_axis_label='p_{T} threshold (GeV)',
            do_legend=True,
    )
        # dm.write(name='eg_TDRvsSummer20_matchig_eff')
        h_name = f'hScaling_{objs[0]}_{objs_sel[0]}_{gen_sel[0]}'
        if h_name_sfx != '':
            h_name = f'{h_name}_{h_name_sfx}'
        dm.toWeb(name=h_name, page_creator=wc_eff)
