# %load ./python/utilities.py
import uuid
import ROOT
from drawingTools import draw
import math
import numpy as np


def effSigma(hist):
    xaxis = hist.GetXaxis()
    nb = xaxis.GetNbins()
    if nb < 10:
        print("effsigma: Not a valid histo. nbins = {}".format(nb))
        return -1

    bwid = xaxis.GetBinWidth(1)
    if bwid == 0:
        print("effsigma: Not a valid histo. bwid = {}".format(bwid))
        return -1

    xmax = xaxis.GetXmax()
    xmin = xaxis.GetXmin()
    ave = hist.GetMean()
    rms = hist.GetRMS()

#     print 'xmax: {}, xmin: {}, ave: {}, rms: {}'.format(xmax, xmin, ave, rms)

    total = 0.
    for i in range(0, nb+2):
        total += hist.GetBinContent(i)

    if total < 100.:
        print("effsigma: Too few entries {}".format(total))
        return -1

    ierr = 0
    ismin = 999

    rlim = 0.683*total
    # Set scan size to +/- rms
    nrms = int(rms/(bwid))
    if nrms > nb/10:
        # could be tuned
        nrms = nb/10

    widmin = 9999999.
    # scan the window center
    for iscan in range(-nrms, nrms+1):
        ibm = int((ave-xmin)/bwid+1+iscan)
        x = (ibm-0.5) * bwid + xmin
        xj = x
        xk = x
        jbm = ibm
        kbm = ibm
        bin = hist.GetBinContent(ibm)
        total = bin
        for j in range(1, nb):
            if jbm < nb:
                jbm += 1
                xj += bwid
                bin = hist.GetBinContent(jbm)
                total += bin
                if total > rlim:
                    break
            else:
                ierr = 1
            if kbm > 0:
                kbm -= 1
                xk -= bwid
                bin = hist.GetBinContent(kbm)
                total += bin
                if total > rlim:
                    break
            else:
                ierr = 1
        dxf = (total-rlim)*bwid/bin
        wid = (xj-xk+bwid-dxf)*0.5
        if wid < widmin:
            widmin = wid
            ismin = iscan

    if ismin == nrms or ismin == -nrms:
        ierr = 3
    if ierr != 0:
        print("effsigma: Error of type {}".format(ierr))

    return widmin


def quantiles(yswz, zeroSuppress=True):
    ys = [y for y in yswz if y > 0] if zeroSuppress else yswz[:]
    if len(ys) < 3:
        return (0, 0, 0)
    ys.sort()
    ny = len(ys)
    median = ys[ny/2]
    # if ny > 400e9:
    #     u95 = ys[min(int(math.ceil(ny*0.975)), ny-1)]
    #     l95 = ys[int(math.floor(ny*0.025))]
    #     u68 = 0.5*(median+u95)
    #     l68 = 0.5*(median+l95)
    if ny > 20:
        u68 = ys[min(int(math.ceil(ny*0.84)), ny-1)]
        l68 = ys[int(math.floor(ny*0.16))]
    else:
        rms = math.sqrt(sum((y-median)**2 for y in ys)/ny)
        u68 = median + rms
        l68 = median - rms
    return (median, l68, u68)


def gausstailfit_wc(name, project_hist, bin_limits):
    global cache
    global stuff
    if cache is not None and not cache[(cache.h_name == name) & (cache.bin_limits == bin_limits)].empty:
        print('READ cached fit results h_name: {}, bin_limits: {}'.format(name, bin_limits))
#         print cache[(cache.h_name == name) & (cache.bin_limits == bin_limits)].results
        return cache[(cache.h_name == name) & (cache.bin_limits == bin_limits)].results.values[0]

    max_bin = project_hist.GetMaximumBin()
    max_value = project_hist.GetBinCenter(max_bin)
    rms_value = project_hist.GetRMS()
    max_y = project_hist.GetMaximum()
#     print max_bin, max_value, rms_value

    def gausstail(x, p):
        #         // [Constant] * ROOT::Math::crystalball_function(x, [Alpha], [N], [Sigma], [Mean])
        return p[0] * ROOT.Math.crystalball_function(x[0], p[3], p[4], p[2], p[1])

    fitf = ROOT.TF1('gausstail', gausstail, -1.5, 1.5, 5)
    fitf.SetParNames('norm', 'mean', 'sigma', 'alpha', 'n')
#     fitf.FixParameter(0, 1.)
    fitf.SetParLimits(1, max_value-0.04, max_value+0.04)

    fitf.SetParameters(max_y, max_value, rms_value, 1, 1)
    draw([project_hist], labels=['fit'], text=name)
    stuff.append(fitf)
    project_hist.Draw("same")
#     c.Draw()
#     print '   name: {}, y_max = {}, max_value = {}, RMS = {}'.format(name, max_y, max_value, rms_value)
    result = project_hist.Fit('gausstail', 'QERLS+')
    result.Print()
#     print '   norm = {}, reso_mean = {}, reso_sigma = {}, alpha = {}, n = {}'.format(result.GetParams()[0], result.GetParams()[1], result.GetParams()[2], result.GetParams()[3], result.GetParams()[4])
#     func = project_hist.GetFunction("gaus")
    # print '   NDF = {}, chi2 = {}, prob = {}'.format(fitf.GetNDF(), fitf.GetChisquare(), fitf.GetProb())

    if cache is not None:
        cache = cache.append({'h_name': name,
                              'bin_limits': bin_limits,
                              'results': result}, ignore_index=True)
#         print cache

    return result


def effective_sigma_energy(project_hist):
    eff_sigma = effSigma(project_hist)
    bin_values = [project_hist.GetBinContent(bin) for bin in range(1, project_hist.GetNbinsX()+1)]
    return (eff_sigma,)


def gausstailfit_energy(project_hist):
    global stuff
    max_bin = project_hist.GetMaximumBin()
    max_value = project_hist.GetBinCenter(max_bin)
    rms_value = project_hist.GetRMS()
    max_y = project_hist.GetMaximum()

    def gausstail(x, p):
        return p[0] * ROOT.Math.crystalball_function(x[0], p[3], p[4], p[2], p[1])

    fitf = ROOT.TF1('gausstail', gausstail, -1.0, 0.5, 5)
    fitf.SetParNames('norm', 'mean', 'sigma', 'alpha', 'n')
#     fitf.FixParameter(0, 1.)
    fitf.SetParLimits(1, max_value-0.04, max_value+0.04)
    fitf.SetParameters(max_y, max_value, rms_value, 1, 1)
    stuff.append(fitf)
    project_hist.Draw()
#     c.Draw()
#     print '   y_max = {}, max_value = {}, RMS = {}'.format(max_y, max_value, rms_value)
    result = project_hist.Fit('gausstail', 'QERLS+')
    result.Print()
#     print '   norm = {}, reso_mean = {}, reso_sigma = {}, alpha = {}, n = {}'.format(result.GetParams()[0],
#                                                                                      result.GetParams()[1],
#                                                                                      result.GetParams()[2],
#                                                                                      result.GetParams()[3],
#                                                                                      result.GetParams()[4])
    return result.GetParams()[0], result.GetParams()[1], result.GetParams()[2], result.GetParams()[3], result.GetParams()[4]


def gausstailfit_ptresp(project_hist, x_low=0., x_high=1.2):
    global stuff
    max_bin = project_hist.GetMaximumBin()
    max_value = project_hist.GetBinCenter(max_bin)
    rms_value = project_hist.GetRMS()
    max_y = project_hist.GetMaximum()

    eff_sigma = effSigma(project_hist)

    prob = np.array([0.001, 0.999])

    q = np.array([0., 1.2])
    y = project_hist.GetQuantiles(2, q, prob)
    x_low = q[0]
    x_high = q[1]

    def gausstail(x, p):
        return p[0] * ROOT.Math.crystalball_function(x[0], p[3], p[4], p[2], p[1])

    # print x_low, x_high
    fitf = ROOT.TF1('gausstail', gausstail, x_low, x_high, 5)

    fitf.SetParNames('norm', 'mean', 'sigma', 'alpha', 'n')
#     fitf.FixParameter(0, 1.)
    fitf.SetParLimits(1, max_value-eff_sigma, max_value+eff_sigma)
    fitf.SetParameters(max_y, max_value, eff_sigma, 1, 1)
    stuff.append(fitf)
    project_hist.Draw()

    #     c.Draw()
#     print '   y_max = {}, max_value = {}, RMS = {}'.format(max_y, max_value, rms_value)
    result = project_hist.Fit('gausstail', 'QERLS+')
    result.Print()
    print('CHi2 prob: {}'.format(fitf.GetProb()))
#     print '   norm = {}, reso_mean = {}, reso_sigma = {}, alpha = {}, n = {}'.format(result.GetParams()[0],
#                                                                                      result.GetParams()[1],
#                                                                                      result.GetParams()[2],
#                                                                                      result.GetParams()[3],
#                                                                                      result.GetParams()[4])
    return result.GetParams()[0], result.GetParams()[1], result.GetParams()[2], result.GetParams()[3], result.GetParams()[4], fitf.GetProb()


def computeResolution(histo2d,
                      bin_limits,
                      y_axis_range,
                      fit_function,
                      result_index,
                      cache=None):
    global stuff

    def get_results(histo_name,
                    project_hist,
                    bin_limits,
                    fit_function,
                    cache=None):
        if cache is not None:
            if not cache[(cache.h_name == histo_name) &
                         (cache.bin_limits == bin_limits) &
                         (cache.fit_function == fit_function)].empty:
                print('READ cached fit results h_name: {}, bin_limits: {}, fit_function: {}'.format(histo_name,
                                                                                                    bin_limits,
                                                                                                    fit_function))
                return cache[(cache.h_name == histo_name) &
                             (cache.bin_limits == bin_limits) &
                             (cache.fit_function == fit_function)].results.values[0]
            else:
                print("No ENTRY in CACHE")
                result = fit_function(project_hist)
                cache.loc[cache.shape[0]+1] = {
                    'h_name': histo_name,
                    'bin_limits': bin_limits,
                    'fit_function': fit_function,
                    'results': result}
                return result
        return fit_function(project_hist)

    h2d = histo2d.Clone()
    h2d.GetYaxis().SetRangeUser(y_axis_range[0], y_axis_range[1])

    x, y, ex_l, ex_h, ey_l, ey_h = [], [], [], [], [], []
    print('-----------------------')
    for x_bin_low, x_bin_high in bin_limits:
        y_proj = h2d.ProjectionY(uuid.uuid4().hex[:6]+'_y', x_bin_low, x_bin_high)
        stuff.append(y_proj)
        x_low = h2d.GetXaxis().GetBinLowEdge(x_bin_low)
        x_high = h2d.GetXaxis().GetBinUpEdge(x_bin_high)
#         print 'x_low: {} x_high: {}'.format(x_low, x_high)
        draw([y_proj], labels=['fit'], text='BIN: ({}, {}) = ({}, {}) GeV, RES: {}'.format(
                                            x_bin_low, x_bin_high, x_low, x_high, 0))

        fit_result = get_results(histo2d.GetName(),
                                 y_proj,
                                 (x_bin_low, x_bin_high),
                                 fit_function,
                                 cache)
#         draw([y_proj], labels=['fit'], text='BIN: ({}, {}) = ({}, {}) GeV, RES: {}'.format(
#                                             x_bin_low, x_bin_high, x_low, x_high, fit_result[result_index]))

        h2d.SetAxisRange(x_low, x_high)
        x_mean = h2d.GetMean()
        x.append(x_mean)
        ex_l.append(0)
        ex_h.append(0)
        y.append(fit_result[result_index])
        ey_l.append(0)
        ey_h.append(0)
    return x, y, ex_l, ex_h, ey_l, ey_h


def computeEResolution(h2d_orig,
                       bins_limits=[(3, 6), (7, 12), (13, 23), (24, 34), (35, 49), (50, 100)],
                       cache=None):
    global stuff
    h2d = h2d_orig.Clone()
    h2d.GetYaxis().SetRangeUser(-100, 100)
    x, y, ex_l, ex_h, ey_l, ey_h = [], [], [], [], [], []
    print('-----------------------')
    for x_bin_low, x_bin_high in bins_limits:
        y_proj = h2d.ProjectionY(uuid.uuid4().hex[:6]+'_y', x_bin_low, x_bin_high)
        stuff.append(y_proj)
        x_low = h2d.GetXaxis().GetBinLowEdge(x_bin_low)
        x_high = h2d.GetXaxis().GetBinUpEdge(x_bin_high)
#         print 'x_low: {} x_high: {}'.format(x_low, x_high)
#         fit_result = gausstailfit(h2d_orig.GetName(), y_proj)
        fit_result = gausstailfit_wc(h2d_orig.GetName(), y_proj, (x_bin_low, x_bin_high))
        h2d.SetAxisRange(x_low, x_high)
        x_mean = h2d.GetMean()
#         print 'mean: {}'.format(x_mean)
    #     x_value = h2d.GetXaxis().GetBinCenter(x_bin)
    #     x_err_min = x_value - h2d.GetXaxis().GetBinLowEdge(x_bin)
    #     x_err_plus = h2d.GetXaxis().GetBinUpEdge(x_bin) - x_value

        x.append(x_mean)
        ex_l.append(0)
        ex_h.append(0)
        y.append(fit_result.GetParams()[2])

#         y.append(fit_result.GetParams()[2]/x_mean)
        ey_l.append(0)
        ey_h.append(0)
    return x, y, ex_l, ex_h, ey_l, ey_h


def computeEResolutionMean(h2d_orig,
                           bins_limits=[(3, 6), (7, 12), (13, 23), (24, 34), (35, 49), (50, 100)],
                           cache=None):
    global stuff
    h2d = h2d_orig.Clone()
    h2d.GetYaxis().SetRangeUser(-100, 100)
    x, y, ex_l, ex_h, ey_l, ey_h = [], [], [], [], [], []
    print('-----------------------')
    for x_bin_low, x_bin_high in bins_limits:
        y_proj = h2d.ProjectionY(uuid.uuid4().hex[:6]+'_y', x_bin_low, x_bin_high)
        stuff.append(y_proj)
        x_low = h2d.GetXaxis().GetBinLowEdge(x_bin_low)
        x_high = h2d.GetXaxis().GetBinUpEdge(x_bin_high)
        print('x_low: {} x_high: {}'.format(x_low, x_high))
#         fit_result = gausstailfit(h2d_orig.GetName(), y_proj)
        fit_result = gausstailfit_wc(h2d_orig.GetName(), y_proj, (x_bin_low, x_bin_high))

        h2d.SetAxisRange(x_low, x_high)
        x_mean = h2d.GetMean()
#         print 'mean: {}'.format(x_mean)
    #     x_value = h2d.GetXaxis().GetBinCenter(x_bin)
    #     x_err_min = x_value - h2d.GetXaxis().GetBinLowEdge(x_bin)
    #     x_err_plus = h2d.GetXaxis().GetBinUpEdge(x_bin) - x_value

        x.append(x_mean)
        ex_l.append(0)
        ex_h.append(0)

        y.append(fit_result.GetParams()[1])
        ey_l.append(0)
        ey_h.append(0)
    return x, y, ex_l, ex_h, ey_l, ey_h


def get_gauss_avg_sigma(ys):
    (median, lo, hi) = quantiles(ys, False)
    # print median,lo,hi
    avg = median
    rms2 = (hi - lo)
    for niter in range(3):
        truncated = [y for y in ys if abs(y-avg) < rms2]
        if len(truncated) <= 2:
            break
        avg = sum(truncated)/len(truncated)
        rms2 = 2*math.sqrt(sum((t-avg)**2 for t in truncated)/(len(truncated)-1))
    return avg, rms2/2
