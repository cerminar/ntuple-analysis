import ROOT


def convertRateToGraph(histo_rate, params=None):
    """
    Converts a rate TH1F to a graph.
    Optionally applyes a linear scale mapping to the X axis.
    """
    
    xaxis = histo_rate.GetXaxis()
    fNpoints = xaxis.GetNbins()
    fX = [0]*fNpoints
    fEX = [0]*fNpoints
    fY = [0]*fNpoints
    fEY = [0]*fNpoints

    graph = ROOT.TGraphAsymmErrors()
    graph.Set(fNpoints)
    for i in range(0, fNpoints):
        if params is None:
            fX[i] = xaxis.GetBinLowEdge(i + 1)
        else:
            fX[i] = params[0] + params[1]*xaxis.GetBinLowEdge(i + 1)
        fY[i] = histo_rate.GetBinContent(i + 1)
        fEX[i] = histo_rate.GetBinWidth(i + 1) * ROOT.gStyle.GetErrorX()
        fEY[i] = histo_rate.GetBinError(i + 1)
    return fX, fY, fEX, fEX, fEY, fEY


def cutAtRate(histo, rate):
    """
    Determines the X value (pt cut) corresponding to Y value (rate)
    """
    cut = None
    for ix in range(1, histo.GetNbinsX()+1):
        if histo.GetBinContent(ix) <= rate:
            cut = histo.GetXaxis().GetBinLowEdge(ix)
            break
    return cut
