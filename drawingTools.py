import ROOT
import math
import uuid

# some useful globals, mainly to deal with ROOT idiosyncrasies
c_idx = 0
p_idx = 0
colors = range(1, 6)
stuff = []


# define some utility functions
def newCanvas(name=None, title=None, xdiv=0, ydiv=0, form=4):
    global c_idx
    if name is None:
        name = 'c_{}'.format(uuid.uuid4().hex[:6])
        c_idx += 1
    if title is None:
        title = name
    # print name, title
    canvas = ROOT.TCanvas(name, title, form)
    if(xdiv*ydiv != 0):
        canvas.Divide(xdiv, ydiv)
    return canvas


def draw(plot, options=''):
    c = newCanvas()
    c.cd()
    plot.Draw(options)
    c.Draw()
    return


def getLegend(x1=0.7, y1=0.71, x2=0.95, y2=0.85):
    global stuff
    legend = ROOT.TLegend(x1, y1, x2, y2)
    stuff.append(legend)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    # legend.SetTextSize(30)
    return legend


def drawAndProfileX(plot2d, miny=None, maxy=None, do_profile=True, options=''):
    global p_idx
    if miny and maxy:
        plot2d.GetYaxis().SetRangeUser(miny, maxy)
    c = newCanvas()
    c.SetGrid(1, 1)
    c.cd()
    plot2d.Draw(options)
    ROOT.gPad.SetGrid(1, 1)
    ROOT.gStyle.SetGridColor(15)

    if do_profile:
        profname = plot2d.GetName()+'_prof_'+str(p_idx)
        p_idx += 1
        firstbin = 1
        lastbin = -1
        prof = plot2d.ProfileX(profname, firstbin, lastbin, 's')
        prof.SetMarkerColor(2)
        prof.SetLineColor(2)
        prof.Draw('same')
    c.Draw()


class Sample():
    def __init__(cls, name, pu_tag, version=None):
        cls.name = name
        cls.pu_tag = pu_tag
        if version:
            version = '_'+version
        else:
            version = ''
        cls.histo_filename = '../plots/histos_{}_{}{}.root'.format(cls.name, cls.pu_tag, version)
        cls.histo_file = ROOT.TFile(cls.histo_filename)


def drawSame(histograms, labels, options='', norm=False, logy=False):
    global colors
    c = newCanvas()
    c.cd()
    leg = getLegend()
    if norm:
        for hist in histograms:
            hist.Scale(1./hist.Integral())

    max_value = max([hist.GetMaximum() for hist in histograms])
    min_value = min([hist.GetMinimum() for hist in histograms])
    for hidx in range(0, len(histograms)):
        histograms[hidx].SetLineColor(colors[hidx])
        histograms[hidx].Draw('same'+','+options)
        leg.AddEntry(histograms[hidx], labels[hidx], 'l')

    histograms[0].GetYaxis().SetRangeUser(min_value, max_value*1.1)
    leg.Draw()
    c.Draw()
    if logy:
        c.SetLogy()


def drawProfileX(histograms, labels, options=''):
    profiles = [hist.ProfileX() for hist in histograms]
    drawSame(profiles, labels, options)


def drawSeveral(histograms, labels, options='', do_profile=False):
    ydiv = int(math.ceil(float(len(histograms))/2))
    for hidx in range(0, len(histograms)):
        if do_profile:
            drawAndProfileX(histograms[hidx], options=options, do_profile=do_profile)
        else:
            draw(histograms[hidx], options=options)


def drawProfileRatio(prof1, prof2, ymin=None, ymax=None):
    hist1 = prof1.ProjectionX(uuid.uuid4().hex[:6])
    hist2 = prof2.ProjectionX(uuid.uuid4().hex[:6])
    hist1.Divide(hist2)
    draw(hist1)
    if ymin is not None and ymax is not None:
        hist1.GetYaxis().SetRangeUser(ymin, ymax)
