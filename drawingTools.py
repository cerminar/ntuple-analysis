import ROOT
import math
import uuid

# some useful globals, mainly to deal with ROOT idiosyncrasies
c_idx = 0
p_idx = 0
colors = range(1, 12)
stuff = []
f_idx = 0

ROOT.gStyle.SetOptTitle(False)
ROOT.gStyle.SetPadBottomMargin(0.13)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.13)

# ROOT.gStyle.SetCanvasBorderMode(0)
# ROOT.gStyle.SetCanvasColor(0)


def getText(text, ndc_x, ndc_y):
    global stuff
    rtext = ROOT.TLatex(ndc_x, ndc_y, text)
    stuff.append(rtext)
    rtext.SetNDC(True)
    # rtext.SetTextFont(40)
    rtext.SetTextSize(0.03)
    return rtext

def getLegend(x1=0.7, y1=0.71, x2=0.95, y2=0.85):
    global stuff
    legend = ROOT.TLegend(x1, y1, x2, y2)
    stuff.append(legend)
    legend.SetFillColor(0)
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.05)
    return legend

def newCanvas(name=None, title=None, height=600, width=800, xdiv=0, ydiv=0, form=4):
    global c_idx
    if name is None:
        name = 'c_{}'.format(uuid.uuid4().hex[:6])
        c_idx += 1
    if title is None:
        title = name
    # print name, title, width, height
    canvas = ROOT.TCanvas(name, title, width, height)
    if(xdiv*ydiv != 0):
        canvas.Divide(xdiv, ydiv)
    global stuff
    stuff.append(canvas)
    return canvas

def drawAndProfileX(plot2d, miny=None, maxy=None, do_profile=True, options='', text=None):
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

    if text:
        rtext = getText(text, 0.15, 0.85)
        rtext.Draw('same')

    c.Draw()


def draw(histograms,
         labels,
         options='',
         norm=False,
         logy=False,
         min_y=None,
         max_y=None,
         text=None,
         y_axis_label=None,
         x_axis_label=None,
         v_lines=None,
         h_lines=None,
         do_stats=False,
         do_profile=False,
         do_ratio=False):
    global colors
    global stuff
    global p_idx


    # 0 - determine kind and # of histos
    n_histos = len(histograms)
    if n_histos == 0:
        print '[draw]**ERROR: empy histo list'
        return

    histo_class = histograms[0].ClassName()

    # 1 - check argument consisntency
    if 'TH2' not in histo_class and do_profile:
        print '[draw]**ERROR: do_profile set for histo of class: {}'.format(histo_class)
        return

    # TH1 histograms are overlayed
    # TH2 histograms are drawn side by side in the same canvas
    overlay = True
    if n_histos > 1:
        if 'TH2' in histo_class or 'TH3' in histo_class:
            overlay = False

    # 2 - determine canvas properties
    xdiv = 0
    ydiv = 0
    c_width = 800
    c_height = 600
    # do_ratio = False

    if do_ratio:
        c_height = 800

    if not overlay:
        xdiv = 2
        c_width = 1000.
        ydiv = math.ceil(float(n_histos)/2)
        c_height = 500*ydiv
    canvas = newCanvas(name=None,
                       title=None,
                       height=int(c_height),
                       width=int(c_width),
                       xdiv=int(xdiv),
                       ydiv=int(ydiv))
    if overlay:
        canvas.SetRightMargin(0.30)

    canvas.cd()
    leg = getLegend()

    drawn_histos = []
    # drawn_histos = histograms
    for hidx, hist in enumerate(histograms):
        opt = options
        if 'TGraph' not in histo_class:
            hist.SetStats(do_stats)
        else:
            opt = 'P'+options
        if overlay:
            hist.SetLineColor(colors[hidx])
            if 'TGraph' in histo_class:
                hist.SetMarkerColor(colors[hidx])
            if hidx:
                opt = 'same,'+opt
            else:
                if 'TGraph' in histo_class:
                    opt = opt+'A'
        else:
            canvas.cd(hidx+1)
        # print hidx, opt
        d_hist = hist
        if norm:
            d_hist = hist.DrawNormalized(opt, 1.)
        else:
            hist.Draw(opt)

        if do_profile:
            profname = d_hist.GetName()+'_prof_'+str(p_idx)
            p_idx += 1
            firstbin = 1
            lastbin = -1
            prof = d_hist.ProfileX(profname, firstbin, lastbin, 's')
            prof.SetMarkerColor(2)
            prof.SetLineColor(2)
            prof.Draw('same')

        if overlay:
            if 'TGraph' not in histo_class:
                leg.AddEntry(hist, labels[hidx], 'l')
            else:
                leg.AddEntry(hist, labels[hidx], 'P')
        else:
            if text:
                newtext = '{}: {}'.format(labels[hidx], text)
                rtext = getText(newtext, 0.15, 0.85)
                rtext.Draw('same')
        drawn_histos.append(d_hist)

    if do_ratio:
        h_ratio = ROOT.TRatioPlot(drawn_histos[0], drawn_histos[1])
        stuff.append(h_ratio)
        h_ratio.Draw()

        pad = canvas.cd(1)
        h_ratio.GetUpperPad().SetBottomMargin(0)
        h_ratio.GetUpperPad().SetRightMargin(0.3)
        if logy:
            h_ratio.GetUpperPad().SetLogy()
        pad.Update()
        pad = canvas.cd(2)
        h_ratio.GetLowerPad().SetTopMargin(0)
        h_ratio.GetLowerPad().SetRightMargin(0.3)
        pad.Update()
        # hratio.GetLowerRefGraph().GetYaxis().SetRangeUser(-6, 6)

    canvas.Update()

    # we now set the axis properties
    max_value = max_y
    min_value = min_y

    if min_y is None:
        min_value = min([hist.GetBinContent(hist.GetMinimumBin()) for hist in drawn_histos])
    if max_y is None:
        max_value = max([hist.GetBinContent(hist.GetMaximumBin()) for hist in drawn_histos])*1.2
    # print min_value, max_value

    for hist in drawn_histos:
        hist.GetYaxis().SetRangeUser(min_value, max_value)
        if y_axis_label:
            hist.GetYaxis().SetTitle(y_axis_label)
        if x_axis_label:
            hist.GetXaxis().SetTitle(x_axis_label)

    canvas.Draw()

    # we draw additional stuff if needed
    if overlay:
        canvas.cd()
        leg.Draw()
        if text:
            rtext = getText(text, 0.15, 0.85)
            rtext.Draw("same")

    pad_range = range(0, 1)
    if not overlay:
        pad_range = range(1, len(histograms)+1)

    for pad_id in pad_range:
        canvas.cd(pad_id)
        if v_lines:
            for v_line_x in v_lines:
                aline = ROOT.TLine(v_line_x, ROOT.gPad.GetUymin(), v_line_x,  ROOT.gPad.GetUymax())
                aline.SetLineStyle(2)
                aline.Draw("same")
                stuff.append(aline)
        if h_lines:
            for h_line_y in h_lines:
                aline = ROOT.TLine(ROOT.gPad.GetUxmin(), h_line_y, ROOT.gPad.GetUxmax(),  h_line_y)
                aline.SetLineStyle(2)
                aline.Draw("same")
                stuff.append(aline)
        if logy:
            if not do_ratio:
                ROOT.gPad.SetLogy()

        ROOT.gPad.Update()
    canvas.Update()
    canvas.Draw()

    return canvas
