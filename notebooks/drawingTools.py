import ROOT
import math
import uuid
import pandas as pd

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


def DrawPrelimLabel(canvas):
    canvas.cd()
    tex = ROOT.TLatex()
    global stuff
    stuff.append(tex)
    tex.SetTextSize(0.03)
    tex.DrawLatexNDC(0.13,0.91,"#scale[1.5]{CMS} #scale[1.]{Phase-2 Simulation}")
    tex.Draw("same");
    return


# void DrawLumiLabel(TCanvas* c, TString Lumi = "35.9")
# {
#   c->cd();

#   TLatex tex;
#   tex.SetTextSize(0.035);
#   TString toDisplay = Lumi + " fb^{-1} (13 TeV)";
#   tex.DrawLatexNDC(0.66,0.91,toDisplay.Data());
#   tex.Draw("same");

#   return;
# }


def SaveCanvas(canvas, name):
    canvas.cd()
    canvas.SaveAs(name+'.pdf')


# void SaveCanvas(TCanvas* c, TString PlotName = "myPlotName")
# {
#   c->cd();
#   c->SaveAs(PlotName + ".pdf");
#   c->SaveAs(PlotName + ".root");

#   return;
# }



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
    legend.SetTextSize(0.03)
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
         min_x=None,
         max_x=None,
         text=None,
         y_axis_label=None,
         x_axis_label=None,
         v_lines=None,
         h_lines=None,
         do_stats=False,
         do_profile=False,
         do_ratio=False,
         do_write=False,
         write_name=None,
         legend_position=None):
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
    if do_write:
        c_width = 600

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
        if not do_write:
            canvas.SetRightMargin(0.30)


    canvas.cd()
    leg = getLegend()
    if do_write and legend_position is None:
        leg = getLegend(0.6, 0.7, 0.86, 0.8)
    elif legend_position is not None:
        leg = getLegend(legend_position[0], legend_position[1], legend_position[2], legend_position[3])

    drawn_histos = []
    # drawn_histos = histograms
    for hidx, hist in enumerate(histograms):
        opt = options
        if 'TGraph' not in histo_class:
            hist.SetStats(do_stats)
            hist.SetTitle("")
        else:
            opt = 'P'+options
            hist.SetMarkerSize(3)
            hist.SetMarkerStyle(7)
        if overlay:
            hist.SetLineColor(colors[hidx])
            if 'TGraph' in histo_class:
                hist.SetMarkerColor(colors[hidx])
#                 hist.SetMarkerSize(3)
#                 hist.SetMarkerStyle(7)

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
        if min_x and max_x:
            hist.GetXaxis().SetRangeUser(min_x, max_x)


    canvas.Draw()

    # we draw additional stuff if needed
    if overlay:
        canvas.cd()
        if len(labels) > 1:
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
    if(do_write):
        DrawPrelimLabel(canvas)

    canvas.Draw()
    if(do_write):
        SaveCanvas(canvas, write_name)

    return canvas


files = {}
file_keys = {}


class RootFile:
    def __init__(self, file_name):
        global file
        self.file_name = file_name
        if self.file_name not in files.keys():
            print 'get file: {}'.format(self.file_name)
            files[self.file_name] = ROOT.TFile(self.file_name)
        self._file = files[self.file_name]
        self._file_keys = None

    def cd(self):
        self._file.cd()

    def GetListOfKeys(self):
        global file_keys
        if self.file_name not in file_keys.keys():
            print 'get list'
            file_keys[self.file_name] = self._file.GetListOfKeys()
        self._file_keys = file_keys[self.file_name]
        return self._file_keys


class Sample():
    def __init__(self, name, label, version=None, type=None):
        self.name = name
        self.label = label
        if version:
            version = '_'+version
        else:
            version = ''
        self.histo_filename = '../plots1/histos_{}{}.root'.format(self.name, version)
        self.histo_file = None
        self.type = type

    def __repr__(self):
        return '<{} {}, {}>'.format(self.__class__.__name__, self.histo_filename, self.type)

    def open_file(self):
        self.histo_file = ROOT.TFile(self.histo_filename, 'r')

# sample_names = ['ele_flat2to100_PU0',
#                 'ele_flat2to100_PU200',
#                 'photonPt35_PU0',
#                 'photonPt35_PU200']


class HProxy:
    def __init__(self, classtype, tp, tp_sel, gen_sel, root_file):
        self.classtype = classtype
        self.tp = tp
        self.tp_sel = tp_sel
        self.gen_sel = gen_sel
        self.root_file = root_file
        self.instance = None

    def get(self, debug=False):
        if self.instance is None:
            name = '{}_{}_{}'.format(self.tp, self.tp_sel, self.gen_sel)
            if self.gen_sel is None:
                name = '{}_{}'.format(self.tp, self.tp_sel)
            if debug:
                print '-- HProxy:Get: {}'.format(name)
            self.instance = self.classtype(name, self.root_file, debug)
        return self.instance


class HPlot:
    def __init__(self, samples, tp_sets, tp_selections, gen_selections):
        self.tp_sets = tp_sets
        # self.tp_selections = tp_selections
        # self.gen_selections = gen_selections
        self.pus = []
        self.labels_dict = {}

        for sample in samples:
            self.pus.append(sample.label)
            self.labels_dict[sample.type] = sample.type

        self.data = pd.DataFrame(columns=['sample', 'pu', 'tp', 'tp_sel', 'gen_sel', 'classtype', 'histo'])

        self.labels_dict.update(tp_sets)
        self.labels_dict.update(tp_selections)
        self.labels_dict.update(gen_selections)
        self.labels_dict.update({'PU0': 'PU0', 'PU200': 'PU200'})

    def cache_histo(self,
                    classtype,
                    samples,
                    pus,
                    tps,
                    tp_sels,
                    gen_sels):
        if gen_sels is None:
            gen_sels = [None]

        for sample in samples:
            print sample
            for tp in tps:
                for tp_sel in tp_sels:
                    for gen_sel in gen_sels:
                        print sample, tp, tp_sel, gen_sel
                        self.data = self.data.append({'sample': sample.type,
                                                      'pu': sample.label,
                                                      'tp': tp,
                                                      'tp_sel': tp_sel,
                                                      'gen_sel': gen_sel,
                                                      'classtype': classtype,
                                                      'histo': HProxy(classtype, tp, tp_sel, gen_sel, sample.histo_file)},
                                                     ignore_index=True)

    def get_histo(self,
                  classtype,
                  sample=None,
                  pu=None,
                  tp=None,
                  tp_sel=None,
                  gen_sel=None,
                  debug=False):
        histo = None
        labels = []
        text = ''

        query = '(pu == @pu) & (tp == @tp) & (tp_sel == @tp_sel) & (classtype == @classtype)'
        if gen_sel is not None:
            query += ' & (gen_sel == @gen_sel)'
        else:
            query += ' & (gen_sel.isnull())'
        if sample is not None:
            query += '& (sample == @sample)'

        histo_df = self.data.query(query)

        if histo_df.empty:
            print 'No match found for: pu: {}, tp: {}, tp_sel: {}, gen_sel: {}, classtype: {}'.format(pu, tp, tp_sel, gen_sel, classtype)
            return None, None, None
        if debug:
            print histo_df

        field_counts = histo_df.apply(lambda x: len(x.unique()))
        label_fields = []
        text_fields = []
        # print field_counts
        for field in field_counts.iteritems():
            if(field[1] > 1 and field[0] != 'histo'):
                label_fields.append(field[0])
            if(field[1] == 1 and field[0] != 'histo' and field[0] != 'classtype' and field[0] != 'sample'):
                if(gen_sel is None and field[0] == 'gen_sel'):
                    continue
                text_fields.append(field[0])
#         print 'label fields: {}'.format(label_fields)
#         print 'text fields: {}'.format(text_fields)

        for item in histo_df[label_fields].iterrows():
            labels.append(', '.join([self.labels_dict[tx] for tx in item[1].values if self.labels_dict[tx] != '']))

        # print labels
        text = ', '.join([self.labels_dict[fl] for fl in histo_df[text_fields].iloc[0].values if self.labels_dict[fl] != ''])
        histo = [his.get(debug) for his in histo_df['histo'].values]
        return histo, labels, text
