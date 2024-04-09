import hist.dask as dah
import hist
import awkward as ak

def TH1F(name, title, nbins, bin_low, bin_high):
    b_axis_name = 'X'
    title_split = title.split(';')
    if len(title_split) > 1:
        b_axis_name = title_split[1]
    b_name = title_split[0]
    b_label = name
        
    return hist.dask.Hist(
         hist.axis.Regular(bins=nbins, start=bin_low, stop=bin_high, name=b_axis_name),
         label=b_label,
         name=b_name,
         storage=hist.storage.Weight()
         )

def TH2F(name, title, x_nbins, x_bin_low, x_bin_high, y_nbins, y_bin_low, y_bin_high):
    b_x_axis_name = 'X'
    b_y_axis_name = 'Y'
    title_split = title.split(';')
    if len(title_split) > 1:
        b_x_axis_name = title_split[1]
    if len(title_split) > 2:
        b_y_axis_name = title_split[2]
    b_name = title_split[0]
    b_label = name
    
    return hist.dask.Hist(
        hist.axis.Regular(bins=x_nbins, start=x_bin_low, stop=x_bin_high, name=b_x_axis_name), 
        hist.axis.Regular(bins=y_nbins, start=y_bin_low, stop=y_bin_high, name=b_y_axis_name), 
        label=b_label, 
        name=b_name,
        storage=hist.storage.Weight()
        )

def fill_1Dhist(hist, array, weights=None):
    flar = ak.drop_none(ak.flatten(array))
    
    if weights is None:
        hist.fill(flar, threads=None)
        # ROOT.fill_1Dhist(hist=hist, array=flar)
    else:
        hist.fill(flar, weights)
        # ROOT.fill_1Dhist(hist=hist, array=flar, weights=weights)
        
def fill_2Dhist(hist, arrayX, arrayY, weights=None):
    flar_x = ak.drop_none(ak.flatten(arrayX))
    flar_y = ak.drop_none(ak.flatten(arrayY))

    if weights is None:
        # ROOT.fill_2Dhist(hist=hist, arrayX=flar_x, arrayY=flar_y)
        hist.fill(flar_x, flar_y, threads=None)
    else:
        # ROOT.fill_2Dhist(hist=hist, arrayX=flar_x, arrayY=flar_y, weights=weights)
        hist.fill(flar_x, flar_y, weights)