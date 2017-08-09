from pylab import *
import pandas as pd
import plotly.plotly as py
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from root_pandas import read_root

def LoadEvent(evtid,df):
    
    rech_2dcl = df.rechit_cluster2d[evtid]
    rech_mult = df.cluster2d_multicluster[evtid][rech_2dcl]
    rech_mult_init = -2
    rech_e = df.rechit_energy[evtid][(rech_mult>rech_mult_init)]
    rech_x = df.rechit_x[evtid][(rech_mult>rech_mult_init)]
    rech_y = df.rechit_y[evtid][(rech_mult>rech_mult_init)]
    rech_z = df.rechit_z[evtid][(rech_mult>rech_mult_init)]
    rech_multid = rech_mult[rech_mult>rech_mult_init]
    rech = pd.DataFrame({"e":rech_e,"x":rech_x,"y":rech_y,"z":rech_z,"multid":rech_multid})

    mult_e = df.multiclus_energy[evtid]
    mult_theta = 2*np.arctan(np.exp(- df.multiclus_eta[evtid]))
    mult_x = df.multiclus_z[evtid] * np.tan(mult_theta) * np.cos(df.multiclus_phi[evtid])
    mult_y = df.multiclus_z[evtid] * np.tan(mult_theta) * np.sin(df.multiclus_phi[evtid])
    mult_z = df.multiclus_z[evtid]
    
    mult_nlayer = []
    for i in df.multiclus_cluster2d[evtid]:
        mult_nlayer.append(i.size)
    mult_nlayer =  array(mult_nlayer)
    mult = pd.DataFrame({"e":mult_e,"x":mult_x,"y":mult_y,"z":mult_z,"nlayer":mult_nlayer})
    
    isgen = df.genpart_gen[evtid]>-1
    gen_eta = df.genpart_eta[evtid][isgen]
    gen_phi = df.genpart_phi[evtid][isgen]
    gen_theta = 2*np.arctan(np.exp(- gen_eta))
    gen_z = 300 * np.sign(gen_eta)
    gen_x = gen_z * np.tan(gen_theta) * np.cos(gen_phi)
    gen_y = gen_z * np.tan(gen_theta) * np.sin(gen_phi)
    gen_e = df.genpart_energy[evtid][isgen]
    gen = pd.DataFrame({"e":gen_e,"x":gen_x,"y":gen_y,"z":gen_z})
    
    simclus_eta = df.simcluster_eta[evtid]
    simclus_phi = df.simcluster_phi[evtid]
    simclus_theta =  2*np.arctan(np.exp(- simclus_eta))
    simclus_z = 300 * np.sign(simclus_eta)
    simclus_x = simclus_z * np.tan(simclus_theta) * np.cos(simclus_phi)
    simclus_y = simclus_z * np.tan(simclus_theta) * np.sin(simclus_phi)
    simclus_e = df.simcluster_energy[evtid]
    simclus = pd.DataFrame({"e":simclus_e,"x":simclus_x,"y":simclus_y,"z":simclus_z})
    
    pfclus_eta = df.pfcluster_eta[evtid]
    pfclus_phi = df.pfcluster_phi[evtid]
    pfclus_theta =  2*np.arctan(np.exp(- pfclus_eta))
    pfclus_z = 300 * np.sign(pfclus_eta)
    pfclus_x = pfclus_z * np.tan(pfclus_theta) * np.cos(pfclus_phi)
    pfclus_y = pfclus_z * np.tan(pfclus_theta) * np.sin(pfclus_phi)
    pfclus_e = df.pfcluster_energy[evtid]
    pfclus = pd.DataFrame({"e":pfclus_e,"x":pfclus_x,"y":pfclus_y,"z":pfclus_z})
    
    return gen,rech,mult,simclus,pfclus

def DisplayEvent(gen,rech,mult,simclus=None):
    
    partgun = go.Scatter3d(x=0.0,y=0.0,z=0.0,mode='markers',
                           marker=dict(size=6,color='gray',symbol='sphere'))
    
    genx = reshape(reshape(r_[gen.x,zeros(gen.x.size)],(2,gen.x.size)).T,(2*gen.x.size))
    geny = reshape(reshape(r_[gen.y,zeros(gen.y.size)],(2,gen.y.size)).T,(2*gen.y.size))
    genz = reshape(reshape(r_[gen.z,zeros(gen.z.size)],(2,gen.z.size)).T,(2*gen.z.size))
    genpart = go.Scatter3d(x=genx,y=genz,z=geny,mode='lines',
                           marker=dict(size=0,color="lightgrey",opacity=0.1)) 
        
    rechits = go.Scatter3d(x=rech.x,y=rech.z,z=rech.y,mode='markers',
                           marker=dict(size=1.2,cmax=0.1,cmin=0.01,
                                        color=rech.e, 
                                        colorscale='Reds',   # choose a colorscale
                                        showscale = True,colorbar=dict(x=1.0,len=0.6,title="RechitE/GeV"),
                                        opacity=0.4,symbol='square'))
    
    multicluster = go.Scatter3d(x=mult.x,y=mult.z,z=mult.y,mode='markers',
                                marker=dict(size=5*(mult.e**0.5),
                                            cmax=30,cmin=0,
                                            #size=20,
                                            color=mult.e, 
                                            colorscale='Viridis',   # choose a colorscale
                                            showscale = True, colorbar=dict(x=1.1,len=0.6,title="MclusterE/GeV"),
                                            opacity=0.3,symbol='square'))
    if simclus is None:
        data = [partgun,genpart,rechits,multicluster]
    else:
        simcluster = go.Scatter3d(x=simclus.x,y=simclus.z,z=simclus.y,mode='markers',
                                  marker=dict(size=5*(simclus.e**0.2),color='gray',symbol='sphere'))
        data = [partgun,genpart,rechits,multicluster,simcluster]

        
    layout = go.Layout(
        scene=dict(xaxis=dict(title='x (cm)'), yaxis=dict(title='z (cm)'), zaxis=dict(title='y (cm)')),
        margin=dict(l=0,r=0,b=0,t=0)
    )

    fig = go.Figure(data=data, layout=layout)
    return fig

