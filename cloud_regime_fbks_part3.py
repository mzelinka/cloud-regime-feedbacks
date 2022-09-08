########################################################################
# Third part of the analysis used in Zelinka et al (2022) paper
# "Detailing cloud property feedbacks with a regime-based decomposition"
# Climate Dynamics
# https://doi.org/10.1007/s00382-022-06488-7
########################################################################

# This code reads in data saved in part2 and generates multi-model mean/std figures

####################################################################################
# PART 2:
####################################################################################
import cdutil
import glob
import matplotlib.pyplot as pl
import cdms2 as cdms
import numpy as np
import MV2 as MV
import string
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import MDZ_utils as MU
from mpl_toolkits.axes_grid1 import make_axes_locatable

pl.rcParams['lines.linewidth'] = 2
pl.rcParams['font.size']=12
pl.rcParams['pcolor.shading']='auto'
letters = string.ascii_lowercase

########################################################################
# START USER INPUT
########################################################################
# Decision 1 - Observational Centroids:
#obsname = 'MODIS'
#obsname = 'ISCCP10' # The 10 Weather States on the ISCCP website
obsname = 'ISCCP8' # The 8 weather states in Tselioudis et al (2021)

# Decision 2 - Which scalars:
#scalar_flag = 'compute' # Compute the 3 scalars from the joint histograms (do not use those that are provided)
scalar_flag = 'grab' # Grab the 3 scalars that are provided in the papers and from COSP 'lite' diags
#scalar_flag = 'all42' # Grab the 3 scalars that are provided in the papers and from COSP 'lite' diags

# Decision 3 - experiment type:
#exptype = 'amip'
exptype = 'p4K'

# Decision 3 -- forward or average anomalies
anom_flag = 'fwd'
#anom_flag = 'avg'
########################################################################

if anom_flag == 'fwd':
    EXPS = ['p4K']
else:
    EXPS = ['amip','p4K']

#------------------------------------------------------#
# FIGURE HELPER FUNCTIONS    
#------------------------------------------------------#
def avg_data(D5,D6,mip_era):  
    # D5 and D6 should be size [lat,lon,regime,model]
    avg6 = np.ma.average(np.ma.sum(D6,-2),-1) # sum across regimes, average across models
    std6 =     np.ma.std(np.ma.sum(D6,-2),-1) # sum across regimes, std across models
    avg5 = np.ma.average(np.ma.sum(D5,-2),-1) # sum across regimes, average across models
    std5 =     np.ma.std(np.ma.sum(D5,-2),-1) # sum across regimes, std across models
    DATA=None
    if mip_era == 'CMIP5p6':
        DATA = np.ma.concatenate((D5,D6),axis=-1)
        avg = np.ma.average(np.ma.sum(DATA,-2),-1) # sum across regimes, average across models
        std =     np.ma.std(np.ma.sum(DATA,-2),-1) # sum across regimes, std across models
    elif mip_era == 'CMIP6m5':
        avg = avg6 - avg5
        std = np.ma.sqrt(std5**2+std6**2)
    elif mip_era == 'CMIP5':
        avg = avg5
        std = std5
    elif mip_era == 'CMIP6':
        avg = avg6
        std = std6
        
    avg=MV.array(avg)
    avg.setAxisList(AXL)
    std=MV.array(std)
    avg.setAxisList(AXL)
    return(avg,std,DATA)                 

#------------------------------------------------------#
def get_dots(DATA,region):
    # DATA is size [lats,lons,models]
    dots = MV.array(DATA) 
    dots.setAxis(0,AXL[0])
    dots.setAxis(1,AXL[1])
    TR = cdutil.region.domain(latitude = (-30,30))
    TRavg = cdutil.averager(TR.select(dots),axis='xy',weights='generate')
    GLavg = cdutil.averager(dots,axis='xy',weights='generate')
    EXTRavg = 2*GLavg - TRavg
    if region=='GL':
        return(GLavg)
    if region=='TR':
        return(TRavg)
    if region=='EXTR':
        return(EXTRavg) 

#------------------------------------------------------#
def plot_Chist(ax,data,showcb=False):
    #bounds = [0.1,0.2,0.4,0.8,1.5,3.,5.,7.,9,11,13,15.]
    bounds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    bounds=np.arange(0,11.5,0.5)
    #cmap = pl.cm.get_cmap('GnBu',len(bounds))
    cmap = pl.cm.get_cmap('PuBu',len(bounds))
    #cmap = pl.cm.get_cmap('twilight',len(bounds))
    colors = list(cmap(np.arange(len(bounds))))
    cmap = mpl.colors.ListedColormap(colors[:-1], "")
    # set over-color to last color of list 
    cmap.set_over(colors[-1])
    cmap.set_under('white', 1.0)  # 1.0 represents not transparent       
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N) # this is the critical line for making sure the colors vary linearly even if the desired color boundaries are at varied intervals          
    im1 = ax.pcolor(data,cmap=cmap,norm=norm)
    ax.set_xticklabels(tau, minor=False)
    ax.set_yticks(np.arange(len(ctp)))
    ax.set_yticklabels(ctp, minor=False)
    #ax.set_ylabel('Cloud Top Pressure (hPa)')
    #ax.set_xlabel('Optical Depth')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb=pl.colorbar(im1, cax=cax, ticks=bounds[::2], extend='max')
    if showcb==False:
        cb.remove()
    else:
        cb.set_label('%')
                
#------------------------------------------------------#
def pcolor_lat_regime(ax,DATA,cbar=True):
    DATA0 = np.average(DATA,1) # average across longitude
    avg = np.average(DATA0,-1) # average across models
    NR = DATA.shape[2]
    im1 = ax.pcolor(lats[:],np.arange(NR),avg[:,-1::-1].T,vmin=-0.4,vmax=0.4,cmap='RdBu_r',shading='auto')
    # assess where at least 80%, or 8 of 10 models agree on sign (so sum must be at least 8-2=6):
    thresh=80
    CNT = np.int32(MV.count(DATA0[0,0,:],-1))
    sDATA=np.abs(MV.sum(np.sign(DATA0),axis=-1))           
    A = np.int32(np.ceil((thresh/100.)*CNT))  # 0.8*10= 8
    B = CNT-A                               # 10-8=2
    C = A-B                                 # 8-2 = 6            
    # If >80% of the models agree on the sign, put a dot
    for r in range(NR):
        y = (NR-r)*np.ones(90)-1#0.5
        y = np.ma.masked_where(sDATA[:,r]<C,y)
        pl.plot(lats[:],y,'.',ms=2,color='k')
    pl.yticks(np.arange(0,NR,1),np.arange(NR,0,-1))
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb=pl.colorbar(im1, cax=cax, extend='both')
        cb.set_label('W/m$^2$/K')     
        
#------------------------------------------------------#
def pcolor_lat_model(ax,DATA,ytcks=False,cbar=True):
    DATA0 = np.average(DATA,1) # average across longitude
    avg = np.sum(DATA0,1) # sum over regimes
    im1 = ax.pcolor(lats[:],np.arange(len(allmodels)),avg[:,-1::-1].T,vmin=-2.5,vmax=2.5,cmap='RdBu_r',shading='auto')
    ax.axhline(y=4.5,color='gray')
    if ytcks:
        pl.yticks(np.arange(0,len(allmodels),1),allmodels[::-1])
    else:
        pl.yticks(np.arange(0,len(allmodels),1),'')
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb=pl.colorbar(im1, cax=cax, extend='both')
        cb.set_label('W/m$^2$/K')          

#------------------------------------------------------#   
def plot_robin(ax,data56,bounds,cmap='RdBu_r',extend='both',cbar=True,stippling=True,cbunits='',digits=2):

    if extend=='both':
        bounds2 = np.append(np.append(-500,bounds),500) # This is only needed for norm if colorbar is extended
    elif extend=='max':
        bounds2 = np.append(bounds,500)     # This is only needed for norm if colorbar is extended
    elif extend=='min':
        bounds2 = np.append(-500,bounds)    # This is only needed for norm if colorbar is extended
    elif extend=='neither':
        bounds2 = bounds    # This is only needed for norm if colorbar is extended
                        
    # data56 is size [lat,lon,regime,models]
    data56.setAxis(0,AXL[0])
    data56.setAxis(1,AXL[1])
    sumDATA = MV.sum(data56,-2) # sum across regimes    
    avg_DATA = MV.average(sumDATA,-1)   
    #avg_DATA[abs(avg_DATA) < 1e-4] = 0.0 # set tiny numbers to zero (cartopy complains otherwise)
    CMAP = pl.cm.get_cmap(cmap,len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds2, CMAP.N) # ensure the colors vary linearly even if the desired color boundaries are at varied intervals
    im1 = ax.contourf(LON,LAT,avg_DATA,bounds,transform=ccrs.PlateCarree(),transform_first=True,cmap=CMAP,norm=norm,extend=extend)
    ax.coastlines()
    ax.set_global()
    if cbar:
        cb = pl.colorbar(im1,orientation='vertical',drawedges=True,ticks=bounds[::2])
        cb.set_label(cbunits)   
    if stippling:
        MU.add_stippling_v2(ax,LON,LAT,sumDATA)#,thresh=80)
    GLavg = cdutil.averager(avg_DATA, axis="xy", weights='weighted')      
    ax.set_title('['+MU.signif_digits(GLavg,digits)+']'+extra_space,fontsize=16,loc='right')
    pl.tight_layout(rect=(0.025,0.025,1,1))
    return bounds2
    
#------------------------------------------------------# 
def plot_Zmeans(ax,AVG,STD,LABEL,**kwargs):#COLOR,LABEL):
    ax.plot(lats, AVG.filled(np.nan), label=LABEL, **kwargs)#, color=COLOR)
    ax.fill_between(lats, AVG-STD, AVG+STD, alpha=0.2, label='_nolegend_', **kwargs)#, color=COLOR) 
    pl.axhline(y=0,color='gray')
    pl.ylim(-1.75,1.75)
    pl.xlim(-90,90)      
    
########################################################################
    

figdir = '/home/zelinka1/figures/regimes/'+obsname+'/'
coastlat, coastlon = MU.loadCoast()

if obsname == 'ISCCP8':
    nregimes=8+1 
    ROWS,COLS = 3,3
elif obsname == 'MODIS':
    nregimes=11+1
    ROWS,COLS = 4,3

sec='ALL'
bands=['LW','SW','NET']
types=['tot','amt','alt','tau','err']
Etypes0=['Total','Amount','Altitude','Optical Depth','Residual']
Etypes = dict(zip(types,Etypes0))  
typeCOL={}
typeCOL['tot']='k'
typeCOL['amt']='C3'
typeCOL['alt']='C8'
typeCOL['tau']='C4'
typeCOL['err']='C9'

if anom_flag=='fwd':
    flavors=['between_prime','between','within','covary']
else:
    flavors=['between_prime','between','within']

catdata={}
for EXP in EXPS:
    catdata[EXP]={}
    allmodels=[]
    for mip,mip_era in enumerate(['CMIP5','CMIP6']):
        if EXP=='p4K':
            if mip_era=='CMIP5':
                experiment = 'amip4K'
            elif mip_era=='CMIP6':
                experiment = 'amip-p4K'
        else:
            experiment = 'amip'
        outdir = '/p/user_pub/climate_work/zelinka1/cmip'+mip_era[-1]+'/'+experiment+'/'
        gg=glob.glob(outdir+sec+'_cld_fbk_maps_'+obsname+'_'+scalar_flag+'*_'+anom_flag+'.nc')
        gg.sort()
        DATA={}
        for g in gg:
            f=cdms.open(g)
            model = g.split('/')[-1].split('.')[0].split('_')[-1]
            ripf = g.split('/')[-1].split('.')[-2].split('_')[0]
            print(model+'.'+ripf+'.'+f.timeperiod)
            if model=='MRI-CGCM3' and ripf=='r5i1p3':
                continue # just take the first variant of this model
            if model=='IPSL-CM6A-LR' and ripf=='r22i1p1f1':
                continue # just take the first variant of this model
            allmodels.append(model)
            DATA[model] = {}
            for band in bands[:-1]:
                for ty in types:
                    for flavor in flavors:
                        name = sec+'_'+band+'cld_'+ty+'_'+flavor
                        DATA[model][name] = f(name) # (lat,regime)

            lats=f(name).getLatitude()[:]
            lons=f(name).getLongitude()[:]
            AXL = f(name)[:,:,0].getAxisList()
            try: # I forgot to save this for amip/avg
                DATA[model]['CRhist'] = np.reshape(f('CRhist').T,(49,nregimes)) # (49,regime)
                DATA[model]['dCRhist'] = np.reshape(f('dCRhist').T,(49,nregimes)) # (49,regime)
            except:
                print('no CRhist or dCRhist data for '+model) 
                pass
            DATA[model]['RFO'] = f('RFO')                  # (lat,lon,regime)
            DATA[model]['dRFO'] = f('dRFO')                # (lat,lon,regime)
            DATA[model]['LW_TRUTH'] = f('LW_TRUTH')        # (lat,lon)
            DATA[model]['SW_TRUTH'] = f('SW_TRUTH')        # (lat,lon)
            ALB0 = np.moveaxis(f('regime_ALB')[0,:],0,-1)  # (lat,lon,regime)
            PCT0 = np.moveaxis(f('regime_PCT')[0,:],0,-1)  # (lat,lon,regime)
            CLT0 = np.moveaxis(f('regime_CLT')[0,:],0,-1)  # (lat,lon,regime)
            ALB1 = np.moveaxis(f('regime_ALB')[1,:],0,-1)  # (lat,lon,regime)
            PCT1 = np.moveaxis(f('regime_PCT')[1,:],0,-1)  # (lat,lon,regime)
            CLT1 = np.moveaxis(f('regime_CLT')[1,:],0,-1)  # (lat,lon,regime)
            DATA[model]['regime_alb'] = ALB0
            DATA[model]['regime_pct'] = PCT0
            DATA[model]['regime_clt'] = CLT0
            if obsname=='MODIS':
                anomT = 4.4 # we forgot to save this
            else:
                anomT = f('anomT')                             # scalar (4.4#)
            DATA[model]['dregime_alb'] = (ALB1 - ALB0)/anomT
            DATA[model]['dregime_pct'] = (PCT1 - PCT0)/anomT
            DATA[model]['dregime_clt'] = (CLT1 - CLT0)/anomT
            f.close()

        catdata[EXP][mip_era]={}
        for band in bands:
            catdata[EXP][mip_era][band]={}
            for fl,flavor in enumerate(flavors):
                catdata[EXP][mip_era][band][flavor]={}
                catdata[EXP][mip_era][band]['sum']={}
                catdata[EXP][mip_era][band]['sum_prime']={}
                if band=='NET':
                    continue
                for t,ty in enumerate(types):
                    name = sec+'_'+band+'cld_'+ty+'_'+flavor
                    for m,mod in enumerate(DATA.keys()):
                        data0 = np.ma.array(DATA[mod][name])
                        if m==0:
                            data = np.expand_dims(data0,3)
                        else:   
                            data=np.ma.concatenate((data,np.expand_dims(data0,-1)),axis=-1)
                    catdata[EXP][mip_era][band][flavor][ty] = data # (lat,lon,regime,model)
            
        if anom_flag == 'avg': # I forgot to save this for amip/avg
            NAMES = ['RFO','dRFO','LW_TRUTH','SW_TRUTH','regime_alb','regime_pct','regime_clt','dregime_alb','dregime_pct','dregime_clt']
        else:
            NAMES = ['CRhist','dCRhist','RFO','dRFO','LW_TRUTH','SW_TRUTH','regime_alb','regime_pct','regime_clt','dregime_alb','dregime_pct','dregime_clt']
        for name in NAMES:
            for m,mod in enumerate(DATA.keys()):
                data0 = np.ma.array(DATA[mod][name])
                if m==0:
                    data = np.expand_dims(data0,-1)
                else:   
                    data=np.ma.concatenate((data,np.expand_dims(data0,-1)),axis=-1)
                catdata[EXP][mip_era][name] = data # (lat,regime,model)
        catdata[EXP][mip_era]['modelnames'] = DATA.keys()

        band='NET' 
        for fl,flavor in enumerate(flavors):
            catdata[EXP][mip_era][band][flavor]={}
            for t,ty in enumerate(types):             
                catdata[EXP][mip_era]['NET'][flavor][ty] = catdata[EXP][mip_era]['LW'][flavor][ty] + catdata[EXP][mip_era]['SW'][flavor][ty]          

    # Also, compute the sum of within, covary, and between:
    if anom_flag=='fwd':
        sum_prime_components = ['within','covary','between_prime']
        sum_components = ['within','covary','between']
    else:
        sum_prime_components = ['within','between_prime']
        sum_components = ['within','between']


    for mip,mip_era in enumerate(['CMIP5','CMIP6']):
        for band in bands:
            for t,ty in enumerate(types):    
                data=0
                for flavor in sum_prime_components:
                    data+=catdata[EXP][mip_era][band][flavor][ty]
                catdata[EXP][mip_era][band]['sum_prime'][ty] = data
                data=0
                for flavor in sum_components:
                    data+=catdata[EXP][mip_era][band][flavor][ty]
                catdata[EXP][mip_era][band]['sum'][ty] = data
            
# Done loading in stuff

if anom_flag == 'avg':
    # Ensure that the models are in the same order for amip and amip-p4K:
    if catdata['amip']['CMIP5']['modelnames'] != catdata['p4K']['CMIP5']['modelnames']:
        raise RuntimeError('amip and amip+4k models are not in same order!')
    if catdata['amip']['CMIP6']['modelnames'] != catdata['p4K']['CMIP6']['modelnames']:
        raise RuntimeError('amip and amip+4k models are not in same order!')


if exptype=='p4K':
    experiment = 'amip-p4K'
else:
    experiment = 'amip'
    
if anom_flag=='fwd':
    flavors=['sum_prime','between_prime','between','within','covary','sum']
    Eflavors0=["Total","Across","Across (orig)","Within","Covariance","Total (orig)"]
else:
    flavors=['sum_prime','between_prime','between','within','sum']
    Eflavors0=["Total","Across","Across (orig)","Within","Total (orig)"]

Eflavors = dict(zip(flavors,Eflavors0))  
flavCOL={}
flavCOL['sum_prime']='k'
flavCOL['sum']='gray'
flavCOL['between_prime']='C0'
flavCOL['between']='C0'
flavCOL['within']='C1'
flavCOL['covary']='C2'

LON, LAT = np.meshgrid(lons,lats)

####################################################################################
# MMM Feedback Maps
# Columns: Across, Within, Total 
# Rows: total, amount, optical depth
####################################################################################
extra_space = '     '
mip_era = 'CMIP5p6'
band='SW'
cmap = pl.cm.RdBu_r
extend = 'both'
bounds = np.arange(-2.5,2.75,0.25)          
fig=pl.figure(figsize=(18,12))
gs = gridspec.GridSpec(3,3)#(ROWS,COLS) 
cnt=0
row=-1
for ty in ['tot', 'amt', 'tau']:
    row+=1
    col=-1
    for flavor in ['sum_prime', 'between_prime', 'within']:
        col+=1
        cnt+=1
        data5 = catdata[exptype]['CMIP5'][band][flavor][ty] 
        data6 = catdata[exptype]['CMIP6'][band][flavor][ty]
        avg,std,data56 = avg_data(data5,data6,mip_era)
        ax = fig.add_subplot(gs[row,col],projection=ccrs.Robinson(central_longitude=180.))
        bounds2 = plot_robin(ax,data56,bounds,
            cmap = cmap,
            extend = extend,
            cbar = False,
            stippling = True,
            cbunits = 'W/m$^2$/K')   
        label1 = Eflavors[flavor]
        label2 = Etypes[ty]
        label = label1+' '+label2
        if label1=='Total':
            label = label2
        if label2=='Total':
            label = label1
        pl.title(extra_space+letters[cnt-1]+') '+label,fontsize=16,loc='left')
        if row==2 and col==1:
            left,bottom,width,height = MU.get_axis_locs(ax)
ax2 = fig.add_axes([left, bottom-0.03, width,0.02]) # create a second axes for the colorbar
norm = mpl.colors.BoundaryNorm(bounds2, cmap.N) # ensure the colors vary linearly even if the desired color boundaries are at varied intervals
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, ticks=bounds[::2], boundaries=bounds2,extend=extend,orientation='horizontal')       
cb.set_label('W/m$^2$/K')   
pl.savefig(figdir+mip_era+'_'+experiment+'_MMM_'+sec+'_decomp4a_maps_'+obsname+'_'+scalar_flag+'_'+anom_flag+'.pdf',bbox_inches='tight')

####################################################################################
# Similar _decomp4a_maps_, but show zonal mean for all regimes
####################################################################################               
# Columns: Across, Within, Total 
# Rows: total, amount, optical depth
extra_space = ''
for mip_era in ['CMIP5','CMIP6','CMIP5p6']:
    band='SW'
    fig=pl.figure(figsize=(18,12))
    gs = gridspec.GridSpec(3,3)#(ROWS,COLS) 
    cnt=0
    row=-1
    for t,ty in enumerate(types): 
        if ty!='tot' and ty!='amt' and ty!='tau': 
            continue   
        row+=1
        col=-1
        for fl,flavor in enumerate(flavors):  # flavors=['between_prime','between','within','covary','sum_prime','sum']
            if flavor!='between_prime' and flavor!='within' and flavor!='sum_prime':
                continue
            col+=1
            cnt+=1
            ax = fig.add_subplot(gs[row,col])
            if mip_era == 'CMIP5p6':
                DATA5 = catdata[exptype]['CMIP5']['SW'][flavor][ty] 
                DATA6 = catdata[exptype]['CMIP6']['SW'][flavor][ty] 
                DATA0 = np.ma.concatenate((DATA5,DATA6),axis=-1)
            elif mip_era == 'CMIP5':
                DATA0 = catdata[exptype]['CMIP5']['SW'][flavor][ty] 
            elif mip_era == 'CMIP6':
                DATA0 = catdata[exptype]['CMIP6']['SW'][flavor][ty] 
            cbar=False
            if col==2:
                cbar=True

            # ensure  that the clear-sky regime is zero, not masked:
            DATA0[:,:,-1,:] = 0
            
            pcolor_lat_regime(ax,DATA0,cbar) 
            label1 = Eflavors[flavor]
            label2 = Etypes[ty]
            label = label1+' '+label2
            if label1=='Total':
                label = label2
            if label2=='Total':
                label = label1
            ax.set_title(extra_space+letters[cnt-1]+') '+label,fontsize=14,loc='left')
            ax.set_xticks(np.arange(-90,90+30,30))
            if row==2:
                ax.set_xlabel('Latitude')
            if col==0:
                ax.set_ylabel('Cloud Regime')
    pl.savefig(figdir+mip_era+'_'+experiment+'_MMM_'+sec+'_combo_'+obsname+'_'+scalar_flag+'.pdf',bbox_inches='tight')   



####################################################################################
# Same as above, but plot zonal average for each model, summed over regimes
####################################################################################               
# Columns: Across, Within, Total 
# Rows: total, amount, optical depth
extra_space = ''
mip_era = 'CMIP5p6'
band='SW'
fig=pl.figure(figsize=(18,12))
gs = gridspec.GridSpec(3,3)#ROWS,COLS) 
cnt=0
row=-1
for t,ty in enumerate(types): 
    if ty!='tot' and ty!='amt' and ty!='tau': 
        continue   
    row+=1
    col=-1
    for fl,flavor in enumerate(flavors):  # flavors=['between_prime','between','within','covary','sum_prime','sum']
        if flavor!='between_prime' and flavor!='within' and flavor!='sum_prime':
            continue
        col+=1
        cnt+=1
        ax = fig.add_subplot(gs[row,col])
        DATA5 = catdata[exptype]['CMIP5']['SW'][flavor][ty] 
        DATA6 = catdata[exptype]['CMIP6']['SW'][flavor][ty] 
        DATA0 = np.ma.concatenate((DATA5,DATA6),axis=-1)
        cbar=False
        if col==2:
            cbar=True
        ytcks=False
        if col==0:
            ytcks=True
        pcolor_lat_model(ax,DATA0,ytcks,cbar)
        label1 = Eflavors[flavor]
        label2 = Etypes[ty]
        label = label1+' '+label2
        if label1=='Total':
            label = label2
        if label2=='Total':
            label = label1
        ax.set_title(extra_space+letters[cnt-1]+') '+label,fontsize=14,loc='left')
        ax.set_xticks(np.arange(-90,90+30,30))
        if row==2:
            ax.set_xlabel('Latitude')
pl.savefig(figdir+mip_era+'_'+experiment+'_allmods_'+sec+'_combo_'+obsname+'_'+scalar_flag+'.pdf',bbox_inches='tight')   


####################################################################################
# Same as above, but plot CMIP5 and CMIP6 zonal mean feedbacks:
####################################################################################               
# Columns: Across, Within, Total 
# Rows: total, amount, optical depth
extra_space = ''
band='SW'
fig=pl.figure(figsize=(18,12))
gs = gridspec.GridSpec(3,3)#(ROWS,COLS) 
cnt=0
row=-1
for t,ty in enumerate(types): 
    if ty!='tot' and ty!='amt' and ty!='tau': 
        continue   
    row+=1
    col=-1
    for fl,flavor in enumerate(flavors):  # flavors=['between_prime','between','within','covary','sum_prime','sum']
        if flavor!='between_prime' and flavor!='within' and flavor!='sum_prime':
            continue
        col+=1
        cnt+=1
        ax = fig.add_subplot(gs[row,col])
        data5 = catdata[exptype]['CMIP5']['SW'][flavor][ty] 
        data6 = catdata[exptype]['CMIP6']['SW'][flavor][ty] 

        avg5 = np.ma.average(np.ma.average(np.ma.sum(data5,-2),1),-1) # sum across regimes, Zavg, average across models
        std5 =     np.ma.std(np.ma.average(np.ma.sum(data5,-2),1),-1) # sum across regimes, Zavg, std across models       
        plot_Zmeans(ax,avg5,std5,'CMIP5',color='C0')
        avg6 = np.ma.average(np.ma.average(np.ma.sum(data6,-2),1),-1) # sum across regimes, Zavg, average across models
        std6 =     np.ma.std(np.ma.average(np.ma.sum(data6,-2),1),-1) # sum across regimes, Zavg, std across models       
        plot_Zmeans(ax,avg6,std6,'CMIP6',color='C1')      
        avg = avg6 - avg5
        std = np.ma.sqrt(std5**2+std6**2)   
        plot_Zmeans(ax,avg,std,'CMIP6-CMIP5',color='k')
        
        GLavg5 = get_dots(np.ma.sum(data5,-2),'GL').mean()
        GLavg6 = get_dots(np.ma.sum(data6,-2),'GL').mean()
        GLavg6m5 = GLavg6 - GLavg5
        pl.text(1,0.95,str(np.round(GLavg5,2)),color='C0', ha='right', va='center', transform=ax.transAxes)
        pl.text(1,0.89,str(np.round(GLavg6,2)),color='C1', ha='right', va='center', transform=ax.transAxes)
        pl.text(1,0.83,str(np.round(GLavg6m5,2)),color='k', ha='right', va='center', transform=ax.transAxes)
        
        ax.axhline(y=0,color='gray')
        label1 = Eflavors[flavor]
        label2 = Etypes[ty]
        label = label1+' '+label2
        if label1=='Total':
            label = label2
        if label2=='Total':
            label = label1
        ax.set_title(extra_space+letters[cnt-1]+') '+label,fontsize=14,loc='left')
        ax.set_xticks(np.arange(-90,90+30,30))
        if row==2:
            ax.set_xlabel('Latitude')
        if col==0:
            ax.set_ylabel('W/m$^2$/K') 
        if col==0 and row==0:
            ax.legend(loc=8,handletextpad=0.4, fontsize=10)#,frameon=0)
pl.savefig(figdir+mip_era+'_'+experiment+'_CMIP5v6_'+sec+'_combo_'+obsname+'_'+scalar_flag+'.pdf',bbox_inches='tight')   



####################################################################################
# MMM RFO MAPS
####################################################################################
extra_space = '     '
mip_era = 'CMIP5p6'
bounds = np.array([0,2.5,5,7.5,10,15,20,25,30,40,50,60,70,80,90,100])
cmap = pl.cm.PuBu
extend = 'neither'
fig=pl.figure(figsize=(18,12))
gs = gridspec.GridSpec(ROWS,COLS) 
data5 = catdata[exptype]['CMIP5']['RFO']
data6 = catdata[exptype]['CMIP6']['RFO']
avg,std,data56 = avg_data(data5,data6,mip_era)
for r in range(nregimes):
    DATA = np.expand_dims(data56[:,:,r,:],-2)
    ax = fig.add_subplot(gs[r],projection=ccrs.Robinson(central_longitude=180.))
    bounds2 = plot_robin(ax,100*DATA,bounds,
            cmap = cmap,
            extend = extend,
            cbar = False,
            stippling = False,
            cbunits = '%',
            digits=1)    
    pl.title(extra_space+letters[r]+') Regime '+str(r+1),fontsize=16,loc='left')
    if r==nregimes-1:
        left,bottom,width,height = MU.get_axis_locs(ax)
  
ax2 = fig.add_axes([left+width+0.01, bottom, 0.02,height]) # create a second axes for the colorbar
norm = mpl.colors.BoundaryNorm(bounds2, cmap.N) # ensure the colors vary linearly even if the desired color boundaries are at varied intervals
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, ticks=bounds[::2], boundaries=bounds2,extend=extend,orientation='vertical')       
cb.set_label('%')   
pl.savefig(figdir+mip_era+'_'+experiment+'_MMM_RFO_maps_'+obsname+'_'+scalar_flag+'_'+anom_flag+'.pdf',bbox_inches='tight')



####################################################################################
# MMM DELTA RFO MAPS
####################################################################################
extra_space = '     '
mip_era = 'CMIP5p6'
cmap = pl.cm.RdBu_r
extend = 'both'
bounds = np.arange(-2.5,2.75,0.25)
fig=pl.figure(figsize=(18,12))
gs = gridspec.GridSpec(ROWS,COLS) 
data5 = catdata[exptype]['CMIP5']['dRFO']
data6 = catdata[exptype]['CMIP6']['dRFO']
avg,std,data56 = avg_data(data5,data6,mip_era)
for r in range(nregimes):
    DATA = np.expand_dims(data56[:,:,r,:],-2)
    ax = fig.add_subplot(gs[r],projection=ccrs.Robinson(central_longitude=180.))
    bounds2 = plot_robin(ax,100*DATA,bounds,
            cmap = cmap,
            extend = extend,
            cbar = False,
            stippling = True,
            cbunits = '%')    
    pl.title(extra_space+letters[r]+') Regime '+str(r+1),fontsize=16,loc='left')
    if r==nregimes-1:
        left,bottom,width,height = MU.get_axis_locs(ax)
  
ax2 = fig.add_axes([left+width+0.01, bottom, 0.02,height]) # create a second axes for the colorbar
norm = mpl.colors.BoundaryNorm(bounds2, cmap.N) # ensure the colors vary linearly even if the desired color boundaries are at varied intervals
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, ticks=bounds[::2], boundaries=bounds2,extend=extend,orientation='vertical')       
cb.set_label('%/K')   
pl.savefig(figdir+mip_era+'_'+experiment+'_MMM_dRFO_maps_'+obsname+'_'+scalar_flag+'_'+anom_flag+'.pdf',bbox_inches='tight')




####################################################################################
# GL avg bars and dots
#################################################################################### 
if anom_flag == 'avg':
    these_flavors = ['between_prime', 'within', 'sum_prime']
else:
    these_flavors = ['covary', 'between_prime', 'within', 'sum_prime']
pl.figure(figsize=(18,12))
r,region = 0,'GL'
cnt=r+1
pl.subplot(2,1,cnt)
band='SW'
xlabs=[]
xlocs=[]
x=-1
cnt=-1
# LEFT BREAKDOWN
for ty in ['tau','alt', 'amt', 'tot']:
    pl.axvline(x+0.5,color='gray',ls='-',lw=0.5) 
    cnt+=1
    pl.text(x+0.6,1.5,letters[cnt]+') '+Etypes[ty],ha='left',va='center')
    x+=0.5
    for fl,flavor in enumerate(these_flavors):
        avg5 = get_dots(np.ma.sum(catdata[exptype]['CMIP5'][band][flavor][ty],-2),region)
        avg6 = get_dots(np.ma.sum(catdata[exptype]['CMIP6'][band][flavor][ty],-2),region)
        avg = np.append(avg5,avg6)
        x+=0.5
        xlocs.append(x)
        xlabs.append(Eflavors[flavor])
        pl.plot(x*np.ones(avg.shape),avg.filled(np.nan),'o',mec=flavCOL[flavor],mfc='none',zorder=10)
        pl.bar(x,np.ma.average(avg),width=0.47,edgecolor=flavCOL[flavor],facecolor='none',lw=2,zorder=10)
pl.axvline(x+0.5,color='gray',ls='-',lw=0.5) 

# MIDDLE -- Raw (no regime break-down) truth
avg5 = get_dots(catdata[exptype]['CMIP5']['SW_TRUTH'],region)
avg6 = get_dots(catdata[exptype]['CMIP6']['SW_TRUTH'],region)
avg = np.append(avg5,avg6)
x+=1       
pl.plot(x*np.ones(avg.shape),avg.filled(np.nan),'o',mec='k',mfc='none',zorder=10)
pl.bar(x,np.ma.average(avg),width=0.47,edgecolor='k',facecolor='none',lw=2,zorder=10)
xlabs.append('No Breakdown')
xlocs.append(x)

# RIGHT
for flavor in these_flavors[-1::-1]:
    pl.axvline(x+0.5,color='gray',ls='-',lw=0.5)
    cnt+=1
    pl.text(x+0.6,1.5,letters[cnt]+') '+Eflavors[flavor],ha='left',va='center')
    x+=0.5
    for ty in ['tot','amt','alt','tau']:
        avg5 = get_dots(np.ma.sum(catdata[exptype]['CMIP5'][band][flavor][ty],-2),region)
        avg6 = get_dots(np.ma.sum(catdata[exptype]['CMIP6'][band][flavor][ty],-2),region)
        avg = np.append(avg5,avg6)
        x+=0.5
        xlocs.append(x)
        xlabs.append(Etypes[ty])
        pl.plot(x*np.ones(avg.shape),avg.filled(np.nan),'o',mec=flavCOL[flavor],mfc='none',zorder=10)
        pl.bar(x,np.ma.average(avg),width=0.47,edgecolor=flavCOL[flavor],facecolor='none',lw=2,zorder=10)
pl.axvline(x+0.5,color='gray',ls='-',lw=0.5)
pl.ylim(-0.75,1.75)
pl.axhline(y=0,color='gray',zorder=1)
pl.xticks(xlocs,xlabs,rotation=90,fontsize=12)
pl.ylabel('W/m$^2$/K',fontsize=12)
pl.title('Global Mean SW Cloud Feedback Components',fontsize=14,loc='left')
pl.xlim(xlocs[0]-0.5,xlocs[-1]+0.5)
pl.savefig(figdir+mip_era+'_'+experiment+'_bar_dots3_'+obsname+'_'+scalar_flag+'_'+anom_flag+'.pdf',bbox_inches='tight')   


####################################################################################    
# Compute some model-, and globally-averaged quantities for each regime
####################################################################################    
# Compare regime_clt with sum of regime_C:
hist5=catdata[exptype]['CMIP5']['CRhist']
hist6=catdata[exptype]['CMIP6']['CRhist']
hist56 = np.ma.concatenate((hist5,hist6),axis=-1)
AVG = np.ma.average(hist56,-1)
clt5 = catdata[exptype]['CMIP5']['regime_clt']
clt6 = catdata[exptype]['CMIP6']['regime_clt']
clt56 = np.ma.concatenate((clt5,clt6),axis=-1)
AVG2 = MV.average(clt56,-1)
AVG2.setAxis(0,AXL[0])
AVG2.setAxis(1,AXL[1])

NAME = ['regime_clt','regime_alb','regime_pct','RFO','dregime_clt','dregime_alb','dregime_pct','dRFO']
avgs,stds = {},{}
for i in range(nregimes):
    avgs[i]={}
    stds[i]={}
    for name in NAME:
        if 'RFO' in name or 'alb' in name: 
            scale=100
        else:
            scale=1
        D5 = catdata[exptype]['CMIP5'][name]
        D6 = catdata[exptype]['CMIP6'][name]
        D56 = np.ma.concatenate((D5,D6),axis=-1)   
        GLavgs = scale*get_dots(D56[:,:,i,:],'GL')
        avgs[i][name] = GLavgs.mean()
        stds[i][name] = GLavgs.std()
print(NAME[:4])
for i in range(nregimes):
    namevec = str(i+1)
    for name in NAME[:4]:
        namevec+=' & '+str(np.round(avgs[i][name],1))+' ('+str(np.round(stds[i][name],1))+')'
    namevec+=' \\'
    print(namevec)
  
print(NAME[4:])
for i in range(nregimes):
    namevec = str(i+1)
    for name in NAME[4:]:
        namevec+=' & '+str(MU.signif_digits(avgs[i][name],2))+' ('+str(MU.signif_digits(stds[i][name],2))+')'
    namevec+=' \\'
    print(namevec)

                  
####################################################################################           
# Plot the global mean control climate histograms for each cloud regime:
# Compare to Figure 2 of Cho, Tan, and Oreopoulos (2021)
####################################################################################
tau=np.array([0.,0.3,1.3,3.6,9.4,23,60,380])
ctp=np.array([1000,800,680,560,440,310,180,50])
 
pl.figure(figsize=(18,12))
hist5=catdata[exptype]['CMIP5']['CRhist']
hist6=catdata[exptype]['CMIP6']['CRhist']
hist56 = np.ma.concatenate((hist5,hist6),axis=-1)
AVG = np.ma.average(hist56,-1)
CRhist = np.reshape(AVG,(7,7,nregimes))
for i in range(nregimes):
    ax = pl.subplot(ROWS,COLS,i+1)
    if i==nregimes-1:
        plot_Chist(ax,CRhist[:,:,i],True)
    else:
        plot_Chist(ax,CRhist[:,:,i],False)
    ax.set_title('Regime '+str(i+1),fontsize=14,loc='left')
    if i+1==1 or i+1==4 or i+1==7:
        ax.set_ylabel('Cloud Top Pressure (hPa)')
    if i>nregimes-4:
        ax.set_xlabel('Optical Depth')
pl.savefig(figdir+mip_era+'_'+experiment+'_MMM_CRhists_'+obsname+'_'+scalar_flag+'.pdf',bbox_inches='tight')   

          
####################################################################################           
# Plot the change in global mean control climate histograms for each cloud regime:
####################################################################################
pl.figure(figsize=(18,12))
cmap = pl.cm.RdBu_r
hist5=catdata[exptype]['CMIP5']['dCRhist']
hist6=catdata[exptype]['CMIP6']['dCRhist']
hist56 = np.ma.concatenate((hist5,hist6),axis=-1)
AVG = np.ma.average(hist56,-1)
dCRhist = np.reshape(AVG,(7,7,nregimes))
for i in range(nregimes):
    ax = pl.subplot(ROWS,COLS,i+1)
    DATA=dCRhist[:,:,i]
    im1 = ax.pcolor(DATA,cmap='RdBu_r',vmin=-0.45,vmax=0.45)

    # assess where at least 80%, or 8 of 10 models agree on sign (so sum must be at least 8-2=6):
    thresh=80
    CNT = np.int32(MV.count(hist56[0,0,:],-1))
    sDATA=np.abs(MV.sum(np.sign(hist56[:,i,:]),axis=-1))           
    sDATA = np.reshape(sDATA,(7,7))
    A = np.int32(np.ceil((thresh/100.)*CNT))  # 0.8*10= 8
    B = CNT-A                               # 10-8=2
    C = A-B                                 # 8-2 = 6            
    # If >80% of the models agree on the sign, put a dot
    for row in range(7):
        x = np.arange(7)+0.5
        y = (row+0.5)*np.ones(7)
        y = np.ma.masked_where(sDATA[row,:]<C,y)
        pl.plot(x,y,'+',ms=8,color='k')   
    ax.set_xticklabels(tau, minor=False)
    ax.set_yticks(np.arange(len(ctp)))
    ax.set_yticklabels(ctp, minor=False)
    pl.title('Regime '+str(i+1),fontsize=14,loc='left')
    if i+1==1 or i+1==4 or i+1==7:
        pl.ylabel('Cloud Top Pressure (hPa)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb=pl.colorbar(im1, cax=cax, extend='both')
    if i!=nregimes-1:
        cb.remove()
    else:
        cb.set_label('%/K')
    if i>nregimes-4:
        ax.set_xlabel('Optical Depth')
pl.savefig(figdir+mip_era+'_'+experiment+'_MMM_dCRhists_'+obsname+'_'+scalar_flag+'.pdf',bbox_inches='tight')   


####################################################################################
# Plot the change in regime frequency of occurrence as a function of latitude
####################################################################################
   
# Compare CMIP5 and CMIP6 delta RFO
for mip,mip_era in enumerate(['CMIP5','CMIP6','CMIP5p6']):
    if mip_era=='CMIP5p6':
        RFO5 = catdata[exptype]['CMIP5']['RFO']
        RFO6 = catdata[exptype]['CMIP6']['RFO']
        RFO=np.ma.concatenate((RFO5,RFO6),axis=-1)
        dRFO5 = catdata[exptype]['CMIP5']['dRFO']
        dRFO6 = catdata[exptype]['CMIP6']['dRFO']
        dRFO=np.ma.concatenate((dRFO5,dRFO6),axis=-1)
    else:
        #continue
        RFO = catdata[exptype][mip_era]['RFO']
        dRFO = catdata[exptype][mip_era]['dRFO']
    pl.figure(figsize=(18,12))
    cnt=0        
    cnt+=1
    pl.subplot(2,2,cnt)
    avg = 100*MV.average(np.average(RFO,-1),1) # average across models, then across longitude
    pl.pcolor(lats[:],np.arange(nregimes),avg[:,-1::-1].T,vmin=0,vmax=50,cmap='PuBu')
    pl.yticks(np.arange(0,nregimes,1),np.arange(nregimes,0,-1))
    cb=pl.colorbar(extend='max')
    cb.set_label('%')
    pl.title(letters[cnt-1]+') Relative Frequency of Occurrence',fontsize=14,loc='left')
    pl.xlabel('Latitude')
    pl.ylabel('Regime #')
    
    cnt+=1
    pl.subplot(2,2,cnt)
    avg = 100*MV.average(np.average(dRFO,-1),1) # average across models, then across longitude
    pl.pcolor(lats[:],np.arange(nregimes),avg[:,-1::-1].T,vmin=-2,vmax=2,cmap='RdBu_r')
    # assess where at least 80%, or 8 of 10 models agree on sign (so sum must be at least 8-2=6):
    thresh=80
    CNT = np.int32(MV.count(dRFO[0,0,0,:],-1))
    sDATA=np.abs(MV.sum(np.sign(np.average(dRFO,1)),axis=-1))           
    A = np.int32(np.ceil((thresh/100.)*CNT))  # 0.8*10= 8
    B = CNT-A                               # 10-8=2
    C = A-B                                 # 8-2 = 6            
    # If >80% of the models agree on the sign, put a dot
    for r in range(nregimes):
        y = (nregimes-r)*np.ones(90)-1#0.5
        y = np.ma.masked_where(sDATA[:,r]<C,y)
        pl.plot(lats[:],y,'.',ms=2,color='k')
            
    pl.yticks(np.arange(0,nregimes,1),np.arange(nregimes,0,-1))
    cb=pl.colorbar(extend='both')
    cb.set_label('%/K')
    pl.title(letters[cnt-1]+') $\Delta$Relative Frequency of Occurrence',fontsize=14,loc='left')
    pl.xlabel('Latitude')
    pl.ylabel('Regime #')
    pl.savefig(figdir+mip_era+'_'+experiment+'_MMM_'+sec+'_RFO_'+obsname+'_'+scalar_flag+'.pdf',bbox_inches='tight')   
