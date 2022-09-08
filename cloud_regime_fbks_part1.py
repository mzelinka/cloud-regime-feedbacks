########################################################################
# First part of the analysis used in Zelinka et al (2022) paper
# "Detailing cloud property feedbacks with a regime-based decomposition"
# Climate Dynamics
# https://doi.org/10.1007/s00382-022-06488-7
########################################################################

# Script was initially developed based on the Williams and Webb 2009 code
# to map GCMs to the observed ISCCP cloud regimes
# https://github.com/tsussi/cloud-regime-error-metric

import cdutil
import glob
from regrid2 import Horizontal
import matplotlib.pyplot as pl
import MDZ_utils as MU
import zelinka_analysis as ZA
import CMIP6_utils as CU
import MV2 as MV
import numpy as np
import cdms2 as cdms
import matplotlib as mpl


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

# Decision 3 - Start year for the 3-yr chunks
# [1979, 1982, 1985, 1988, 1991, 1994, 1997, 2000, 2003, 2006, 2009]
START = 2000#1991
END = 2008 # CMIP5 does not extend beyond 2008; most of CMIP6 does not extend beyond 2014 

########################################################################
# END USER INPUT
########################################################################
exps5 = ['amip','amip4K']#['piControl','abrupt4xCO2']#
exps6 = ['amip','amip-p4K']#['piControl','abrupt-4xCO2']#

if exps5[-1] == 'abrupt4xCO2':
    START = 2450
    END = START+8
    
figdir = '/home/zelinka1/figures/regimes/'+obsname+'/'
outdir = '/p/user_pub/climate_work/zelinka1/cloud_regimes/'
coastlat, coastlon = MU.loadCoast()

# ----------------------------------------------------------------------
# Zelinka et al 2012 kernels:
f=cdms.open('/home/zelinka1/kernels/zelinka/cloud_kernels2.nc')
LWkernel=f('LWkernel')
SWkernel=f('SWkernel')
f.close()
# Define the cloud kernel axis attributes
lats=cdms.createAxis(LWkernel.getLatitude()[:])
lats.id="lat" 
lats.units="degrees_N"
lats.designateLatitude()
lons=cdms.createAxis(np.arange(1.25,360,2.5))
lons.id="lon" 
lons.units="degrees_E"
lons.designateLongitude()
kern_grid = cdms.createGenericGrid(lats,lons)
kern_grid.getLatitude().id='lat'
kern_grid.getLongitude().id='lon'
LWkernel.setGrid(kern_grid)
LWkernel=MV.masked_where(np.isnan(LWkernel),LWkernel)
SWkernel=MV.masked_where(np.isnan(SWkernel),SWkernel)
KALB=np.arange(0.0,1.5,0.5) # the clear-sky albedos over which the kernel is computed
lons2 = kern_grid.getLongitude()[:] 
lats2 = kern_grid.getLatitude()[:] 


# ----------------------------------------------------------------------
# TAU AND CTP INFORMATION:
f=cdms.open('/p/user_pub/climate_work/zelinka1/cloud_regimes/MODIS_CRs_equal_angle/MODIS_C61_CRs_2014.nc')
TAU=f('Cloud_Regime_COT_boundaries')
CTP=f('Cloud_Regime_CTP_boundaries')[-1::-1] # SFC to TOA, in hPa
f.close()
# ALB = f(TAU) expression used in ISCCP simulator: https://github.com/CFMIP/COSPv2.0/blob/master/src/simulator/icarus/icarus.F90#L580
ALB = (TAU**0.895)/((TAU**0.895)+6.82)
ALB_midpt = np.average(ALB,1)
CTP = CTP # hPa
#CTP[-1][-1] = 1  # swap out 1100 for 1000 hPa in the lowest bin
CTP_midpt = np.average(CTP,1)

#MAKE CLASSIC NETCDF:
#===========================
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)

########################################################################
def get_xml_list(mip_era):
    xml_list={}
    if mip_era=='CMIP5':
        exps = exps5
        freq = 'day'
    elif mip_era=='CMIP6':
        exps = exps6
        freq = '*ay'  
    flag = 'mo' 
    
    good_models = CU.available_models(mip_era,exps,variables,flag,'atmos',freq)        

    for model in good_models:
        if 'IPSL-CM5' in model: # simulator implementation is flawed
            continue
        if model=='BCC-CSM2-MR': # clisccp is zero everywhere but the first timestep in each file!
            continue
        xml_list[model]={}
                   
        experiment = exps[-1] # this is more likely to be the limiting exp
        activity = '*'#'CMIP'
        realm = 'atmos'
        institution = '*'
        variant_label = '*' # don't forget about this
        gg = MU.search_xmls_v2(mip_era,activity,experiment,realm,freq,'tas',institution,model,variant_label)
        if gg==None:
            #print('No xmls for '+experiment+' '+freq+' tas '+model+' '+variant_label)
            continue
        ripfs = gg[5]
        xmls = gg[0]
        for ri,ripf in enumerate(ripfs):
            xml_list[model][ripf]={}
            for e,exp in enumerate(exps):
                xml_list[model][ripf][exp]={}
                all_xmls=[]
                for var in variables:
                    hh = MU.search_xmls_v2(mip_era,activity,exp,realm,freq,var,institution,model,ripf)             
                    if hh==None:
                        #print('No xmls for '+exp+' '+freq+' '+var+' '+model+' '+ripf)
                        continue         
                    all_xmls.append(hh[0][0])
                    xml_list[model][ripf][exp] = all_xmls
                        
    return(xml_list)
    
########################################################################
def get_observations(obsname,scalar_flag):

    if 'MODIS' in obsname:
        # ----------------------------------------------------------------------
        # MODIS CLOUD REGIMES
        f=cdms.open('/p/user_pub/climate_work/zelinka1/cloud_regimes/MODIS_CRs_equal_angle/MODIS_C61_CRs_2014.nc')
        # the above file was downloaded from a dropbox file provided by Ivy Tan on 8/28/20
        Centroids = f('Cloud_Regime_Centroids') # [12,7,6]
        Centroids = np.moveaxis(Centroids,-1,-2) # [12,6,7]
        Centroids = Centroids[:,:,-1::-1] # SFC to TOA
        latitude=f('latitude')  
        longitude0=f('longitude')  
        MODIS_CRs0=f('MODIS_AQUA_CRs')
        #MODIS_CRs0=f('MODIS_TERRA_CRs')
        f.close()
        longitude=MU.nanarray(longitude0.shape)
        OBS_CRs = MU.nanarray(MODIS_CRs0.shape)
        longitude[:180] = longitude0[180:]
        longitude[180:] = longitude0[:180]+360
        OBS_CRs[:,:,:180] = MODIS_CRs0[:,:,180:]
        OBS_CRs[:,:,180:] = MODIS_CRs0[:,:,:180]
        del(MODIS_CRs0)

    elif 'ISCCP' in obsname:
        # ----------------------------------------------------------------------
        # ISCCP HGG CLOUD REGIMES
        # https://isccp.giss.nasa.gov/wstates/hggws.html
        f=cdms.open('/p/user_pub/climate_work/zelinka1/cloud_regimes/ISCCP_HGG/2014.nc')
        ws=f('ws') # [10 centroids, 42 pctau]
        NR = ws.shape[0]+1 # number of regimes plus one for the clear regime
        Centroids = np.zeros((NR,6,7))# [11 centroids, 6 tau, 7 ctps]
        for i in range(NR-1):
            Centroids[i,:] = ws[i,:].reshape(6,7)
        Centroids = Centroids[:,:,-1::-1] # SFC to TOA
        OBS_CRs = np.zeros((NR,12,360,180))
        OBS_Ntot = np.zeros((NR,12,360,180))
        for m,month in enumerate(['january','february','march','april','may','june','july','august','september','october','november','december']):
            data = f(month)
            for ws in range(NR-1):
                wsdata = np.where(data==ws+1,1,0)
                # Add the number of counts each gridcell was assigned to the WS
                OBS_CRs[ws,m,:] = np.sum(wsdata,axis=0)
                OBS_Ntot[ws,m,:] = data.shape[0]
        f.close()
        longitude = data.getLongitude()[:]
        latitude = np.arange(-90,90)
        OBS = np.moveaxis(np.sum(OBS_CRs,1)/np.sum(OBS_Ntot,1),0,-1)
        OBS = np.moveaxis(OBS,0,1) # [180, 360, 11]
        if obsname=='ISCCP8':
            # Tselioudis et al (2021) combined some pairs of weather states:
            OBS2 = np.zeros((180,360,9))
            OBS2[:,:,0] = OBS[:,:,0]
            OBS2[:,:,1] = OBS[:,:,1]
            OBS2[:,:,2] = OBS[:,:,2]+OBS[:,:,5] # Combine 3 and 6
            OBS2[:,:,3] = OBS[:,:,3]
            OBS2[:,:,4] = OBS[:,:,4]
            OBS2[:,:,5] = OBS[:,:,6]
            OBS2[:,:,6] = OBS[:,:,7]        
            OBS2[:,:,7] = OBS[:,:,8]+OBS[:,:,9] # Combine 9 and 10
            OBS2[:,:,8] = OBS[:,:,10]
            OBS = OBS2
            Centroids2 = np.zeros((9,6,7))
            Centroids2[0,:] = Centroids[0,:]
            Centroids2[1,:] = Centroids[1,:]
            Centroids2[2,:] = Centroids[2,:]+Centroids[5,:] # Combine 3 and 6
            Centroids2[3,:] = Centroids[3,:]
            Centroids2[4,:] = Centroids[4,:]
            Centroids2[5,:] = Centroids[6,:]
            Centroids2[6,:] = Centroids[7,:]        
            Centroids2[7,:] = Centroids[8,:]+Centroids[9,:] # Combine 9 and 10
            Centroids2[8,:] = Centroids[10,:]
            Centroids = Centroids2
        
    
    ######################################################################## 
    # Ignore the clear-sky Centroid, as this will not be determined via minimum 
    # Euclidean distance but rather from simply knowing that the scene is clear
    ######################################################################## 
    Centroids = 100*Centroids[:-1,:] # also put in units of %
    
    ######################################################################## 
    # COMPUTE THE 3 SCALARS OF THE CENTROID
    ######################################################################## 
    NR,b,c=Centroids.shape
    nregimes = NR # +1 # add 1 for undefined points
    obs_pct = np.zeros(nregimes,) 
    obs_alb = np.zeros(nregimes,)
    obs_clt = np.zeros(nregimes,)
    if scalar_flag == 'grab':
        # Take the centroid scalars reported in the papers:
        # with this option, we should use albisccp, pctisccp, and cltisccp
        if 'MODIS' in obsname:
            """
            # Take the clt values from Figure 2 of Cho et al (2021)
            obs_clt[:-1] = np.array([96.6,77.9,96.5,84.4,89.3,84.8,88.9,92.1,82.1,53.2,27.2,0.0])/100.
            # Take the pct and tau values from Table 1 of Cho et al (2021)
            obs_pct[:-2] = np.array([179,209,273,290,405,578,752,874,880,867,712])
            obs_tau = np.array([28.2,2.8,23.5,5.9,18.4,23,17.6,19.3,8.1,5.0,9.9])
            """
            # FOR EQUAL ANGLE CENTROIDS WE WILL USE THE FOLLOWING PROVIDED BY CHO ON 5/20/21:
            obs_clt = np.array([96.6,78.5,96.5,84.3,89.6,84.9,89.1,93.8,82.4,52.7,27.5])
            obs_pct = np.array([180.2,234.2,273.1,305.0,409.5,583.8,757.4,866.8,878.5,876.6,679.0])
            obs_tau = np.array([26.9,2.7,23.2,5.7,18.1,21.8,16.6,17.9,7.8,4.5,7.8])        
            # convert tau to albedo using same equation as in ISCCP simulator: https://github.com/CFMIP/COSPv2.0/blob/master/src/simulator/icarus/icarus.F90#L580    
            obs_alb = (obs_tau**0.895)/((obs_tau**0.895)+6.82)   
        elif obsname=='ISCCP8':
            # Take the values from Table 2 of Tselioudis et al (2021)
            obs_clt = np.array([99.5,99.2,79.9,84.5,97.2,40.,79.6,90.7])
            obs_pct = np.array([242.6,433.6,316.3,395.6,606.9,645.1,840.1,725.5])
            obs_tau = np.array([10.5,10.4,1.2,2.2,9.5,3.2,4.,6.3])
            # convert tau to albedo using same equation as in ISCCP simulator: https://github.com/CFMIP/COSPv2.0/blob/master/src/simulator/icarus/icarus.F90#L580    
            obs_alb = (obs_tau**0.895)/((obs_tau**0.895)+6.82)
    else:
        # Compute the scalars yourself
        # with this option, we should NOT use albisccp, pctisccp, and cltisccp
        #obs_pct[:-1],obs_alb[:-1],obs_clt[:-1] = compute_scalars(Centroids,ALB_midpt,CTP_midpt)
        obs_pct,obs_alb,obs_clt = compute_scalars(Centroids,ALB_midpt,CTP_midpt)

    if 'MODIS' in obsname:
        ######################################################################## 
        # COMPUTE RFO OF OBS REGIMES
        ######################################################################## 
        A,B,C=OBS_CRs.shape
        OBS = np.zeros((B,C,nregimes))
        for i in range(1,1+nregimes):
            mem=(OBS_CRs==i)
            OBS[:,:,i-1] = np.ma.sum(mem,0)/A

    ######################################################################## 
    # OBS RFO MAP
    ######################################################################## 
    cmap = mpl.cm.viridis
    bounds = [0.01,0.02,0.03,0.04,0.06,0.08,0.10,0.13,0.16,0.20,0.25,0.30,0.40,0.50,0.70,0.90]
    bounds2 = np.append(np.append(-20,bounds),20) # This is only needed for norm if colorbar is extended
    cticks=bounds[::2]
    norm = mpl.colors.BoundaryNorm(bounds2, cmap.N) # this is the critical line for making sure the colors vary linearly even if the desired color boundaries are at varied intervals
    pl.figure(figsize=(18,12))
    for r in range(nregimes):
        pl.subplot(4,3,r+1)
        #pl.pcolor(LON,LAT,np.ma.sum(CNTS[:,:,:,r],0)/a,vmin=0,vmax=1.0,cmap='Blues_r')
        pl.contourf(longitude,latitude,OBS[:,:,r],bounds,cmap=cmap,norm=norm,extend='both')
        pl.colorbar(ticks=cticks)
        pl.plot(coastlon, coastlat, color='k', linewidth=0.5) # overlay coastlines
        pl.xlim(0, 360)
        pl.ylim(latitude[0],latitude[-1])  
        pl.title('CR'+str(r+1))
    pl.suptitle(obsname+' 2014 CRs',y=0.95,fontsize=16)
    pl.savefig(figdir+obsname+'_2014_CRs.png',bbox_inches='tight')

    ######################################################################## 
    # OBS CENTROID HISTOGRAMS
    ######################################################################## 
    tau=np.array([0.,1.3,3.6,9.4,23.,60.,380.])
    ctp=np.array([1000,800,680,560,440,310,180,50])
    #cmap = mpl.cm.viridis
    if 'ISCCP' in obsname:
        bounds = [0.2,1.,2.,3.,4.,6.,8.,10.,15.,99.]
    else:
        bounds = [0.1,0.2,0.4,0.8,1.5,3.,6.,10.,15.,20.,25.,35.]
    cmap = pl.cm.get_cmap('viridis',len(bounds))
    colors = list(cmap(np.arange(len(bounds))))
    cmap = mpl.colors.ListedColormap(colors[:-1], "")
    # set over-color to last color of list 
    cmap.set_over(colors[-1])
    #from copy import copy
    #palette = copy(cmap)
    cmap.set_under('white', 1.0)  # 1.0 represents not transparent    
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N) # this is the critical line for making sure the colors vary linearly even if the desired color boundaries are at varied intervals          
    fig=pl.figure(figsize=(18,12))
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    for i in range(nregimes):
        ax = pl.subplot(4,3,i+1)
        if i+1 == 2:
            pl.title(obsname,fontsize=16)                    
        im1 = ax.pcolor(Centroids[i,:].T,cmap=cmap,norm=norm)
        pl.text(6,0,'Regime '+str(i+1),ha='right',va='bottom',color='gray',fontsize=14)
        ax.set_xticklabels(tau, minor=False)
        ax.set_yticklabels(ctp, minor=False)
        MU.add_colorbar(fig,im1,ax, ticks=bounds, extend='max')
    pl.savefig(figdir+obsname+'_CR_hists.png',bbox_inches='tight')
    
    return(Centroids,obs_clt,obs_pct,obs_alb)


########################################################################
def compute_scalars(histogram,ALB_midpt,CTP_midpt):
    # COMPUTE 3-SCALAR VERSIONS OF CENTROIDS FOLLOWING WILLIAMS AND WEBB 2009
    # histogram is size [time,tau,ctp,...]
    clt = np.sum(np.sum(histogram,1),1)
    C = np.moveaxis(histogram,1,-1) # move tau to the end
    alb = np.sum(np.sum(C,1)*np.array(ALB_midpt),-1)/clt
    C = np.moveaxis(histogram,2,-1) # move ctp to the end
    pct = np.sum(np.sum(C,1)*np.array(CTP_midpt),-1)/clt
    
    return(pct,alb,clt)

########################################################################
def cnt_wtd_avg(modata,counts):
    # modata is [days,lat,lon]
    # counts is [days,lat,lon,regime]
    # count-weighted average over days; counts are simply 0s and 1s:
    
    frac_of_month=counts/np.ma.sum(np.ma.sum(counts,-1,keepdims=True),0,keepdims=True) # normalize by total number of VALID days in a month
    
    this = np.sum(modata*np.moveaxis(frac_of_month,-1,0),1) # sum over days --> [regime,lat,lon]
    that = np.sum(np.moveaxis(frac_of_month,-1,0),1) # sum over days --> [regime,lat,lon]
    monthly_data = this/that
    monthly_count = that # fraction of valid times in the month occupied by regime r; sum over all regimes equals 1
    return(monthly_data,monthly_count)
    
########################################################################
def go_to_monthly(data,CNTS,TIME):
    nregimes = CNTS.shape[-1]
    monthly_data=np.zeros((nregimes,lenmos,90,144))
    monthly_count=np.zeros((nregimes,lenmos,90,144))
    for m,month in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']):
        modata,indices = extract_month(data,TIME,month) # day,lat,lon
        counts,indices = extract_month(CNTS,TIME,month) # day,lat,lon,regime          
        for y in range(lenyrs):
            modata = np.take(data,indices[y],axis=0)
            counts = np.take(CNTS,indices[y],axis=0) 
            monthly_data[:,y*12+m,:],monthly_count[:,y*12+m,:] = cnt_wtd_avg(modata,counts)
                
    return(monthly_data,monthly_count)
        
########################################################################    
def extract_month(DATA,TIME,month): 
   
    DATA.setAxis(0,TIME)
    # Retrieve and concatenate all indices from a given month together
    indices, bounds, starts = cdutil.monthBasedSlicer(TIME,month)
    catDATA = None
    for i, sub in enumerate(indices):
        tmp = DATA(time=slice(sub[0],sub[-1]+1))
        #print("Year",i,"shape of "+month,tmp.shape)
        if catDATA is None:
            catDATA = tmp
        else:
            catDATA = MV.concatenate((catDATA, tmp))
                   
    return(catDATA,indices)
    
########################################################################    
def day_to_month_v2(DATA,TIME):
    out = MU.nanarray((DATA[1,:].shape))
    OUT = np.rollaxis(np.tile(np.expand_dims(out,axis=-1),lenmos),-1)
    for m,month in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']):
        # extract the days in this month
        catDATA,indices = extract_month(DATA,TIME,month) 
        for y in range(lenyrs):
            modata = np.take(DATA,indices[y],axis=0)
            OUT[y*12+m,:] = np.nanmean(modata,0) 
            
    return (OUT)
        
########################################################################    
def do_regime_analysis(pctisccp,cltisccp,albisccp,PCTAU,albcs,LWkernel,SWkernel,normalize):
    """
    pctisccp must be in hPa
    albisccp must be in the range 0-1
    cltisccp must be in %
    """
    print('Assigning data to observational cloud regimes')

    AXL = PCTAU[:,0,0,:].getAxisList()
    TIME=albcs.getTime()
    a,b,c = albisccp.shape
    
    if scalar_flag=='all42':     

        # Create a place to catch undefined data:
        obs_PC = np.append(Centroids,-9999*np.ones((1,6,7)),axis=0)

        # If PCTAU is masked, set it to -9999
        bigmask = PCTAU.mask
        PCTAU = MV.where(bigmask,-9999,PCTAU)
           
        #######################################
        # Assign model data to observed regimes
        #######################################
        PC = np.moveaxis(np.moveaxis(PCTAU[:,1:,:],1,-1),1,-1)
        ed=np.zeros((a,b,c,nregimes+1))
        for i in range(nregimes+1): # go to nregimes+1 so as to include the undefined regime so it does not get assigned to one of the true cloud regimes
            df2 = (PC-obs_PC[i])**2
            this = np.sqrt(np.sum(np.sum(df2,-1),-1))
            ed[...,i] = this 
            
        group = np.ma.argmin(ed,axis=-1)
        distance = np.min(ed,-1) # the actual Euclidean distance for this assignment 

        CNTS=np.zeros((a,b,c,nregimes))
        DIST=np.zeros((a,b,c,nregimes))
        for i in range(nregimes): # do not go to nregimes+1 as we do not want to include missing data locations in denom
            mem=(group==i)
            CNTS[:,:,:,i] = mem
            dummy=MU.nanarray(distance.shape)
            dummy[mem]=distance[mem]
            DIST[:,:,:,i] = dummy
        CNTS = MV.array(CNTS)
            
            
    else:
        pctisccp.setAxisList(AXL)
        cltisccp.setAxisList(AXL)
        albisccp.setAxisList(AXL)    
        LAT = cltisccp.getLatitude()
        LON = cltisccp.getLongitude()   

        # Reshape these to be vectors
        ALB = albisccp.flatten()#np.reshape(albisccp,(a*b*c,))
        PCT = pctisccp.flatten()#np.reshape(pctisccp,(a*b*c,))
        CLT = cltisccp.flatten()#np.reshape(cltisccp,(a*b*c,))
        ALBCS = albcs.flatten()#np.reshape(albcs,(a*b*c,))

        ################################################################################        
        # Apply the normalization used in Jin et al (2017) DOI 10.1007/s00382-016-3107-6
        ################################################################################
        # normalize both model and obs by a single standard deviation computed across a 
        # concatenated vector of all grid points and all days from the model 
        ALB_norm = ALB/normalize['albisccp']#ALB.std()
        obs_alb_norm = obs_alb/normalize['albisccp']#ALB.std()
        PCT_norm = PCT/normalize['pctisccp']#PCT.std()
        obs_pct_norm = obs_pct/normalize['pctisccp']#PCT.std()
        CLT_norm = CLT/normalize['cltisccp']#CLT.std()
        obs_clt_norm = obs_clt/normalize['cltisccp']#CLT.std()
    
        # Create a place to catch clear-sky (9999) and undefined data (-9999):
        obs_alb_norm=np.append(np.append(obs_alb_norm,9999),-9999)
        obs_pct_norm=np.append(np.append(obs_pct_norm,9999),-9999)
        obs_clt_norm=np.append(np.append(obs_clt_norm,9999),-9999)

        # If clear-sky, re-set everyone to +9999 (this has to be done again because we normalized)
        ALB_norm = MV.where(CLT==9999,9999,ALB_norm)
        PCT_norm = MV.where(CLT==9999,9999,PCT_norm)
        CLT_norm = MV.where(CLT==9999,9999,CLT_norm)

        # If CLT, ALB, or PCT are masked, set everyone to -9999
        bigmask = (ALBCS+ALB+PCT+CLT).mask
        ALB_norm = MV.where(bigmask,-9999,ALB_norm)
        PCT_norm = MV.where(bigmask,-9999,PCT_norm)
        CLT_norm = MV.where(bigmask,-9999,CLT_norm)

        #######################################
        # Assign model data to observed regimes
        #######################################
        ed=np.zeros((len(ALB),nregimes+2))
        for i in range(nregimes+2): # go to nregimes+1 so as to include clear-sky and undefined regimes so they do not get assigned to one of the true cloud regimes
            this = np.sqrt(((ALB_norm-obs_alb_norm[i])**2)+((PCT_norm-obs_pct_norm[i])**2)+((CLT_norm-obs_clt_norm[i])**2))
            ed[:,i] = this 
            
        group = np.ma.argmin(ed,axis=-1)
        distance = np.min(ed,-1) # the actual Euclidean distance for this assignment 

        CNTS=np.zeros((a,b,c,nregimes+1))
        DIST=np.zeros((a,b,c,nregimes+1))
        for i in range(nregimes+1): # do not go to nregimes+2 as we do not want to include missing data locations in denom (but we do want clear-sky)
            mem=(group==i)
            CNTS[:,:,:,i] = np.reshape(mem,(a,b,c))
            dummy=MU.nanarray(distance.shape)
            dummy[mem]=distance[mem]
            DIST[:,:,:,i] = np.reshape(dummy,(a,b,c)) 
        CNTS = MV.array(CNTS) 

                
    ################################################################### 
    # SAVE CNTS MATRIX FOR MASKING OTHER VARIABLES
    ###################################################################
    savefile = mip_era+'.'+exp+'.'+mo+'.'+ripf+'.CNTS.'+str(styr)+'-'+str(fnyr)+'.'+obsname+'.'+scalar_flag+'.nc'
    ff = cdms.open(outdir+savefile,'w')
    CNTS.id = 'CNTS'
    ff.write(CNTS)
    ff.history='Written by /home/zelinka1/scripts/cloud_regime_error_metric_part0_maps.py on feedback2.llnl.gov'
    ff.close()    
    print('-----Saved '+outdir+savefile)
    
    ################################################################### 
    # Aggregate data to monthly resolution (4)
    ###################################################################
    # cltisccp [days,lat,lon]
    # PCTAU  [days,TAU,CTP,lat,lon]
    # CNTS     [days,lat,lon,regime]; np.sum(CNTS,-1).min()==1; np.sum(CNTS,-1).max()==1; 

    # sum(X4*N4) across regimes = monthly mean X
    # N4 = fraction of valid times in the month occupied by regime r; sum over all regimes equals 1, except where no data exists, where it is 0

    albcs4,N4 = go_to_monthly(albcs,CNTS,TIME)
    clt4,N4 = go_to_monthly(cltisccp,CNTS,TIME)
    pct4,N4 = go_to_monthly(pctisccp,CNTS,TIME)
    alb4,N4 = go_to_monthly(albisccp,CNTS,TIME)
    Nmos = clt4.shape[1]
    C4 = MU.nanarray((nregimes+1,Nmos,7,7,90,144))
    KLW4 = MU.nanarray((nregimes+1,Nmos,7,7,90,144))
    KSW4 = MU.nanarray((nregimes+1,Nmos,7,7,90,144))
    
    # Assign the appropriate SW kernel based on the zonal mean albedo:
    # this is equivalent to doing this mapping to lat/lon, then zonally averaging (but is faster)
    # repeat the kernel to span the full range of months , if necessary:
    indices=np.tile(np.arange(12),(lenyrs,))
    repSWkernel = np.take(SWkernel,indices,axis=0)       # [month,TAU,CTP,lat,albcs]
    repLWkernel = np.take(LWkernel,indices,axis=0)       # [month,TAU,CTP,lat,albcs]
    for i in range(nregimes+1): 
        KLW4[i,:] = np.tile(np.expand_dims(repLWkernel[...,0],axis=4),(1,1,1,1,144))
        KSW4[i,:] = ZA.map_SWkern_to_lon(repSWkernel,albcs4[i,:]) # expects albcsmap as [time,lats,lons], but we are giving it [time,lats,regimes]
    KSW4 = MV.masked_where(np.isnan(KSW4),KSW4)

    # set this so that sum over all regimes equals masked where no data exists
    Nreconstr = np.sum(N4,0) # 12,90,144
    for r in range(nregimes+1):
        N4[r,:] = N4[r,:]/Nreconstr # this will divide by zero if no valid data exists, yielding a nan
    N4 = MV.masked_where(np.isnan(N4),N4) # set all nans to masked
        
    for t in range(7):
        for p in range(7):
            C4[:,:,t,p,:],dummy = go_to_monthly(PCTAU[:,t,p,:],CNTS,TIME)
    
    # Compute global mean regime cloud albedo, CTP, and clt:
    regime_clt = MV.array(np.sum(clt4*N4,1)/np.sum(N4,1)) # regime,lat,lon   
    regime_pct = MV.array(np.sum(pct4*N4,1)/np.sum(N4,1)) # regime,lat,lon   
    regime_alb = MV.array(np.sum(alb4*N4,1)/np.sum(N4,1)) # regime,lat,lon
    regime_clt.setAxis(1,AXL[1])
    regime_pct.setAxis(1,AXL[1])
    regime_alb.setAxis(1,AXL[1])
    regime_clt.setAxis(2,AXL[2])
    regime_pct.setAxis(2,AXL[2])
    regime_alb.setAxis(2,AXL[2])

    # N4 is [nregimes+1,lenmos,90,144]
    
    # Define RFO such that summing it over regime = 1:
    # sum rfo over all regimes equals 1, except where no data exists, where it is masked
    rfo = N4/np.ma.sum(N4,0,keepdims=True)
    
    print('Done with do_regime_analysis()')
    return (C4,KLW4,KSW4,rfo,regime_clt,regime_pct,regime_alb)
    
########################################################################                  


########################################################################                  
# START OF MAIN CODE
########################################################################                  

Centroids,obs_clt,obs_pct,obs_alb = get_observations(obsname,scalar_flag)
nregimes=len(obs_alb)

# Let's find all models that have all the variables we need:
if scalar_flag == 'grab':
    variables=['albisccp','pctisccp','cltisccp', 'rsuscs', 'rsdscs', 'clisccp','tas']
else:
    variables=['rsuscs', 'rsdscs', 'clisccp','tas']


mip_eras = ['CMIP5','CMIP6']
for mip,mip_era in enumerate(mip_eras):
    tic1 = MU.tic()
    xml_list = get_xml_list(mip_era)
    if mip_era=='CMIP5':
        #continue
        exps = exps5
    elif mip_era=='CMIP6':
        exps = exps6
    good_models = list(xml_list.keys())

    for mo in good_models:#[-1::-1]:
        tic2 = MU.tic()
        ripfs = list(xml_list[mo].keys())
        for ri,ripf in enumerate(ripfs):
            if mo=='MRI-CGCM3' and ripf=='r5i1p3':
                continue
            if mo=='IPSL-CM6A-LR' and ripf=='r22i1p1f1':
                continue             
                
            tic3 = MU.tic()                      

            
            # MAKE A REGRIDDER FUNCTION
            # https://cdms.readthedocs.io/en/latest/manual/cdms_4.html#cdms-horizontal-regrider
            xmls = xml_list[mo][ripf][exps[0]]
            f=cdms.open(xmls[0])
            var = xmls[0].split('.')[-7]
            yrs,mos=MU.get_plottable_time(f[var])
            # get last complete year:                    
            endyr = np.int32(yrs[np.where(mos==12)[-1][-1]])
            this=f(var,time=slice(0,1))           
            ingrid = this.getGrid()
            f.close()
            regridfunc = Horizontal(ingrid, kern_grid)


            NY = 3 # 3 yr chunks
            chunks = np.arange(START,endyr,NY)

            
            # Get normalizations
            normalize={}
            for xml in xmls:
                var = xml.split('.')[-7]
                if var=='albisccp' or var=='pctisccp' or var=='cltisccp':   
                    f=cdms.open(xml) 
                    # if it is one of the three scalars, read in 3 yrs to get the normalization values:
                    srcData0 = f(var,time=("2003-01-01 0:0:0.0","2005-12-31 23:59:0.0") ) 
                    std=srcData0.flatten().std()
                    if var=='pctisccp' and mo!='MPI-ESM-LR': 
                        normalize[var]=std/100. # convert to hPa
                    else:
                        normalize[var]=std
                    f.close()              
            
            for styr in chunks: # still need to do IPSL amip4K last 2 chunks 
                tic4 = MU.tic()
                fnyr = styr+NY-1
                # take whichever is smaller:
                smallyr = np.min((endyr,fnyr))
                   
                #========================================
                # FOR NOW LETS TAKE ONLY THESE PERIODS
                if smallyr>END: 
                    continue
                #========================================                  
                
                tslice = (str(styr)+"-01-01 0:0:0.0",str(smallyr)+"-12-31 23:59:0.0")
                lenyrs = NY 
                lenmos=12*lenyrs
                           
                for e,exp in enumerate(exps):
                    tic5 = MU.tic()
                    filename = mip_era+'.'+exp+'.'+mo+'.'+ripf+'.processed.maps.'+str(styr)+'-'+str(fnyr)+'.'+obsname+'.'+scalar_flag+'.nc'
                    gg=glob.glob(outdir+filename)

                    """
                    if len(gg)>0:
                        print('Already saved '+filename)
                        continue
                    """
                        
                    xmls = xml_list[mo][ripf][exp]
                    if len(xmls)!=len(variables):
                        continue                        

                    skip='n'
                    for xml in xmls:
                        #if 'cltisccp' not in xml:
                        #    continue
                        if skip=='y':
                            continue
                        print('Reading in '+xml)
                        var = xml.split('.')[-7]
                        # Read in and regrid input data    
                        f=cdms.open(xml,'r')
                        try:
                            srcData = f(var,time=tslice) 
                            f.close()              
                        except:
                            f.close()   
                            moot
                            print('Could not load that in; bailing...')
                            skip='y'
                            continue           
                        if var=='clisccp':
                            cgrid = srcData.regrid(kern_grid,regridTool="esmf")
                            clisccp_grid,swapflag = MU.clisccp_axes_check(cgrid) # in %
                            if scalar_flag != 'grab':
                                pctisccp_grid,albisccp_grid,cltisccp_grid = compute_scalars(clisccp_grid[:,1:,:],ALB_midpt,CTP_midpt)      
                        elif var=='pctisccp':
                            if mo=='MPI-ESM-LR': # this one says its in Pa but its actually in hPa
                                pctisccp_grid = regridfunc(srcData) # do not convert to hPa
                            else:
                                pctisccp_grid = regridfunc(srcData)/100. # convert to hPa
                        else:
                            exec(var+'_grid = regridfunc(srcData)')
                    # end for xml in xmls    (all variables for this model/ripf/exp)
                    
                    if skip=='y':
                        toc = MU.toc(tic5)
                        print('--------Done with '+mip_era+'.'+mo+'.'+ripf+'.'+str(styr)+'-'+str(fnyr)+'.'+exp+' in '+str(np.round(toc/60,1))+' minutes')                    
                        continue
                     
                    # cltisccp_grid in %         

                    albcs = rsuscs_grid/rsdscs_grid
                    albcs = MV.masked_less(albcs, 0)
                    albcs = MV.masked_greater(albcs, 1)
                    TIME=albcs.getTime()
                    TIME.setBounds(TIME.genGenericBounds())
                    tas_yrs,tas_mos = MU.get_plottable_time(tas_grid)
                    YEARS = 1+np.int32(np.floor(tas_yrs[-1]) - np.floor(tas_yrs[0]))                    
                    
                    # sometimes only one element of the histogram is masked, and this wreaks havoc.
                    # I think this only happens where the rest of the histogram is 0, so set that bin to 0:
                    summask = np.sum(np.sum(clisccp_grid.mask,1),1) # [days,lat,lon]
                    newclisccp = MU.nanarray(clisccp_grid.shape)
                    for t in range(7):
                        for p in range(7):
                            this = np.ma.where(summask==1,0,clisccp_grid[:,t,p,:])
                            newclisccp[:,t,p,:] = np.ma.where(cltisccp_grid==0,0,this)
                    clisccp_grid = MV.array(newclisccp)  
                    del(summask)   

                    # Set all clear-sky points to +9999
                    pctisccp_grid = MV.where(cltisccp_grid==0,9999,pctisccp_grid)
                    albisccp_grid = MV.where(cltisccp_grid==0,9999,albisccp_grid)
                    cltisccp_grid = MV.where(cltisccp_grid==0,9999,cltisccp_grid)
                                                
                    # Mask everyone in the same place:
                    clt = np.sum(np.sum(clisccp_grid,1),1)
                    bigmask = (clt+pctisccp_grid+cltisccp_grid+albisccp_grid+rsdscs_grid+rsuscs_grid).mask
                    pctisccp_grid = MV.masked_where(bigmask,pctisccp_grid)
                    cltisccp_grid = MV.masked_where(bigmask,cltisccp_grid)
                    albisccp_grid = MV.masked_where(bigmask,albisccp_grid)
                    for T in range(7):
                        for P in range(7):
                            clisccp_grid[:,T,P,:] = MV.masked_where(bigmask,clisccp_grid[:,T,P,:])       

                    RAout = do_regime_analysis(pctisccp_grid,cltisccp_grid,albisccp_grid,clisccp_grid,albcs,LWkernel,SWkernel,normalize)
                    C, KLW, KSW, RFO, regime_CLT, regime_PCT, regime_ALB = RAout
                
                    RSDSCS = day_to_month_v2(rsdscs_grid,TIME)
                    RSUSCS = day_to_month_v2(rsuscs_grid,TIME)
                    tas = np.nanmean(day_to_month_v2(tas_grid,TIME),-1)
                    clisccp = day_to_month_v2(clisccp_grid,TIME)
                    
                    ALB = RSUSCS/RSDSCS
                    ALB = MV.masked_less(ALB, 0)
                    ALB = MV.masked_greater(ALB, 1)             
                
                    ################################################################
                    stuff_to_save={}      
                    print('Saving '+outdir+filename)
                    ff = cdms.open(outdir+filename,'w')
                    stuff_to_save['ALB'] = ALB
                    stuff_to_save['tas'] = tas
                    stuff_to_save['clisccp'] = clisccp
                    stuff_to_save['C'] = C
                    stuff_to_save['KLW'] = KLW
                    stuff_to_save['KSW'] = KSW
                    stuff_to_save['RFO'] = RFO
                    stuff_to_save['regime_CLT'] = regime_CLT
                    stuff_to_save['regime_PCT'] = regime_PCT
                    stuff_to_save['regime_ALB'] = regime_ALB    
                    for name in list(stuff_to_save.keys()):
                        tmp = stuff_to_save[name]
                        tmp.id = name
                        ff.write(tmp)
                        del(tmp)  
                    ff.history='Written by /home/zelinka1/scripts/cloud_regime_error_metric_part0_maps.py on feedback2.llnl.gov'
                    ff.close()
                    print(ff.id+' saved')
                    del(stuff_to_save)  
                    ################################################################                   

                    toc = MU.toc(tic5)
                    print('--------Done with '+mip_era+'.'+mo+'.'+ripf+'.'+str(styr)+'-'+str(fnyr)+'.'+exp+' in '+str(np.round(toc/60,1))+' minutes')                
                # end for e,exp in enumerate(exps):
                toc = MU.toc(tic4)
                print('------Done with '+mip_era+'.'+mo+'.'+ripf+'.'+str(styr)+'-'+str(fnyr)+' in '+str(np.round(toc/60,1))+' minutes')
            # end for styr in chunks            
            toc = MU.toc(tic3)
            print('----Done with '+mip_era+'.'+mo+'.'+ripf+' in '+str(np.round(toc/60,1))+' minutes')
        # end ripf loop
        toc = MU.toc(tic2)
        print('--Done with '+mip_era+'.'+mo+' in '+str(np.round(toc/60,1))+' minutes')
    # end model loop
    toc = MU.toc(tic1)
    print('Done with '+mip_era+' in '+str(np.round(toc/60,1))+' minutes')
# end MIP loop


##########################################################################
# If you are not summing over regimes, you don't need to multiply by RFO, just take an average
# straight monthly mean of X is equal to sum over regime of X(regime)*RFO(regime)
##########################################################################
