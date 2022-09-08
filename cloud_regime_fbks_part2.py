########################################################################
# Second part of the analysis used in Zelinka et al (2022) paper
# "Detailing cloud property feedbacks with a regime-based decomposition"
# Climate Dynamics
# https://doi.org/10.1007/s00382-022-06488-7
########################################################################

import cdutil
import glob
import gc
import matplotlib.pyplot as pl
import MDZ_utils as MU
import zelinka_analysis as ZA
import MV2 as MV
import numpy as np
import cdms2 as cdms

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
#exptype = 'abrupt-4xCO2'

# Decision 4 - Start and end years to consider
START = 2000#1991
END = 2008 # CMIP5 does not extend beyond 2008; most of CMIP6 does not extend beyond 2014 
########################################################################
# END USER INPUT
########################################################################

if exptype == 'amip':
    exps5 = ['amip']
    exps6 = ['amip']
elif exptype == 'p4K':
    exps5 = ['amip','amip4K']
    exps6 = ['amip','amip-p4K']
elif exptype == 'abrupt-4xCO2':
    exps5 = ['piControl','abrupt4xCO2']#['amip','amip4K']
    exps6 = ['piControl','abrupt-4xCO2']#['amip','amip-p4K']
    START = 2450
    END = START+8
    
figdir = '/home/zelinka1/figures/regimes/'+obsname+'/'
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

# Get lat/lon axis info from random file:
f=cdms.open('/p/user_pub/climate_work/zelinka1/cmip6/amip-p4K/cloud_feedback_maps_MIROC6.r1i1p1f1.nc')
AX0 = f('ALL_LWcld_tot').getLatitude()      
AX1 = f('ALL_LWcld_tot').getLongitude()      
f.close()

#MAKE CLASSIC NETCDF:
#===========================
cdms.setNetcdfShuffleFlag(0)
cdms.setNetcdfDeflateFlag(0)
cdms.setNetcdfDeflateLevelFlag(0)
    
    
########################################################################    
def notime(indict,anomT):
    # Function to either sum over time (for +4K response) or regress on tas (for amip)
    outdict={}
    names=list(indict.keys())
    for name in names:
        if exptype == 'amip': # Detrend the fields and regress on tas
            anomY,trend = MU.detrend(np.arange(lenmos),indict[name])
            outdict[name] = MU.linreg(anomT,anomY) # regress over month --> [lat,regime]
        else:
            # Avg over time:
            #outdict[name] = np.sum(indict[name],0) # sum over month --> [lat,regime]
            #outdict[name] = np.ma.average(indict[name],0) # avg over month --> [lat,regime]
            outdict[name] = np.ma.average(indict[name],0)/anomT # avg over month --> [lat,regime]
    return(outdict)
########################################################################                  


########################################################################                  
# START OF MAIN CODE
########################################################################                  
indir = '/p/user_pub/climate_work/zelinka1/cloud_regimes/'

mip_eras=['CMIP5','CMIP6']
for mip,mip_era in enumerate(mip_eras):
    tic1 = MU.tic()
    if mip_era=='CMIP5':
        exps = exps5
    else:
        exps = exps6
    source='*'
    ripf='*'
    styr = START
    fnyr = '*'    
    outdir = '/p/user_pub/climate_work/zelinka1/cmip'+mip_era[-1]+'/'+exps[-1]+'/'
    search = mip_era+'.'+exps[-1]+'.'+source+'.'+ripf+'.processed.maps.'+str(styr)+'-'+str(fnyr)+'.'+obsname+'.'+scalar_flag+'.nc'
    gg = glob.glob(indir+search) # should turn up one file per model/ripf
    models = []
    for g in gg:
        models.append(g.split('.')[2])
    sources = np.unique(models)        
    for mo in sources:
        tic2 = MU.tic()
        styr = '*'
        search = mip_era+'.'+exps[-1]+'.'+mo+'.'+ripf+'.processed.maps.'+str(styr)+'-'+str(fnyr)+'.'+obsname+'.'+scalar_flag+'.nc'
        hh = glob.glob(indir+search) # should turn up all the ripfs for this model
        ripfs0 = []
        for h in hh:
            ripfs0.append(h.split('.')[3])
        ripfs = np.unique(ripfs0)
        for member in ripfs:
            tic3 = MU.tic()
            """
            check=glob.glob(outdir+'ALL_cld_fbk_maps_'+obsname+'_'+scalar_flag+'_'+mo+'.'+member+'_fwd.nc')
            if len(check)>0:
                print('Already saved '+mo+'.'+member)
                continue            
            """
            if mo=='MRI-CGCM3' and member=='r5i1p3':
                continue
            if mo=='IPSL-CM6A-LR' and member=='r22i1p1f1':
                continue
                
            BIGDATA={}
            for exp in exps:
                tic4 = MU.tic()                
                BIGDATA[exp]={}
                search = mip_era+'.'+exp+'.'+mo+'.'+member+'.processed.maps.[!r]*.'+obsname+'.'+scalar_flag+'.nc'   
                # should turn up multiple files per model (which need to be spliced)        
                xmls0 = glob.glob(indir+search)
                xmls0.sort()
                xmls=[]
                for xml in xmls0:
                    yr0 = int(xml.split('.')[-4].split('-')[0])
                    yr1 = int(xml.split('.')[-4].split('-')[1])
                    
                    #========================================
                    # FOR NOW LETS TAKE ONLY THESE PERIODS
                    if yr0>=START and yr1<=END: 
                        xmls.append(xml)                     
                    #========================================  
                    
                tas=[]
                for xml in xmls: # this is looping over chunks of time
                    ff = cdms.open(xml,'r')
                    print('Reading in '+xml)
                    if len(tas)==0:
                        clisccp = ff('clisccp')         # [lenmos,TAU,CTP,90,144]
                        ALB = ff('ALB')                 # [lenmos,90,144]
                        tas = ff('tas')                 # [lenmos,90]
                        C = ff('C')                     # [nregimes,lenmos,TAU,CTP,90]
                        KLW = ff('KLW')                 # [nregimes,lenmos,TAU,CTP,90]
                        KSW = ff('KSW')                 # [nregimes,lenmos,TAU,CTP,90]
                        RFO = ff('RFO')                 # [nregimes,lenmos,90]
                        regime_CLT = ff('regime_CLT')   # [nregimes,90]
                        regime_PCT = ff('regime_PCT')   # [nregimes,90]
                        regime_ALB = ff('regime_ALB')   # [nregimes,90]
                    else: # splice these along the time axis
                        clisccp = np.ma.concatenate((clisccp, ff('clisccp')), axis=0)
                        ALB = np.ma.concatenate((ALB, ff('ALB')), axis=0)
                        tas = np.ma.concatenate((tas, ff('tas')), axis=0)
                        C = np.ma.concatenate((C, ff('C')), axis=1)
                        KLW = np.ma.concatenate((KLW, ff('KLW')), axis=1)
                        KSW = np.ma.concatenate((KSW, ff('KSW')), axis=1)
                        RFO = np.ma.concatenate((RFO, ff('RFO')), axis=1)
                    ff.close() 
                    
    
                # end loop over chunks of time
                # we now have concatenated matrices spanning the full period analyzed
                begin = xmls[0].split('.')[-4].split('-')[0]
                end = xmls[-1].split('.')[-4].split('-')[1]
                tag = begin+'-'+end

                # If doing +4K, compute monthly resolved climatologies
                print('Compute monthly-resolved climatologies')
                if exptype == 'amip': 
                    BIGDATA[exp]['clisccp'] = clisccp
                    BIGDATA[exp]['ALB'] = ALB
                    BIGDATA[exp]['tas'] = tas
                    BIGDATA[exp]['C'] = C
                    BIGDATA[exp]['KLW'] = KLW
                    BIGDATA[exp]['KSW'] = KSW
                    BIGDATA[exp]['RFO'] = RFO
                else:
                    for m in range(12):
                        # Compute RFO-weighted averages over all years                        
                        that = RFO[:,m::12,:]
                        this = np.moveaxis(np.moveaxis(C[:,m::12,:],3,0),3,0)
                        prod = np.ma.sum(this*that,-3)/np.ma.sum(that,-3)
                        C[:,m,:] = np.moveaxis(np.moveaxis(prod,0,2),0,2)
                        #C[:,m,:] = np.ma.average(C[:,m::12,:],1)
                        
                        this = np.moveaxis(np.moveaxis(KLW[:,m::12,:],3,0),3,0)
                        prod = np.ma.sum(this*that,-3)/np.ma.sum(that,-3)
                        KLW[:,m,:] = np.moveaxis(np.moveaxis(prod,0,2),0,2)
                        #KLW[:,m,:] = np.ma.average(KLW[:,m::12,:],1)
                        
                        this = np.moveaxis(np.moveaxis(KSW[:,m::12,:],3,0),3,0)
                        prod = np.ma.sum(this*that,-3)/np.ma.sum(that,-3)
                        KSW[:,m,:] = np.moveaxis(np.moveaxis(prod,0,2),0,2)
                        #KSW[:,m,:] = np.ma.average(KSW[:,m::12,:],1)
                        
                        RFO[:,m,:] = np.ma.average(RFO[:,m::12,:],1)
                        
                        clisccp[m,:] = np.ma.average(clisccp[m::12,:],0)
                        ALB[m,:] = np.ma.average(ALB[m::12,:],0)
                        tas[m,:] = np.ma.average(tas[m::12,:],0)
                        
                    BIGDATA[exp]['clisccp'] = clisccp[:12,:]
                    BIGDATA[exp]['ALB'] = ALB[:12,:]
                    BIGDATA[exp]['tas'] = tas[:12,:]
                    BIGDATA[exp]['C'] = C[:,:12,:]
                    BIGDATA[exp]['KLW'] = KLW[:,:12,:]
                    BIGDATA[exp]['KSW'] = KSW[:,:12,:]
                    BIGDATA[exp]['RFO'] = RFO[:,:12,:]
                BIGDATA[exp]['regime_CLT'] = regime_CLT
                BIGDATA[exp]['regime_PCT'] = regime_PCT
                BIGDATA[exp]['regime_ALB'] = regime_ALB

                toc = MU.toc(tic4)
                print('------Done reading in '+mip_era+'.'+mo+'.'+member+'.'+exp+' in '+str(np.round(toc/60,1))+' minutes')                    
            # end loop over exps


            print('Stack the control and perturbed experiments')
            
            nregimes = BIGDATA[exp]['C'].shape[0]-1 # minus 1 because the last regime is clear-sky
            lenmos = BIGDATA[exp]['C'].shape[1]
            lenyrs = np.int32(lenmos/12)
            
            clisccp =       MU.nanarray((2,         lenmos,7,7,90,144))
            ALB =           MU.nanarray((2,         lenmos,    90,144))
            tas =           MU.nanarray((2,         lenmos,    90)) # accidentally saved the zonal mean
            C =             MU.nanarray((2,nregimes+1,lenmos,7,7,90,144))
            KLW =           MU.nanarray((2,nregimes+1,lenmos,7,7,90,144))
            KSW =           MU.nanarray((2,nregimes+1,lenmos,7,7,90,144))
            RFO =           MU.nanarray((2,nregimes+1,lenmos,    90,144))
            #RFOfull =       MU.nanarray((2,nregimes+1,lenmos,  90,144)) # use this for maps that include clear-sky regime
            regime_CLT =    MU.nanarray((2,nregimes+1,90,144))
            regime_PCT =    MU.nanarray((2,nregimes+1,90,144))
            regime_ALB =    MU.nanarray((2,nregimes+1,90,144))
            
            clisccp[0,:] =      BIGDATA[exps[0]]['clisccp']
            ALB[0,:] =          BIGDATA[exps[0]]['ALB']
            tas[0,:] =          BIGDATA[exps[0]]['tas']
            C[0,:] =            BIGDATA[exps[0]]['C']#[:-1,:] # leave off the last bin which is the clear-sky regime
            KLW[0,:] =          BIGDATA[exps[0]]['KLW']#[:-1,:]
            KSW[0,:] =          BIGDATA[exps[0]]['KSW']#[:-1,:]
            RFO[0,:] =          BIGDATA[exps[0]]['RFO']#[:-1,:]
            #RFOfull[0,:] =      BIGDATA[exps[0]]['RFO']
            regime_CLT[0,:] =   BIGDATA[exps[0]]['regime_CLT']#[:-1,:]
            regime_PCT[0,:] =   BIGDATA[exps[0]]['regime_PCT']#[:-1,:]
            regime_ALB[0,:] =   BIGDATA[exps[0]]['regime_ALB']#[:-1,:]
        
            clisccp[1,:] =      BIGDATA[exps[-1]]['clisccp']
            ALB[1,:] =          BIGDATA[exps[-1]]['ALB']
            tas[1,:] =          BIGDATA[exps[-1]]['tas']
            C[1,:] =            BIGDATA[exps[-1]]['C']#[:-1,:] # leave off the last bin which is the clear-sky regime
            KLW[1,:] =          BIGDATA[exps[-1]]['KLW']#[:-1,:]
            KSW[1,:] =          BIGDATA[exps[-1]]['KSW']#[:-1,:]
            RFO[1,:] =          BIGDATA[exps[-1]]['RFO']#[:-1,:]
            #RFOfull[1,:] =      BIGDATA[exps[-1]]['RFO']
            regime_CLT[1,:] =   BIGDATA[exps[-1]]['regime_CLT']#[:-1,:]
            regime_PCT[1,:] =   BIGDATA[exps[-1]]['regime_PCT']#[:-1,:]
            regime_ALB[1,:] =   BIGDATA[exps[-1]]['regime_ALB']#[:-1,:]

            del(BIGDATA)
        
            ##########################################################################
            # If you are not summing over regimes, you don't need to multiply by RFO, just take an average
            # straight monthly mean of X is equal to sum over regime of X(regime)*RFO(regime)
            ##########################################################################

            
            print('Compute climate change response')
            ###############################################
            # COMPUTE CLIMATE CHANGE RESPONSE            
            if exptype == 'amip': # if we are dealing with amip (interannual) feedbacks,
                # index 0 currently contains total, want climo in 0, total in 1
                for m in range(12):
                    that = RFO[0,:,m::12,:]
                    # Compute RFO-weighted averages over all years
                    this = np.moveaxis(np.moveaxis(C[0,:,m::12,:],3,0),3,0)
                    prod = np.ma.sum(this*that,-3)/np.ma.sum(that,-3)
                    Cavg = np.moveaxis(np.moveaxis(prod,0,2),0,2)
                    for tt in range(7):
                        for pp in range(7):                       
                            #this = np.average(C[0,:,m::12,tt,pp,:],1) # [regime,lat,lon]
                            this = Cavg[:,tt,pp,:] # [regime,lat,lon]
                            C[0,:,m::12,tt,pp,:] = np.tile(np.expand_dims(this,axis=1),(1,lenyrs,1,1))
                            this = np.average(clisccp[0,m::12,tt,pp,:],0) # [lat,lon]
                            clisccp[0,m::12,tt,pp,:] = np.tile(np.expand_dims(this,axis=0),(lenyrs,1,1))
                    this = np.average(tas[0,m::12,:],0) # [,lat]
                    tas[0,m::12,:] = np.tile(this,(lenyrs,1))                    
                    this = np.average(RFO[0,:,m::12,:],1) # [regime,lat,lon]
                    RFO[0,:,m::12,:] = np.tile(np.expand_dims(this,axis=1),(1,lenyrs,1,1))
            dtas = tas[1,:] - tas[0,:]
            dtas.setAxis(1,lats)
            dC = C[1,:] - C[0,:]
            dRFO = RFO[1,:] - RFO[0,:]
            #dRFOfull = RFOfull[1,:] - RFOfull[0,:]
            dclisccp = clisccp[1,:] - clisccp[0,:]
            if exptype == 'amip': # if we are dealing with amip (interannual) feedbacks,
                DT = cdutil.averager(dtas,axis='y',weights='generate')
                anomT,trend = MU.detrend(np.arange(lenmos),DT)
                # for amip (interannual) feedbacks we will normalize by DT later
            else:
                DT = MV.average(cdutil.averager(dtas,axis='y',weights='generate'),0)
                anomT = DT
            ###############################################
            
            # Where RFO==0, make the clouds=0
            for p in range(7):
                for t in range(7):
                    C[:,:,:,t,p,:] = MV.where(RFO==0,0,C[:,:,:,t,p,:])
                    KSW[:,:,:,t,p,:] = MV.where(RFO==0,0,KSW[:,:,:,t,p,:])
                    KLW[:,:,:,t,p,:] = MV.where(RFO==0,0,KLW[:,:,:,t,p,:])
                    
            C.setAxis(-2,lats)
            C.setAxis(-1,lons)
            CRhist = cdutil.averager(MV.average(C[0,:],1),axis='xy',weights='generate')                           
                                                            
            if exptype == 'amip': # Detrend the fields and regress on tas
                anomY,trend = MU.detrend(np.arange(lenmos),np.moveaxis(dC,1,0))
                DATA = MU.linreg(anomT,anomY) # regress over month --> [lat,]
            else: 
                DATA = MV.average(dC,1)/DT        # average over months
            DATA.setAxis(-2,lats)
            DATA.setAxis(-1,lons)        
            dC_to_save = cdutil.averager(DATA,axis='xy',weights='generate')           
            
            # Map the SW kernel to the appropriate longitudes
            #SWK = np.nanmean(ZA.map_SWkern_to_lon(SWkernel,ALB[0,:]),-1) # function expects Ksw to be [12,7,7,lats,3] and albcsmap to be [time,lats,lons]
            SWK = ZA.map_SWkern_to_lon(SWkernel,ALB[0,:]) # function expects Ksw to be [12,7,7,lats,3] and albcsmap to be [time,lats,lons]
            sundown = np.isnan(SWK[:,0,0,:])
            LWK = np.tile(np.expand_dims(LWkernel[...,0],axis=-1),144)

            # repeat the kernel to span the full range of months , if necessary:
            indices=np.tile(np.arange(12),(lenyrs,))
            LWK = np.take(LWK,indices,axis=0)       # [month,TAU,CTP,lat,lon,regime]
            KLW = np.take(KLW,indices,axis=2)       # [month,TAU,CTP,lat,lon,regime]
            KSW = np.take(KSW,indices,axis=2)       # [month,TAU,CTP,lat,lon,regime]


            RFO_to_save = np.average(RFO[0,:,:12,:],1) # avg over months

            if exptype == 'amip': # Detrend the fields and regress on tas
                anomY,trend = MU.detrend(np.arange(lenmos),np.moveaxis(dRFO,1,0))
                DATA = MU.linreg(anomT,anomY) # regress over month --> [lat,lon,]          
                bounds = np.arange(-0.5,0.6,0.1) 
            else:          
                DATA = np.average(dRFO,1)/DT # average over months
                bounds = np.arange(-0.05,0.06,0.01)  
            dRFO_to_save = DATA

            print('Decompose the within- and between-regime components into amt/alt/tau')
           
            ####################################################################################
            # Decompose the within- and between-regime components into amt/alt/tau:
            ####################################################################################
            c0 = np.moveaxis(C[0,:],0,-1)            # [month,TAU,CTP,lat,lon,regime]
            c1 = np.moveaxis(C[1,:],0,-1)            # [month,TAU,CTP,lat,lon,regime]
            Klw = np.moveaxis(KLW[0,:],0,-1)         # [12,TAU,CTP,lat,lon,regime]
            Ksw = np.moveaxis(KSW[0,:],0,-1)         # [12,TAU,CTP,lat,lon,regime]
            
            # no breakdown -- should equal within + between + covariance
            b = np.moveaxis(np.moveaxis(C[0,:],3,0),3,0)    # [TAU,CTP,regime,month,lat,lon]
            bRFO = np.moveaxis(b*RFO[0,:],3,0)              # [month,TAU,CTP,regime,lat,lon]
            C1N1 = np.moveaxis(bRFO,3,-1)                   # [month,TAU,CTP,lat,lon,regime]
            b = np.moveaxis(np.moveaxis(C[1,:],3,0),3,0)    # [TAU,CTP,regime,month,lat,lon]
            bRFO = np.moveaxis(b*RFO[1,:],3,0)              # [month,TAU,CTP,regime,lat,lon]
            C2N2 = np.moveaxis(bRFO,3,-1)                   # [month,TAU,CTP,lat,lon,regime]
            control = C1N1
            dc = C2N2 - C1N1
            pert = control + dc
            output = ZA.KT_decomposition_general(control,pert,Klw,Ksw) # expects input [month,TAU,CTP,lat,lon?]
            NOBREAK = notime(output,anomT)
            
            # "within regime" feedbacks
            b = np.moveaxis(np.moveaxis(dC,3,0),3,0)    # [TAU,CTP,regime,month,lat,lon]
            within0 = np.moveaxis(b*RFO[0,:],3,0)         # [month,TAU,CTP,regime,lat,lon]
            within = np.moveaxis(within0,3,-1)          # [month,TAU,CTP,lat,lon,regime]
            
            # Standard "between regime" feedbacks -- only the amount component is nonzero by definition 
            b = np.moveaxis(np.moveaxis(C[0,:],3,0),3,0)    # [TAU,CTP,regime,month,lat,lon]
            bdRFO = np.moveaxis(b*dRFO,3,0)               # [month,TAU,CTP,regime,lat,lon]
            between = np.moveaxis(bdRFO,3,-1)               # [month,TAU,CTP,lat,lon,regime]
            
            # "covariance" feedbacks
            b = np.moveaxis(np.moveaxis(dC,3,0),3,0)    # [TAU,CTP,regime,month,lat,lon]
            bdRFO = np.moveaxis(b*dRFO,3,0)        # [month,TAU,CTP,regime,lat,lon]
            covary = np.moveaxis(bdRFO,3,-1)    # [month,TAU,CTP,lat,lon,regime]
            
            # Alternative "between regime" feedbacks
            # express CdRFOK as the sum of the following 4 components:
            # Cbar*dRFO*Kbar + Cbar*dRFO*K' + C'dRFOKbar  + C'dRFOK'
            # where X' = X - Xbar and bar means average across both regime and month
            b = np.moveaxis(np.moveaxis(C[0,:-1,:],3,0),3,0)  # DO NOT INCLUDE THE CLEAR-SKY REGIME IN AVG
            # weighted sum over regime [2], but straight avg over time [3]:
            avg = np.ma.mean(np.sum(b*RFO[0,:-1,:],2,keepdims=True),3,keepdims=True) # DO NOT INCLUDE THE CLEAR-SKY REGIME IN AVG
            Cbar = np.moveaxis(np.moveaxis(avg,0,3),0,3)    # [1,1,TAU,CTP,lat,lon]
            Cprime = C[0,:] - Cbar                  # C minus C averaged across regimes
            Cprime[-1,:] = 0 # ENFORCE THAT THE CLEAR SKY REGIME IS ZERO
            
            # for this "between-bar" term, use a regime- and monthly-averaged kernel instead
            this = np.ma.mean(np.ma.mean(Ksw[...,:-1],0,keepdims=True),-1,keepdims=True)    # [1,TAU,CTP,lat,lon,1]
            Kswbar = np.moveaxis(np.tile(np.moveaxis(this,0,-1),(lenmos,nregimes+1)),-2,0)
            this = np.mean(np.mean(Klw[...,:-1],0,keepdims=True),-1,keepdims=True)    # [1,TAU,CTP,lat,lon,1]
            Klwbar = np.moveaxis(np.tile(np.moveaxis(this,0,-1),(lenmos,nregimes+1)),-2,0)
                        
            Kswprime = Ksw - Kswbar
            Klwprime = Klw - Klwbar
            
            b = np.moveaxis(np.moveaxis(Cprime,3,0),3,0)    # [TAU,CTP,regime,month,lat,lon]
            bdRFO = np.moveaxis(b*dRFO,3,0)                     # [month,TAU,CTP,regime,lat,lon]
            between_prime = np.moveaxis(bdRFO,3,-1)           # [month,TAU,CTP,lat,lon,regime]
            between_prime[...,-1] = 0 # ENFORCE THAT THE CLEAR SKY REGIME IS ZERO
            
            b = np.moveaxis(np.moveaxis(Cbar,3,0),3,0)      # [TAU,CTP,regime,month,lat,lon]
            bdRFO = np.moveaxis(b*dRFO,3,0)                     # [month,TAU,CTP,regime,lat,lon]
            between_bar = np.moveaxis(bdRFO,3,-1)             # [month,TAU,CTP,lat,lon,regime]
            between_bar[...,-1] = 0 # ENFORCE THAT THE CLEAR SKY REGIME IS ZERO
            
            sections=['ALL','HI680','LO680']
            Psections =[slice(0,7),slice(2,7),slice(0,2)]
            sec_dic=dict(zip(sections,Psections))  

            for sec in sections:  
                if sec!='ALL':
                    continue
                PP=sec_dic[sec] 
                
                names = ['LWcld_tot','LWcld_amt','LWcld_alt','LWcld_tau','LWcld_err','SWcld_tot','SWcld_amt','SWcld_alt','SWcld_tau','SWcld_err']                
                c2 = c0 + between
                output = ZA.KT_decomposition_general(c0[:,:,PP,:],c2[:,:,PP,:],Klw[:,:,PP,:],Ksw[:,:,PP,:]) # expects input [month,TAU,CTP,lat,lon?]
                BETWEEN = notime(output,anomT)
                
                c2 = c0 + within
                output = ZA.KT_decomposition_general(c0[:,:,PP,:],c2[:,:,PP,:],Klw[:,:,PP,:],Ksw[:,:,PP,:]) # expects input [month,TAU,CTP,lat,lon?]
                WITHIN = notime(output,anomT)

                c2 = c0 + covary
                output = ZA.KT_decomposition_general(c0[:,:,PP,:],c2[:,:,PP,:],Klw[:,:,PP,:],Ksw[:,:,PP,:]) # expects input [month,TAU,CTP,lat,lon?]
                COVARY = notime(output,anomT)
                    
                # Now do the four alternative between-regime components 
                # 1) Cbar*dRFO*Kbar
                c2 = c0 + between_bar
                output = ZA.KT_decomposition_general(c0[:,:,PP,:],c2[:,:,PP,:],Klwbar[:,:,PP,:],Kswbar[:,:,PP,:]) # expects input [month,TAU,CTP,lat,lon?]
                BETWEEN_CbarKbar = notime(output,anomT)
                    
                # 2) Cbar*dRFO*K'
                c2 = c0 + between_bar
                output = ZA.KT_decomposition_general(c0[:,:,PP,:],c2[:,:,PP,:],Klwprime[:,:,PP,:],Kswprime[:,:,PP,:]) # expects input [month,TAU,CTP,lat,lon?]
                BETWEEN_CbarKprime = notime(output,anomT)
                               
                # 3) C'dRFOKbar     
                c2 = c0 + between_prime
                output = ZA.KT_decomposition_general(c0[:,:,PP,:],c2[:,:,PP,:],Klwbar[:,:,PP,:],Kswbar[:,:,PP,:]) # expects input [month,TAU,CTP,lat,lon?]
                BETWEEN_CprimeKbar = notime(output,anomT)
                             
                # 4) C'dRFOK'         
                c2 = c0 + between_prime
                output = ZA.KT_decomposition_general(c0[:,:,PP,:],c2[:,:,PP,:],Klwprime[:,:,PP,:],Kswprime[:,:,PP,:]) # expects input [month,TAU,CTP,lat,lon?]                
                BETWEEN_CprimeKprime = notime(output,anomT)
                    
                output = ZA.KT_decomposition_general(clisccp[0,:,:,PP,:],clisccp[1,:,:,PP,:],LWK[:,:,PP,:],SWK[:,:,PP,:]) # expects input [month,TAU,CTP,lat,lon?]               
                TRUTH = notime(output,anomT)
                
                LW_TRUTH = np.ma.sum(np.ma.sum(dclisccp[:,:,PP,:] * LWK[:,:,PP,:],1),1) # [12,90,144]
                SW_TRUTH = np.ma.sum(np.ma.sum(dclisccp[:,:,PP,:] * SWK[:,:,PP,:],1),1) # [12,90,144]
                # set to zero in the polar night
                SW_TRUTH = MV.where(sundown,0,SW_TRUTH)
                if exptype == 'amip': # Detrend the fields and regress on tas
                    anomY,trend = MU.detrend(np.arange(lenmos),SW_TRUTH)
                    SW_TRUTH = MU.linreg(anomT,anomY) # regress over month --> [lat,]
                    anomY,trend = MU.detrend(np.arange(lenmos),LW_TRUTH)
                    LW_TRUTH = MU.linreg(anomT,anomY) # regress over month --> [lat,]
                else: 
                    SW_TRUTH = np.average(SW_TRUTH,0)/DT # average over mon --> [lat,lon]
                    LW_TRUTH = np.average(LW_TRUTH,0)/DT # average over mon --> [lat,lon]
                
                # Output the regime-resolved data to NC file  
                # set up CR axis info:
                AX2 = cdms.createAxis(np.arange(nregimes+1))
                AX2.id="cloud_regime" 
                AX2.comment="CRs corresponding to "+obsname+"_CRs" 
                AXL = [AX0,AX1,AX2]

                savefile = outdir+sec+'_cld_fbk_maps_'+obsname+'_'+scalar_flag+'_'+mo+'.'+member+'_fwd.nc'
                out1 = cdms.open(savefile,'w')    
                for name in names:
                    tmp = WITHIN[name]
                    tmp.id = sec+'_'+name+'_within'
                    tmp.setAxisList(AXL)
                    out1.write(tmp)
                    del(tmp)    
                    tmp = BETWEEN[name]
                    tmp.id = sec+'_'+name+'_between'
                    tmp.setAxisList(AXL)
                    out1.write(tmp)
                    del(tmp)   
                    tmp = BETWEEN_CbarKbar[name]
                    tmp.id = sec+'_'+name+'_between_CbarKbar'
                    tmp.setAxisList(AXL)
                    out1.write(tmp)
                    del(tmp)    
                    tmp = BETWEEN_CbarKprime[name]
                    tmp.id = sec+'_'+name+'_between_CbarKprime'
                    tmp.setAxisList(AXL)
                    out1.write(tmp)
                    del(tmp)     
                    tmp = BETWEEN_CprimeKbar[name]
                    tmp.id = sec+'_'+name+'_between_prime' # Cprime*Kbar <-- dominant term
                    tmp.setAxisList(AXL)
                    out1.write(tmp)
                    del(tmp)    
                    tmp = BETWEEN_CprimeKprime[name]
                    tmp.id = sec+'_'+name+'_between_CprimeKprime'
                    tmp.setAxisList(AXL)
                    out1.write(tmp)
                    del(tmp)    
                    tmp = COVARY[name]
                    tmp.id = sec+'_'+name+'_covary'
                    tmp.setAxisList(AXL)
                    out1.write(tmp)
                    del(tmp)              
                    

                # And the TRUE feedback (which does not involve regime decomposition at all)
                tmp = LW_TRUTH
                tmp.id = 'LW_TRUTH'
                tmp.setAxisList((AX0,AX1)) # [lat,lon]
                out1.write(tmp)
                del(tmp)  
                tmp = SW_TRUTH
                tmp.id = 'SW_TRUTH'
                tmp.setAxisList((AX0,AX1)) # [lat,lon]
                out1.write(tmp)
                del(tmp)  
                
                # The following do not depend on section:
                if sec=='ALL':    
                    # RFO
                    tmp = np.moveaxis(RFO_to_save,0,-1) # [lat,lon,nregimes]
                    tmp.setAxisList(AXL)
                    tmp.id = 'RFO'
                    out1.write(tmp)
                    del(tmp)      
                    
                    # dRFO
                    tmp = np.moveaxis(dRFO_to_save,0,-1) # [lat,lon,nregimes]
                    tmp.setAxisList(AXL)
                    tmp.id = 'dRFO'
                    out1.write(tmp)
                    del(tmp)  
                    
                    # CRhist
                    tmp = CRhist # [nregimes,7,7]
                    tmp.setAxis(0,AX2)
                    tmp.setAxis(1,SWkernel.getAxis(1))
                    tmp.setAxis(2,SWkernel.getAxis(2))
                    tmp.id = 'CRhist'
                    out1.write(tmp)
                    del(tmp) 
                    
                    # dCRhist
                    tmp = dC_to_save # [nregimes,7,7]
                    tmp.setAxis(0,AX2)
                    tmp.setAxis(1,SWkernel.getAxis(1))
                    tmp.setAxis(2,SWkernel.getAxis(2))
                    tmp.id = 'dCRhist'
                    out1.write(tmp)
                    del(tmp)     
                
                    # Regime-mean cloud properties
                    tmp = regime_ALB
                    tmp.id = 'regime_ALB'
                    tmp.setAxis(-1,AX1) # [exps,nregimes,lat,lon]
                    tmp.setAxis(-2,AX0) # [exps,nregimes,lat,lon]
                    out1.write(tmp)
                    del(tmp)  
                    tmp = regime_PCT
                    tmp.id = 'regime_PCT'
                    tmp.setAxis(-1,AX1) # [exps,nregimes,lat,lon]
                    tmp.setAxis(-2,AX0) # [exps,nregimes,lat,lon]
                    out1.write(tmp)
                    del(tmp)  
                    tmp = regime_CLT
                    tmp.id = 'regime_CLT'
                    tmp.setAxis(-1,AX1) # [exps,nregimes,lat,lon]
                    tmp.setAxis(-2,AX0) # [exps,nregimes,lat,lon]
                    out1.write(tmp)
                    del(tmp)  
                                         
                    tmp = anomT
                    tmp.id = 'anomT'
                    out1.write(tmp)
                    del(tmp) 

                out1.timeperiod = tag
                out1.history='Written by /home/zelinka1/scripts/cloud_regime_error_metric_part1.py on feedback2.llnl.gov'
                out1.close()
                print(out1.id+' saved')
                
            # end CTP section loop
        
                        
            gc.collect()
            pl.close('all')

            toc = MU.toc(tic3)
            print('----Done with '+mip_era+'.'+mo+'.'+member+' in '+str(np.round(toc/60,1))+' minutes')
        # end member loop
        toc = MU.toc(tic2)
        print('--Done with '+mip_era+'.'+mo+' in '+str(np.round(toc/60,1))+' minutes')
    # end model loop
    toc = MU.toc(tic1)
    print('Done with '+mip_era+' in '+str(np.round(toc/60,1))+' minutes')
# end MIP loop

