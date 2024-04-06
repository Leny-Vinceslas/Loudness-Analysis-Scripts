#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import os
import pwlf

def dB2Pa(dB):
    Paref=0.00002
    return 10**(dB/20)*Paref

def Pa2dB(Pa):
    Paref=0.00002
    return 20*np.log10(Pa/Paref)

def nrmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())/len(targets)

work_path=os.getcwd()
csv_path=work_path+"\\csv_data" 

# participantID='3634'
participantIDs=['6915','3634']
tests=['DP','GR']
grFreqs=['1414','4242']
sides=['L','R']

DP_file_names=[]
GR_file_names=[]
for ID in participantIDs:
    for test in tests:
        for side in sides:
            if test=='GR':
                for ff in grFreqs:
                    GR_file_names.append(test+'_'+side+'_'+ff+'_'+ID+'.csv')
            else:
                DP_file_names.append(test+'_'+side+'_'+ID+'.csv')

#%% read dpGram data

# df={}
DP_data=[]
OAEs=[]
for ID in participantIDs:
    DP_data=[]
    GR_data=[]
    OAEs.append({})
    OAEs[-1]['ID']=ID
    
    for DP_file in DP_file_names:
        if DP_file.find(ID) <0:
            data=pd.read_csv(csv_path+'\\'+DP_file)  

            df = pd.DataFrame(data)

            data={}
            data['ID']=ID
            data['side']=DP_file[3]
            data['f']=df['Freq (Hz)']
            data['f1']=df['F1 (dB)']
            data['f2']=df['F2 (dB)']	
            data['dp']=df['DP (dB)']	
            data['noise2']=df['Noise+2sd (dB)']	
            data['noise1']=df['Noise+1sd (dB)']
            data['f22_1']=df['2F2-F1 (dB)']
            data['f31_22']=df['3F1-2F2 (dB)']	
            data['f32_21']=df['3F2-2F1 (dB)']	
            data['f41_32']=df['4F1-3F2 (dB)']
            data['pdsnr']=df['DP (dB)']-df['Noise+1sd (dB)']
        
            DP_data.append(data)
    OAEs[-1]['DP']=DP_data  
    
    for GR_file in GR_file_names:
        if GR_file.find(ID) <0:
            data=pd.read_csv(csv_path+'\\'+GR_file)  

            df = pd.DataFrame(data)

            data={}
            data['ID']=ID
            data['side']=GR_file[3]
            data['f']=df['Freq (Hz)']
            data['f1']=df['F1 (dB)']
            data['f2']=df['F2 (dB)']	
            data['dp']=df['DP (dB)']	
            data['noise2']=df['Noise+2sd (dB)']	
            data['noise1']=df['Noise+1sd (dB)']
            data['f22_1']=df['2F2-F1 (dB)']
            data['f31_22']=df['3F1-2F2 (dB)']	
            data['f32_21']=df['3F2-2F1 (dB)']	
            data['f41_32']=df['4F1-3F2 (dB)']
            data['pdsnr']=df['DP (dB)']-df['Noise+1sd (dB)']
            
            GR_data.append(data)
    OAEs[-1]['GR']=GR_data



# %% Plot DPGRAM

# plt.figure()
# plt.figure(figsize=(15, 5))
colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
colors.insert(1,colors[3]) 
for OAE in OAEs:
    plt.figure(figsize=(15, 5))
    for ii, DP in enumerate(OAE['DP']):

        ylim=[-25,25]
        ax = plt.subplot(1, 2, ii + 1)
        ax.plot(DP['f'], DP['dp'],'o-',color=colors[ii],label='DP')
        ax.fill_between(DP['f'], DP['noise2'],ylim[0], color='gray', alpha=0.3)
        ax.plot(DP['f'], DP['noise1'],':',label='Noise 1sd')
        ax.plot(DP['f'], DP['noise2'],':',label='Noise 2sd')
        ax.set_xscale('log')
        ax.set_title('DPOAE, ID: '+DP['ID']+' Side: '+DP['side'])
        ax.set_xlabel('Frequency [kHz]')
        ax.set_ylabel('DP level [dB SPL]')
        ax.grid(True, which="both")
        ax.set_xticks(np.array([1, 2, 3, 4,5,6,7,8])*1000)
        ax.set_xticklabels(['1', '2', '3', '4','5','6','7','8'])
        ax.set_ylim(ylim[0], ylim[1])
        plt.legend()

    # plt.figure()
    # for ii, side in enumerate(sides):

    #     plt.plot(DP_data[ii]['f'], DP_data[ii]['dp'],'o-',label='DP '+side)
    #     # plt.plot(DP_data[ii]['f'], DP_data[ii]['noise1'],'o-',label='Noise+1sd')
    #     # plt.plot(DP_data[ii]['f'], DP_data[ii]['noise2'],'o-',label='Noise+2sd')

    # plt.grid('DPOAE, side: L+R')
    # plt.title('')
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('DP Level [dB SPL]')
    # plt.xscale('log')
    # plt.xticks(np.array([1, 2, 3, 4,5,6,7,8])*1000,['1', '2', '3', '4','5','6','7','8'])
    # # plt.xticklabels(['1', '2', '3', '4','5','6','7'])
    # plt.legend()
    # plt.grid(True, which="both")

#%%--------------------------------- Compute fittings 
# look at guinan and backus paper about DPgrowth function for fitting 
# OAEuPa=slope(l2-L2interscept)
# Pref = 20 microPascals
# es_DP80=20 log (SLOPE(80-L2Intercept)/Pref)

for OAE in OAEs:
    for ii, GR in enumerate(OAE['GR']):
        

        m,b = np.polyfit(GR['f2'], dB2Pa(GR['dp']), 1)
        GR['slope']=[m,b]
        
        # Square fit
        square=np.polyfit(GR['f2'], GR['dp'], 2)
        squareFitObj=np.poly1d(square)
        GR['squareFit']=[GR['f2'],squareFitObj(GR['f2'])]

        # Cubic fit fit
        cubic=np.polyfit(GR['f2'], GR['dp'], 3)
        cubicFitObj=np.poly1d(cubic)
        GR['cubicFit']=[GR['f2'],cubicFitObj(GR['f2'])]

        # Log fit 
        logFit=np.polyfit(GR['f2'], np.log10(np.sqrt(GR['dp']**2)), 1)
        logFitObj=np.poly1d(logFit)
        GR['logFit']=[GR['f2'],logFitObj(GR['f2'])]

        #Then compute the derivative 
        GR['derivative']=[GR['cubicFit'][0][:-1],np.diff(GR['cubicFit'][1])]
        
        # piecewise fit 

        # initialize piecewise linear fit with your x and y data
        my_pwlf = pwlf.PiecewiseLinFit(GR['f2'], dB2Pa(GR['dp']))
        #y2-y1 / x2-x1
        
        # define custom bounds for the interior break points
        n_segments = 3
        bounds = np.zeros((n_segments-1, 2))
        # first breakpoint
        bounds[0, 0] = 0.0  # lower bound
        bounds[0, 1] = 50  # upper bound
        # second breakpoint
        bounds[1, 0] = 50  # lower bound
        bounds[1, 1] = 65  # upper bound

        # breaks = my_pwlf.fit_guess([35, 55] )

        # fit the data for four line segments
        # res = my_pwlf.fit(3)
        res = my_pwlf.fitfast(3, pop=100)
        # res = my_pwlf.fitfast(n_segments, pop=100, bounds=bounds)
        GR['slopeValFit']=my_pwlf.slopes

        # predict for the determined points
        xHat = np.linspace(min(GR['f2']), max(GR['f2']), num=100)
        yHat = my_pwlf.predict(xHat)
        GR['fit']=[xHat,yHat]

        pwlf_error=nrmse(my_pwlf.predict(GR['f2']),GR['dp'])
        slope_error=nrmse(m*GR['f2']+b,GR['dp'])

#%%---------------------------- plot DP Growth Function




for OAE in OAEs:
    plt.figure(figsize=(15, 10))
    
    for ii, GR in enumerate(OAE['GR']):
        if GR['side']=='L' and GR['f'][0]==1416: n_sub=1
        if GR['side']=='R' and GR['f'][0]==1416: n_sub=2
        if GR['side']=='L' and GR['f'][0]==4248: n_sub=3
        if GR['side']=='R' and GR['f'][0]==4248: n_sub=4
   
        ax = plt.subplot(2, 2, n_sub)
        ax.plot(GR['f2'], GR['dp'],'o-',label='DP')
        ax.fill_between(GR['f2'], GR['noise2'],
                        min([min(GR['noise1']),min(GR['dp'])]), 
                        color='gray', alpha=0.3)
        ax.plot(GR['f2'], GR['noise1'],':',label='Noise 1sd')
        ax.plot(GR['f2'], GR['noise2'],':',label='Noise 2sd')
        
        # ax.plot(GR['squareFit'][0], GR['squareFit'][1],'--',label='Square fit')
        ax.plot(GR['cubicFit'][0], GR['cubicFit'][1],'--',label='Cubic fit')
        # ax.plot(GR['logFit'][0], GR['logFit'][1],'--',label='Log fit')
        ax.plot(GR['derivative'][0], GR['derivative'][1],'--',label='derivative')
        
        # ax.set_ylim(-25, 15) 
        ax.set_title(' DP I/O [dB]. ID:'+GR['ID']+ ', '+str(GR['f'][0])+ ' Hz, side: '+GR['side'])
        ax.set_xlabel('L2 level [dB SPL]')
        ax.set_ylabel('DP [dB SPL]')
        ax.grid(True)
        ax.legend()
    plt.show()
# plt.legend()


for OAE in OAEs:
    plt.figure(figsize=(15, 10))
    
    for ii, GR in enumerate(OAE['GR']):
        
        if GR['side']=='L' and GR['f'][0]==1416: n_sub=1
        if GR['side']=='R' and GR['f'][0]==1416: n_sub=2
        if GR['side']=='L' and GR['f'][0]==4248: n_sub=3
        if GR['side']=='R' and GR['f'][0]==4248: n_sub=4
        
        ax = plt.subplot(2, 2, n_sub)
        ax.plot(GR['f2'], dB2Pa(GR['dp']),'o-',label='DP')
        ax.plot(GR['f2'], dB2Pa(GR['noise1']),':',label='Noise 1sd')
        ax.plot(GR['f2'], dB2Pa(GR['noise2']),':',label='Noise 2sd')
        ax.fill_between(GR['f2'], dB2Pa(min([min(GR['noise1']),min(GR['dp'])])), dB2Pa(GR['noise2']), color='gray', alpha=0.3)
        ax.plot(GR['f2'], GR['slope'][0]*
                 GR['f2']+GR['slope'][1],'--',label='slope: '+ format(GR['slope'][0]*10**6,'.2f'))
        ax.plot(GR['fit'][0],GR['fit'][1],'--',label='PWLF: ' + format(GR['slopeValFit'][1]*10**6,'.2f'))
        # ax.set_xlim(xmin, xmax)
        # ax.set_ylim(-.5*10**-5, 20*10**-5) 
        ax.set_title(' DP I/O [µPa]. ID:'+GR['ID']+ ', '+str(GR['f'][0])+ ' Hz, side: '+GR['side'])
        ax.set_xlabel('L2 level [dB SPL]')
        ax.set_ylabel('DP [µPa]')
        ax.grid(True)
        ax.legend()
    plt.show()
# plt.legend()

# look at the aoes paper with the fitting
# Distortion Product Otoacoustic Emission (DPOAE) Growth in Aging Ears with Clinically Normal Behavioral Thresholds

# %%
