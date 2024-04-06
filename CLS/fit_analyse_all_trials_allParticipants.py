
#%% Load Modules and definitions
import parseXML
import os
import numpy as np
import fit_loudness_function
import numpy as np 
import os
import scipy.io as spio
import pwlf_bez_collection
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

work_path=os.getcwd()
xml_path=work_path+"\\xml_data" 
participantIDs=['6915','3634']
tests=['CLS']
# xmlfile_names=["CLS_3634.xml"]
# xmlfile_names=["CLS_6915.xml"]
save_path=work_path+"\\save_matfiles" 
xmlfile_names=[]

for ID in participantIDs:
    for test in tests:
        xmlfile_names.append(test+'_'+ID+'.xml')

def x2y_lin(x, x0, y0, m):
    return y0 + m*(x-x0)

def y2x_lin(y, x0, y0, m):
    return (y-y0)/m + x0

def loudnessFunc(x,L):
    m_l=x[0]
    m_h=x[1]
    L_cut=x[2] 

    L_15=y2x_lin(15, L_cut, 25, m_l)
    L_35=y2x_lin(35, L_cut, 25, m_h)

    F_L=np.zeros(len(L))
    for i,Li in enumerate(L):
        if Li<=L_15:
            F_L[i]=25+m_l*(Li-L_cut)
        if Li>=L_35:
            F_L[i]=25+m_h*(Li-L_cut)
    for i,Li in enumerate(L):
        if L_15<Li<L_35:
            F_L[i]=bezSmoothing(Li,L_cut,L_15,L_35)
            # F_L[i]=0
            
    return F_L 

def bezSmoothing(L,L_cut,L_15,L_35):
    x0, y0 = L_15, 15
    x1 = 2 * L_cut - 2 * L_15
    y1 = 2 * 25 - 2 * 15
    x2 = L_15 - 2 * L_cut + L_35
    y2 = 15 - 2 * 25 + 35 
    xk, yk = x1, y1
    xa, ya = x0, y0
    xb, yb = x2, y2
    if ((yk-ya)/(xk-xa)) < ((yk-yb)/(xk-xb)):
        #Is the sign correct here for the bez smothing ???? 
        t = -( x1 / ( 2*x2 )) - np.sqrt((( L - x0 ) / x2) + ( x1**2 / ( 4 * x2**2 )))
        # print('1')
    elif ((yk-ya)/(xk-xa)) > ((yk-yb)/(xk-xb)):
        t = -( x1 / ( 2*x2 )) - np.sqrt(( L - x0 ) / x2 + ( x1**2 / ( 4 * x2**2 )))
        # print('2')
    
    return y0 + y1 * t + y2 * t**2

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def nrmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())/len(targets)


#%% 
# Class to parse XML and import blocks obj-------------
blockFromCls = parseXML.Cls2dic()
CLSs=[] 

for ID in participantIDs:

    file_name=(test+'_'+ID+'.xml')
    blocks=[]
    CLSs.append({})
    CLSs[-1]['ID']=ID

    blocks=blockFromCls.parse(path = xml_path + '\\' + file_name)
    CLSs[-1]['Blocks']=blocks  
    print("-----> xml data parsed:", str(file_name))

# %% 
# Perform all the fit----------------------------------

# min_methods=['MAT','NEL','CG','SLS','TRU']
# min_methods=['NEL','CG','SLS','TRU']
# # min_methods=['CG']
# fit_methods=['BX','BY','BTX','BTUX','BTUY','BTPX']
min_methods=['NEL']
fit_methods=['BTUX']
data=[]
# blocks=blocks[3:5]
for CLS in CLSs:
    for ii, block in enumerate(CLS['Blocks']):
        data=np.array([block['Level'], block['CU'],]).flatten(order='F')
        print('------- Fitting block '+str(ii)+' from '+CLS['ID']+'---------')

        for mm, min_method in enumerate(min_methods):

            for ff, fit_method in enumerate(fit_methods):
                fit = np.array(fit_loudness_function.fit_loudness_function(data, fit_method,min_method))

                print(min_method + '_' + fit_method + ': ' + str(fit)) 
                block[min_method + '_' + fit_method]=fit
# %%
# Compute pwlf Bz and RSME--------------------------
# min_methods=['MAT','NEL','CG','SLS','TRU']
# min_methods=['NEL','CG','SLS','TRU']
# min_methods=['NEL']
# fit_methods=['BX','BY','BTX','BTUX','BTUY','BTPX']
markers=["o","*","+","x","v","^","<",">","s","p","P","h","H"]

for CLS in CLSs:
    for ii, block in enumerate(CLS['Blocks']):
        nPoint=100
        
        try:
            nPointBez=len(block['Data_fit_level'])
            slopeL=float(block['SlopeL'])
            slopeU=float(block['SlopeU'])

            # recover broken sticks from intersection and slopes from xml data
            intersect=float(block['Intersection'])

            yu=50 
            yl=0
            xu=y2x_lin(yu,intersect,25,slopeU)
            xl=y2x_lin(yl,intersect,25,slopeL)
            x15=y2x_lin(15,intersect,25,slopeL)
            x35=y2x_lin(35,intersect,25,slopeU)

            #Interpolate from the 3 points
            block['Pwlf-Artc']=[np.linspace(xl, xu, num=nPoint),
                                np.interp(np.linspace(xl, xu, num=nPoint), [xl,intersect,xu], [yl,25,yu])]

            block['PwlfBz-Artc']=[np.linspace(xl, xu, num=nPoint),loudnessFunc([slopeL,slopeU,intersect],np.linspace(xl, xu, num=nPoint))]
            
            block['RMSE_Artc']=rmse(block['CU'],loudnessFunc([slopeL,slopeU,intersect],(block['Level'])))
            block['NRMSE_Artc']=nrmse(block['CU'],loudnessFunc([slopeL,slopeU,intersect],(block['Level'])))
            
        except:
            print('Auralec data missing')
        
        pwlf1=pwlf_bez_collection.Pwlf_bez()
        y_l,res_lsq=pwlf1.Pwlf_bez_BrandT_true(block['Level'],block['CU'])
        block['fit_BY_own']=[y_l[2],y_l[0],y_l[1]]

        keys_plot_list=[]
        dataPlots=[]
        dataPlotsBz=[]
        
        
        for mm, min_method in enumerate(min_methods):

            for ff, fit_method in enumerate(fit_methods):
                key=min_method + '_' + fit_method

                intersect=block[key][0]
                slopeL=block[key][1]
                slopeU=block[key][2]

                xu=y2x_lin(yu,intersect,25,slopeU)
                xl=y2x_lin(yl,intersect,25,slopeL)

                
                x15=y2x_lin(15,intersect,25,slopeL)
                x35=y2x_lin(35,intersect,25,slopeU)

                # key_plot='plot_'+key[4:] + '_Pwlf'
                key_plot=key[4:] 
                keys_plot_list.append(key_plot)
                
                block['Pwlf_'+key]=([np.linspace(xl, xu, num=nPoint),
                                    np.interp(np.linspace(xl, xu, num=nPoint), [xl,intersect,xu], [yl,25,yu])])
            
                block['PwlfBz_'+key]=[np.linspace(xl, xu, num=nPoint),loudnessFunc([slopeL,slopeU,intersect],np.linspace(xl, xu, num=nPoint))]
            
                block['RMSE_'+key]=rmse(block['CU'],loudnessFunc([slopeL,slopeU,intersect],block['Level']))
                block['NRMSE_'+key]=nrmse(block['CU'],loudnessFunc([slopeL,slopeU,intersect],block['Level']))
                # block['STD_'+key]=np.std(block['CU'],loudnessFunc([slopeL,slopeU,intersect],block['Level']))



#%%
# Plot min_methods for each block--------------------------------------------------------

# min_methods=['MAT','NEL','CG','SLS','TRU']
# min_methods=['NEL']
# fit_methods=['BX','BY','BTX','BTUX','BTUY','BTPX']

min_methods=['NEL']
fit_methods=['BTUX']

for CLS in CLSs:
    for ii, block in enumerate(CLS['Blocks']):
        for mm, fit_method in enumerate(fit_methods):  
            print('Block: '+ str(ii)+' fit: '+fit_method)     
            plt.figure()
            plt.plot([10,110],[25,25],':',color='grey')
            plt.scatter(block['Level'], block['CU'],color='black', label='Data')
            try:
                plt.plot(block['Pwlf-Artc'][0],block['Pwlf-Artc'][1],'-',label='Artc'+' '+str(round(block['RMSE_Artc'],4)))
            
                plt.plot(block['PwlfBz-Artc'][0],block['PwlfBz-Artc'][1],'-',label='Artc'+' e:'+str(round(block['NRMSE_Artc'],4)))
            
            except:
                print('Auralec data missing for plot')


            for ff, min_method in enumerate(min_methods):
                key=min_method + '_' + fit_method
                plt.plot(block['Pwlf_'+key][0],block['Pwlf_'+key][1],'--',label=min_method+'_'+fit_method+' '+str(round(block['RMSE_'+key],4)))
                plt.plot(block['PwlfBz_'+key][0],block['PwlfBz_'+key][1],'--',label=min_method+'_'+fit_method+' e:'+str(round(block['NRMSE_'+key],4)) )
            plt.title('Block: '+ str(ii)+' fit: '+fit_method)
            plt.grid()
            plt.legend()
            plt.show()

#%%
# Plot each block in a subplot -------------------------- 

for CLS in CLSs:
    plt.figure(figsize=(20, 15))
    for ii, block in enumerate(CLS['Blocks']):
        for mm, fit_method in enumerate(fit_methods):     
            ax = plt.subplot(3, 3, ii + 1)
            ax.plot([10,110],[25,25],':',color='grey')
            ax.scatter(block['Level'], block['CU'],color='black', label='Data')
            try:
                # ax.plot(block['Pwlf-Artc'][0],block['Pwlf-Artc'][1],'-',label='Artc'+' '+str(round(block['RMSE_Artc'],4)))
            
                ax.plot(block['PwlfBz-Artc'][0],block['PwlfBz-Artc'][1],'-',label='Artc'+' e:'+str(round(block['NRMSE_Artc'],4)))
            
            except:
                print('Auralec data missing for plot')


            for ff, min_method in enumerate(min_methods):
                key=min_method + '_' + fit_method
                ax.plot(block['Pwlf_'+key][0],block['Pwlf_'+key][1],'--',label=min_method+'_'+fit_method+' '+str(round(block['RMSE_'+key],4)))
                # ax.plot(block['PwlfBz_'+key][0],block['PwlfBz_'+key][1],'--',label=min_method+'_'+fit_method+' e:'+str(round(block['NRMSE_'+key],4)) )
            ax.set_title('Max level: '+ str(block['AdaptiveMaxLevel']) + 'dB')
            plt.xlabel('Level [dB SPL]')
            plt.ylabel('Categorical Units')
            ax.grid()
            ax.legend()
            # ax.show()

#%% get rid of the trainning blocks 
# for CLS in CLSs:
del CLSs[0]['Blocks'][0]

#%%
# Compute mean over each block -------------------------------------------------------------

# min_methods=['MAT','NEL','CG','SLS','TRU']
# min_methods=['NEL']
# fit_methods=['BX','BY','BTX','BTUX','BTUY','BTPX']
# fit_methods=['BTPX']

key=min_method + '_' + fit_method

allPwlf_this_level=[]
allPwlfBz_this_level=[]

blockMeans=[]
blockMeans.append({})
# blockMeans[-1]['GroupId']='Means'

for CLS in CLSs:

    allPwlfBz_this_level=[]
    allPwlf_this_level=[]
    allPwlfBz_this_level2=[]
    allPwlf_this_level2=[]
    allPoints_this_block_level=[]
    allPoints_this_block_CU=[]

    for ii, block in enumerate(CLS['Blocks']):
        
        allPwlf_this_level.append(block['Pwlf-Artc'][1])
        allPwlfBz_this_level.append(block['PwlfBz-Artc'][1])
        allPoints_this_block_level.extend(block['Level'])
        allPoints_this_block_CU.extend(block['CU']) 
        
        for mm, fit_method in enumerate(fit_methods):
                
            for ff, min_method in enumerate(min_methods):
                key=min_method + '_' + fit_method
                allPwlf_this_level2.append(block['Pwlf_'+key][1])
                allPwlfBz_this_level2.append(block['PwlfBz_'+key][1])
                
    blockMeans[-1]['meanPwlf-Artc']= [block['Pwlf-Artc'][0], np.mean(allPwlf_this_level,0)]
    blockMeans[-1]['meanPwlfBz-Artc']= [block['PwlfBz-Artc'][0], np.mean(allPwlfBz_this_level,0)]
    blockMeans[-1]['meanPwlf_'+str(key)]= [block['Pwlf_'+str(key)][0], np.mean(allPwlf_this_level2,0)]
    blockMeans[-1]['meanPwlfBz_'+str(key)]= [block['PwlfBz_'+str(key)][0], np.mean(allPwlfBz_this_level2,0)]
    CLS['meanPwlf-Artc']= [block['Pwlf-Artc'][0], np.mean(allPwlf_this_level,0)]
    CLS['meanPwlfBz-Artc']= [block['PwlfBz-Artc'][0], np.mean(allPwlfBz_this_level,0)]
    CLS['meanPwlf_'+str(key)]= [block['Pwlf_'+str(key)][0], np.mean(allPwlf_this_level2,0)]
    CLS['meanPwlfBz_'+str(key)]= [block['PwlfBz_'+str(key)][0], np.mean(allPwlfBz_this_level2,0)]
    CLS['All_Levels']= allPoints_this_block_level
    CLS['All_CUs']= allPoints_this_block_CU

#%% 
# Compute refit, means, pwlf Bz and RSME over gathered block data -----------------------------


min_methods=['NEL']
fit_methods=['BTUX']
data=[]

for CLS in CLSs:
    data=np.array([CLS['All_Levels'], CLS['All_CUs'],]).flatten(order='F')
    for mm, min_method in enumerate(min_methods):
        for ff, fit_method in enumerate(fit_methods):
            fit = np.array(fit_loudness_function.fit_loudness_function(data, fit_method,min_method))  
            print(min_method + '_' + fit_method + ': ' + str(fit)) 
            CLS['Refit_inter_slopes_'+str(key)]=fit

            intersect=CLS['Refit_inter_slopes_'+str(key)][0]
            slopeL=CLS['Refit_inter_slopes_'+str(key)][1]
            slopeU=CLS['Refit_inter_slopes_'+str(key)][2]
            

            xu=y2x_lin(yu,intersect,25,slopeU)
            xl=y2x_lin(yl,intersect,25,slopeL)

            
            x15=y2x_lin(15,intersect,25,slopeL)
            x35=y2x_lin(35,intersect,25,slopeU)

            key_plot=key[4:] 
            keys_plot_list.append(key_plot)
            
            CLS['Refit_meanPwlf_'+key]=([np.linspace(xl, xu, num=nPoint),
                                np.interp(np.linspace(xl, xu, num=nPoint), [xl,intersect,xu], [yl,25,yu])])
        
            CLS['Refit_meanPwlfBz_'+key]=[np.linspace(xl, xu, num=nPoint),loudnessFunc([slopeL,slopeU,intersect],np.linspace(xl, xu, num=nPoint))]
        
            CLS['Refit_RMSE_'+key]=rmse(CLS['All_CUs'],loudnessFunc([slopeL,slopeU,intersect],CLS['All_Levels']))
            CLS['Refit_NRMSE_'+key]=nrmse(CLS['All_CUs'],loudnessFunc([slopeL,slopeU,intersect],CLS['All_Levels']))

#%%
# Plot means ------------------------------------------------------------

for CLS in CLSs:
    plt.figure()
    plt.plot([10,110],[25,25],':',color='grey')
    plt.plot(CLS['meanPwlf-Artc'][0],CLS['meanPwlf-Artc'][1],'-',label='mean Artc' )

    for mm, fit_method in enumerate(fit_methods):
        for ff, min_method in enumerate(min_methods):
            key=min_method + '_' + fit_method
            plt.plot([10,110],[25,25],':',color='grey')
            # plt.plot(CLS['meanPwlf_'+str(key)][0],CLS['meanPwlf_'+str(key)][1],'-',label='mean '+str(key))
            plt.plot(CLS['meanPwlfBz_'+str(key)][0],CLS['meanPwlfBz_'+str(key)][1],'-',label='mean Bz '+str(key))
    plt.title('ID: '+CLS['ID']+ ' fit: '+fit_method)
    plt.xlabel('Level [dB SPL]')
    plt.ylabel('Categorical Units')
    plt.grid()
    plt.legend()
    plt.show()

plt.figure()        
for CLS in CLSs:
    
    plt.plot([10,110],[25,25],':',color='grey')
    plt.plot(CLS['meanPwlf-Artc'][0],CLS['meanPwlf-Artc'][1],'-',label=CLS['ID']+'mean Artc' )

    for mm, fit_method in enumerate(fit_methods):
        for ff, min_method in enumerate(min_methods):
            key=min_method + '_' + fit_method
            plt.plot([10,110],[25,25],':',color='grey')
            # plt.plot(CLS['meanPwlf_'+str(key)][0],CLS['meanPwlf_'+str(key)][1],'-',label='mean '+str(key))
            plt.plot(CLS['meanPwlfBz_'+str(key)][0],CLS['meanPwlfBz_'+str(key)][1],'-',label=CLS['ID']+'mean '+str(key))

plt.title('fit: '+fit_method)
plt.xlabel('Level [dB SPL]')
plt.ylabel('Categorical Units')
plt.grid()
plt.legend()
plt.show()

#%%
# Plot mean Refit ------------------------------------------------------------

for CLS in CLSs:
    plt.figure()
    plt.plot([10,110],[25,25],':',color='grey')
    plt.scatter(CLS['All_Levels'], CLS['All_CUs'],color='black', label='Data')

    for mm, fit_method in enumerate(fit_methods):
        for ff, min_method in enumerate(min_methods):
            key=min_method + '_' + fit_method
            plt.plot([10,110],[25,25],':',color='grey')
            # plt.plot(CLS['meanPwlf_'+str(key)][0],CLS['meanPwlf_'+str(key)][1],'-',label='mean '+str(key))
            plt.plot(CLS['Refit_meanPwlfBz_'+str(key)][0],CLS['Refit_meanPwlfBz_'+str(key)][1],'-',label='refit Bz '+str(key))
    plt.title('ID: '+CLS['ID']+ ' refit: '+fit_method)
    plt.xlabel('Level [dB SPL]')
    plt.ylabel('Categorical Units')
    plt.grid()
    plt.legend()
    plt.show()

plt.figure()        
for CLS in CLSs:
    
    plt.plot([10,110],[25,25],':',color='grey')

    for mm, fit_method in enumerate(fit_methods):
        for ff, min_method in enumerate(min_methods):
            key=min_method + '_' + fit_method
            plt.plot([10,110],[25,25],':',color='grey')
            # plt.plot(CLS['meanPwlf_'+str(key)][0],CLS['meanPwlf_'+str(key)][1],'-',label='mean '+str(key))
            plt.plot(CLS['Refit_meanPwlfBz_'+str(key)][0],CLS['Refit_meanPwlfBz_'+str(key)][1],'-',label= CLS['ID']+' refit Bz '+str(key))

plt.title('refit: '+fit_method)
plt.xlabel('Level [dB SPL]')
plt.ylabel('Categorical Units')
plt.grid()
plt.legend()
plt.show()

#%%
# Plot mean vs Refit ------------------------------------------------------------

for CLS in CLSs:
    plt.figure() 
    plt.plot([10,110],[25,25],':',color='grey')
    plt.plot(CLS['meanPwlf-Artc'][0],CLS['meanPwlf-Artc'][1],'-',label=CLS['ID']+' mean Artc' )

    for mm, fit_method in enumerate(fit_methods):
        for ff, min_method in enumerate(min_methods):
            key=min_method + '_' + fit_method
            plt.plot([10,110],[25,25],':',color='grey')
            # plt.plot(CLS['meanPwlf_'+str(key)][0],CLS['meanPwlf_'+str(key)][1],'-',label='mean '+str(key))
            plt.plot(CLS['meanPwlfBz_'+str(key)][0],CLS['meanPwlfBz_'+str(key)][1],'-',label=CLS['ID']+' mean '+str(key))
            plt.plot(CLS['Refit_meanPwlfBz_'+str(key)][0],CLS['Refit_meanPwlfBz_'+str(key)][1],'-',label= CLS['ID']+' refit mean Bz '+str(key))

    plt.title('ID: '+CLS['ID']+' refit vs means: '+fit_method)
    plt.xlabel('Level [dB SPL]')
    plt.ylabel('Categorical Units')
    plt.grid()
    plt.legend()
    plt.show()

#%%

#Can BTUX be refited???????? it seems it can. it implements rules for when data above 35CU is missing

# %%
# look at difference between binaural and mono 
# plot participant data