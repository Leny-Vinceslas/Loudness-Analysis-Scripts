
#%% ---------------------------Load Modules and definitions
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
import re

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


#%% ---------------------------Find all CLS.xml files
# Find all CLS.xml files ---------------------------------------------
# directory_path = os.getcwd()+'\\'
# directory_path = 'c:\\Users\\lenyv\\OneDrive - University College London\\UCL\\Hyperacusis-project\\Python\\Loudness Analysis Scripts\\Participants\\'
directory_path =os.path.dirname(os.getcwd())+"\\Participants\\"
extension = ".xml"
# name_pattern =r'(CLS)_(\d{4})\.xml$'  
# name_pattern =r'(\d{4})_(CLS)\.xml$'
# name_pattern = re.compile(r'^((CLS)_(\d{4})\.xml|(\d{4})_(CLS)\.xml)$')
name_pattern = re.compile(r'^(CLS_\d{4}\.xml|\d{4}_CLS\.xml)$')
matched_files = []
IDs=[]

for root, _, files in os.walk(directory_path):
    
    for file in files:
        if file.endswith(extension) and name_pattern.match( file):
            matched_files.append(os.path.join(root, file))

ID_pattern1 = re.compile(r'\\\w{3}_(\d{4})\.xml$')
ID_pattern2 = re.compile(r'\\(\d{4})_\w{3}\.xml$')


# ID_pattern = (r'\\\w{3}_(\d{4})\.xml$' | '\\w{3}_(\d{4})\.xml$')
# ID_pattern = re.compile(r'^(\\?\w{3}_\d{4}\.xml|\\?\d{4}_\w{3}\.xml)$')


# Iterate over file paths
for matched_file in matched_files:
    # Search for pattern in each file path
    match = ID_pattern1.search( matched_file) or ID_pattern2.search( matched_file)
    if match:
        ID = match.group(1)
        if ID not in IDs:
            IDs.append(ID)

participantIDs=IDs
# Print the extracted IDs
print("IDs in the list:")
print(IDs)
print(matched_files)
#%% ---------------------------Parse XML and import blocks obj
# Parse XML and import blocks obj-------------
blockFromCls = parseXML.Cls2dic()
CLSs=[]

for ID in participantIDs:

    blocks=[]
    CLSs.append({})
    CLSs[-1]['ID']=ID

    for file in matched_files:
        if re.search(rf'\w+_{ID}\.xml$' , os.path.basename(file)):
            blocks=blockFromCls.parse(file)
            CLSs[-1]['Blocks']=blocks  
            print("-----> xml data parsed:", str(file))

#%% ---------------------------get rid of some failed blocks -----------------------------
# get rid of some failed blocks -----------------------------

del CLSs[1]['Blocks'][-1]
del CLSs[2]['Blocks'][-1]

#%% ---------------------------get rid of the trainning blocks -----------------------------
# get rid of the trainning blocks -----------------------------

nTrials=3
[CLS['Blocks'].pop(0) for CLS in CLSs if len(CLS['Blocks']) > nTrials]

#%% ---------------------------gather all data points -----------------------------
#gather all data points -----------------------------

for CLS in CLSs:
    allPoints_this_block_level=[]
    allPoints_this_block_CU=[]

    for ii, block in enumerate(CLS['Blocks']):
        allPoints_this_block_level.extend(block['Level'])
        allPoints_this_block_CU.extend(block['CU']) 

    CLS['All_Levels']= allPoints_this_block_level
    CLS['All_CUs']= allPoints_this_block_CU

#%% ---------------------------get rid of 0dB threshold -----------------------------
#get rid of 0dB threshold -----------------------------

for CLS in CLSs:
    data=np.array([CLS['All_Levels'], CLS['All_CUs']]).flatten(order='F')
    pairs = list(zip(CLS['All_Levels'], CLS['All_CUs']))
    filtered_pairs = [(x, y) for x, y in pairs if y != 0]
    CLS['filt_All_Levels'] = [pair[0] for pair in filtered_pairs]
    CLS['filt_All_CUs'] = [pair[1] for pair in filtered_pairs]
    


# %% --------------------------Perform all the fit per block
# Perform all the fit per block----------------------------------

# min_methods=['MAT','NEL','CG','SLS','TRU']
# fit_methods=['BX','BY','BTX','BTUX','BTUY','BTPX']
min_methods=['NEL']
fit_methods=['BTUX']
fit_methods.append('BTX')
# we can change/delete the min_method to simplify and integrate the Artc <-------------------
data=[]

for CLS in CLSs:
    maxLoudness=[]
    for ii, block in enumerate(CLS['Blocks']):
        data=np.array([block['Level'], block['CU'],]).flatten(order='F')
        maxLoudness=block['AdaptiveMaxLevel']
        print('------- Fitting block '+str(ii)+' from '+CLS['ID']+'---------')

        for mm, min_method in enumerate(min_methods):

            for ff, fit_method in enumerate(fit_methods):
                fit = np.array(fit_loudness_function.fit_loudness_function(data, fit_method,min_method))

                print(min_method + '_' + fit_method + ': ' + str(fit)) 
                block[min_method + '_' + fit_method]=fit
    CLS['AdaptiveMaxLevel']=maxLoudness
    
conditions=['Artc']
for mm, min_method in enumerate(min_methods):
    for ff, fit_method in enumerate(fit_methods):
        conditions.append(min_method + '_' + fit_method)
# %% --------------------------Compute pwlf Bz and RSME -----------------------------
# Compute pwlf Bz and RSME --------------------------

markers=["o","*","+","x","v","^","<",">","s","p","P","h","H"]

for CLS in CLSs:
    temp_slopeL=[]
    temp_slopeU=[]
    temp_intersect=[]
    temp2_slopeL=[]
    temp2_slopeU=[]
    temp2_intersect=[]
    # temp={}
    temp=[[[],[],[]],[[],[],[]],[[],[],[]]]
    

    for ii, block in enumerate(CLS['Blocks']):
        nPoint=100
        collection_conditions=['Artc']
        
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
            block['Pwlf_Artc']=[np.linspace(xl, xu, num=nPoint),
                                np.interp(np.linspace(xl, xu, num=nPoint), [xl,intersect,xu], [yl,25,yu])]

            block['PwlfBz_Artc']=[np.linspace(xl, xu, num=nPoint),loudnessFunc([slopeL,slopeU,intersect],np.linspace(xl, xu, num=nPoint))]
            
            block['RMSE_Artc']=rmse(block['CU'],loudnessFunc([slopeL,slopeU,intersect],(block['Level'])))
            block['NRMSE_Artc']=nrmse(block['CU'],loudnessFunc([slopeL,slopeU,intersect],(block['Level'])))
            #Compute average of intersec and slopes
            temp_slopeL.append(float(block['SlopeL']))
            temp_slopeU.append(float(block['SlopeU']))
            temp_intersect.append(float(block['Intersection']))

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

                #Compute average intercept and slopes 
                # temp2_slopeL.append(block[str(key)][1])
                # temp2_slopeU.append(block[str(key)][2])
                # temp2_intersect.append(block[str(key)][0])
                # temp[key+'_slopeL'].append(block[str(key)][1])
                # temp[key+'_slopeU'].append(block[str(key)][2])
                # temp[key+'_intersect'].append(block[str(key)][0])
                
                temp[ff][1].append(block[str(key)][1])
                temp[ff][2].append(block[str(key)][2])
                temp[ff][0].append(block[str(key)][0])
                
                collection_conditions.append(key)
    CLS['Artc_inter_slopes']=[np.mean(temp_intersect), np.mean(temp_slopeL), np.mean(temp_slopeU)]
    # CLS[str(key)+'_inter_slopes']=[np.mean(temp2_intersect), np.mean(temp2_slopeL), np.mean(temp2_slopeU)]
    CLS['NEL_BTUX_inter_slopes']=[np.mean(temp[0][0]), np.mean(temp[0][1]), np.mean(temp[0][2])]
    CLS['NEL_BTX_inter_slopes']=[np.mean(temp[1][0]), np.mean(temp[1][1]), np.mean(temp[1][2])]
    
    # collection_conditions=['Artc',str(key)]
    for condition in collection_conditions:
        intersect=CLS[condition+'_inter_slopes'][0]
        slopeL=CLS[condition+'_inter_slopes'][1]
        slopeU=CLS[condition+'_inter_slopes'][2]

        xu=y2x_lin(yu,intersect,25,slopeU)
        xl=y2x_lin(yl,intersect,25,slopeL)

        
        x15=y2x_lin(15,intersect,25,slopeL)
        x35=y2x_lin(35,intersect,25,slopeU)

        # key_plot='plot_'+key[4:] + '_Pwlf'
        key_plot=key[4:] 
        keys_plot_list.append(key_plot)
        
        CLS['meanPwlf_'+condition]=([np.linspace(xl, xu, num=nPoint),
                            np.interp(np.linspace(xl, xu, num=nPoint), [xl,intersect,xu], [yl,25,yu])])

        CLS['meanPwlfBz_'+condition]=[np.linspace(xl, xu, num=nPoint),loudnessFunc([slopeL,slopeU,intersect],np.linspace(xl, xu, num=nPoint))]

        CLS['RMSE_'+condition]=rmse(CLS['All_CUs'],loudnessFunc([slopeL,slopeU,intersect],CLS['All_Levels']))
        CLS['NRMSE_'+condition]=nrmse(CLS['All_CUs'],loudnessFunc([slopeL,slopeU,intersect],CLS['All_Levels']))
                

#%% ---------------------------Compute refit, means, pwlf Bz and RSME over gathered block data
# Compute refit, means, pwlf Bz and RSME over gathered block data -----------------------------


# min_methods=['NEL']
# fit_methods=['BTUX']
data=[]

for CLS in CLSs:
    data=np.array([CLS['All_Levels'], CLS['All_CUs']]).flatten(order='F')
    # data=np.array([CLS['filt_All_Levels'], CLS['filt_All_CUs']]).flatten(order='F')
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

#%% ---------------------------Plot each block in a subplot -----------------------------
# Plot each block in a subplot -------------------------- 

for CLS in CLSs:
    # plt.figure(figsize=(20, 5))
    # plt.figure()
    num_cols = 3
    num_rows = (len(CLS['Blocks']) + 2) // 3 
    fig,axes = plt.subplots(num_rows, num_cols, figsize=(20, 4.5*num_rows))
    axes = axes.flatten()
    
    
    for ii, block in enumerate(CLS['Blocks']):
        ax=axes[ii]
        ax.grid()
        ax.plot([-10,120],[25,25],':',color='grey')
        ax.plot([int(block['AdaptiveMaxLevel']),int(block['AdaptiveMaxLevel'])],[-10,70],'--',color='red', alpha=0.3)
        ax.scatter(block['Level'], block['CU'],color='black', label='Data')
        try:
            # ax.plot(block['Pwlf-Artc'][0],block['Pwlf-Artc'][1],'-',label='Artc'+' '+str(round(block['RMSE_Artc'],4)))
            ax.plot(block['PwlfBz_Artc'][0],block['PwlfBz_Artc'][1],'-',label='Artc'+' e:'+str(round(block['NRMSE_Artc'],4)))
        except:
            print('Auritec data missing for plot')
        for mm, fit_method in enumerate(fit_methods): 
            


            for ff, min_method in enumerate(min_methods):
                key=min_method + '_' + fit_method
                # ax.plot(block['Pwlf_'+key][0],block['Pwlf_'+key][1],'--',label=min_method+'_'+fit_method+' '+str(round(block['RMSE_'+key],4)))
                ax.plot(block['PwlfBz_'+key][0],block['PwlfBz_'+key][1],'--',label=min_method+'_'+fit_method+' e:'+str(round(block['NRMSE_'+key],4)) )
            ax.set_title(' Block: '+ str(ii) )
            # ax.set_xlim(-2,100)
            ax.set_ylim(-2,52)
            plt.xlabel('Level [dB SPL]')
            plt.ylabel('Categorical Units')
            plt.suptitle('ID: ' + CLS['ID'] +', Max level: '+ str(block['AdaptiveMaxLevel']) + 'dB', fontsize=10)
            
            ax.legend()
    for j in range(len(CLS['Blocks']), num_rows * 3):
        axes[j].axis('off')
    
    
        
    plt.savefig('Figures//CLS_sub_'+CLS['ID'] +'.svg', format="svg")

#%% ---------------------------Compute filtered refit, means, pwlf Bz and RSME over gathered block data
# Compute filtered refit, means, pwlf Bz and RSME over gathered block data -----------------------------

data=[]

for CLS in CLSs:
    # data=np.array([CLS['All_Levels'], CLS['All_CUs']]).flatten(order='F')
    data=np.array([CLS['filt_All_Levels'], CLS['filt_All_CUs']]).flatten(order='F')
    for mm, min_method in enumerate(min_methods):
        for ff, fit_method in enumerate(fit_methods):
            fit = np.array(fit_loudness_function.fit_loudness_function(data, fit_method,min_method))  
            print(min_method + '_' + fit_method + ': ' + str(fit)) 
            CLS['filt_Refit_inter_slopes_'+str(key)]=fit

            intersect=CLS['filt_Refit_inter_slopes_'+str(key)][0]
            slopeL=CLS['filt_Refit_inter_slopes_'+str(key)][1]
            slopeU=CLS['filt_Refit_inter_slopes_'+str(key)][2]
            

            xu=y2x_lin(yu,intersect,25,slopeU)
            xl=y2x_lin(yl,intersect,25,slopeL)

            
            x15=y2x_lin(15,intersect,25,slopeL)
            x35=y2x_lin(35,intersect,25,slopeU)

            key_plot=key[4:] 
            keys_plot_list.append(key_plot)
            
            CLS['filt_Refit_meanPwlf_'+key]=([np.linspace(xl, xu, num=nPoint),
                                np.interp(np.linspace(xl, xu, num=nPoint), [xl,intersect,xu], [yl,25,yu])])
        
            CLS['filt_Refit_meanPwlfBz_'+key]=[np.linspace(xl, xu, num=nPoint),loudnessFunc([slopeL,slopeU,intersect],np.linspace(xl, xu, num=nPoint))]
        
            CLS['filt_Refit_RMSE_'+key]=rmse(CLS['filt_All_CUs'],loudnessFunc([slopeL,slopeU,intersect],CLS['filt_All_Levels']))
            CLS['filt_Refit_NRMSE_'+key]=nrmse(CLS['filt_All_CUs'],loudnessFunc([slopeL,slopeU,intersect],CLS['filt_All_Levels']))
#%% ---------------------------Plot means
# Plot means ------------------------------------------------------------


colorList=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']

plt.figure(figsize=(20, 15))
for ii, CLS in enumerate(CLSs):
    # plt.figure()
    ax = plt.subplot(3, 3, ii + 1)
    ax.plot([10,110],[25,25],':',color='grey')
    ax.plot(CLS['meanPwlfBz_Artc'][0],CLS['meanPwlfBz_Artc'][1],'-',label='mean Artc' )
    
    for mm, fit_method in enumerate(fit_methods):
        for ff, min_method in enumerate(min_methods):
            key=min_method + '_' + fit_method
            ax.plot([10,110],[25,25],':',color='grey')
            # plt.plot(CLS['meanPwlf_'+str(key)][0],CLS['meanPwlf_'+str(key)][1],'-',label='mean '+str(key))
            ax.plot(CLS['meanPwlfBz_'+str(key)][0],CLS['meanPwlfBz_'+str(key)][1],'-',label='mean Bz '+str(key))
    plt.title('ID: '+CLS['ID']+ ' fit: '+fit_method)
    plt.xlabel('Level [dB SPL]')
    plt.ylabel('Categorical Units')
    plt.grid()
    plt.legend()
    # plt.show()

plt.figure(figsize=(10, 4))  
   
for ii, CLS in enumerate(CLSs):
    ax = plt.subplot(1, 2, 1) 
    ax.plot([10,110],[25,25],':',color='grey')
    ax.plot(CLS['meanPwlf_Artc'][0],CLS['meanPwlf_Artc'][1],'-',color=colorList[ii],label=CLS['ID'] )
    ax.set_title('mean Artc' )
    ax.grid()
    ax.legend()
    
    for mm, fit_method in enumerate(fit_methods):
        for ff, min_method in enumerate(min_methods):
            key=min_method + '_' + fit_method
            bx = plt.subplot(1, 2, 2)
            bx.plot([10,110],[25,25],':',color='grey')
            # plt.plot(CLS['meanPwlf_'+str(key)][0],CLS['meanPwlf_'+str(key)][1],'-',label='mean '+str(key))
            bx.plot(CLS['meanPwlfBz_'+str(key)][0],CLS['meanPwlfBz_'+str(key)][1],'-',color=colorList[ii],label=CLS['ID'])
            bx.set_title(' mean  '+str(key))
            
# plt.suptitle('fit: '+fit_method)
plt.xlabel('Level [dB SPL]')
plt.ylabel('Categorical Units')
plt.grid()
plt.legend()
plt.show()

#%% ---------------------------Plot mean Refit
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

#%% ---------------------------Plot mean vs Refit
# Plot mean vs Refit ------------------------------------------------------------

    
colorList=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']
# param_list=['meanPwlfBz_Artc','meanPwlfBz_NEL_BTUX','Refit_meanPwlfBz_NEL_BTUX','filt_Refit_meanPwlfBz_NEL_BTUX']
# name_param_list=['Artc','BTUX','Refit_BTUX','filt_Refit_BTUX']

param_list=['meanPwlfBz_Artc','meanPwlfBz_NEL_BTUX','Refit_meanPwlfBz_NEL_BTUX','meanPwlfBz_NEL_BTX','Refit_meanPwlfBz_NEL_BTX']
name_param_list=['Artc mean','BTUX mean','Refit_BTUX','BTX mean','Refit_BTX']

plt.figure(figsize=(13, 7))
for ii, CLS in enumerate(CLSs):
    # plt.figure()
    ax = plt.subplot(2, 3, ii + 1)
    ax.plot([-2,110],[25,25],':',color='grey')
    ax.plot([int(CLS['AdaptiveMaxLevel']),int(CLS['AdaptiveMaxLevel'])],[-10,70],'--',color='red', alpha=0.3)   
    ax.scatter(CLS['All_Levels'], CLS['All_CUs'],color='black', label='Data')
    ax.plot(CLS['meanPwlfBz_Artc'][0],CLS['meanPwlfBz_Artc'][1],'-',label='Artc'+' e:'+str(round(CLS['NRMSE_Artc'],4)) )
    
    for mm, fit_method in enumerate(fit_methods):
        for ff, min_method in enumerate(min_methods):
            key=min_method + '_' + fit_method
            # plt.plot(CLS['meanPwlf_'+str(key)][0],CLS['meanPwlf_'+str(key)][1],'-',label='mean '+str(key))
            ax.plot(CLS['meanPwlfBz_'+str(key)][0],CLS['meanPwlfBz_'+str(key)][1],'-',label='mean_'+fit_method+' e:'+str(round(CLS['NRMSE_'+key],4)))
            ax.plot(CLS['Refit_meanPwlfBz_'+str(key)][0],CLS['Refit_meanPwlfBz_'+str(key)][1],'--',label= 'refit_' +fit_method+' e:'+str(round(CLS['Refit_NRMSE_'+key],4)))
            # ax.plot(CLS['filt_Refit_meanPwlfBz_'+str(key)][0],CLS['filt_Refit_meanPwlfBz_'+str(key)][1],'--',label= 'filt_refit_BTUX '+' e:'+str(round(CLS['filt_Refit_NRMSE_'+key],4)))
            
    plt.title('ID: '+CLS['ID'])
    ax.set_xlabel('Level [dB SPL]')
    ax.set_ylabel('Categorical Units')
    # ax.set_xlim(-2,100)
    ax.set_ylim(-2,52)
    plt.grid()
    plt.legend()
    # plt.show()
plt.suptitle('Means vs Refits',size=15)
plt.tight_layout()
plt.savefig('Figures//CLS_sub_mean_refit_perID.svg', format="svg",dpi=300, bbox_inches = "tight")






# plt.figure(figsize=(20, 15))     
# for n,CLS in enumerate(CLSs):
#     ax = plt.subplot(3, 3, n + 1)
#     ax.plot([10,110],[25,25],':',color='grey')
#     for ii, param in enumerate(param_list):
#         ax.plot(CLS[param][0],CLS[param][1],'-',color=colorList[ii],label='Artc'+' e:'+str(round(CLS['NRMSE_Artc'],4)) )
#     ax.set_title(CLS['ID'] )
#     ax.grid()
#     ax.legend()
#     ax.set_xlabel('Level [dB SPL]')
#     ax.set_ylabel('Categorical Units')

# param_list=['Refit_meanPwlfBz_NEL_BTUX']
# name_param_list=['Refit_BTUX']
# name_param_list=['Artc mean','BTUX mean','Refit_BTUX','BTX mean','Refit_BTX']
plt.figure(figsize=(13, 4*2))     
for n, param in enumerate(param_list):
    ax = plt.subplot(2, 3, n + 1)
    ax.plot([-2,110],[25,25],':',color='grey')

    for ii,CLS in enumerate(CLSs):
        ax.plot(CLS[param][0],CLS[param][1],'-',color=colorList[ii],label=CLS['ID'] )
    ax.set_title(name_param_list[n] )
    ax.grid()
    ax.legend()
    ax.set_xlabel('Level [dB SPL]')
    ax.set_ylabel('Categorical Units')
    # ax.set_xlim(-2,100)
    ax.set_ylim(-2,52)
plt.suptitle('Means vs Refits',size=15)
plt.tight_layout()
plt.savefig('Figures//CLS_sub_mean_refit.svg', format="svg",dpi=300, bbox_inches = "tight")

#

#%% compare slopes and angles and error
#
# get rid of 0dB threshold

#%%

#Can BTUX be refited???????? it seems it can. it implements rules for when data above 35CU is missing



#%% Compute averaged slope and refit slopes
IDs=[]
artc_slopes=[]
artc_intercept=[]
fit_slopes=[]
fit_intercept=[]
refit_slopes=[]
refit_intercept=[]
filtRefit_slopes=[]
filtRefit_intercept=[]
artc_error=[]
fit_error=[]
refit_error=[]
filtRefit_error=[]


for CLS in CLSs:
    IDs.append(CLS['ID'])
                    
    artc_slopes.append([CLS['Artc_inter_slopes'][1],CLS['Artc_inter_slopes'][2]])
    artc_intercept.append(CLS['Artc_inter_slopes'][0])
    artc_error.append(CLS['NRMSE_Artc'])

    fit_slopes.append([CLS[str(key)+'_inter_slopes'][1],CLS[str(key)+'_inter_slopes'][2]])
    fit_intercept.append(CLS[str(key)+'_inter_slopes'][0])
    fit_error.append(CLS['NRMSE_'+key])

    refit_slopes.append([CLS['Refit_inter_slopes_'+str(key)][1],CLS['Refit_inter_slopes_'+str(key)][2]])
    refit_intercept.append(CLS['Refit_inter_slopes_'+str(key)][0])
    refit_error.append(CLS['Refit_NRMSE_'+key])

    filtRefit_slopes.append([CLS['filt_Refit_inter_slopes_'+str(key)][1],CLS['filt_Refit_inter_slopes_'+str(key)][2]])
    filtRefit_intercept.append(CLS['filt_Refit_inter_slopes_'+str(key)][0])           
    filtRefit_error.append(CLS['filt_Refit_NRMSE_'+key])

data_intercepts= {'IDs': IDs, 'artc_intercept': artc_intercept, 
                  'fit_intercept': fit_intercept,
                  'refit_intercept': refit_intercept,
                  'filt_refit_intercept': filtRefit_intercept,}
data_slopeL= {'IDs': IDs, 'artc_slopeL': [s[0] for s in artc_slopes],
              'fit_slopeL': [s[0] for s in fit_slopes],
              'refit_slopeL': [s[0] for s in refit_slopes],
              'filt_refit_slopeL': [s[0] for s in filtRefit_slopes]}
data_slopeU= {'IDs': IDs, 'artc_slopeU': [s[1] for s in artc_slopes],
              'fit_slopeU': [s[1] for s in fit_slopes],
              'refit_slopeU': [s[1] for s in refit_slopes],
              'filt_refit_slopeU': [s[1] for s in filtRefit_slopes]}
data_errors= {'IDs': IDs, 'artc_error':artc_error,'fit_error':fit_error,'Refit_error':refit_error,'filt_Refit_error':filtRefit_error,}


data_all={}
data_all.update(data_intercepts)
data_all.update(data_slopeL)
data_all.update(data_slopeU)
data_all.update(data_errors)






df_all=pd.DataFrame(data_all)

# Displaying the DataFrame
# print(df_all[['IDs', 'artc_intercept','fit_intercept','refit_intercept','filt_refit_intercept']])


# df.style.hide(axis="index")

df_all.style.format(precision=2).background_gradient()
df_slopesL = pd.DataFrame(data_slopeL)
df_slopesL.style.format(precision=2).background_gradient()
df_slopesU = pd.DataFrame(data_slopeU)
df_slopesU.style.format(precision=2).background_gradient()
df_intercept = pd.DataFrame(data_intercepts)
df_intercept.style.format(precision=2).background_gradient()
df_errors = pd.DataFrame(data_errors)
df_errors.style.format(precision=2).background_gradient()


# df_all.plot(kind='scatter',
#         x='filt_refit_slopeL',
#         y='filt_refit_intercept',
#         color='red')

# df_all.plot(kind='scatter',
#         x='filt_refit_slopeU',
#         y='filt_refit_intercept',
#         color='red')

# df_all.plot(kind='scatter',
#         x='filt_refit_slopeU',
#         y='filt_refit_slopeL',
#         color='red')

# plt.figure()
# plt.plot(filtRefit_intercept,[s[0] for s in filtRefit_slopes],'o')
# plt.plot(refit_intercept,[s[0] for s in refit_slopes],'o')
# plt.plot(fit_intercept,[s[0] for s in fit_slopes],'o')

# plt.figure()
# plt.plot(filtRefit_slopes,'o')
# plt.plot(refit_slopes,'o')
# plt.plot( fit_slopes,'o')

#look at paper plot