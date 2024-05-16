
#%% Load Modules and definitions
#Load Modules and definitions
import parseXML
import os
import numpy as np
import numpy as np 
import os
import scipy.io as spio
import matplotlib.pyplot as plt
import pandas as pd
import re
from importlib import reload 
from fit_psyche.psychometric_curve import PsychometricCurve
from sklearn.model_selection import RandomizedSearchCV
from scipy.optimize import curve_fit

colors=colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

level_colors = {
    '50': colors[0],
    '60': colors[1],
    '70': colors[2]
}

#%% Find all files ---------------------------------------------
# Find all files ---------------------------------------------
# directory_path = os.getcwd(../)+'\\'
# directory_path = 'c:\\Users\\lenyv\\OneDrive - University College London\\UCL\\Hyperacusis-project\\Python\\Loudness Analysis Scripts\\Participants\\'
directory_path =os.path.dirname(os.getcwd())+"\\Participants\\"
extension = ".xml"
name_pattern =r'(Matrix)_(\d{4})\.xml$'  
matched_files = []
IDs=[]

for root, _, files in os.walk(directory_path):
    
    for file in files:
        if file.endswith(extension) and re.match(name_pattern, file):
            matched_files.append(os.path.join(root, file))

ID_pattern = r'\\\w{6}_(\d{4})\.xml$'

# List to store extracted IDs

# Iterate over file paths
for matched_file in matched_files:
    # Search for pattern in each file path
    match = re.search(ID_pattern, matched_file)
    if match:
        ID = match.group(1)
        if ID not in IDs:
            # IDs.append(ID)
            IDs.append({})
            IDs[-1]['ID']=ID

print("IDs in the list:")
print(IDs)
#%% Parse XML and import blocks obj-------------
# Parse XML and import blocks obj-------------
reload(parseXML)
blockFromCls = parseXML.Matrix2dic()

for ID in IDs:
    blocks=[]
    for file in matched_files:
        IDnum=ID['ID']
        if re.search(rf'\w+_{IDnum}\.xml$' , os.path.basename(file)):
            Matrix=blockFromCls.parse(file)
            # IDs[-1]['Matrix']=Matrix  
            ID['Matrix']=Matrix  
            print("-----> xml data parsed:", str(file))




#%% ---------------------------get rid of the trainning blocks -----------------------------
#get rid of the trainning blocks -----------------------------
for ID in IDs:
    ID['Matrix']['Blocks'] = [block for block in reversed(ID['Matrix']['Blocks']) if block['TrialsDone'] != '10']

# %% retrive sentences SNR and plot -------------------------------------
# retrive sentences SNR and plot -------------------------------------

# psychometric function
# def func(x, alpha, beta):
#     return 1. / (1 + np.exp( -(x-alpha)/beta ))

def func(x, alpha, beta):
    return 1 / (1 + np.exp( -alpha*(x-beta ))) 

for ID in IDs:
    fig, ax1 = plt.subplots()
    plt.figure(figsize=(15, 10))
    
    for n,block in enumerate(ID['Matrix']['Blocks']):
        # for block in blocks:
            L50=[]
            slope=[]
            SNRs=np.array([])
            trialIntel=np.array([])
            for sentence in block['Sentences']:
                # for sentence in sentences:
                SNRs=np.append(SNRs,float(sentence['SNR']))
                trialIntel=np.append(trialIntel,float(sentence['TrialIntelligibility']))
                    # SNRs.append(float(sentence['SNR']))
                    # trialIntel.append(float(sentence['TrialIntelligibility']))
            # print("SNR:", str(SNRs))
            # print("ITL:", str(trialIntel))

            trialIntel=trialIntel[np.argsort(SNRs)]
            SNRs=np.sort(SNRs)
            
            # trialIntel=trialIntel()
            block['SNRs']=SNRs
            block['TrialIntel']=trialIntel
            if "L50" in block:
                
                if any(chr.isdigit() for chr in block['L50']):
                    L50=float(block['L50'])
                    slope=float(block['Slope'])/100 #slope is %dB
                    
                # else:L50=[]
                
            print("L50:",str(block['L50']))
            print("slope:",str(block['Slope']))
                
            if "SRT 0.5" in block:
                # print("SRT:",str(block['SRT 0.5']))
                L50=float(block['SRT 0.5'])
                # slope=[]
                
            # print("L50:",str(block['L50']))
            
            # plt.figure()
            ax=plt.subplot(3,3,n+1)
            ax.plot(SNRs,trialIntel,'o')
            
            xdata=SNRs
            ydata=trialIntel
            popt, pcov = curve_fit(func, xdata, ydata) 
            ax.plot(xdata, func(xdata, *popt), '-')
            
            if L50 and slope:
                
                ax.plot(L50,0.5,'+')
                
                slopeY=np.array([0.4,0.5,0.6])
                
                # Calculate y-intercept (b) using the point-slope form: y - y1 = m(x - x1)
                b = 0.5 - slope * L50

                # Calculate x-coordinates for y=0.4 and y=0.6 using the linear function equation y = mx + b
                slopeX = (slopeY - b) / slope

                # Points on the linear function for y=0.4 and y=0.6
                point_04 = (slopeX, slopeY)
                
            
                ax.plot(slopeX,slopeY,'+--')

            ax.grid()
            ax.set_xlabel('SNR [dB]')
            ax.set_ylabel('Intelligibility')
            ax.set_title('Speech level: '+block['SpeechLevel']+' dB')
            
            # if block['SpeechLevel']== '50': colorLevel=colors[0]
            # if block['SpeechLevel']== '60': colorLevel=colors[1]
            # if block['SpeechLevel']== '70': colorLevel=colors[2]
            
            ax1.plot(xdata, func(xdata, *popt), '-',color=level_colors.get(block['SpeechLevel']),label=block['SpeechLevel']+'dB')
            ax1.grid()
            ax1.set_xlabel('SNR [dB]')
            ax1.set_ylabel('Intelligibility')
            ax1.set_title('Speech levels: 50, 60, 70dB')
            ax1.legend()
    plt.tight_layout(h_pad=2)
    # plt.savefig('Figures//sub_matrix.svg', format="svg")
    # plt.suptitle(' DP I/O [µPa]. ID:'+GR['ID'])


#%%
#gather same level toguether
#%%
            # fitting

            # pc = PsychometricCurve(model='wh').fit(SNRs, trialIntel)
            # plt.figure()
            # pc.plot(SNRs, trialIntel)

            
            

            xdata=(SNRs+5)/10
            ydata=trialIntel/max(trialIntel)
            plt.figure
            pc = PsychometricCurve(model='wh').fit(xdata, ydata)
            pc.plot(xdata, ydata)

#%% process intelligibility per word
# process intelligibility per word
for ID in IDs:
    for block in ID['Matrix']['Blocks']:
        # for block in blocks:
            SNRs=np.array([])
            trialIntel=np.array([])
            for sentence in block['Sentences']:
                target=list(sentence['TargetWords'].split(" "))
                answer=list(sentence['SelectedWords'].split(" "))
                score = [(target[i] == answer[i]) for i in range(len(target))]
            
            print("target:", str(target))
            print("answer:", str(answer))
            print("score:", str(score))

  
# %%
