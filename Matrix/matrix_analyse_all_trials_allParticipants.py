
#%% Load Modules and definitions
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
from scipy.optimize import curve_fit



#%%
# Find all OAE files ---------------------------------------------
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

# participantIDs=IDs
# Print the extracted IDs
print("IDs in the list:")
print(IDs)
#%% 
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

# %%
# retrive sentences SNR and plot

# psychometric function
def func(x, alpha, beta):
    return 1. / (1 + np.exp( -(x-alpha)/beta ))

for ID in IDs:
    for block in ID['Matrix']['Blocks']:
        # for block in blocks:
            SNRs=np.array([])
            trialIntel=np.array([])
            for sentence in block['Sentences']:
                # for sentence in sentences:
                SNRs=np.append(SNRs,float(sentence['SNR']))
                trialIntel=np.append(trialIntel,float(sentence['TrialIntelligibility']))
                    # SNRs.append(float(sentence['SNR']))
                    # trialIntel.append(float(sentence['TrialIntelligibility']))
            print("SNR:", str(SNRs))
            print("ITL:", str(trialIntel))

            trialIntel=trialIntel[np.argsort(SNRs)]
            SNRs=np.sort(SNRs)
            
            # trialIntel=trialIntel()
            block['SNRs']=SNRs
            block['TrialIntel']=trialIntel
            if "L50" in block:
                print("L50:")
                if any(chr.isdigit() for chr in block['L50']):
                    L50=float(block['L50'])
                    slope=float(block['Slope'])
                else:L50=0
                
            if "SRT 0.5" in block:
                print("SRT:")
                L50=float(block['SRT 0.5'])
                slope=[]
            
            plt.figure()
            plt.plot(SNRs,trialIntel,'o')
            plt.plot(L50,0.5,'+')
            if slope:
                slopeY=np.array([0.4,0.5,0.6])
                b=0.5-slope*L50
                slopeX=(slopeY-b)/slope    #problem here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                plt.plot(slopeX,slopeY,'r')
            plt.grid()

            # fitting

            # pc = PsychometricCurve(model='wh').fit(SNRs, trialIntel)
            # pc.plot(SNRs, trialIntel)

            
            xdata=SNRs
            ydata=trialIntel
            # xdata=[0,1,2,3,4,5]

            # ydata=[0,1,2,3,4,5]
            popt, pcov = curve_fit(func, xdata, ydata)

            plt.plot(xdata, func(xdata, *popt), 'r-')

            # popt, pcov = curve_fit(func, xdata, ydata,bounds=(0.5,L50))

            # plt.plot(xdata, func(xdata, *popt), 'g-')