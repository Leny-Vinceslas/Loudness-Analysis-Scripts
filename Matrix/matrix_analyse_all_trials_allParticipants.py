
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
            IDs.append(ID)

participantIDs=IDs
# Print the extracted IDs
print("IDs in the list:")
print(IDs)
#%% 
# Parse XML and import blocks obj-------------
# reload(parseXML)
blockFromCls = parseXML.Matrix2dic()
CLSs=[]

for ID in participantIDs:

    blocks=[]
    CLSs.append({})
    CLSs[-1]['ID']=ID

    for file in matched_files:
        if re.search(rf'\w+_{ID}\.xml$' , os.path.basename(file)):
            matrix, blocks=blockFromCls.parse(file)
            CLSs[-1]=matrix  
            print("-----> xml data parsed:", str(file))

