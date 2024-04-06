from xml.etree.ElementTree import parse
import numpy as np 
from itertools import islice


class Cls2dic:
    def __init__(self):
        self.path = []
        self.blocks=[]
        
 
    def parse(self, path):
        self.path=path
        level_list=[]
        CU_list=[]
        data_fit_CU=[]
        data_fit_level=[]
        self.blocks=[]

        tree = parse(self.path)
        root = tree.getroot()

        for elem in root :
            # print(elem.tag, elem.attrib)
            
            for child1 in elem:
                # print('-->', child1.tag, child1.attrib)
                for child12 in child1:
                    for child2 in islice(child12.iter(), 1, None):
                        # print('---->', child2.tag, child2.attrib)
                        # for child3 in child2:
                        #     print('------>', child3.tag, child3.attrib)
                        #     if child3.tag == "Value" and child3.attrib['Name']== "ChannelLevels"  :
                                # print('======CHILD3', child3.attrib['Name'],child3.attrib['Value'])
                                
                        if child2.tag == "Value" and child2.attrib['Name']== "GroupId"  : 
                            # print('------ New block -------------->')
                            self.blocks.append({})      
                            
                        if child2.tag == "Value" and child2.attrib['Name']== "ChannelLevels"  :
                            
                            if child2.attrib['Value']=='- 0':
                                 self.blocks[-1]['Side']=1
                                 self.blocks[-1]['SideStr']='Left'
                            elif child2.attrib['Value']=='0':
                                 self.blocks[-1]['Side']=0
                                 self.blocks[-1]['SideStr']='Right'
                            elif child2.attrib['Value']=='0 0':
                                 self.blocks[-1]['Side']=00
                                 self.blocks[-1]['SideStr']='diotic'
                                
                        if child2.tag=='Subheader' and child2.attrib['Name']=='Data':
                            for child3 in child2:
                                data_fit_level.append(float(child3.get('Value').split(' ')[0]))
                                data_fit_CU.append(float(child3.get('Value').split(' ')[1]))
                            
                            self.blocks[-1]['Data_fit_level']=(np.asarray(data_fit_level))
                            self.blocks[-1]['Data_fit_CU']=(np.asarray(data_fit_CU))    
                            
                        elif child2.tag=='Value' and not child2.attrib['Name']=='':
                            self.blocks[-1][child2.attrib['Name']]=child2.attrib['Value']
                                
                            # for child4 in child3:
                            #     print('-------->', child4.tag, child4.attrib)
                                
                                
                            #     for child5 in child4:
                            #         print('---------->', child5.tag, child5.attrib)
                            #         for child5 in child4:
                            #             print('+----------->', child5.tag, child5.attrib)
                            #             # for child6 in child5:
                            #             for child6 in islice(child5.iter(), 1, None):
                            #                 print('X------------->', child6.tag, child6.attrib)
                            #                 # for child6 in child5:
                            #                 #     print('------------>', child6.tag, child6.attrib)
                            #                 #     for child7 in child5:
                            #                 #         for child8 in child7:
                            #                 #             print('-------------->', child8.tag, child8.attrib)
                                    
                        if child2.tag == "Value" and child2.attrib['Name']== "CU":
                                                CU_list.append(child2.get('Value'))
                        if child2.tag == "Value" and child2.attrib['Name']== "Level":
                                                level_list.append(child2.get('Value'))
                    if level_list and CU_list:  
                        self.blocks[-1]['CU'] = np.asanyarray([float(i) for i in CU_list])
                        self.blocks[-1]['Level'] = np.asanyarray([float(i) for i in level_list])
                        level_list=[]
                        CU_list=[]
                        data_fit_level=[]
                        data_fit_CU=[]
        return self.blocks
                        
 
 


