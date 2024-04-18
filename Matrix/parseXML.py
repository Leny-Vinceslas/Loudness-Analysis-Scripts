from xml.etree.ElementTree import parse
import numpy as np 
from itertools import islice


class Matrix2dic:
    def __init__(self):
        self.path = []
        self.blocks=[]
        self.matrix={}
        
 
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
            if elem.tag == 'Value': self.matrix[elem.attrib['Name']]=elem.attrib['Value']
            for child1 in elem:
                # print('-->', child1.tag, child1.attrib)
                if child1.tag == 'Value': self.matrix[child1.attrib['Name']]=child1.attrib['Value']
                for child12 in child1:
                    for child2 in islice(child12.iter(), 1, None):
                        # print('---->', child2.tag, child2.attrib)   
                        if child2.tag == 'Subheader' and child2.attrib['Name']== "Blocks" and child2.attrib['Comment']== "" :
                            # print('------ enter data -------------->')
                            for child3 in child2:
                                   if child3.tag == 'Subheader' and child3.attrib['Name']== "" and child3.attrib['Comment']== "" :
                                    # print('----- New block -------------->')
                                    self.blocks.append({}) 

                                    
                                    for child4 in child3:
                                        # print('1------->', child4.tag, child4.attrib)
                                        # if child4.tag == 'Value': self.matrix[child4.attrib['Name']]=child4.attrib['Value']
                                        if child4.tag == 'Value': self.blocks[-1][child4.attrib['Name']]=child4.attrib['Value']
                                        if child4.tag == 'Value' and child4.attrib['Name']== "Status" and child4.attrib['Value']== "initialised" :
                                            del self.blocks[-1]
                                            break
                                        if child4.tag == 'Subheader' and child4.attrib['Name']== "Trials" and child4.attrib['Comment']== "" :                             
                                            
                                            
                                            data_sentances=[]
                                            for child5 in child4:
                                                # print('2-------->', child5.tag, child5.attrib)
                                                if child5.tag == 'Subheader' and child5.attrib['Name']== "" and child5.attrib['Comment']== "" :
                                                    # print('----- New sentense-------------->')
                                                    data_sentances.append({})
                                                    for child6 in child5:
                                                        # print('4--------->', child6.tag, child6.attrib)
                                                        data_sentances[-1][child6.attrib['Name']]=child6.attrib['Value']
                                            self.blocks[-1]['Sentences']=data_sentances
                    self.matrix['Blocks']=self.blocks                   

        return self.matrix
                        
 
 


