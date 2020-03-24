# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:05:26 2020

@author: wahab
"""

import sys
import pathlib

#sys.path.append("C:/Users/wahab/Töölaud/XAI/XAI-SPRING2020/CLEAR_DSG_RandomForest/")

#path=str((pathlib.Path(__file__).parent.absolute()) )+ '/'
#print("path",path)
def RUN_CLEAR(dataset,start,end):
    from CLEAR import run_Clear
    results_df=[]
    time_consumed=0
    test_data=[]
    for i in range(start,end+1):
        out = run_Clear('BreastC',i,i)
        results_df.append(out[0])
        time_consumed += out[1]
        test_data = out[2]
    return results_df,time_consumed,test_data
        





#results1,time1,test_data2=RUN_CLEAR('BreastC',1,2)
#print(pathlib.Path(__file__).parent.absolute())a