# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:05:26 2020

@author: wahab
"""

import sys
import pathlib

sys.path.append("C:/Users/wahab/Töölaud/CLEAR-master/")

#path=str((pathlib.Path(__file__).parent.absolute()) )+ '/'
#print("path",path)

from CLEAR import run_Clear

results_df,time_consumed,test_data = run_Clear('BreastC',1,2)

#print(pathlib.Path(__file__).parent.absolute())