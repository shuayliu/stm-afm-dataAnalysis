# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:19:05 2017

@author: jonah
"""

import sys

def PicoViewDealing(filename):
    
    if not filename.endswith('.txt') or filename.endswith('.mi'):
        print('%s is not PicoView file'%filename)
        sys.exit(3)
        
    import numpy as np
    ParaList = dict()
    data = dict()
    skiprows = 0
    
    with open(filename,'r') as f:
        lines = f.readlines()

    # 寻找SkipRow，SkipRow之前的数据作为ParaList
    for line in lines:
        if  line.startswith('data'):
            key = line[:14].strip()
            value = line[14:].strip()
            ParaList[key]=value
            skiprows += 1
            break
        else:
            key = line[:14].strip()
            value = line[14:].strip()
            ParaList[key]=value
            skiprows += 1
    
    if filename.endswith('.txt'):
        # SkipRow之后的数据作为data
        keys = lines[skiprows].strip().split('\t')
        colNums = len(keys)
        for key in keys:
            key = key[:-3].strip()
            data[key]= []
        # 这里有意见比较危险的事情，假设key是按这顺序作为index
        keys = list(data.keys())
        for line in lines[skiprows+1:]:
            for index in range(0,colNums):
                _d = line.split('\t')
                data[keys[index]].append(_d[index])
                
        for key in keys:
            data[key]=np.array(data[key],dtype=float)
            
    return ParaList,data