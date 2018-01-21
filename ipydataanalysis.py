# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:05:44 2017

@author: jonah
"""

VERSION = '1.0.1'
import numpy as np
import sys,os
from SpectroscopyTools.ForceTools import ForceTools
from AnalysisTools.AnalysisFunctions import ForceCurveAnalysis
import warnings
warnings.filterwarnings("ignore")



def getConfigs(basePath):
    
    configs = dict()
    
    configs['mode'] = 'single'
    configs['olnyDSA'] = False
    configs['doDSA'] = True
    configs['doSlopeAnalysis'] = False
    configs['miEXTS']=['.txt','.mi']



#    opts,args = getopt.getopt(sys.argv[1:],shortArgs,longArgs)

#    transformMethod = 'FixedLength'

    # 读取配置文件
    confPath = os.path.join(basePath,'config.ini')
    conf = configparser.ConfigParser()
    
    if os.path.isfile(confPath):
        conf.read(confPath)
        configs['filename'] = os.path.abspath(conf.get('Configs','File_Path'))
        configs['SKIPROWS'] = conf.getint('DataFormat','Skiprows')
        configs['mode'] = conf.get('Configs','Mode')
        configs['forceConst'] = conf.getfloat('Configs','ForceConstant')
        configs['DELIMITER'] = str(conf.get('DataFormat','Delimiter')).lower()
        configs['DefSens'] = conf.getfloat('Configs','Deflection_Sensitive')
        configs['transformMethod'] = str(conf.get('Configs','Transform_Method')).lower()
        configs['stepLength'] = int(conf.get('Configs','Step_Length'))
        print('read config.ini success!')
        # print(DATAFORMAT[DELIMITER])


        #Switch Options:
        switchOpts = argparse.ArgumentParser(description='Switch Options:')
       
        switchOpts.add_argument('--DSA','--onlyDSA',action='store_true',default=False,
                                dest='onlyDSA',help='Switch Deflection Sensitive Analysis On.')        
        switchOpts.add_argument('--noDSA',action='store_false',default=True,
                                dest='doDSA',help='Switch Deflection Sensitive Analysis OFF.')     
        switchOpts.add_argument('--SlopeAnalysis','--SA',action='store_true',default=False,
                                dest='doSlopeAnalysis',help='Switch Slope Analysis ON.')        
        switchOpts.add_argument('-m','--model=',choices=['a'],default='a',
                                dest='model',help='Select Analysis Model')  
        
        #Parse Options
        switchOpts.add_argument('--filename', nargs='+', action='store',
                               dest='filenames',help='file(s) to Analysis. \n ')  

        
        switchOpts.add_argument('--forceConstant','--fc',nargs='?', type=float,default=np.nan,
                                dest='forceConst', help='Manually set Force Constant for F-D Cycle')
        
        
        if not switchOpts.parse_args().filenames == None:
            configs['filename'] = os.path.abspath(switchOpts.parse_args().filenames)
            
        if not switchOpts.parse_args().forceConst == np.nan:
            configs['forceConst'] = switchOpts.parse_args().forceConst

            
        configs['onlyDSA'] = switchOpts.parse_args().onlyDSA
        configs['doDSA'] = switchOpts.parse_args().doDSA
        configs['doSlopeAnalysis'] = switchOpts.parse_args().doSlopeAnalysis
        
        
        
        return configs


def dealingwithSingleDir(dirName):
    print("#"*int(30)+"--  Now Start Dealing  --"+"#"*int(27))
    print(" %s is dir"%os.path.split(dirName)[1])
    print(" Now dealing single file...")
    sys.stdout.write("#"*int(80)+'|')
    resultPath = os.path.join(dirName,'Result')
    
    if(os.path.exists(resultPath)):
        __import__('shutil').rmtree(resultPath)
    j=0
    for file in os.listdir(dirName):
        _filename = os.path.join(dirName,file)
        if os.path.isfile(_filename):
#                print(_filename)
            if file.endswith('.txt'):
                savePath = ForceCurveAnalysis(_filename,configs)
            j+=1
            sys.stdout.write('\r'+(j*80//len(os.listdir(dirName)))*'-'+'-->|'+"\b"*3)
            sys.stdout.flush()


    # 将所有的单条力曲线整合，并做2D Frequency bins统计
    print("\n Finish with sigle F-D Curve. Times: %.2f \n Now merging all data..."
		%(time.clock()-startTime))
    sys.stdout.write("#"*int(80)+'|')
    j=0
    allData=[-999.9,-999.9]
    for file in os.listdir(savePath):
        _filename = os.path.join(savePath,file)
        if os.path.isfile(_filename) and file.endswith('.txt') and file.startswith('FC_'):             
            singleData = np.loadtxt(_filename,comments=' ',unpack=True).T
            
            allData=np.vstack((allData,singleData))
            j+=1
            sys.stdout.write('\r'+(j*80//len(os.listdir(savePath)))*'-'+'-->|'+"\b"*3)
            sys.stdout.flush()
#        print(allData=[xData,yData])
    # 数据重新排序，按x升序
    allData=np.delete(allData,0,axis=0)
    I = np.argsort(allData.T[0])
    allData.T[0][:]=allData.T[0][I]
    allData.T[1][:]=allData.T[1][I]
    
    mergeDataPath = os.path.join(savePath,'mergeData_%s.dat'%os.path.split(dirName)[1])
    print("\n Finish with merging F-D Curves. \n Now writing all data to: '%s'..."%os.path.relpath(mergeDataPath))
		
    np.savetxt(mergeDataPath,allData,delimiter='\t')
    print(" Data Saved! \n\n Now histograming all data...")
    # histogram 2D:
    x,y = allData.T
    H,xedges,yedges = np.histogram2d(x,y,bins=(155,310),
                                     range=[[-0.5,15],[-0.5,15]],normed=False)
    histDataPath = os.path.join(savePath,'histData_%s.dat'%os.path.split(dirName)[1])
    np.savetxt(histDataPath,H.T)

    
    
    # 对参数文档进行分析
    # 这部分等TODO完善之后再进行uncomment
    print(" hist data saved at: '%s'! \n\n Now doing describe statistic on all parameters..."%os.path.relpath(histDataPath))
    import pandas as pd

    paraFile = os.path.join(dirName,"Result/Analysis/FDParameters.txt")
    stat = pd.read_csv(paraFile,sep='\t')
    gp = stat.groupby(['tipX','tipY'])
    gpStat = gp.describe()
    gpStat.to_csv(os.path.join(dirName,"Result/Analysis/statstistic.txt"),sep='\t')
    

    times = time.clock() - startTime
    
    sys.stdout.write("\n FINISHED! \n Times: %.2f s\n"%times)





if __name__ == '__main__':

    import configparser,argparse
    import time
    startTime = time.clock()

    basePath = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(basePath,"Fc.txt")
    
    configs = getConfigs(basePath)
    filename = configs['filename']


#    判断力常数，如果为1.0则认为力常数未作修改，抛出警告
#    并且，力常数不能小于零
    if configs['forceConst'] == 1.0:
        print('you MAY NOT set the proper ForceConstant, but the programme will keep going' )
    elif configs['forceConst'] <0:
        print("ForceConstant < 0 ,it is illegal")
        sys.exit(3)
   
    


    if os.path.isfile(filename):
        print("data is file")
        if filename.endswith('.txt'):
            ForceCurveAnalysis(filename,configs)

    elif os.path.isdir(filename):
        dirs = [d for d in os.listdir(filename) if not 'esult' in d and os.path.isdir(os.path.join(filename,d))]
        if dirs:
            for d in dirs:
                dealingwithSingleDir(os.path.join(filename,d))            
        else:
            dealingwithSingleDir(filename)
        
        print('\n' + "#"*int(82) + '\n')
        times = time.clock()-startTime
        print("Total time:%.2f"%times)
                

    else:
        print("Parameter of programme error")
        sys.exit(2)

      
