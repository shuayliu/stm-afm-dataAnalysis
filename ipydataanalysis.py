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


if __name__ == '__main__':

    import configparser,argparse
    import pandas as pd
    import time
    startTime = time.clock()


    def usage():
        print('''
        args:
        
        -f finelame
        -v version
        -m mode
        
        
        ''')

    basePath = os.path.dirname(os.path.abspath(__file__))

    mode = 'single'
    filename = os.path.join(basePath,"Fc.txt")
    olnyDSA = False
    doDSA = True
    doSlopeAnalysis = False

#    opts,args = getopt.getopt(sys.argv[1:],shortArgs,longArgs)

    transformMethod = 'FixedLength'

    # 读取配置文件
    confPath = os.path.join(basePath,'config.ini')
    conf = configparser.ConfigParser()
    
    if os.path.isfile(confPath):
        conf.read(confPath)
        filename = conf.get('Configs','File_Path')
        SKIPROWS = conf.getint('DataFormat','Skiprows')
        mode = conf.get('Configs','Mode')
        forceConst = conf.getfloat('Configs','ForceConstant')
        DELIMITER = str(conf.get('DataFormat','Delimiter')).lower()
        DefSens = conf.getfloat('Configs','Deflection_Sensitive')
        transformMethod = str(conf.get('Configs','Transform_Method')).lower()
        stepLength = int(conf.get('Configs','Step_Length'))
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
        
        switchs = dict()
        
        if not switchOpts.parse_args().filenames == None:
            filename = switchOpts.parse_args().filenames
            
        switchs['onlyDSA'] = switchOpts.parse_args().onlyDSA
        switchs['doDSA'] = switchOpts.parse_args().doDSA
        switchs['doSlopeAnalysis'] = switchOpts.parse_args().doSlopeAnalysis
        forceConst = switchOpts.parse_args().forceConst   
        
    
    

#    判断力常数，如果为1.0则认为力常数未作修改，抛出警告
#    并且，力常数不能小于零
    if forceConst == 1.0:
        print('you MAY NOT set the proper ForceConstant, but the programme will keep going' )
    elif forceConst <0:
        print("ForceConstant < 0 ,it is illegal")
        sys.exit(3)
   
    


    if os.path.isfile(filename):
        print("data is file")
        if filename.endswith('.txt'):
            ForceCurveAnalysis(filename,switchs)

    elif os.path.isdir(filename):
        print(" data is dir")
        print(" Now dealing single file...")
        sys.stdout.write("#"*int(80)+'|')
        resultPath = os.path.join(filename,'/Result')
        if(os.path.exists(resultPath)):
            __import__('shutil').rmtree(resultPath)
        j=0
        for file in os.listdir(filename):
            _filename = os.path.join(filename,file)
            if os.path.isfile(_filename):
#                print(_filename)
                if file.endswith('.txt'):
                    savePath = ForceCurveAnalysis(_filename,switchs)
                j+=1
                sys.stdout.write('\r'+(j*80//len(os.listdir(filename)))*'-'+'-->|'+"\b"*3)
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
        
        mergeDataPath = os.path.join(savePath,'mergeData.dat')
        print("\n Finish with merging F-D Curves. Now writing all data to %s..."%os.path.relpath(mergeDataPath))
		
        np.savetxt(mergeDataPath,allData,delimiter='\t')
        print("\n Data Saved!,\n Now histograming all data...")
        # histogram 2D:
        x,y = allData.T
        H,xedges,yedges = np.histogram2d(x,y,bins=(155,310),
                                         range=[[-0.5,15],[-0.5,15]],normed=False)
        histDataPath = os.path.join(savePath,'histData.dat')
        np.savetxt(histDataPath,H.T)

        
        
        # 对参数文档进行分析
        # 这部分等TODO完善之后再进行uncomment
#        paraFile = os.path.join(filename,"result/paras.txt")
#        stat = pd.read_csv(paraFile,sep='\t',index_col=-1)
#        gp = stat.groupby('potential')
#        gpStat = gp.describe()
#        gpStat.to_csv(os.path.join(filename,"result/statstistic.txt"),sep='\t')
        

        times = time.clock() - startTime
        
        sys.stdout.write("\n \nFINISHED! at %s/result \n Times: %.2f s\n"%(filename,times))


    else:
        print("Parameter of programme error")
        sys.exit(2)

      