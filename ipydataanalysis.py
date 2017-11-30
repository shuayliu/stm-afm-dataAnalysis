# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:05:44 2017

@author: jonah
"""

VERSION = '1.0.1'
import numpy as np
import sys,os
from SpectroscopyTools.ForceTools import ForceTools
import warnings
warnings.filterwarnings("ignore")

def SlopeAnalysis(xData,yData,steplength=25):
    from scipy import stats
    Slope = []
    Intercept = []
    R2 = []
    halfStep = steplength//2

    for i in range(halfStep,len(xData)-halfStep):
        k,y0,r = stats.linregress(xData[i-halfStep:i+halfStep],
                             yData[i-halfStep:i+halfStep])[0:3]
        Slope.append(k)
        Intercept.append(y0)
        R2.append(r**2)
    #TODO: 后面想改成所有的线性回归参数
    return xData[halfStep:-halfStep],Slope,Intercept,R2

def PicoViewAnalysis(filename):

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


def ForceCurveAnalysis(pathname):
    ParameterList=dict()
    Data = dict()
    sens = np.nan
    kcant= np.nan
    tipRadius =np.nan
    steplength=np.nan
    
    
    if os.path.isfile(pathname):
        ParameterList,Data = PicoViewAnalysis(pathname)
        
        sens = float(ParameterList['deflectionSens'])
        forceConstant = float(ParameterList['forceConstant'])
        kcant = np.equal(forceConstant,1.0) and forceConstant or np.nan
        
        xData = Data['Distance']
        yData = Data['Force'] #in Voltage it is Deflection


        
        FDCycle = ForceTools(xData=xData,yData=yData,sens=sens,kcant=kcant)
        # Calculate Force
        Separation =  FDCycle.TraceX + FDCycle.TraceY * FDCycle.DeflSens;
        Force = FDCycle.ForceConst * FDCycle.DeflSens * FDCycle.TraceY


        # Extra Correction
        Separation =  Separation - FDCycle.MaxIndentation()
        
        # Get F-D cycle's parameters
        DeflSens = FDCycle.DeflectionSensitivity()
        AdhesionWork = FDCycle.AdhesionWork()
        MaxAdhesionForce = FDCycle.MaxAdhesionForce()
        MaxIndentation = FDCycle.MaxIndentation()
        ContactPoint = FDCycle.ContactPoint
 
       # Save F-D Cycle with Header
        SavedHeader = '''Separation(nm)\tForce(nN)
DeflSens = %.4e
AdhesionWork = %.4e
MaxAdhesionForce = %.4e
MaxIndetation = %.4e
ContactPoint  = %.4e
ForceConstant = %.4e
'''%(
DeflSens,
AdhesionWork,
MaxAdhesionForce,
MaxIndentation,
ContactPoint,
forceConstant)
        
        SavedData = np.array([Separation,Force]).T * 1e9
        
        filepath,filename = os.path.split(pathname)
        
        SavedPath = os.path.join(filepath,'Result')
        if not os.path.exists(SavedPath):
            os.makedirs(SavedPath+'/Analysis')
        
        np.savetxt(SavedPath+'/FC_'+filename,X=SavedData,delimiter='\t',
                   header=SavedHeader,comments=' ')
        
        
        # Save F-D Ananysis File 
        AnalysisFilename = SavedPath + '/Analysis/FDParameters.txt'
        # Write Header
        if not os.path.exists(AnalysisFilename):
            with open(AnalysisFilename,'w') as f:
                print('\t'.join(['DeflSens', 'AdhesionWork',
                                 'MaxAdhesionForce','MaxIndentation']),
                      file=f, sep='\t', end='\n')
        
        # Write parameters in append mode
        SavedParaList = '%.4e\t%.4e\t%.4e\t%.4e'%(DeflSens,AdhesionWork,
                                                  MaxAdhesionForce,MaxIndentation)
        with open(AnalysisFilename,'a') as f:
            print(SavedParaList, file=f, sep='\t', end='\n')  
            
       
        # Analysis slope for FD-Cycle
        if True == doSlopeAnalysis:
            steplength == np.nan and 20 or steplength
            # Default Tip Radius
            tipRadius == np.nan and 10e-9 or tipRadius # 
            
            SlopeAnalysisFilename = SavedPath +'/Analysis/Slope_' + filename
            # SlopeResult = np.array(SlopeAnalysis(FDCycle.TraceX*1e9,FDCycle.TraceY,
            # SlopeResult = np.array(SlopeAnalysis(Separation,Force/tipRadius,
            SlopeResult = np.array(SlopeAnalysis(Separation,Force/tipRadius,
                                                 steplength),dtype=float).T
                                                 
            np.savetxt(SlopeAnalysisFilename,X=SlopeResult,delimiter='\t',
                       header='Separation(nm) \t Slope(nN/nm) \t Intercept \t R^2 \n stepLength=%i'%steplength)
        
        return SavedPath
        
        
        
if __name__ == '__main__':

    import sys,os,getopt
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
        
        ffname = switchOpts.parse_args().filenames
        onlyDSA = switchOpts.parse_args().onlyDSA
        doDSA = switchOpts.parse_args().doDSA
        doSlopeAnalysis = switchOpts.parse_args().doSlopeAnalysis
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
            ForceCurveAnalysis(filename)

    elif os.path.isdir(filename):
        print("data is dir")
        print("Now dealing single file...")
        sys.stdout.write("#"*int(80)+'|')
        j=0
        for file in os.listdir(filename):
            _filename = os.path.join(filename,file)
            if os.path.isfile(_filename):
#                print(_filename)
                if file.endswith('.txt'):
                    savePath = ForceCurveAnalysis(_filename)
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
        np.savetxt(mergeDataPath,allData,delimiter='\t')
        
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
        
        sys.stdout.write("\n \n FINISHED! at %s/result \n Times: %.2f s\n"%(filename,times))


    else:
        print("Parameter of programme error")
        sys.exit(2)

      