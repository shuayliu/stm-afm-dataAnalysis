# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:05:44 2017

@author: jonah
"""
import numpy as np
import sys,os
from SpectroscopyTools.ForceTools import ForceTools

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
        
        # Save F-D Cycle with Header
        SavedHeader = '''Separation(nm)\tForce(nN)
DeflSens = %.4e
AdhesionWork = %.4e
MaxAdhesionForce = %.4e
MaxIndetation = %.4e
ForceConstant = %.4e'''%(DeflSens,AdhesionWork,MaxAdhesionForce,MaxIndentation,forceConstant)
        
        SavedData = np.array([Separation,Force]).T * 1e9
        
        filepath,filename = os.path.split(pathname)
        
        SavedPath = os.path.join(filepath,'Result')
        if not os.path.exists(SavedPath):
            os.makedirs(SavedPath+'/Analysis')
        
        np.savetxt(SavedPath+'/FC_'+filename,X=SavedData,delimiter='\t',header=SavedHeader)
        
        
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
        steplength = 20
        SlopeAnalysisFilename = SavedPath +'/Analysis/Slope_' + filename
        # SlopeResult = np.array(SlopeAnalysis(FDCycle.TraceX*1e9,FDCycle.TraceY,
        SlopeResult = np.array(SlopeAnalysis(Separation,Force,
                                             steplength),dtype=float).T
        np.savetxt(SlopeAnalysisFilename,X=SlopeResult,delimiter='\t',
                   header='Separation(nm) \t Slope \t Intercept \t R^2 \n stepLength=%i'%steplength)
        
        
        
        
if __name__ == '__main__':

    import sys,os,getopt
    import configparser
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

    shortArgs = 'f:m:v:tfm'
    longArgs = ['file=', 
				'mode=', 
				'version',
				'noDSA',
				'onlyDSA','DSA'
				'tfMethod=',
				'stepLength=',
				'ForceConstant=']

    mode = 'single'
    filename = os.path.join(basePath,"Fc.txt")

    opts,args = getopt.getopt(sys.argv[1:],shortArgs,longArgs)

    transformMethod = 'FixedLength'

    # 读取配置文件
    confPath = os.path.join(basePath,'config.ini')
    if os.path.isfile(confPath):
        conf = configparser.ConfigParser()
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


    if not len(opts) == 0:
        for opt, val in opts:
            if opt in ('-f', '--filename='):
                filename = val
            elif opt in ('-m', '--model='):
                modelName = val
                modelPath = os.path.join(basePath, 'Models/' + modelName + '.py')
                if not os.path.exists(modelPath):
                    print("model does not exit", file=sys.stdout)
                    sys.exit(2)
            elif opt in ('-v', '--version'):
                    print('VERSION %s'.format(VERSIONS))
                    usage()
                    sys.exit(2)
            elif opt in ('--noDSA'):
                DSA = False
            elif opt in ('--onlyDSA','--DSA'):
                onlyDSA = True
            elif opt in ('-tfm','--tfMethod='):
                transformMethod = str(val).lower()
            elif opt in ('--stepLength=','--steplength='):
                stepLength = int(val)
            elif opt in ('--ForceConstant='):
                forceConst = float(val)



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
        print("Now dealing...")
        sys.stdout.write("#"*int(80)+'|')
        j=0
        for file in os.listdir(filename):
            _filename = os.path.join(filename,file)
            if os.path.isfile(_filename):
#                print(_filename)
                if file.endswith('.txt'):
                    ForceCurveAnalysis(_filename)
                j+=1
                sys.stdout.write('\r'+(j*80//len(os.listdir(filename)))*'-'+'-->|'+"\b"*3)
                sys.stdout.flush()

        # 对参数文档进行分析
        # 这部分等TODO完善之后再进行uncomment
#        paraFile = os.path.join(filename,"result/paras.txt")
#        stat = pd.read_csv(paraFile,sep='\t',index_col=-1)
#        gp = stat.groupby('potential')
#        gpStat = gp.describe()
#        gpStat.to_csv(os.path.join(filename,"result/statstistic.txt"),sep='\t')
        

        times = time.clock() - startTime
        
        sys.stdout.write("\nFINISHED! at %s/result \n Times: %.2f s\n"%(filename,times))


    else:
        print("Parameter of programme error")
        sys.exit(2)

      