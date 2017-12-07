# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:20:13 2017

@author: jonah
"""
VERSION = '1.0.1'
import numpy as np
import sys,os
from SpectroscopyTools.ForceTools import ForceTools
from scipy import stats
from AnalysisTools.PicoViewAnalysis import PicoViewDealing


def SlopeAnalysis(xData,yData,steplength=25):
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




def ForceCurveAnalysis(pathname,switchs):
    
    if not os.path.isfile(pathname):        
        return  print('%s is not a file! \n Please check.'%pathname)
    
    ParameterList=dict()
    Data = dict()
    sens = np.nan
    kcant= np.nan
    tipRadius =np.nan
    steplength=np.nan   
    
    
    
    ParameterList,Data = PicoViewDealing(pathname)
    
    sens = float(ParameterList['deflectionSens'])
    forceConstant = float(ParameterList['forceConstant'])
    kcant = np.equal(forceConstant,1.0) and forceConstant or np.nan
    # meter to top-left
    tipX = float(ParameterList['tipX']) / float(ParameterList['xPixels']) * float(ParameterList['xLength'])
    tipY = float(ParameterList['tipY']) / float(ParameterList['yPixels']) * float(ParameterList['yLength'])
    
    
    xData = Data['Distance']
    yData = Data['Force'] #in Voltage it is Deflection


    # for single Force-Distance cycle:
    FDCycle = ForceTools(xData=xData,yData=yData,sens=sens,kcant=kcant)
    # Calculate Force
    Separation =  FDCycle.TraceX + FDCycle.TraceY * FDCycle.DeflSens;
    Force = FDCycle.ForceConst * FDCycle.DeflSens * FDCycle.TraceY


    # Extra Correct Max Indentation to Zero;
    # for our system in Gold substrate it is not necessary to calculate max indentation.
    Separation =  Separation - FDCycle.MaxIndentation()
    
    # Get F-D cycle's parameters
    DeflSens = FDCycle.DeflectionSensitivity()
    AdhesionWork = FDCycle.AdhesionWork()
    MaxAdhesionForce = FDCycle.MaxAdhesionForce()
    MaxIndentation = FDCycle.MaxIndentation()
    ContactPoint = FDCycle.ContactPoint
 
   # Save F-D Cycle with Header
    SavedHeader = '''Separation(nm)\tForce(nN)
DeflSens = %.4g
AdhesionWork = %.4g
MaxAdhesionForce = %.4g
MaxIndetation = %.4g
ContactPoint  = %.4g
ForceConstant = %.4g
tipX = %.4g
tipY = %.4g
'''%(
DeflSens,
AdhesionWork,
MaxAdhesionForce,
MaxIndentation,
ContactPoint,
forceConstant,
tipX,
tipY
)
    
    SavedData = np.array([Separation,Force]).T * 1e9
    
    filepath,filename = os.path.split(pathname)
    
    SavedPath = os.path.join(filepath,'Result')
    if not os.path.exists(SavedPath):
        os.makedirs(SavedPath+'/Analysis')
    
    np.savetxt(SavedPath+'/FC_'+filename,X=SavedData,delimiter='\t',
               header=SavedHeader,comments=' ')
    
    
    # Save F-D Ananysis File 
    AnalysisFilename = SavedPath + '/Analysis/FDParameters.txt'

    
    # Write parameters in append mode
  
    SavedParaHeader = ['DeflSens', 
                       'AdhesionWork',
                       'MaxAdhesionForce',
                       'MaxIndentation',
                       'tipX',
                       'tipY'
                       ]
    SavedParaList =   ['%.4e'%DeflSens,
                       '%.4e'%AdhesionWork,
                       '%.4e'%MaxAdhesionForce,
                       '%.4e'%MaxIndentation,
                       '%.4e'%tipX,
                       '%.4e'%tipY
                       ]
  
    if not os.path.exists(AnalysisFilename):
        with open(AnalysisFilename,'w') as f:
            print('\t'.join(SavedParaHeader), file=f, end='\n')
    
    
    with open(AnalysisFilename,'a') as f:
        print('\t'.join(SavedParaList), file=f, end='\n')  
        
   
    # Analysis slope for FD-Cycle
    if True == switchs['doSlopeAnalysis']:
        steplength == np.nan and 20 or steplength
        # Default Tip Radius
        tipRadius == np.nan and 10e-9 or tipRadius # 
        
        SlopeAnalysisFilename = SavedPath +'/Analysis/Slope_' + filename
        
        SlopeResult = np.array(SlopeAnalysis(Separation,Force/tipRadius,
                                             steplength),dtype=float).T
                                             
        np.savetxt(SlopeAnalysisFilename,X=SlopeResult,delimiter='\t',
                   header='Separation(nm) \t Slope(nN/nm) \t Intercept \t R^2 \n stepLength=%i'%steplength)
    
    return SavedPath
