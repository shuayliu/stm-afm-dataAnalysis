# -*- coding: utf-8 -*-
"""


"""

#:author: Christian Rankl <Christian_Rankl@agilent.com>
#
#:license: See LICENSE.txt for licensing terms

from PicoviewPlugin import PicoviewPlugin

class ForcePlugin(PicoviewPlugin):
    """
    A class specialized for providing support in writing analysis plug-ins for
    force distance curves. By default it receives the deflection sensitivity and 
    spring constant from PicoView. It provides a valid :func:`PicoviewPlugin.MyPlugin` function.
    In order to write your own plug-in derive a new class from 
    :class:`ForcePlugin` and overwrite the :func:`GetValue`. 
    
    *Example:* A plug-in extracting the x-position at the turning point, which could be 
    used to get a topography image::
    
        class Topography(ForcePlugin):
            def GetValue(FDCycle):
                return FDCycle.TraceX[0]
        
        if __name__ == '__main__':
            plugin = Topography(Unit = 'm')
            plugin.Main()
            
    """
    def __init__(self, ParameterList = ['deflectionSensitivity', 'forceConstant'], Unit = 'V'):
        PicoviewPlugin.__init__(self, ParameterList=ParameterList, Unit=Unit)
        
    def GetValue(self, FDCycle):
        """
        Function used to determine a single measure per force distance cycle.
        
        Args:
            FDCycle: :class:`ForceTools` object
                class used for manipulating the force distance curve
            
        Returns: 
            :class:`float` - Extracted property
        """
        return -2.0
        
    def Reduction(self, Values):
        """
        Function used to reduce the list of measures of a list of force distance 
        cycles into a single value. The default is to calculate the mean.
        
        Args:
            Values: :class:`numpy.array`
            
        Returns:
            :class:`float` - Value representing the list of values.                                     
        """
        from numpy import mean
        return mean(Values)
        
    def MyPlugin(self, xData, yData):
        if len(xData) == 0 or len(yData) == 0:
            return 0.0
            
        if len(xData) != len(yData):
            return -1.0             
            
        from numpy import nan
        
        if 'deflectionSensitivity' in self.parameters:
            sens = self.parameters['deflectionSensitivity']
        else:
            sens = nan
            
        if 'forceConstant' in self.parameters:
            kcant = self.parameters['forceConstant']
        else:
            kcant = nan            
			
        from numpy import array, zeros
        from ForceTools import ForceTools
                
        Values = zeros(len(xData))
        
        for ind in range(0, len(xData)):
            x = array(xData[ind])
            y = array(yData[ind])            
                
            try:
                FDCycle = ForceTools(x, y, sens, kcant)    
            except:
                return -2.5
            
            try:                
                val =  self.GetValue(FDCycle)               
            except:
                return -3.0
                pass
                
            Values[ind] = val
            
        return self.Reduction(Values)

