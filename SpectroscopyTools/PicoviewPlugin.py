# -*- coding: utf-8 -*-
"""
PicoviewPlugin:
---------------
This module provides the PicoviewPlugin class. This class can be used to easily 
write new plug-ins, by deriving a new subclass from it and overwriting the MyPlugin
function.
"""

#:author: Christian Rankl <Christian_Rankl@agilent.com>
#
#:license: See LICENSE.txt for licensing terms

import sys

class PicoviewPlugin(object):
    """
    This class can be used to easily create Picoview plug-ins in Python. The new 
    plug-in can be created by deriving a new class and overwriting the :func:`MyPlugin`
    function.        
       
    Args:   
        ParameterList: :class:`list`
            List of Picoview parameters this plug-ins wants to use.
            The parameters are stored as dict in self.parameters.                 
        Unit: :class:`str`, optional
            Unit of the return value
        Connect: boolean, optional
            If True (default) then a connection to Picoview is established,
            and the plug-in is functional. If False then Picoview connection 
            is not established rendering the plug-in nonfunctional, but the 
            MyPlugin() function can be tested and debugged with offline
            data                            
    
    *Example:* A plug-in calculating the min Voltage level of a curve::
    
        class MinVoltPlugin(Picoview):
            def MyPlugin(self, xData, yData):
                minV = 10
                for data in yData:
                    tmpmin = min(data)
                    minV = min(minV, tmpmin)
                
                return minV
            
            plugin = MinVoltPlugin()
            plugin.Main()
    """
    def __init__(self, ParameterList = None, Unit = 'V', Connect = True):   
        self.connect = Connect;
        if self.connect:
            import SpectroscopyPlugIn
            
            # Setup
            self.plugin = SpectroscopyPlugIn.SpectroscopyPlugIn(sys.argv)
            
            if Unit != None:
                # Set unit
                self.plugin.SetParameter('Unit', Unit)
    
            if ParameterList != None:
                # Set parameter list
                self.plugin.SetParameterList(ParameterList)
    
                # Pass name of parameter handling function
                self.plugin.RegisterUpdateParameterList(self.HandleParameterList)
            else:
                self.parameters = dict()
                
            fun = lambda x,y: self.MyPlugin(x, y)            
    
            # Pass name of processing function
            self.plugin.RegisterProcessSpectroscopyData(fun)
        
    def Main(self):
        """
        Main loop, taking care of communication with PicoView.
        """
        if self.connect:
            # Processing loop
            while True:
                self.plugin.ProcessCommand()
            
    # Parameter handling function
    def HandleParameterList(self, param):
        """
        Function receiving the parameters from PicoView
        """
        self.parameters = param        
        
    def MyPlugin(self, xData, yData):
        """
        function used to analyze spectroscopy curves. It receives a list of curves and must return 
        a :class:`float` number. In addition it must return 0 if xData or yData is empty, otherwise the
        plug-in will not be recognized by PicoView.
        
        Args:
            xData: :class:`list`
                list of list of x positions of the individual force curves
            yData: :class:`list`
                list of list of y positions of the individual force curves
            
        Returns:
            Extracted property: :class:`float`
        """
        length = 0
        summation = 0
        for data in yData:
            length += len(data)
	    summation += sum(data)

        if length > 0:
            return summation / length
		
        return 0
