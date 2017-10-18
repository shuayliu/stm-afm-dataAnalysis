# -*- coding: utf-8 -*-
#:author: Christian Rankl <Christian_Rankl@agilent.com>   
#
#:license: See LICENSE.txt for licensing terms

"""
DeflectionSensitivity
---------------------
  This plug-in determines the deflection sensitivity.

Usage
^^^^^
  - Make force distance cycles

Algorithm
^^^^^^^^^
  The sensitivity is reciprocal value of the slope of the trace at contact.
  

Assumptions
^^^^^^^^^^^    
  The plug-in does not check if the force curve is invalid, such as curves without
  touching the surface or curves having a plateau on the left side due to photo-diode
  saturation.

"""

#######################################
#
# DO NOT CHANGE SECTION BELOW
#
#######################################
from SpectroscopyTools import ForcePlugin

class DeflSens(ForcePlugin):
    def GetValue(self, FDCycle):
        return FDCycle.DeflectionSensitivity()

if __name__ == '__main__':
    plugin = DeflSens(Unit = 'm/V')
    plugin.Main()
