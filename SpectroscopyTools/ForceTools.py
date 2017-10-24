# -*- coding: utf-8 -*-
"""
ForceTools:
-----------

This module provides methods for determining several properties of force distance cycles, 
such as elastictiy or adhesion work.
"""

#:author: Christian Rankl <Christian_Rankl@agilent.com>
#
#:license: See LICENSE.txt for licensing terms

import numpy
import configparser as ConfigParser
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.optimize import curve_fit, fsolve, newton, minimize

def FitLine(xy, x, y, x2, y2):
    '''
    Fit a line through x, y data points and returns the found parameters as well 
    as the sum of the residual.
    Args:
        xy: 1-D :class:`numpy.array`
            sum (x*y)
        x : 1-D :class:`numpy.array`
            sum (x)
        y : 1-D :class:`numpy.array`
            sum (y)
        x2: 1-D :class:`numpy.array` 
            sum (x**2) 
        y2 : 1-D :class:`numpy.array`
            sum (y2)
    Returns: 
        (slope, intersect, ressum)
        
        slope: 
            the slope of the line
        intersect: 
            the intersection of the line with the y axis
        ressum: 
            the sum of the residuals
    '''
    n = numpy.arange(1, len(xy)+1);
    
    slope = (xy - x*y/n) / (x2-x*x/n)
    
    intersect = (y - slope*x) / n
    
    ressum = y2 + slope*slope * x2 + intersect*intersect * n - 2 * slope * xy \
        - 2 * intersect * y + 2 * slope * intersect * x
    
    return slope, intersect, ressum
    
def FitAllLineSegments(x, y, start = 0, ende = - 1):
    '''
    Fits all possible line segments starting which starts at start
    i.e. [slope[0], intersect[0]] = polyfit(x[start:start], y[start:start], 1)
         [slope[1], intersect[1]] = polyfit(x[start:start+1], y[start:start+1], 1)
         ...
         [slope[i], intersect[i]] = polyfit(x[start:start+i], y[start:start+i], 1)
         
   Args:
        x: 1-D :class:`numpy.array`
            xdata of points
        y: 1-D :class:`numpy.array`
            ydata of points     
        start: :class:`float`, optional
            start value, default = 0
        ende: :class:`float`, optional
            end value, default = -1
            
    Returns:
        slope: 1-D :class:`numpy.array` 
            slopes of the segments
        intersect: 1-D :class:`numpy.array` 
            y-axis intersections of the segments
        ressum: 1-D :class:`numpy.array`
            sums of residuals of the segments
    '''
    
    if ende == -1:
        ende = len(x)
    
    cxy = numpy.hstack([0, numpy.cumsum(x * y)])
    cx = numpy.hstack([0, numpy.cumsum(x)])
    cy = numpy.hstack([0, numpy.cumsum(y)])
    cx2 = numpy.hstack([0, numpy.cumsum(x * x)])
    cy2 = numpy.hstack([0, numpy.cumsum(y * y)])
    
    slope, intersect, ressum = FitLine(cxy[start+1:ende+1]-cxy[start], cx[start+1:ende+1]-cx[start], cy[start+1:ende+1]-cy[start], cx2[start+1:ende+1]-cx2[start], cy2[start+1:ende+1] - cy2[start])
    
    return slope, intersect, ressum
    
#def RobustFit(x, y, order=1, frac=0.95, n=3):
#    """
#    Very simple robust fit implementation by iterative leaving out the worst
#    points.
#    """
#        
#    p = numpy.polyfit(x, y, order)
#    
#    res = y - numpy.polyval(p, x)
#    res *= res
#    
#    if n > 0:
#        sresind = numpy.argsort(res)
#        ind = numpy.arange(0, int(numpy.floor(frac*len(sresind))))
#        goodind = sresind[ind]
#        if len(goodind) > order+2:
#            p = RobustFit(x[goodind], y[goodind], order, frac, n-1)
#        
#    return p
    
def RobustFit(x, y, frac = 0.95, n = 3):
    '''
    Simple way of fitting a linear to sample data in a robust fashion by 
    repetitive fitting and eliminating the worst data points
    
    Args:
        x:1-D :class:`numpy.array`
            x-values
        y:1-D :class:`numpy.array`
            y-values
        frac: :class:`float`, optional
            fraction of points to keep
        n: :class:`integer`, optional
            number of iterations
            
    Returns:
        returns best fit polynomial coefficient 
    '''
    
    sxy = numpy.sum(x*y)
    sx = numpy.sum(x)
    sy = numpy.sum(y)
    sx2 = numpy.sum(x*x)

    nr = len(x)        
    slope = (sxy - sx*sy/nr) / (sx2-sx*sx/nr)    
    intersect = (sy - slope*sx) / nr
    
    
    for k in range(0, n):
        res = slope*x+intersect-y
        res *= res
        
        sresind = numpy.argsort(res)
        
        cut = int(numpy.floor(frac*len(sresind)))
        
        if cut < 5:
            break
        
        ind = numpy.arange(cut, len(sresind))
        badind = sresind[ind]
        
        sxy -= numpy.sum(x[badind]*y[badind])
        sx -= numpy.sum(x[badind])
        sy -= numpy.sum(y[badind])
        sx2 -= numpy.sum(x[badind]*x[badind])
              
        ind = numpy.arange(0, cut)
        goodind = sresind[ind]
        x = x[goodind]
        y = y[goodind]        
        
        nr = len(x)        
        slope = (sxy - sx*sy/nr) / (sx2-sx*sx/nr)    
        intersect = (sy - slope*sx) / nr
        
    p = numpy.array([slope, intersect])
    
    return p        
        
        
class ForceTools:
    ''' 
        This class is used to analyze force distance curves.
    
        Args:
            xdata: :class:`numpy.array`
                piezo travel distance [m]
            ydata: :class:`numpy.array`
                cantilever deflection [V]
            sens: :class:`float`, optional
                sensitivity [m/V]
            kcant: :class:`float`, optional
                Spring constant of used cantilever [N/m]    
            ConfigFile: :class:`str`, optional
                Filename used to read configuration, default='ForceTools.cfg'
    '''    
        
    def __init__(self, xData, yData, sens=numpy.nan, kcant=numpy.nan, 
                 ConfigFile='ForceTools.cfg'):                

        
        self.ConfigFile = ConfigFile #: Name of the configuration file
        
        self.Config = ConfigParser.SafeConfigParser()
                
        succ = self.Config.read(self.ConfigFile)  
        
        if len(succ) == 0:
            self.WriteDefaultConfig()
            self.Config.read(self.ConfigFile)
                
        if numpy.isnan(sens):
            sens = self.Config.getfloat('Common', 'Sensitivity')
            
        if numpy.isnan(kcant):
            kcant = self.Config.getfloat('Common', 'ForceConstant')
                
        
        self.DeflSens = sens #: Deflection sensitivity [m/V]
        self.ForceConst = kcant #: Cantilever spring constant [N/m]
        self.SetData(xData, yData)
        
    def WriteDefaultConfig(self):
        defcfg = """# Parameter common to all ForceTools modules
[Common]
# Deflection sensitvity [m/V]
Sensitivity = 15e-9

# Cantilever spring constant [N/m]
ForceConstant = 0.3

# Parameters for Elastictiy determination
[Elasticity]
# Elasticity of the Tip [Pa]
E_tip = 310e9

# Poisson ration of tip [1]
nu_tip = 0.27

# Estimated Radius of the tip [m] for sphere model
TipRadius = 10e-9

# Radius of Cylindrical tip [m]
CylinderRadius = 0.9e-6

# Opening angle of pyramid [°]
PyramidAngle = 30

# Poisson ratio of the sample [1]
nu = 0.5

# If UseTrace is True the trace curve will be used for elastictiy determination
# otherwise the retrace curve will be used.
UseTrace = True

[AdhesionWork]

# Defines if the Force versus tip sample distance curve (UseIndentation = True) 
# or the usual Force versus piezo travel distance curve (UseIndentation = False)
# will be used to calculatet the Adhesion Work
UseIndentation = False
        """
        
        f = open(self.ConfigFile, 'wt')
        f.writelines(defcfg)
        f.close()
        
    def SetData(self, xData, yData):
        '''
        Assigns the given force distance cycle to the class and
        does some processing of it:
        
            - Splits the curve into trace and retrace 
            - Align the trace and retrace curve so that the both base lines 
              have the same level and subtracts this base line level
            - Determine the contact point and subtract it from the x data
            
        In addition deflection versus tip-sample distance curves are calculated 
        and saved in IndTraceX/Y and IndRetraceX/Y for trace and retrace, respectively. 
            
        
        Args:        
            xData: :class:`numpy.array`
                Piezo travel distance [m]
                
            yData: :class:`numpy.array`
                Cantilever deflection [V]
        '''
        
        # Save the raw data
        self.x = xData #: raw xData
        self.y = yData #: raw yData
        
        # Extrace Trace and Retrace part
        self.TraceX, self.TraceY, self.RetraceX, self.RetraceY = self.TraceRetrace(self.x, self.y)        
        
        
        # Align Trace and Retrace
        self.TraceX, self.TraceY, self.RetraceX, self.RetraceY = self.AlignTraceRetrace(self.TraceX, self.TraceY, self.RetraceX, self.RetraceY)
        
        # Find Contact Point 
        PosY = self.TraceY-min(self.TraceY)+0.1    
        lPosY = numpy.log(PosY)
        OptSplit, OptP1, OptP2 = self.BaseSlopeSplit(self.TraceX, lPosY, 2)
               
        # Contact point is the intersection between base line and contact slope
        #DiffOpt = OptP1 - OptP2
        #self.ContactPoint = -DiffOpt[1]/DiffOpt[0]
        
        # New contact point algorithm
        # Contact point is the point where all points are above the baseline
        tmp = lPosY - numpy.polyval(OptP2, self.TraceX)
        CptIdx = numpy.flatnonzero(numpy.diff(numpy.flatnonzero(tmp>0))!=1)
        if len(CptIdx) == 0:
            CptIdx = [0]
        
        self.ContactPoint = self.TraceX[CptIdx[0]]
        
        # Rescale X values that the origin is at the contact point
        self.RetraceX = self.RetraceX - self.ContactPoint
        self.TraceX = self.TraceX - self.ContactPoint
                
        # Convert from Piezo travel distance to tip-sample separation
        self.IndTraceX, self.IndTraceY = self.PiezoMovementToIndentation(self.TraceX, self.TraceY)
        self.IndRetraceX, self.IndRetraceY = self.PiezoMovementToIndentation(self.RetraceX, self.RetraceY)        
        
        #self.OptSplit, self.OptP1, self.OptP2 = self.BaseSlopeSplit(self.TraceX,
        #                                                            self.TraceY)
        
    def AlignTraceRetrace(self, tx, ty, rx, ry):
        '''
        Align Trace and Retrace curve to have the same base level and set this
        common level to zero. Returned are the aligned and leveled curves.
        
        Args:        
            tx: :class:`numpy.array`
                trace x-coordinates
            ty: :class:`numpy.array`
                trace y-coordinates
            rx: :class:`numpy.array`
                retrace x-coordinates
            ry: :class:`numpy.array`
                retrace y-coordinates
                
        Returns:     
            (tx, ty, rx, ry)
            
            tx: :class:`numpy.array`
                trace x-coordinates
            ty: :class:`numpy.array`
                retrace y-coordinates
            rx: :class:`numpy.array`
                trace x-coordinates
            ry: :class:`numpy.array`
                retrace y-coordinates
        '''
        # Evaluate ry at the positions of rx
        ry2 = interp1d(rx, ry, bounds_error=False)(tx);    
        
        ind = numpy.arange(int(round(2*len(tx)/3)), len(tx))
        
        diff = ty[ind]-ry2[ind]
        diff = diff[numpy.isfinite(diff)]
        diff = numpy.median(diff)
        
        ty = ty - diff/2
        ry = ry + diff/2
        
        # Determine zero level for deflection
        base = numpy.median(ty[ind])
        
        # Rescale deflection to start from zero
        ty = ty - base
        ry = ry - base
        
        return tx, ty, rx, ry        
    
    def PiezoMovementToIndentation(self, x, y):
        '''
        Calculate a deflection versus tip-sample distance curve
        '''
        x = x + y * self.DeflSens
        # sort the data will result a confusing on structure information
        #I = numpy.argsort(x);
        #x = x[I]
        #y = y[I]
        
        return x, y        
        
    def BaseSlopeSplit(self, x, y, robust = 0):
        '''
        Splits the force curve into two linear segments: baseline and slope region. 
        
        Args:
            x: :class:`numpy.array`
                x coordinates of curve
            y: :class:`numpy.array`
                y coordinates of curve
            robust: :class:`integer`, optional
                Bitfield indicating which segments to refine by :func:`RobustFit`. 
                Set first bit to refine first segment, and so on.
                
        Returns:
            (OptSplit, OptP1, OptP2)
            
            OptSplit: 
                Index indicating the best split point. First segment is ``range(0, OptSplit)``,
                second segment is ``range(OptSplit, len(x))``
            OptP1: :class:`numpy.array`
                coefficient for first segment
            OptP2: :class:`numpy.array`
                coefficient for second segment
        '''
        
        # First fit all LineSegments which starts at 0
        k1, d1, r1 = FitAllLineSegments(x, y)
        
        # Now fit all LineSegments which starts at the end
        k2, d2, r2 = FitAllLineSegments(x[::-1], y[::-1])
        k2 = k2[::-1]
        d2 = d2[::-1]
        r2 = r2[::-1]
        
        # Find best fit
        R = r1 + r2;
        OptSplit = numpy.nonzero(numpy.logical_and(R == numpy.nanmin(R), numpy.isfinite(R)))
        Optp1 = numpy.array([k1[OptSplit], d1[OptSplit]])
        Optp2 = numpy.array([k2[OptSplit], d2[OptSplit]])
        
        # Refine first segment
        if robust&1 == 1:
            ind = numpy.arange(0, int(OptSplit[0]))
            Optp1 = RobustFit(x[ind], y[ind], 0.9, 2)
            
        if robust&2 == 2:
            ind = numpy.arange(int(OptSplit[0]), len(x))
            Optp2 = RobustFit(x[ind], y[ind], 0.9, 2)            
        
        return OptSplit, Optp1, Optp2

        
    def TraceRetrace(self, x,y):
        '''
        Splits the force distance cycle in trace and retrace
        
        Args:
            x: :class:`numpy.array`
                x points of curve
            y: :class:`numpy.array`
                y points of curve
            
        Returns:            
            (TraceX, TraceY, RetraceX, RetraceY)
        '''
        
        split = int(numpy.floor(len(x)/2))
        
        ind1 = numpy.arange(0, split)
        ind2 = numpy.arange(split, len(x))
        
        # Direction of travel for the first segment
        # smaller than zero means approach
        direction = numpy.mean(numpy.diff(x[ind1]))
        
        if direction < 0:            
            trcx = x[ind1]
            trcy = y[ind1]
            
            rtcx = x[ind2]
            rtcy = y[ind2]
        else:
            trcx = x[ind2]
            trcy = y[ind2]
            
            rtcx = x[ind1]
            rtcy = y[ind1]
            
        # sort trace/retrace ==> contact part is consistently 
        # at the beginning of the arrays
        sind = numpy.argsort(trcx)
        trcx = trcx[sind]
        trcy = trcy[sind]
        
        sind = numpy.argsort(rtcx)
        rtcx = rtcx[sind]
        rtcy = rtcy[sind]
        
        return trcx, trcy, rtcx, rtcy
        
    
    def GetElasticity(self, TipRadius = numpy.nan, nu = numpy.nan, 
                      UseTrace = numpy.nan, Model = 'Sphere', 
                      debug= False):
        '''
        Determines the Elasticity[Pa] from a force distance curve using the 
        Hertzian contact mechanics for different intender shapes.
        Currently, supported indenter shapes are sphere, pyramid or cylinder pressing
        For details about the algorithm see Fuhrmann, A et al, Phys.Biol 8 (2011) 015007        
        and http://en.wikipedia.org/wiki/Contact_mechanics
        
        Args:
            TipRadius: :class:`float`, optinal
                Model depending parameter. For sphere it corresponds to the radius [m] of the indenter,
                for a cylinder it's the cylinder radius [m] and in case of a pyramid it's the 
                opening angle [°]. 
            nu: :class:`float`, optional
                Possion ratio of the sample used to convert from reduced Young's modulus to Young's modulus
            UseTrace: :class:`float`, optional
                If True the Trace is used for analysis, otherwise the Retrace curve is used.
            Model: :class:`string`, optional
                indenter shape, must be one of ['Sphere', 'Cylinder', 'Pyramid']
            debug:boolean, optional
                If True extra information will be returned
                
        Returns:
            If debug == False
                Young's modulus [Pa]
            otherwise
                (Young's modulus, Dist, Force, Dist, ModForce, Dist, model)
        '''                
            
        if numpy.isnan(nu):
            nu = self.Config.getfloat('Elasticity', 'nu')
            
        if numpy.isnan(UseTrace):
            UseTrace = self.Config.getboolean('Elasticity', 'UseTrace')
            
        # Convert to proper force units
        if UseTrace:
            Force = self.IndTraceY * self.DeflSens * self.ForceConst
            Dist = self.IndTraceX
        else:
            Force = self.IndRetraceY * self.DeflSens * self.ForceConst
            Dist = self.IndRetraceX
            
        if Model == 'Cylinder':
            if numpy.isnan(TipRadius):
                TipRadius = self.Config.getfloat('Elasticity', 'CylinderRadius')
                
            ModForce = Force
                
            # Get slope            
            cp , p1, p2 = self.BaseSlopeSplit(Dist, ModForce, 1)
                        
            
#            if abs(p1[0]) > abs(p2[0]):
#                slope = abs(p1[0])
#            else:
#                slope = abs(p2[0])
            # lower x values means pressing harder into the surface therefore 
            # the slope must be negative, if not something is wrong            
            if p1[0] < 0:
                slope = -p1[0]
            else:
                slope = numpy.inf         
                
            ERed = slope / (2 * TipRadius)
        elif Model == 'Sphere':            
            if numpy.isnan(TipRadius):
                TipRadius = self.Config.getfloat('Elasticity', 'TipRadius')
                
            # Make force positive
            Force = Force - numpy.min(Force)
            
            # Transform force into domain where elasticity is proprtional to slope
            ModForce = Force**(2./3)
        
            cp, p1, p2 = self.BaseSlopeSplit(Dist, ModForce, 1)            
            
            #
            #if abs(p1[0]) > abs(p2[0]):
            #    slope = abs(p1[0])
            #else:
            #    slope = abs(p2[0])
            
            # lower x values means pressing harder into the surface therefore 
            # the slope must be negative, if not something is wrong            
            if p1[0] < 0:
                slope = -p1[0]
            else:
                slope = numpy.inf                            
            
            ERed = slope**1.5 * 0.75/numpy.sqrt(TipRadius)                    
        elif Model == 'Pyramid':
            if numpy.isnan(TipRadius):
                Angle = self.Config.getfloat('Elasticity', 'PyramidAngle') / 180*numpy.pi
            else:
                Angle = TipRadius/180.*numpy.pi
            
            # Calculate angle between plane and side of pyramid
            Angle = numpy.pi / 2 - Angle
            # Make force positive
            Force = Force - numpy.min(Force)
            
            # Transform force into domain where elasticity is proprtional to slope
            ModForce = numpy.sqrt(Force)
        
            cp, p1, p2 = self.BaseSlopeSplit(Dist, ModForce, 1)
            
#            if abs(p1[0]) > abs(p2[0]):
#                slope = abs(p1[0])
#            else:
#                slope = abs(p2[0])
            # lower x values means pressing harder into the surface therefore 
            # the slope must be negative, if not something is wrong            
            if p1[0] < 0:
                slope = -p1[0]
            else:
                slope = numpy.inf         
            
            ERed = slope*slope * numpy.pi / 2 * numpy.tan(Angle)
        else:
            return numpy.nan
    
        
        E = self.Ered2E(ERed, nu)
        
        if debug == False:
            return E          
        else:
            cp = cp[0]
            model = Dist.copy()
            model[0:cp] = numpy.polyval(p1, Dist[0:cp])
            model[cp:] = numpy.polyval(p2, Dist[cp:])
            
            return (E, Dist, Force, Dist, ModForce, Dist, model)
        
    def Hertz(self, i, i0, Ered, TipRadius, FPull = 0, F0 = 0):
        newi = i - i0
        F = numpy.zeros(i.shape)
        ind = newi > 0
        F[ind] = 4./3*Ered*numpy.sqrt(TipRadius)*newi[ind]**1.5
        F-= F0
        return F

    def DMT(self, i, i0, Ered, TipRadius, Fpull, F0 = 0):        
        return self.Hertz(i, i0, Ered, TipRadius) + Fpull - F0
        
    def JKR(self, i, i0, Ered, TipRadius, Fpull, F0 = 0):
        
        newi = i - i0
        Fpull -= F0        
               
        wadh2 = -4./3*Fpull/TipRadius
        
        indent = lambda a, i, TipRadius, Ered, wadh2: a*a/TipRadius- \
            numpy.sqrt(a*wadh2/Ered) - i
            
#        dindent = lambda a, i, TipRadius, Ered, wadh2: 2*a/TipRadius- \
#            0.5 * numpy.sqrt(wadh2/(Ered*a))           
#        dindent2 = lambda a, i , TipRadius, Ered, wadh2 : 2 / TipRadius + \
#            0.25 * numpy.sqrt(wadh2/Ered) * a**(-1.5)            
            
        fzero = lambda x0, i: newton(indent, x0, 
                                     args= (i, TipRadius, Ered, wadh2))    
                                     
#        # For better numerical stability convert everything into nN, nm, etc.
#        # therefore the numbers will be closer to 1
#        newi *= 1e9;
#        TipRadius *= 1e9;
#        Ered *= 1e-9;                                             
#                                                 
        a = numpy.zeros(len(newi))
        
        #a[0] = newton(indent, 0, args=(newi[0], TipRadius, Ered, wadh))
        x0 = numpy.sqrt(newi[0]*TipRadius) # Start point
        a[0] = fzero(x0, newi[0])
        
        for (k, h) in enumerate(newi[1:], 1):
            try:
                a[k] = fzero(a[k-1], h)
            except:
                a[k] = numpy.nan
            
        Ea3 = Ered * a * a *a    
            
        F = 4.*Ea3/(3.*TipRadius) - 2* numpy.sqrt(Ea3*wadh2)
        
        F-=F0
        
#        # Convert Output back to N
#        F *= 1e-9

        return F     

    def difference(self, A, B):
        dif = A - B;
        dif = dif[numpy.isfinite(dif)]        
        return numpy.dot(dif, dif)/len(dif)
    

    def GetElasticity2(self, TipRadius = numpy.nan, nu = numpy.nan, 
                      UseTrace = numpy.nan, 
                      adhesion = 'DMT', debug= False):
        '''
        Determines the Elasticity from a force distance curve in case of adhesion. 
        Currently two adhesion models are supported: DMT and JKR. 
        
        Args:
            adhesion: :class:`string`
                Determines the model to use. Must be one of ["DMT", "JKR"]
            others: see :func:`GetElasticity`
        
        Returns:
            :func:`GetElasticity`
            
        '''
                          
        if numpy.isnan(TipRadius):
                TipRadius = self.Config.getfloat('Elasticity', 'TipRadius')                          
                                              
        if numpy.isnan(nu):
            nu = self.Config.getfloat('Elasticity', 'nu')
            
        if numpy.isnan(UseTrace):
            UseTrace = self.Config.getboolean('Elasticity', 'UseTrace')
            
        # Convert to proper force units
        if UseTrace:
            Force = self.IndTraceY.copy() * self.DeflSens * self.ForceConst
            Dist = self.IndTraceX.copy()
        else:
            Force = self.IndRetraceY.copy() * self.DeflSens * self.ForceConst
            Dist = self.IndRetraceX.copy()        
            
        # Convert to nN and nm so that numbers are close to 1.             
        Force *= 1e9
        Dist *= 1e9
        TipRadius *= 1e9
            
        ind = Dist < 0            
                        
        Fpull = numpy.min(Force)
        E0 = self.GetElasticity()/1e9
        
        
        if adhesion == 'DMT':
            model = lambda x: self.DMT(-Dist[ind], x[0], x[1], TipRadius, Fpull, 0)
        elif adhesion == 'JKR':
            model = lambda x: self.JKR(-Dist[ind], x[0], x[1], TipRadius, Fpull, 0)
        
        dif = lambda x: self.difference(Force[ind], model(x))
                                                    
        p = minimize(dif, (0, E0), method = 'Nelder-Mead')                                                 
        
        Ered = p.x[1] *1e9
        
        E = self.Ered2E(Ered, nu)
        
        if debug == False:
            return E
        else:
            return (E, Dist, Force, Dist[ind], Force[ind], Dist[ind], model(p.x))
            
    def OliverPharr(self, TipRadius = numpy.nan, nu = numpy.nan, 
                      UseTrace = numpy.nan, epsilon = 0.76, beta = 1., 
                      debug= False):
                          
        if numpy.isnan(TipRadius):
            TipRadius = self.Config.getfloat('Elasticity', 'TipRadius')                          
                                              
        if numpy.isnan(nu):
            nu = self.Config.getfloat('Elasticity', 'nu')
            
        if numpy.isnan(UseTrace):
            UseTrace = self.Config.getboolean('Elasticity', 'UseTrace')
            
        # Convert to proper force units
        if UseTrace:
            Force = self.IndTraceY.copy() * self.DeflSens * self.ForceConst
            Dist = -self.IndTraceX.copy()
        else:
            Force = self.IndRetraceY.copy() * self.DeflSens * self.ForceConst
            Dist = -self.IndRetraceX.copy()            
            
        # Find Stiffness by fitting line to part between 100% and 75% of max force            
        Fmax = numpy.nanmax(Force)
        ind = Force > 0.75*Fmax
        if ind[0] != True:
            return numpy.nan
            
        ind = numpy.flatnonzero(numpy.diff(ind))
        ind = ind[0]
        
        p = RobustFit(Dist[0:ind], Force[0:ind], 0.95, 2)
        
        S = p[0]
        
        ic = numpy.nanmax(Dist)-epsilon*Fmax/S
        
        ics2 = 1/numpy.sqrt(ic)
        
        C = numpy.ones(9)
        ics = numpy.hstack([ic*ic, ic, ics2**numpy.arange(1, 8)])
        Aic = C*ics
        
        Ered = numpy.sqrt(numpy.pi)/(2.*beta)*S/numpy.sqrt(Aic)        
                                              
        return self.Ered2E(Ered, nu)
        
    def Ered2E(self, Ered, nu = 0.5, Et = None, nut = None):
        
        if Et == None:
                Et = self.Config.getfloat('Elasticity', 'E_tip')                          
                                              
        if nut == None:
            nut = self.Config.getfloat('Elasticity', 'nu_tip')
        
        return 1./(1./Ered-(1-nut*nut)/Et)*(1-nu*nu)                         
        
    def MaxIndentation(self):
        '''
        Determines the maximum indentation [m] of the cantilever into the surface
        '''
        
        MaxIndent = self.IndTraceX[0]
        
        return MaxIndent
        
    def MaxAdhesionForce(self):
        '''
        Determines the maximum adhesion force [N] between tip and surface
        '''
        
        MinForce = min(self.RetraceY) * self.DeflSens * self.ForceConst
        return -MinForce
        
    def AdhesionWork(self, UseIndentation = numpy.nan):
        '''
        Calculates the adhesion work, which is the area between trace and retrace.
        
        Args:
            UseIndentation: boolean
                if True (default) the deflection versus tip-sample distance 
                curves to calculate the work, otherwise it uses the deflection versus 
                piezo movement curves
            
        Returns:
            AdhesionWork: :class:`float`
        '''
        
        if numpy.isnan(UseIndentation):
            UseIndentation = self.Config.getboolean('AdhesionWork', 'UseIndentation')
        
        if UseIndentation:
            UseRX = self.IndRetraceX
            UseRY = self.IndRetraceY
            UseTX = self.IndTraceX
            UseTY = self.IndTraceY
        else:
            UseRX = self.RetraceX
            UseRY = self.RetraceY
            UseTX = self.TraceX
            UseTY = self.TraceY
            
        xall = numpy.hstack([UseTX, UseRX])        
        
        xall = numpy.unique(xall)        
        
        InterT = interp1d(UseTX, UseTY, bounds_error=False)(xall)
        InterR = interp1d(UseRX, UseRY, bounds_error=False)(xall)
        
        diff = InterR - InterT
        diff = diff * self.DeflSens * self.ForceConst
        
        isfine = numpy.isfinite(diff)
        
        work = simps(diff[isfine], xall[isfine])
        
        return -work
        
    def DeflectionSensitivity(self):
        """
        Determines the deflection sensitivity of the optical detection system in [m/V] from the 
        force distance curve.
        """
        ind = self.TraceX < 0
        ContactX = self.TraceX[ind]
        ContactY = self.TraceY[ind]
        
        if len(ContactX) > 5:
            p = RobustFit(ContactX, ContactY, frac=0.9, n=3)
        else:
            cp, p, p2 = self.BaseSlopeSplit(self.TraceX, self.TraceY)
                        
        return -1/p[0]
            
    
    
        
        
        
    
