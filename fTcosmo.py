"""
Assorted cosmology functions
Functions assuming flat fT cosmology ie omega_k,omega_r=0 and omega_DE=1-omega_m
by Meryl
"""
import numpy as np

c=299792.458 #speed of light(km/s)

import scipy

from scipy.integrate import quad


def dH(H0):
    """
    Hubble Distance

    Parameters
    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)

    Returns
    ====
    d_H: float or array-like
             Hubble distance, d_H defined as the speed of light times the Hubble time: c/H_0

    """
    return c/H0

def EZ(z,omm,p):   
    """
    The Hubble Function

    Parameters
    ====
            
    p:float or array-like
            p, modification parameter (dimensionless)
            
            
    omm: float or array-like
            omega_m, total matter density (dimensionless)

            
    z: float or array-like
            redshift (dimensionless)

    Returns
    ====
    E(z): float or array-like
            a function in z
            
    """
    
    E=omm*(z + 1)**3 + (1 - omm)*(omm*(z + 1)**3 - (omm - 1)*(omm*(z + 1)**3 - (omm - 1)*(omm*(z + 1)**3 - (omm - 1)*(omm*(z + 1)**3 - (omm - 1)*(omm*(z + 1)**3 - omm + 1)**p)**p)**p)**p)**p

    return np.sqrt(E)

def Integrand(x,omm, p):    
    
        
    """
    Function to be integrated in order to find the comoving distance

    Parameters
    ====
    
    x: float or array-like
            Dummy variable for z
                
    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            
    p: float or array-like
            p, modification parameter (dimensionless)
            

    Returns
    ====
    f(x): float or array-like
            f(x)=1/E(x)

    """
    
    return 1/EZ(x,omm, p)


def DL(z, H0, omm, p):
    
    """
    Luminosity Distance

    Parameters
    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)

    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            

    z: float or array-like
            redshift (dimensionless)
            
    p: float or array-like
            p, modification parameter (dimensionless)

    Returns
    ====
    DL: float or array-like
            dEML, electromagnetic luminosity distance (Mpc)
            
            dEML=(1+z)*dm
            
            where dm is the transverse comoving distance
    """
    
    integral=quad(Integrand, 0, z, args=(omm, p),epsabs=1e-3) #integrate f(x) from x=0 to z
    
    DM=dH(H0)*integral[0]
    
    return (1+z)*DM

def DGW(z, H0,omm, p):
    
    """
    Gravitational Wave Luminosity Distance

    Parameters
    ====

    H0: float or array-like
            Hubble constant, with different prior depending on SHoES or Planck measurement ((km/s)/Mpc)

    
    omm: float or array-like
            omega_m, total matter density (dimensionless)
            

    z: float or array-like
            redshift (dimensionless)
            
    p: float or array-like
            p, modification parameter (dimensionless)

    Returns
    ====
    DLGW: float or array-like
            dLGW, electromagnetic luminosity distance (Mpc)
            
            dLGW=dLEM*sqrt(fT(0)/fT(z))
            
            where dm is the transverse comoving distance
    """
        
    d=DL(z,H0,omm,p)
    E=EZ(z,omm,p)
    fr=(omm*p+p-1)/((2*p-1)+(omm-1)*p*E**(2*p-2))
    return d*np.sqrt(fr)



