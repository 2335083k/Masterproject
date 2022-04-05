"""
Class objects to be used for constaining p 
by Meryl
"""

import fTcosmo as cosmo

import numpy as np

import scipy.interpolate as inter

import astropy.visualization as vis
import emcee
import corner

import matplotlib.pylab as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif',size=12)
rc('legend', fontsize=12) 

c=299792.458 #speed of light(km/s)

class loglikep:
    """
    "loglike" object, takes catalogue data and modifies it according to chosen p and cosmo params. Can then sample for p posterior etc.
    data is a .txt file --- a catalogue containing luminosity distance, relative distance error, and redshift for N events.

    Priors is the set of priors to use for sampling for p and the true cosmo params used to modify the distance data. Can be  'Planck15', 'Planck18', 'Shoes19', 'Shoes22', or 'flat'. Defaults to SH0ES22.
    
    Once class is initialised, you can use:
    
    self.Sample(nsamples, nwalkers) uses emcee to sample the posterior function specified by the class options.
    
    Once the sampler has run you can immediately return:
    self.chain 
    self.samples: all samples
    self.psamples: just the marginalised p samples
    
    or run:
    self.ratio() to print the acceptance ratio
    self.Corner() to plot a corner plot using corner
    self.results() to print p symmetric +/- 95% confidence interval
    self.plot() to plot marginalised p posterior with 95% confindence intervals.
    """

    def __init__(self, data, priors='Planck18',p=0):
        self.p=p

        self.data = np.loadtxt(data) #data catalogue
        self.dLL = self.data.T[1] #retrieve distances
        self.sigmadLpc = self.data.T[2] #retrieve relative distance errors
        self.sigmadL= np.zeros(len(self.data.T[2])) #create empty array to store errors
        
        self.z=self.data.T[3] #retrieve redshifts

        if priors=='Planck18':
            self.H=67.37,0.54
            self.omm=0.3147,0.0074
        
        if priors=='Planck15':
            self.H=67.81, 0.92
            self.omm= 0.308, 0.012
            
        if priors=='SH0ES19':
            self.H=74.03, 1.42
            self.omm=0.30, 0.13
                
        if priors=="SH0ES22":
            self.H=73.30, 1.04
            self.omm=0.327, 0.016
            
        self.priorfun=self.logprior #gaussian priors unless using 'flat' priors
        
        if priors=="flat":
            self.H=73.30,2
            self.omm=0.327,0.03
            self.priorfun=self.flatprior            
        
        self.dgw=np.zeros(len(self.dLL)) #empty array to fill with modified distances
        self.dL=np.zeros(len(self.dLL)) #empty array to fill with scattered distances
        
        for i, I in enumerate(self.sigmadLpc):            
            self.dgw[i]=cosmo.DGW(self.z[i],self.H[0],self.omm[0],self.p) #modify distance
            self.sigmadL[i]=self.dgw[i]*self.sigmadLpc[i] #calculate new error
            self.dL[i]=np.random.normal(self.dgw[i], self.sigmadL[i]) #scatter
        
        if priors=='flat':
            #set prior limits on cosmo params
            self.minH=20
            self.maxH=300
            self.minomm=0.2
            self.maxomm=1
            
        else:
            self.minH=self.H[0]-5*self.H[1]
            self.maxH=self.H[0]+5*self.H[1]
            self.minomm=self.omm[0]-5*self.omm[1]
            self.maxomm=self.omm[0]+5*self.omm[1]
                        
    def  logprior(self,H0, omm):
        
        """
        ln(p(H0,omm))=ln(p(H0)+p(omm))
        
        """
        
        H0p=loggaussx(self.H[0],self.H[1],H0) #gaussian prior
        ommp=loggaussx(self.omm[0],self.omm[1],omm) #gaussian prior
              
        return H0p + ommp

    def flatprior(self, H0, omm):
        """
        ln(p(H0,omm))=ln(p(H0)+p(omm))
        prob is 0 unless params are within range
        
        """
        if self.minH<H0<self.maxH and self.minomm<omm<self.maxomm:
            prior=0
        else:
            prior=-np.inf
        return prior
   
    def logprob(self,theta):
        
        """       
        logprob function for emcee 
        
        inputs theta, an array of length ndim
        
        returns log posterior and log prior as a tuple
        """
        
        p, H0, omm = theta
    
        loglike= -np.inf
        
        logprior=-np.inf
        
        
        #prob=0 for silly parameter values
    
        if -1<p<1 and self.minH<=H0<=self.maxH and self.minomm<=omm<=self.maxomm:
            
            like=np.ones(len(self.z))*(-np.inf)
                  
            for i, I in enumerate(self.z):

                DGW=cosmo.DGW(I,H0,omm,p)
                state=np.isnan(DGW)
                
                if state == True:
                    like[i]=-np.inf
                else:
                    Pxd=loggaussx(self.dL[i],self.sigmadL[i],DGW)
                    priors=self.priorfun(H0, omm)
                    
                    like[i] = Pxd+priors
                
                   
            logprior =self.priorfun(H0,omm)
            if np.isinf(like.any()):
                loglike=-np.inf
            else:
                loglike=np.sum(like)
         
        
        return loglike, logprior       
        
    def Sample(self,sampleno, nwalkers):
        
        """
        run emcee sampler for far away sources in 3 parameters
        """
        
        self.ndim = 3
        self.sampleno=sampleno
        self.nwalkers=nwalkers
            
        sig=1e-5
            
        p0=np.zeros([self.nwalkers,self.ndim]) #empty array to fill with walker starting positions
        for i in range(self.nwalkers):
            #scatter around true values
            p0[i][0]=self.p + sig*np.random.randn() 
            p0[i][1]=self.H[0] + self.H[1]*np.random.randn()
            p0[i][2]=self.omm[0] + self.omm[1]*np.random.randn()
            
        sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.logprob, blobs_dtype="object",a=2) #initiate sampler object
        
        sampler.run_mcmc(p0,self.sampleno) #run sampler
        
        self.chain=sampler.chain
        self.acceptance=sampler.acceptance_fraction       
        self.samples=self.chain[:, 500:, :].reshape((-1, self.ndim))
        self.Dsamples=self.samples.T[0]
                        
        self.histosamp=np.reshape(self.chain[:,500:,0],self.nwalkers*(self.sampleno-500))

        self.Hsamp=np.reshape(self.chain[:,500:,1],self.nwalkers*(self.sampleno-500))

        binnum=40
        
        n, bins, patch = vis.hist(self.histosamp,bins=binnum)
      
        plt.clf() #dinnae
        
        #normalise
        summ=np.sum(n)
        width=bins[1]-bins[0]
        for i, I in enumerate(n):
            n[i]=I/(summ*width)
 
        self.Pm= n, bins
        
    def ratio(self):
        
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(self.acceptance)))
        
    def Corner(self, save=False):
        
        fig = corner.corner(self.samples, labels=["$p$","$H_0$", "$\Omega_m$"])
        if save==True:
            plt.savefig('cornerplot.pdf')

    
    def results(self):
        total=sum(self.Pm[0])
       
        n=self.Pm[0]      
      
        bins=self.Pm[1]
             
        self.shiftt=(bins[1]-bins[0])/2 #used to centre p(p) in bin for interpolating
              
        CDF=np.zeros(len(n)+1)
        for i, I in enumerate(self.Pm[0]):
            summ=0
            for j, J in enumerate(np.arange(i)):
                summ+=n[j]
                CDF[i]=summ/total
        CDF[-1]=1
        self.cdffunc=inter.interp1d(CDF,bins) #define inverse CDF function

        self.Max=self.cdffunc(0.5)+self.shiftt #median
         
        self.P16=self.cdffunc(0.16)+self.shiftt #lower 68
        self.P84=self.cdffunc(0.84)+self.shiftt #upper 68
        self.P05=self.cdffunc(0.05)+self.shiftt #lower 90
        self.P95=self.cdffunc(0.95)+self.shiftt #upper 90
        self.P975=self.cdffunc(0.975)+self.shiftt #upper 95
        self.P025=self.cdffunc(0.025)+self.shiftt #lower 95
        
        #CIs
        
        self.Pup=self.P84-self.Max
        self.Plow=self.Max-self.P16
        
        self.PPup=self.P95-self.Max
        self.PPlow=self.Max-self.P05
        
        self.PPPup=self.P975-self.Max
        self.PPPlow=self.Max-self.P025
        
        
        print('For $m$: {0:.4f}+{1:.4f}-{2:.4f}'.format(self.Max,self.Pup,self.Plow),'68% CI')
        print('For $m$: {0:.4f}+{1:.4f}-{2:.4f}'.format(self.Max,self.PPup,self.PPlow),'90% CI')
        print('For $m$: {0:.4f}+{1:.4f}-{2:.4f}'.format(self.Max,self.PPPup,self.PPPlow),'95% CI')
        print('LISA thinks $p$ is at least ', self.P025)
        print('and at most ', self.P975)
        
    def plot(self, savefig=False):

        bins=self.Pm[1][:-1]+self.shiftt
        nP = self.Pm[0]
        
        ran=np.linspace(bins[0],bins[-1],10000)
        P=inter.CubicSpline(bins,nP)

        rc('font', family='serif',size=18)
        rc('legend', fontsize=12) 

        fig = plt.figure(figsize=(7.1, 5))
        ax = fig.add_subplot(111)
        ax.plot(np.ones(100)*self.p,np.linspace(-np.max(self.Pm[0]),np.max(self.Pm[0])*2,100),c='black',lw=3)
        ax.plot(ran,P(ran),c='C0',lw=3)

        
        MAX=max(nP)

        ax.set_xlabel('$p$')
        ax.set_ylabel('$p(p)$')


        ax.set_ylim(-MAX/10,MAX+MAX/10)
        nonsym_lim_low=self.P025
        nonsym_lim_high=self.P975
        ax.xaxis.set_minor_locator(MultipleLocator(0.025))
        ax.tick_params('both',direction='in', which='major', width=2, length=8)
        ax.tick_params(direction='in', which='minor', width=1, length =5)
        ax.plot([nonsym_lim_low,nonsym_lim_low],[-max(nP),max(nP)*2],'C0',ls='--',lw=3)
        ax.plot([nonsym_lim_high,nonsym_lim_high],[-max(nP),max(nP)*2],'C0',ls='--',lw=3)
        
        if savefig==False:
            plt.show()
        
        else:
            plt.savefig('savefig.pdf', bbox_inches = "tight")
            plt.show()
    
    


"""
Stats stuffs
==================================================================
"""

def loggaussx(m,s,x):
    
    """
    A 1-d gaussian distribution
    Parameters
    ====
    m: float 
            mean
                
    s: float 
            standard deviation
            
    x: float or array-like
            variable
    Returns
    ====
    g(x): float or array-like
            the gaussian with mean m and standard deviation s evaluated at x
    """
    
    gauss = (-(x-m)**2)/(2*s**2)
    
    return gauss
