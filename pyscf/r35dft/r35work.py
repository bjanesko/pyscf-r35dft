#!/usr/bin/env python
# Work routines for R35 DFT 
import time
import numpy
import math 
from scipy import linalg
from scipy.special import lambertw, erf, erfc 
from pyscf import gto 

# Initialize the r35beta array based on requested functional 
def _initialize_r35beta(r35betain):
    if isinstance(r35betain,list):
        return r35betain 
    # Default value is for m11po 
    r35beta= [0.2, 0.6,
         -0.170265960, 0.32610431 , 0,            0.173830277, -0.337808383,
          0,          -0.002379857, 2.026450260, -0.943023593, -4.195668115]
    if isinstance(r35betain,str):
        if('B05' in r35betain.upper()): # Unmodified B05 correlation 
          r35beta= [0.2, 0.6,
            0.,0.,0.,0.,0.,
            0.5,0.,0.,0.,0.]
    return r35beta 

# Overlap-R35 alternative to M11plus Coulomb-R35 energy
# M11plus Coulomb-R35 energy is eq 7 of M11plus paper 
# Input: Spin-polarized density on grid and R35 exponent beta 
# Return: Spin-polarized energy density and derivative array 
# Array rho is assumed to be [rho,gx,gy,gz,EMPTY,tau,eta35]
# Array v is assumed to be [rho,gx,gy,gz,tau,eta35] 
def get_e35(rho,beta,coords=None):
  #fac= -2*(2/math.pi)**0.5*beta**0.5
  # May 5 2026 
  fac=-1
  if(len(rho.shape)<3):
     raise ValueError('Not enough dimensions in R35 e35 rho ')
  nspin,nrho,ngrid = rho.shape 
  if(nrho<7):
     raise ValueError('Not enough density slots in R35 e35 rho ')
  e35  = numpy.zeros((nspin,ngrid))
  n35  = numpy.zeros((nspin,ngrid))
  ve35 = numpy.zeros((nspin,6,ngrid))
  for i in range(nspin):
    r = rho[i,0]
    r = numpy.maximum(r,1e-10*numpy.ones_like(r))
    eta = (rho[i,6]**2)**0.5
    eta = numpy.maximum(eta,1e-10*numpy.ones_like(eta)) 
    n35[i]    = eta**2/r # |EDR(r,d=beta^(-1/2))|^2 
    e35[i]    = fac*r**(2./3.)*eta**(4./3.)
    ve35[i,0] = (2./3.)*fac*r**(-1./3.)*eta**(4./3.)
    ve35[i,5] = (4./3.)*fac*r**( 2./3.)*eta**(1./3.)

  # I don't know why but this term is needed to reproduce Gaussian May 5 2026
  # fixed, there was an issue in the value of fac for general beta. This term
  # is no longer needed after that bugfix. 
  #e35 = e35*2**(0.5)
  #ve35 = ve35*2**(0.5)

  # Debug print
  if(coords is not None):
    if(coords.shape[0]==ngrid):
      for ig in range(ngrid):
        print('AAA %.6f %.6f %.6f %.6e %.6e %.6e %.6e  '%(coords[ig,0],coords[ig,1],coords[ig,2],rho[0,0,ig],rho[0,6,ig],n35[0,ig],e35[0,ig]))
        pass
  return(e35,ve35)

# Overlap-R35 implementation of M11plus R35 normalization, eq19 of M11plus paper 
def get_n35(rho, beta,coords=None):
  if(len(rho.shape)<3):
     raise ValueError('Not enough dimensions in R35 e35 rho ')
  nspin,nrho,ngrid = rho.shape 
  if(nrho<7):
     raise ValueError('Not enough density slots in R35 e35 rho ')
  n35  = numpy.zeros((nspin,ngrid))
  vn35 = numpy.zeros((nspin,6,ngrid))
  e35, ve35 = get_e35(rho,beta)
  enga, venga = get_enga(rho,beta)
  enga = numpy.minimum(enga,-1e-10*numpy.ones_like(enga))
  sma = 1e-6*numpy.ones_like(enga)
  sma0 = sma[0]
  sma2 = 2*sma[0]

  n35 = (e35-2*sma)/(enga-sma) # eq. 19 of M11plus paper, with small regularization to avoid divide by zero

  # Debug print
  if(coords is not None):
    if(coords.shape[0]==ngrid):
      for ig in range(ngrid):
        print('HHH %.6f %.6f %.6f %.6e %.6e %.6e %.6e  '%(coords[ig,0],coords[ig,1],coords[ig,2],rho[0,0,ig],e35[0,ig],enga[0,ig],n35[0,ig]))
        pass

  # Derivative of n35 with respect to rho, using the chain rule and the derivatives of e35 and enga
  for i in range(nspin):
    r = rho[i,0]
    r = numpy.maximum(r,1e-10*numpy.ones_like(r))
    eta = (rho[i,6]**2)**0.5
    eta = numpy.maximum(eta,1e-10*numpy.ones_like(eta))
    venga_rho = venga[i,0]
    ve35_rho = ve35[i,0]

    # Chain rule: dn35/drho = (de35/drho * (enga - 1e-6) - (e35 - 2e-6) * dega/drho) / (enga - 1e-6)^2
    vn35[i,0] = (ve35_rho  * (enga[i] - sma0) - (e35[i] - sma2) * venga_rho) / (enga[i] - sma0)**2
    vn35[i,1] = (ve35[i,1] * (enga[i] - sma0) - (e35[i] - sma2) * venga[i,1]) / (enga[i] - sma0)**2
    vn35[i,2] = (ve35[i,2] * (enga[i] - sma0) - (e35[i] - sma2) * venga[i,2]) / (enga[i] - sma0)**2
    vn35[i,3] = (ve35[i,3] * (enga[i] - sma0) - (e35[i] - sma2) * venga[i,3]) / (enga[i] - sma0)**2
    vn35[i,4] = (ve35[i,4] * (enga[i] - sma0) - (e35[i] - sma2) * venga[i,4]) / (enga[i] - sma0)**2
    vn35[i,5] = (ve35[i,5] * (enga[i] - sma0) - (e35[i] - sma2) * venga[i,5]) / (enga[i] - sma0)**2   
  return(n35,vn35)

def get_n35old(rho):
  if(len(rho.shape)<3):
     raise ValueError('Not enough dimensions in R35 e35 rho ')
  nspin,nrho,ngrid = rho.shape 
  if(nrho<7):
     raise ValueError('Not enough density slots in R35 e35 rho ')
  n35  = numpy.zeros((nspin,ngrid))
  vn35 = numpy.zeros((nspin,6,ngrid))
  for i in range(nspin):
    r = rho[i,0]
    r = numpy.maximum(r,1e-8*numpy.ones_like(r))
    n35[i]    = rho[i,6]**2/r
    vn35[i,0] = -1.0*rho[i,6]**2/r**2
    vn35[i,5] = 2*rho[i,6]/r
  return(n35,vn35)

# FB05 nondynamical correlation factor, spin-polarized terms, each spin case of
# eq 18 of M11plus paper 
def get_Fn35(rho,n35,vn35,Nmax=0.6,coords=None):
  b=5
  nspin,ngrid = n35.shape 
  nmax = Nmax*numpy.ones_like(n35[0])
  Fn35  = numpy.zeros_like(n35)
  vFn35 = numpy.zeros_like(vn35) 
  for i in range(nspin):
    Fn35[i] = .5*(1-erf(b*(n35[i]-nmax)))
    vFn35[i] = vn35[i]*(-b*math.pi**(-0.5)*numpy.exp(-b**2*(n35[i]-nmax)**2))

  # Debug print
  if(coords is not None):
    if(coords.shape[0]==ngrid):
      for ig in range(ngrid):
        print('NNN %.6f %.6f %.6f %.6e %.6e  '%(coords[ig,0],coords[ig,1],coords[ig,2],n35[0,ig],Fn35[0,ig]))
        pass
  return(Fn35,vFn35)

# Assemble all cross-spin terms for the B05-type correlation 
# Input is the corresponding spin-polarized terms, except for the linear part 
def get_eB05(rho,e35,ve35,Fn35,vFn35,Sd,vSd,coords=None):
  nspin,nrho,ngrid = rho.shape 
  eB  = numpy.zeros(ngrid)
  veB = numpy.zeros((nspin,6,ngrid))

  ra=rho[0,0]
  ra = numpy.maximum(ra,1e-10*numpy.ones_like(ra))
  rb=rho[1,0]
  rb = numpy.maximum(rb,1e-10*numpy.ones_like(rb))

  # Initial B05-type cross-spin term 
  eb1 = e35[0]*rb/ra + e35[1]*ra/rb
  veb1 = numpy.zeros_like(ve35)
  veb1[0] = veb1[0] + ve35[0]*rb/ra # Derivs wrt e35
  veb1[1] = veb1[1] + ve35[1]*ra/rb
  veb1[0,0] = veb1[0,0] -e35[0]*rb/ra**2 + e35[1]/rb # Deriv wrt rho_a 
  veb1[1,0] = veb1[1,0] -e35[1]*ra/rb**2 + e35[0]/ra # Deriv wrt rho_b
  
  # Sd density scaling cross-spin term 
  S = Sd[0]*Sd[1] 
  vS = numpy.zeros_like(ve35)
  vS[0] = vSd[0]*Sd[1] 
  vS[1] = vSd[1]*Sd[0] 

  # FB05 nondynamical correlation factor cross term 
  FB = Fn35[0]*Fn35[1]
  vFB = numpy.zeros_like(ve35)
  vFB[0] = vFn35[0]*Fn35[1] 
  vFB[1] = vFn35[1]*Fn35[0] 

  # Combine the terms 
  #eB = eB + FB*eb1*S
  eB = eB + (1/2)*FB*eb1*S # fixed fac 1/2 april 9 2026 eq 17 M11plus paper 
  veB = veB + (1/2)*(vFB*eb1*S + veb1*FB*S + vS*FB*eb1)
  if(coords is not None):
    if(coords.shape[0]==ngrid):
      for ig in range(ngrid):
        print('QQQ %.6f %.6f %.6f %.6e %.6e %.6e %.6e  '%(coords[ig,0],coords[ig,1],coords[ig,2],ra[ig],e35[0,ig],FB[ig],S[ig]))
        pass

  return(eB,veB)

# M11plus density screening S, eq 9, 12 of M11plus paper 
# Input: Spin-polarized density on grid and R35 exponent beta =1/d^2
def get_Sd(rho,beta,coords=None):
  if(len(rho.shape)<3):
     raise ValueError('Not enough dimensions in R35 Sd rho ')
  nspin,nrho,ngrid = rho.shape 
  ymax = (256./3.)**(1./3.)
  yfac = beta**(-0.5) *(4./3.*math.pi)**(1./3.)
  Sd  = numpy.zeros((nspin,ngrid))
  vSd = numpy.zeros((nspin,6,ngrid))
  for i in range(nspin):
    y = yfac*rho[i,0]**(1./3.)
    yrho = (1./3.)*yfac*rho[i,0]**(-2./3.)
    s = .5*(1-erf(y-ymax))
    sy = -1.0*numpy.exp(-(y-ymax)**2)/math.pi**0.5 * yrho 
    Sd[i] = s
    vSd[i,0] = sy

  # Debug print
  if(coords is not None):
    if(coords.shape[0]==ngrid):
      for ig in range(ngrid):
        print('SSS %.6f %.6f %.6f %.6e '%(coords[ig,0],coords[ig,1],coords[ig,2],Sd[0,ig]))
        pass
  return(Sd,vSd)

# M11plus screening 1-l*ELF, eq 6, 13-15, 17
# Input: Spin polarized density, gradient, tau on grid and factor l
# Output: 1-l*ELF and its derivatives 
# Why is ELF not identically 1 in one electron systems 
def get_omELF(rho,l=1,coords=None):
  ctl = 0.3*(6*math.pi**2)**(2./3.) 
  if(len(rho.shape)<3):
     raise ValueError('Not enough dimensions in R35 Sd rho ')
  nspin,nrho,ngrid = rho.shape 
  ome  = numpy.zeros((nspin,ngrid))
  vome = numpy.zeros((nspin,6,ngrid))
  for i in range(nspin):
     r = rho[i,0]
     r = numpy.maximum(r,1e-10*numpy.ones_like(r))
     t = rho[i,5]
     t = numpy.maximum(t,1e-10*numpy.ones_like(t))
     tlda= ctl*r**(5./3.)
     tldarho = (5./3.)*ctl*r**(2./3.)
     gsq = rho[i,1]**2+rho[i,2]**2+rho[i,3]**2
     (gsqx,gsqy,gsqz) = (2*rho[i,1],2*rho[i,2],2*rho[i,3])
     tw = gsq/(8*r)
     twgsq = 1/(8*r)
     twrho = -1.0*gsq/(8*r**2)
     elf0 = 1+((t-tw)/tlda)**2
     elf0rho = 2*(t-tw)/tlda*(-twrho/tlda + -(t-tw)/tlda**2*tldarho ) 
     elf0gsq = 2*(t-tw)/tlda**2 * (-twgsq) 
     elf0tau = 2*(t-tw)/tlda**2 
     elf = 1/elf0
     elfrho = -1/elf0**2 * elf0rho 
     elfgsq = -1/elf0**2 * elf0gsq 

     elftau = -1/elf0**2 * elf0tau 
     # Debug print
     if(coords is not None):
       if(coords.shape[0]==ngrid):
         for ig in range(ngrid):
           print('EEE %.6f %.6f %.6f  %.6e %.6e  '%(coords[ig,0],coords[ig,1],coords[ig,2],rho[i,0,ig],elf[ig]))

     ome[i]    = 1-l*elf
     vome[i,0] = -l*elfrho 
     vome[i,1] = -l*elfgsq*gsqx
     vome[i,2] = -l*elfgsq*gsqy
     vome[i,3] = -l*elfgsq*gsqz
     vome[i,4] = -l*elftau

  return(ome,vome)

# M11plus NGA, eq 8-11
def get_enga(rho,beta):
  a01 = (4*math.pi/3)**(2./3.)
  a11 = 4.869
  b=[[1,1.3493,0],[0.893696,3.642,0],[0.889355,0,0.06202]]
  cl = -(3/4)*(6/math.pi)**(1./3.)
  y2fac = beta**(-1) *(4./3.*math.pi)**(2./3.)
  nspin,nrho,ngrid = rho.shape 
  en  = numpy.zeros((nspin,ngrid))
  ven = numpy.zeros((nspin,6,ngrid))
  nspin,nrho,ngrid = rho.shape 
  for i in range(nspin):
    r = rho[i,0]
    r[r<1e-10]=1e-10
    el = cl*r**(4./3.)
    elrho = (4./3.)*cl*r**(1./3.)
    gsq = rho[i,1]**2+rho[i,2]**2+rho[i,3]**2
    gsq[gsq<1e-12]=1e-12
    (gsqx,gsqy,gsqz) = (2*rho[i,1],2*rho[i,2],2*rho[i,3])
    x32 = gsq**(3./4.)/r**2 
    x32gsq = 0.75*x32/gsq
    x32rho = -2*x32/r
    y2 = y2fac*r**(2./3.)
    y2rho = (2./3.)*y2fac*r**(-1./3.)

    Fnum = a01*y2 + a11*x32*y2
    Fnumrho = a01*y2rho + a11*(x32rho*y2+x32*y2rho)
    Fnumgsq = a11*x32gsq*y2
    (Fden,Fdenrho,Fdengsq) = (0.0000000000000000000001,0,0)
    for ii in range(3):
      for jj in range(3):
        bb = b[jj][ii]
        Fden = Fden+ bb * x32**ii * y2**jj 
        if(ii>0):
          Fdenrho = Fdenrho + bb*ii*x32**(ii-1)*x32rho *y2**jj
          Fdengsq = Fdengsq + bb*ii*x32**(ii-1)*x32gsq *y2**jj
        if(jj>0):
          Fdenrho = Fdenrho + bb*x32**(ii)*jj*y2**(jj-1)*y2rho
  
    F = Fnum/Fden 
    Frho = Fnumrho/Fden -Fnum*Fdenrho/Fden**2
    Fgsq = Fnumgsq/Fden -(Fnum/Fden**2)*Fdengsq

    en[i] = F*el
    ven[i,0] = F*elrho + Frho*el 
    ven[i,1] = Fgsq*el*gsqx
    ven[i,2] = Fgsq*el*gsqy
    ven[i,3] = Fgsq*el*gsqz
    #for ig in range(ngrid):
    # print('FFF %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e '%(rho[i,0,ig],x32[ig],y2[ig],Fnum[ig],Fden[ig],Frho[ig],Fgsq[ig],en[i,ig]))
  return(en,ven) 

# Return a copy of input molecule m with stretched AOs
def stretchAOs(m,beta):
  #print('Calling stretchAOs')
  if m.cart:
     raise ValueError('Cartesian AOs not implemented with stretchAOs')
  if not m._built:
        logger.warn(m, 'Warning: stretchAOs object %s not initialized. Initializing %s',m,m)
        m.build()
  m2 = m.copy() 
  m2.build()

  # Modify _env using the info in _bas . Note that this only works for
  # spherical Gaussians, not Cartesian Gaussians! 
  # _bas[ishell] is for each shell (atom, angmo, nprim, ncenter, kappa, exponent_ptr, coeff_ptr)
  # _bas[ishell,5] points to the start of the nprim locations in _env containing the exponents of this shell 
  # _bas[ishell,6] points to the start of the nprim locations in _env containing the contraction coeffs of this shell 
  # April 2026 need to fix for the case where a shell has multiple contraction coeffs 
  nshell = (m2._bas.shape)[0]
  _oldenv = m._env.copy()
  if(m.verbose>3):
    print('Here is basis _bas \n',m2._bas)
  for ishell in range(nshell):
    lshell = m2._bas[ishell,1]
    nprim = m2._bas[ishell,2]
    ncon  = m2._bas[ishell,3]
    expstart = m2._bas[ishell,5]
    constart = m2._bas[ishell,6]
  
    # Rescale the exponents and contraction coefficients for each shell
    # This version only works if each primitive is only on one contracted shell! 
    if(ncon>1):
       raise ValueError('Unfortunately stretchAOs requires only one contracted shell per primitive')
    for iexp in range(nprim):
      astart = _oldenv[expstart+iexp]
      cstart = _oldenv[constart+iexp]
      aend = astart*beta/(astart+beta) # Stretched exponent 
      cend = cstart*(2*math.pi*beta)**0.75/(astart+beta)**1.5  # Renormalization 
      cend = cend*(beta/(astart+beta))**lshell # Shell dependence 
      m2._env[expstart+iexp] = aend 
      m2._env[constart+iexp] = cend 

#    icon =-1 
#    for iexp in range(nprim):
#      astart = _oldenv[expstart+iexp]
#      aend = astart*beta/(astart+beta) # Stretched exponent 
#      m2._env[expstart+iexp] = aend 
#      for ic in range(ncon):
#        icon=icon+1
#        cstart = _oldenv[constart+icon]
#        # fix 4/8/2026 cend = cstart*(2*math.pi*beta)**0.75/(astart+beta)**1.5  # Renormalization 
#        cend = cstart*(2*math.pi)**0.75/(astart+beta)**0.75  # Renormalization 
#        cend = cend*(beta/(astart+beta))**lshell # Shell dependence 
#        m2._env[constart+icon] = cend 
  return(m2) 


def r35c(ni,rhoin,spin,exc,vxc,r35beta,weight=None,coords=None,verbose=None):
  # Add M11po R35 correlation terms to exc 
  #print('R35c has coords ',coords)

  # TEST do nothing 
  #return(exc,vxc) 

  # Pass the integration grid weights to return the terms in the linear fit.
  # We will print these out after all integration grid batches are done. 
  if((weight is not None) and (not hasattr(ni,'elhs'))):
    ni.elhs=numpy.zeros(5)
    ni.eb05s=numpy.zeros(5)

  # Unpack the parameters 
  ngrid=rhoin.shape[-1]

  # Defaults 
  beta = 0.2
  Nmax = 0.6 
  cs= [-0.170265960,0.326104631,0,0.173830277,-0.337808383]
  ds= [0,-0.002379857,2.026450260,-0.94302359,-4.195668115]
  if(isinstance(r35beta,list)):
    lp = len(r35beta)
    beta = r35beta[0]
    if(lp>1): 
      Nmax = r35beta[1]
    if(lp>7): 
      cs= r35beta[2:7]
    if(lp>8): 
      ds= r35beta[7:12]
 
  # Spin-polarized input densities 
  if(spin<1):
    rhoa = rhoin/2
    rhob = rhoin/2
    rho = numpy.array((rhoa,rhob))
  else:
    rhoa = rhoin[0]
    rhob = rhoin[1]
    rho = rhoin 

  # Print marker for new SCF cycle
  #print('\n\n\n new scf cycle\n')

  # Compute pieces of the XC functional 
  e35,ve35 = get_e35(rho,beta,coords=None)
  n35,vn35 = get_n35(rho, beta,coords=None)
  Fn35,vFn35 = get_Fn35(rho,n35,vn35,Nmax=Nmax,coords=None)
  Sd,vSd = get_Sd(rho,beta,coords=None)
  eB05,veB05 = get_eB05(rho,e35,ve35,Fn35,vFn35,Sd,vSd,coords=None)
  enga,venga = get_enga(rho,beta)
  omelf,vomelf = get_omELF(rho,l=1,coords=None)
  om2elf,vom2elf = get_omELF(rho,l=2,coords=None)

  # Compute gradient magnitude and eta35 for debugging
  gsq_a = rho[0,1]**2 + rho[0,2]**2 + rho[0,3]**2
  eta35_a = rho[0,6]  # rung-3.5 density ingredient

  # Linear fit power series, local-hybrid-like term (eq. 6)
  exclh  = numpy.zeros_like(e35)
  vexclh  = numpy.zeros_like(ve35)
  vexclh1 = numpy.zeros_like(e35)
  efit0 = (enga-e35)*Sd # eq 6 left half of the M11plus paper, without the 1-l*ELF factor
  for s in range(2):
    for i in range(5):
      ometerm = cs[i]*numpy.ones_like(e35[s])
      vometerm = numpy.zeros_like(ve35[s])
      if(i==1):
        ometerm  =             cs[i]*omelf[s]
        vometerm = vomelf[s] * cs[i]*i
      if(i>1):
        ometerm  =             cs[i]*omelf[s]**i
        vometerm = vomelf[s] * cs[i]*i*omelf[s]**(i-1)
      if(weight is not None):
        ni.elhs[i]=ni.elhs[i]+ numpy.dot(weight,efit0[s]*omelf[s]**i)
      exclh[s] = exclh[s] + ometerm*efit0[s]
      vexclh[s] = vexclh[s] + vometerm*efit0[s]
      vexclh1[s] = vexclh1[s] + ometerm 
    vexclh[s] = vexclh[s] + (venga[s]-ve35[s])*Sd[s]*vexclh1[s]
    vexclh[s] = vexclh[s] + vSd[s]*(enga[s]-e35[s])*vexclh1[s]

  # Linear fit power series, B05-like term (eq. 17)
  excB05  = numpy.zeros_like(eB05)
  vexcB05  = numpy.zeros_like(veB05)
  vexcB051 = numpy.zeros_like(eB05)
  for i in range(5):
    om2eterm = ds[i]*numpy.ones_like(e35[s])
    vom2eterm = numpy.zeros_like(ve35)
    if(i>0):
      om2eterm = numpy.zeros_like(e35[s])
      for s in range(2):
        om2eterm  = om2eterm +   ds[i]*om2elf[s]**i
        vom2eterm[s] = vom2elf[s] * ds[i]*i*om2elf[s]**(i-1)
    if(weight is not None):
      ni.eb05s[i]=ni.eb05s[i]+ numpy.dot(weight,eB05*om2elf[s]**i)
    excB05   = excB05   + om2eterm*eB05
    vexcB051 = vexcB051 + om2eterm 
    for s in range(2):
      vexcB05[s] = vexcB05[s] + vom2eterm[s]*eB05
  vexcB05 = vexcB05 + veB05*vexcB051

  # Assemble the results with proper spin polarization. We compute the energy
  # density and return the energy density per particle 
  exc=exc + (exclh[0] + exclh[1] + excB05)/(rhoa[0]+rhob[0])
  if(spin<1):
    vxc=vxc + vexclh[0] + vexcB05[0]
  else: 
    vxc[0] = vxc[0] + vexclh[0] + vexcB05[0]
    vxc[1] = vxc[1] + vexclh[1] + vexcB05[1] 

  # Print: rho e35 enga Sd 
  #nspin,nrho,ngrid = rho.shape 
  #for ig in range(ngrid):
  #  print('GGG %.6f %.6f %.6f %.6e %.6e %.6e %.6e %.6e  '%(coords[ig,0],coords[ig,1],coords[ig,2],rho[0,0,ig],e35[0,ig],enga[0,ig],Sd[0,ig],exclh[0,ig]))
  #  pass

  return(exc,vxc) 

