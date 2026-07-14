from pyscf import scf,gto,dft
from pyscf import lib
import sys 
import numpy 
import math 
from scipy.special import lambertw 
from pyscf import r35dft 
from pyscf.r35dft import roks

m=gto.Mole(atom='Ne',charge=0,spin=0,basis='def2tzvp',verbose=1)
m.build()
mf=scf.UHF(m)
mf.kernel() 
P=mf.make_rdm1()

mp=r35dft.UKS(m,xc='HYB_MGGA_X_M11PO, MGGA_C_M11PO',r35beta='Default')
mp.max_cycle=0
mp.kernel(dm0=P)
print('Neon atom, neutral ground state, def2tzvp basis set')
print('Total post-HF energy, full M11po functional including R35 terms.')
print('Current installation energy (Ha): %22.16f '%(mp.e_tot))
print('Reference value energy (Ha):       -128.8971704215841498')

m=gto.Mole(atom='F', charge=0, spin=1, basis='def2tzvp',verbose=1)
m.build()

# Unrestricted (UKS)
mf=scf.UHF(m)
mf.kernel()
P=mf.make_rdm1()
mp=r35dft.UKS(m,xc='HYB_MGGA_X_M11PO, MGGA_C_M11PO',r35beta='Default')
mp.max_cycle=0
mp.kernel(dm0=P)
print('Fluorine atom, neutral spin-doublet ground state,def2tzvp basis set')
print('Total post-HF energy, full M11po functional including r35 terms')
print('Current installation energy (Ha): %22.16f '%(mp.e_tot))
print('Reference value energy (Ha): -99.69549398755208 ')

# Restricted Open Shell (ROKS)
mf=scf.ROHF(m)
mf.kernel()
P=mf.make_rdm1()
mp=roks.ROKS(m,xc='HYB_MGGA_X_M11PO, MGGA_C_M11PO',r35beta='Default')
mp.max_cycle=0
mp.kernel(dm0=P)
print('Fluorine atom, neutral spin-doublet ground state,def2tzvp basis set')
print('Total post-HF energy, full M11po functional including r35 terms')
print('Current installation energy (Ha): %22.16f '%(mp.e_tot))
print('Reference value energy (Ha): -99.6958974176013')