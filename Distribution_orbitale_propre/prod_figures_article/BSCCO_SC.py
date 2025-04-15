import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from Information_mat import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

fig,ax = plt.subplots(2,1,sharex=True)
plt.gcf().set_size_inches(12.5/2.54, 13/2.54)

#-------------------------------------------------------------------
plt.sca(ax[0])
liste = [8, 9, 9.5, 11, 12]
for U in liste:
    path = f"BSCO/BSCO_U{U}e2.tsv"
    data = np.genfromtxt(path, names=True)
    # data = data[data['mu_ave']>1.56]
    # data = data[data['mu_ave']<1.75]
    D = 3*np.abs(data['D_ave'])
    x = 3*data['mu_ave']-5
    if U == 9:
        plt.plot(x,D, 'o', ms = 3, label = f'$U=8.5, \epsilon_p=2$')
    else:
        plt.plot(x,D, 'o', ms = 3, label = f'$U={U}, \epsilon_p=2$')


# plt.xlabel('$x$',)
plt.xlim(-0.5, 0.3)
plt.ylim(0,0.25)
plt.ylabel("""Paramètre d'ordre SC""")
plt.legend(ncol=2, loc='upper left', fontsize=9)
plt.grid()

#-------------------------------------------------------------------
plt.sca(ax[1])
liste = [10, 12, 14, 16]
for U in liste:
    path = f"BSCO/BSCO_U{U}e{U-8}.tsv"
    data = np.genfromtxt(path, names=True)
    D = 3*np.abs(data['D_ave'])
    x = 3*data['mu_ave']-5
    plt.plot(x,D, 'o', ms = 3, label = f'$U={U}, \epsilon_p={U-8}$')

plt.xlabel('$x$',)
# plt.xlim(-0.5, 0.3)
plt.ylim(0,0.25)
plt.ylabel("""Paramètre d'ordre SC""")
plt.legend(ncol=2, loc='upper left', fontsize=9)
plt.grid()


plt.tight_layout(pad=0.5)
plt.savefig(__file__[:-3]+".pdf")
