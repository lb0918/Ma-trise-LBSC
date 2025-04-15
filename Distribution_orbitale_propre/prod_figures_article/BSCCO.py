import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from Information_mat import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

fig,ax = plt.subplots(1,2,sharey=True)
plt.gcf().set_size_inches(14/2.54, 8/2.54)

#-------------------------------------------------------------------
plt.sca(ax[0])
liste = [8, 9, 9.5, 11, 12]
for U in liste:
    path = f"BSCO/BSCO_U{U}e2.tsv"
    data = np.genfromtxt(path, names=True)
    data = data[data['mu_ave']>1.56]
    data = data[data['mu_ave']<1.75]
    try:
        n_p = 3*data['ave_e']/2
        n_d = 3*(data['ave_mu']-data['ave_e'])
        n = data['ave_mu']
    except:
        n_p = 3*data['e_ave']/2
        n_d = 3*(data['mu_ave']-data['e_ave'])
        n = data['mu_ave']
    mu = data['mu']
    if U == 9:
        plt.plot(4-2*n_p,2-n_d, 'o', ms = 3, label = f'$U=8.5, \epsilon_p=2$')
    else:
        plt.plot(4-2*n_p,2-n_d, 'o', ms = 3, label = f'$U={U}, \epsilon_p=2$')
ligne = np.array([[0,1],[1,0]])
ligne2 = np.array([[0,0.85],[0.85,0]])
ligne3 = np.array([[0,1.15],[1.15,0]])

plt.plot(ligne[0],ligne[1], '-', color = 'gray', lw=1)
plt.plot(ligne2[0],ligne2[1], '-', color = 'blue', lw=1)
plt.plot(ligne3[0],ligne3[1], '-', color = 'blue', lw=1)

off = 0.005
plt.plot([off,off,4],[0,1-off,1-off], '-', color = 'darkblue', lw=2,label='limite atomique')

plt.text(0.3, 0.98, 'dopé-trous', ha='center', va='top')
plt.text(0.07, 0.67, 'dopé-e', ha='center', va='top')
plt.text(0.92, 0.05, '({\\bf a})', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)

plt.xlabel('$4-2n_p$',)
plt.xlim(0,0.49)
plt.ylim(0.2,1)
plt.ylabel('$2-n_d$')
plt.legend(ncol=1, loc='lower left', fontsize=9, handlelength=0.5)
plt.grid()

#-------------------------------------------------------------------
plt.sca(ax[1])
liste = [10, 12, 14, 16]
for U in liste:
    path = f"BSCO/BSCO_U{U}e{U-8}.tsv"
    data = np.genfromtxt(path, names=True)
    data = data[data['mu_ave']>1.56]
    data = data[data['mu_ave']<1.75]
    try:
        n_p = 3*data['ave_e']/2
        n_d = 3*(data['ave_mu']-data['ave_e'])
        n = data['ave_mu']
    except:
        n_p = 3*data['e_ave']/2
        n_d = 3*(data['mu_ave']-data['e_ave'])
        n = data['mu_ave']
    mu = data['mu']
    plt.plot(4-2*n_p,2-n_d, 'o', ms = 3, label = f'$U={U}, \epsilon_p={U-8}$')
ligne = np.array([[0,1],[1,0]])
ligne2 = np.array([[0,0.85],[0.85,0]])
ligne3 = np.array([[0,1.15],[1.15,0]])

plt.plot(ligne[0],ligne[1], '-', color = 'gray', lw=1)
plt.plot(ligne2[0],ligne2[1], '-', color = 'blue', lw=1)
plt.plot(ligne3[0],ligne3[1], '-', color = 'blue', lw=1)

off = 0.005
plt.plot([off,off,4],[0,1-off,1-off], '-', color = 'darkblue', lw=2,label='limite atomique')


plt.text(0.3, 0.98, 'dopé-trous', ha='center', va='top')
plt.text(0.07, 0.67, 'dopé-e', ha='center', va='top')
plt.text(0.92, 0.05, '({\\bf b})', ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)

plt.xlabel('$4-2n_p$',)
plt.xlim(0,0.49)
# plt.ylabel('$2-n_d$')
plt.legend(ncol=1, loc='lower left', fontsize=9, handlelength=0.5)
plt.grid()


plt.tight_layout(pad=0.5)
plt.savefig(__file__[:-3]+".pdf")
