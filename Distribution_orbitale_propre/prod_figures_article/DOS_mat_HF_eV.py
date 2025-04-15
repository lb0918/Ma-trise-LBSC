import numpy as np
import matplotlib.pyplot as plt
import os
import sys


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

label = [r'$\textbf{(a)}$', r'$\textbf{(b)}$', r'$\textbf{(c)}$']
files = ["LCO", "YBCO", "NCCO"]
descr = ["LCO", "YBCO", "NCCO"]
facteur_tpp = [0.64, 0.673, 0.54]
lower_lim = [-2,-1,-15]
upper_lim = [2,1,7]
descr2 = ["$U=14,\epsilon_p=4$", "$U=9,\epsilon_p=3$", "$U=8,\epsilon_p=3$"]

fig,ax = plt.subplots(3,1,sharex=True)
fig.set_size_inches(12/2.54,14/2.54)

for y in range(3):
	file = files[y]+"_dos.tsv"
	data = np.genfromtxt(file,names=True)

	w,Cu,Ox = data["w"],data["up_1"],data["up_2"]
	tot = Cu+2*Ox
	ax[y].text(0.03, 0.80, descr[y], va='top', ha='left', transform=ax[y].transAxes, backgroundcolor = 'w')
	ax[y].text(0.03, 0.68, descr2[y], va='top', ha='left', transform=ax[y].transAxes, backgroundcolor = 'w')
	ax[y].text(0.03, 0.95, label[y], va='top', ha='left', fontsize=13,  transform=ax[y].transAxes, backgroundcolor = 'w')
	ax[y].plot(w*facteur_tpp[y], Cu, 'r-', label="Cu")
	ax[y].plot(w*facteur_tpp[y], Ox, 'b-', label=r"$\mathrm{O}_x$")
	ax[y].axvline(0, c='r', ls='solid', lw=0.5)
	# ax[y].set_xlim(-15,7)
	# ax[y].set_xlim(-0.5,0.5)
	# ax[y].set_xlim(lower_lim[y],upper_lim[y])
	ax[y].set_ylim(0,0.3)

	ax[y].set_ylabel(r'$\rho(\omega)$',fontsize=15)
	if y == 2 : ax[y].set_xlabel(r'$eV$',fontsize=15)
	if y == 0 : ax[y].legend(loc=(0.5,0.65))
ax[0].set_xlim(-2,2)
ax[0].set_xlim(-0.5,0.5)
ax[0].set_xlim(-15,7)
# A = ax[0]
off = 1.5
# A.text(-12+off, 0.05, 'CTB', c='b', ha='center')
# A.text(-1.7+off, 0.027, 'CTG', c='k', ha='center')
# A.text(-4.5+off, 0.13, 'ZRB', c='k', ha='center')
# A.text(1.6+off, 0.03, 'UHB', c='r', ha='center')
# A.text(-17.5+off, 0.13, 'LHB', c='r', ha='center')

plt.tight_layout(pad=0.4)
plt.show()

# plt.savefig(__file__[0:-3]+".pdf")
