import numpy as np
import matplotlib.pyplot as plt
import os
import sys


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

label = [r'$\textbf{(d)}$', r'$\textbf{(e)}$', r'$\textbf{(f)}$']
files = ["LCO", "YBCO", "NCCO"]
# files = ["LCO", "YBa2Cu3O7_U=9_e=3", "NCCO"]
# files = ["LCO", "YBa2Cu3O7_U=9.5_e=3", "NCCO"]
descr = ["LCO", "YBCO", "NCCO"]
facteur_tpp = [0.64, 0.673, 0.54]
lower_lim = [-4,-1,-15]
upper_lim = [4,1,7]
gap_optique = [1.24,0.3]
dim_fleche = [(0.02,0.06),(0.02,0.02)]
decalage_texte = [-0.6,-0.11]
descr2 = ["$U=14,\epsilon_p=4$", "$U=9,\epsilon_p=3$", "$U=8,\epsilon_p=3$"]

fig,ax = plt.subplots(3,1)
# fig,ax = plt.subplots(3,1,sharex=True)
fig.set_size_inches(9.5/2.54,15/2.54)

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
	ax[y].set_xlim(lower_lim[y],upper_lim[y])
	ax[y].set_ylim(0,0.3)

	ax[y].set_ylabel(r'$\rho(\omega)$',fontsize=10)
	if y == 2 : ax[y].set_xlabel(r'$eV$',fontsize=10)
	if y == 0 : ax[y].legend(loc=(0.8,0.65))
	try:
		ax[y].vlines(-gap_optique[y]/2,0,3,linestyle="--",color="black")
		ax[y].vlines(gap_optique[y]/2,0,3,linestyle="--",color="black")
		ax[y].arrow(-gap_optique[y]/2, 0.15, gap_optique[y], 0, head_width=dim_fleche[y][0], head_length=dim_fleche[y][1], linewidth=1, color='black', length_includes_head=True)
		ax[y].arrow(gap_optique[y]/2, 0.15, -gap_optique[y], 0, head_width=dim_fleche[y][0], head_length=dim_fleche[y][1], linewidth=1, color='black', length_includes_head=True)
		ax[y].text(decalage_texte[y], 0.2, s=str(gap_optique[y])+" eV", fontsize=13,backgroundcolor = 'w')
	except:
		continue
# ax[0].set_xlim(-2,2)
# ax[1].set_xlim(-0.5,0.5)
# ax[2].set_xlim(-15,7)
# A = ax[0]
off = 1.5
# A.text(-12+off, 0.05, 'CTB', c='b', ha='center')
# A.text(-1.7+off, 0.027, 'CTG', c='k', ha='center')
# A.text(-4.5+off, 0.13, 'ZRB', c='k', ha='center')
# A.text(1.6+off, 0.03, 'UHB', c='r', ha='center')
# A.text(-17.5+off, 0.13, 'LHB', c='r', ha='center')

plt.tight_layout(pad=0.4)
# plt.savefig(__file__[0:-3]+"_plus_res_U=9.pdf")
plt.savefig(__file__[0:-3]+".pdf")
plt.show()

