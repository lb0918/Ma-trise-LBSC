import numpy as np
import os
import sys
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

fig,ax = plt.subplots(2,2)
plt.gcf().set_size_inches(13/2.54, 10/2.54)

X = np.genfromtxt('BSCO/J.tsv', names=True, dtype=None, encoding='utf8')

Np = X['U'].shape[0]
ms = 5
lw = 0.8
#-------------------------------------------------------------------
AX = ax[0,0]
plt.sca(AX)
x = X['U']-X['e']
y = X['gap_spectral']
plt.plot(x[0:5], y[0:5], 'o-', ms=ms, mew=0.5, c='b', mec='b', mfc='w', lw=lw)
plt.plot(x[5:9], y[5:9], 'o-', ms=ms, mew=0.5, c='r', mec='r', mfc='w', lw=lw)
for i in range(Np):
	plt.annotate('{:d}'.format(X['e'][i]), (x[i], y[i]), xytext=(5,-5), textcoords='offset points', fontsize=8, ha='center', va='center')
plt.xlabel('$U-\epsilon_p$')
plt.ylabel('gap spectral')
plt.text(0.05,0.88, '$\\textbf{(a)}$', transform=AX.transAxes)
plt.grid()

#-------------------------------------------------------------------
AX = ax[0,1]
plt.sca(AX)

x = 1/(X['U']-X['e'])
y = X['J']
plt.plot(x[0:5], y[0:5], 'o-', ms=ms, mew=0.5, c='b', mec='b', mfc='w', lw=lw)
plt.plot(x[5:9], y[5:9], 'o-', ms=ms-1, mew=0.5, c='r', mec='r', mfc='w', lw=lw)
for i in range(5):
	A = plt.annotate('{:d}'.format(X['e'][i]), (x[i], y[i]), xytext=(-5,5), textcoords='offset points', fontsize=8, ha='center', va='center')
xy= [(4,1),(-5,0),(4,-1)]
for i in range(6,9):
	A = plt.annotate('{:d}'.format(X['e'][i]), (x[i], y[i]), xytext=xy[i-6], textcoords='offset points', fontsize=6, ha='center', va='center')
plt.xlabel('$(U-\epsilon_p)^{-1}$')
plt.ylabel('$J$', rotation=0)
plt.text(0.05,0.88, '$\\textbf{(b)}$', transform=AX.transAxes)
plt.grid()

#-------------------------------------------------------------------
AX = ax[1,0]
plt.sca(AX)

x = X['J']
y = X['Dmax']
plt.plot(x[0:5], y[0:5], 'o-', ms=ms, mew=0.5, c='b', mec='b', mfc='w', lw=lw)
plt.plot(x[5:9], y[5:9], 'o-', ms=ms, mew=0.5, c='r', mec='r', mfc='w', lw=lw)
for i in range(Np):
	plt.annotate('{:d}'.format(X['e'][i]), (x[i], y[i]), xytext=(5,-5), textcoords='offset points', fontsize=8, ha='center', va='center')
plt.xlabel('$J$')
plt.ylabel(r"""$\langle D\rangle_{\rm max}$""", rotation=0)
plt.xlim(0.165,0.171)
plt.ylim(0.025,0.043)
plt.text(0.05,0.88, '$\\textbf{(c)}$', transform=AX.transAxes)
plt.grid()

#-------------------------------------------------------------------
AX = ax[1,1]
plt.sca(AX)

x = 1/(X['U']-X['e'])
y = X['Dmax']
plt.plot(x[0:5], y[0:5], 'o-', ms=ms, mew=0.5, c='b', mec='b', mfc='w', lw=lw)
plt.plot(x[5:9], y[5:9], 'o-', ms=ms, mew=0.5, c='r', mec='r', mfc='w', lw=lw)
for i in range(Np):
	plt.annotate('{:d}'.format(X['e'][i]), (x[i], y[i]), xytext=(5,-5), textcoords='offset points', fontsize=8, ha='center', va='center')
plt.xlabel('$(U-\epsilon_p)^{-1}$')
plt.ylabel(r"""$\langle D\rangle_{\rm max}$""", rotation=0)
plt.grid()
plt.text(0.05,0.88, '$\\textbf{(d)}$', transform=AX.transAxes)

plt.tight_layout(pad=0.5)
plt.show()
# plt.savefig(__file__[:-3]+".pdf")
