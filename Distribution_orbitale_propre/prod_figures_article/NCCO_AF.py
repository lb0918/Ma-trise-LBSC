import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'

plt.gcf().set_size_inches(9/2.54, 6/2.54)
path = "NCCO_AF.tsv"
data = np.genfromtxt(path, names=True)
plt.plot(data["U"], 3*abs(data["M_ave"]), "o", ms=3)
plt.xlabel("$U$")
plt.ylim(0,0.75)
plt.ylabel("""Paramètre d'ordre AF""")
plt.grid()
# plt.text(0.95, 0.15, "$\epsilon_p=3$", transform=plt.gca().transAxes, ha='right')
plt.title("NCCO (demi-rempli)")
plt.tight_layout(pad=0.1)
plt.savefig(__file__[:-3]+".pdf")
