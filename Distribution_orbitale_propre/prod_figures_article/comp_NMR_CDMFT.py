import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from Information_mat import *
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'
liste_tuples = [(11,14.0,"LCO","LCO/LCO_U14e4.tsv","LCO $U=14, \epsilon_p=4$", 'b', 'b'),
                (11,20.0,"LCO_U20e10","LCO/LCO_U20e10.tsv", "LCO $U=20, \epsilon_p=10$", 'b', 'w'),
                (18,9.5,"YBCO","YBCO/YBCO_U9.tsv","YBCO $U=9, \epsilon_p=3$", 'g', 'g'),
                (19,8.0,"NCCO","NCCO/cdmft_U8e3.tsv","NCCO $U=8, \epsilon_p=3$", 'r', 'r'),
                (19,12.0,"NCCO","NCCO/cdmft_U12e7.tsv","NCCO $U=12, \epsilon_p=7$", 'r', 'w')
                ]
plt.gcf().set_size_inches(13/2.54,8/2.54)
Phase="supra"
for Tuple in liste_tuples:
    x = Tuple[1]
    ind = Tuple[0]
    material = Dic_mat[ind]
    print(material)
    
    path = Tuple[3]
    data = np.genfromtxt(path, names=True)
    data = data[data['mu_ave']>1.56]
    data = data[data['mu_ave']<1.75]
    tppp = data['tppp'][-1]
    tpp = data['tpp'][-1]
    tpd = data['tpd'][-1]
    e = data['e'][-1]
    try:
        n_p = 3*data['ave_e']/2
        n_d = 3*(data['ave_mu']-data['ave_e'])
        n = data['ave_mu']
        mu = data['mu']
    except:
        n_p = 3*data['e_ave']/2
        n_d = 3*(data['mu_ave']-data['e_ave'])
        n = data['mu_ave']
        mu = data['mu']

    plt.plot(4-2*n_p,2-n_d, 'o', ms = 3, label = Tuple[4], color = Tuple[5], mfc = Tuple[6], mew=0.5)


exp_La214_x = [0.146527,0.167211,0.240381,0.339803]
exp_La214_y = [0.816770,0.824223,0.831677,0.837267]
exp_Y123_x = [0.3280326687484986,0.4037936241034968,0.4176143577777015,0.42439175045468575,0.45755807968154827,0.4672265879688411,0.4838183315603446]
exp_Y123_y = [0.6937888198757763,0.7385093167701863,0.7366459627329192,0.7645962732919254 ,0.7608695652173912 ,0.7608695652173912 , 0.7571428571428571]
exp_NCCO_x = [0.4183435709,0.4022923]
exp_NCCO_y = [0.578260869,0.46459627329]

ligne = np.array([[0,1],[1,0]])
ligne2 = np.array([[0,0.85],[0.85,0]])
ligne3 = np.array([[0,1.15],[1.15,0]])

plt.plot(exp_La214_x,exp_La214_y,"s",color="b", mfc='w', label="LCO (exp.)")
plt.plot(exp_Y123_x,exp_Y123_y,"s",color="g", mfc='w', label="YBCO (exp.)")
plt.plot(exp_NCCO_x,exp_NCCO_y,"s",color="r", mfc='w', label="NCCO (exp.)")

plt.plot(ligne[0],ligne[1], '-', color = 'gray', lw=1)
plt.plot(ligne2[0],ligne2[1], '-', color = 'blue', lw=1)
plt.plot(ligne3[0],ligne3[1], '-', color = 'blue', lw=1)
off = 0.005
plt.plot([off,off,4],[0,1-off,1-off], '-', color = 'darkblue', lw=2, label='atomic limit')
plt.xlabel('$4-2n_p$')
plt.xlim(0,0.8)
plt.ylim(0.3,1)
plt.ylabel('$2-n_d$')
plt.legend(loc="lower right", bbox_to_anchor=(1.2, 0.2))
plt.text(0.05, 0.4, 'electron-doped')
plt.text(0.4, 0.9, 'hole-doped')

plt.tight_layout(pad=0.1)
plt.savefig(__file__[:-3]+".pdf")
# plt.show()