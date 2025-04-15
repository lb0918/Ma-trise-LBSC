import numpy as np
import matplotlib.pyplot as plt
import pyqcm
import pyqcm.cdmft_modif_LB as cdmft
import os, sys
from clusters.cluster_8b_AF import CM, CM2
from models.model_1bande_8b_AF import model

param = "U" # paramètre du modèle sur lequel on boucle
param_cible = int(sys.argv[1]) # argument à passé dans le fichier de remise de la tâche: remise.sh
sol_départ = "solutions/boucle_1bande/boucle_U_AF_supra_1bande_bain_contrainte.tsv" # path de la solution de départ
print("!!!!!!!!!!!!!!!!!!!!!!!!!")
print(sol_départ)
converg = ["self-energy"] #critère de convergence
print("*************************************")
print(converg)
current_file_dir = os.path.dirname(os.path.abspath(__file__)) # variable attribuée au dossier où se trouve ce script
path = f"{current_file_dir}/{sol_départ}"
data = np.genfromtxt(path, names = True)
param_dep = data[param][-1]
if param_cible < param_dep:
    step_mu = -0.02
if param_cible > param_dep:
    step_mu = 0.02
U = data['U'][-1]
t = data['t'][-1]
tp = data['tp'][-1]
mu = data["mu"][-1]


pyqcm.set_global_parameter('Hamiltonian_format','E')
pyqcm.set_global_parameter("max_iter_lanczos",3000)
pyqcm.set_global_parameter('Ground_state_method','P')
pyqcm.set_global_parameter('Ground_state_init_last')
pyqcm.set_global_parameter('parallel_sectors')
pyqcm.set_global_parameter('PRIMME_preconditionning', 1)

model.set_target_sectors(["R0:S0"])

E_bath=CM.E_bath ; S_bath=CM.S_bath ; Tr_bath=CM.Tr_bath
T_bathA_up=CM.T_bathA_up  ; T_bathA_down=CM.T_bathA_down
T_bathB_up=CM.T_bathB_up  ; T_bathB_down=CM.T_bathB_down
no=CM.no ; nb=no-4

E_bath_1=[eb+'_1' for eb in E_bath]
T_bathA_up_1=[tb+'_1' for tb in T_bathA_up]
T_bathA_down_1=[tb+'_1' for tb in T_bathA_down]
T_bathB_up_1=[tb+'_1' for tb in T_bathB_up]
T_bathB_down_1=[tb+'_1' for tb in T_bathB_down]
S_bath_1=[sb+'_1' for sb in S_bath]
Tr_bath_1=[tr+'_1' for tr in Tr_bath]

non_variable_parameters = f"""
U={U}
tp={tp}
t={t}
mu={mu}
D=0
M=0
"""

# Aucune contrainte

# s_bath_param=''
# t_bath_param=''
# e_bath_param=''
# for i in range(nb):
    # s_bath_param+=','+S_bath_1[i]+'=1e-2'
#     s_bath_param+=','+Tr_bath_1[i]+'=1e-3'
#     t_bath_param+=','+T_bathA_up_1[i]+'=0.2'
#     t_bath_param+=','+T_bathA_down_1[i]+'=0.2'
#     t_bath_param+=','+T_bathB_up_1[i]+'=0.2'
#     t_bath_param+=','+T_bathB_down_1[i]+'=0.2'
#     e_bath_param+=','+E_bath_1[i]+'=0.2'
# variable_parameters = t_bath_param+e_bath_param+s_bath_param


# Contraintes pour imposer la symétrie particule-trou.

variable_parameters = """
eb1_1=1e-5
eb2_1=-1*eb1_1
eb3_1=1e-5
eb4_1=-1*eb3_1
eb5_1=1*eb3_1
eb6_1=1*eb4_1
eb7_1=1e-5
eb8_1=-1*eb7_1
sb1_1=1e-5
sb2_1=1e-5
sb3_1=1e-5
sb4_1=1e-5
sb5_1=-1*sb3_1
sb6_1=-1*sb4_1
sb7_1=1e-5
sb8_1=1e-5
tbA1up_1=1e-5
tbB1up_1=1e-5
tbA1down_1=1e-5
tbB1down_1=1e-5
tbA2up_1=1*tbA1down_1
tbB2up_1=1*tbB1down_1
tbA2down_1=1*tbA1up_1
tbB2down_1=1*tbB1up_1
tbA3up_1=1e-5
tbB3up_1=1e-5
tbA3down_1=1e-5
tbB3down_1=1e-5
tbA4up_1=1*tbA3down_1
tbB4up_1=1*tbB3down_1
tbA4down_1=1*tbA3up_1
tbB4down_1=1*tbB3up_1
tbA5up_1=1*tbB3down_1
tbB5up_1=1*tbA3down_1
tbA5down_1=1*tbB3up_1
tbB5down_1=1*tbA3up_1
tbA6up_1=1*tbB4down_1
tbB6up_1=1*tbA4down_1
tbA6down_1=1*tbB4up_1
tbB6down_1=1*tbA4up_1
tbA7up_1=1e-5
tbB7up_1=1e-5
tbA7down_1=1e-5
tbB7down_1=1e-5
tbA8up_1=1*tbA7down_1
tbB8up_1=1*tbB7down_1
tbA8down_1=1*tbA7up_1
tbB8down_1=1*tbB7up_1
trb1_1=1e-5
trb2_1=1e-5
trb3_1=1e-5
trb4_1=1e-5
trb5_1=1e-5
trb6_1=1e-5
trb7_1=1e-5
trb8_1=1e-5
"""


model.set_parameters(non_variable_parameters+variable_parameters)


# Sans symétrie particule-trou
# varia = E_bath_1+T_bathA_up_1+T_bathA_down_1+T_bathB_up_1+T_bathB_down_1+S_bath_1+Tr_bath_1


# Symétrie particule-trou
varia = ["eb1_1","eb3_1","eb7_1","sb1_1","sb2_1","sb3_1","sb4_1","sb7_1","sb8_1","tbA1up_1","tbB1up_1","tbA1down_1","tbB1down_1","tbA3up_1","tbB3up_1","tbA3down_1","tbB3down_1","tbA7up_1","tbB7up_1","tbA7down_1","tbB7down_1","trb1_1","trb2_1","trb3_1","trb4_1","trb5_1","trb6_1","trb7_1","trb8_1"]


filename = f"boucle_{param}_AF_supra_1bande_bain_contrainte3.tsv" # Nom du fichier où on inscrit les solutions CDMFT


def run_cdmft():
    """Fonction qui execute la cdmft"""
    solution = cdmft.CDMFT(model,varia, accur=[1e-4], iteration = "Broyden", method='L-BFGS-B', file = filename,max_value=100, maxiter = 40,convergence = converg,accur_bath=1e-3,beta=35)
    return solution.I


interval = np.arange(param_dep,param_cible,step_mu)


model.set_params_from_file(sol_départ,-1)
os.chdir(f"{current_file_dir}/solutions/boucle_1bande")


model.controlled_loop(task=run_cdmft, varia=varia, loop_param=param, loop_range=(param_dep,param_cible, step_mu),predict=False)