import numpy as np
import matplotlib.pyplot as plt
import pyqcm
import pyqcm.cdmft_modif_LB as cdmft
import os, sys
from clusters.cluster_8b_AF import CM, CM2
from models.model_3bandes_8b_AF import model

param = "mu"
param_cible = float(sys.argv[1])
sol_départ = "solutions/boucles_3bandes/boucle_mu_3b_max_supra_AF.tsv"
print("!!!!!!!!!!!!!!!!!!!!!!!!!")
print(sol_départ)
converg = ["self-energy"]
print("*************************************")
print(converg)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
path = f"{current_file_dir}/{sol_départ}"
data = np.genfromtxt(path, names = True)
param_dep = data[param][-1]


if param_cible < param_dep:
    step_mu = -0.0005
if param_cible > param_dep:
    step_mu = 0.0005


U = data['U'][-1]
ep = data['e'][-1]
tpd = data['tpd'][-1]
tppp = data['tppp'][-1]
mu = data["mu"][-1]
D = data['D'][-1]
M = 0
tpp = data["tpp"][-1]



pyqcm.set_global_parameter('Hamiltonian_format','E')
pyqcm.set_global_parameter("max_iter_lanczos",3000)
pyqcm.set_global_parameter('Ground_state_method','P')
pyqcm.set_global_parameter('Ground_state_init_last')
pyqcm.set_global_parameter('parallel_sectors')
pyqcm.set_global_parameter('PRIMME_preconditionning', 1)

model.set_target_sectors(["R0:S0", "R0:N4:S0", "R0:N4:S0"])

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
D={D}
tpp={tpp}
tppp={tppp}
tpd={tpd}
mu={mu}
M={M}
e={ep}
"""

s_bath_param=''
t_bath_param=''
e_bath_param=''
for i in range(nb):
    s_bath_param+=','+S_bath_1[i]+'=1e-5'
    s_bath_param+=','+Tr_bath_1[i]+'=1e-5'
    t_bath_param+=','+T_bathA_up_1[i]+'=1e-5'
    t_bath_param+=','+T_bathA_down_1[i]+'=1e-5'
    t_bath_param+=','+T_bathB_up_1[i]+'=1e-5'
    t_bath_param+=','+T_bathB_down_1[i]+'=1e-5'
    e_bath_param+=','+E_bath_1[i]+'=1e-5'
variable_parameters = t_bath_param+e_bath_param+s_bath_param

model.set_parameters(non_variable_parameters+variable_parameters)

varia = E_bath_1+T_bathA_up_1+T_bathA_down_1+T_bathB_up_1+T_bathB_down_1+S_bath_1+Tr_bath_1



filename = f"boucle_{param}_3b_max_supra_AF.tsv"


def run_cdmft():
    """Fonction qui execute la cdmft"""
    solution = cdmft.CDMFT(model,varia, accur=[1e-4], iteration = "Broyden", method='L-BFGS-B', file = filename,max_value=100, maxiter = 40,convergence = converg,accur_bath=1e-3,beta=35)
    return solution.I

model.set_params_from_file(sol_départ,-1)
os.chdir(f"{current_file_dir}/solutions/boucles_3bandes")
model.controlled_loop(task=run_cdmft, varia=varia, loop_param=param, loop_range=(param_dep,param_cible, step_mu),predict=True,retry="skip")