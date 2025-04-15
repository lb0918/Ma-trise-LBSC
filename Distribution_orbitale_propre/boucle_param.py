import numpy as np
import pyqcm
import os
import sys
from model_2x2_2C_8b_C2v_L import model
from cluster_2x2_2C_8b_C2v_L import CM
import pyqcm.cdmft_modif_LB as cdmft
from Information_mat import *


"""Boucle sur un paramètre du modèle"""

pyqcm.set_global_parameter("max_iter_lanczos",3000)
pyqcm.set_global_parameter('Hamiltonian_format','E')
pyqcm.set_global_parameter('Ground_state_method','P')
pyqcm.set_global_parameter('Ground_state_init_last')
pyqcm.set_global_parameter('parallel_sectors')
pyqcm.set_global_parameter('PRIMME_preconditionning', 1)


"""Condition de convergence de la procédure CDMFT"""
condition_converg = ['self-energy']
accuracy = [1e-4]


param = "mu" #Paramètre sur lequel on boucle

Phase = "supra"  #Phase normale ou supra

current_file_dir = os.path.dirname(os.path.abspath(__file__)) #Variable désignant le dossier où se trouve ce fichier

mat = Dic_mat[2] #Le matériaux qu'on veut modéliser

path = f"/net/nfs-iq/data/lbsc/Maîtrise_LBSC/Révision_stage_2022/new_supra/Bi2Sr2Ca2Cu3O10_outer/données/données_boucle_mu_U=10.0_t copy_wc=5.tsv" #Path du fichier contenant la solution CDFMT initiale

print("************************************")
print(path)
print("************************************")


data = np.genfromtxt(path, names=True)
param_dep = data[param][-1]
Valeur_U = data["U"][-1]
param_fin = float(sys.argv[1])


if param_dep > param_fin:
    step = -0.05
if param_dep < param_fin:
    step = 0.05

if Phase == "normale":
    non_variable_parameters = ["U",
    "mu",
    "e",
    "tpp",
    "tppp",
    "tpd"
]
if Phase == "supra":
    non_variable_parameters = ["U",
        "mu",
        "D",
        "e",
        "tpp",
        "tppp",
        "tpd"
    ]
variable_parameters = [
    "sb1_1",
    "sb2_1",
    "sb3_1",
    "sb4_1",
    "sb5_1",
    "sb6_1",
    "sb7_1",
    "sb8_1",
    "eb1_1",
    "eb2_1",
    "eb3_1",
    "eb4_1",
    "eb5_1",
    "eb6_1",
    "eb7_1",
    "eb8_1",
    "tb1_1",
    "tb2_1",
    "tb3_1",
    "tb4_1",
    "tb5_1",
    "tb6_1",
    "tb7_1",
    "tb8_1"
]




if Phase =="normale":
    target_sectors = ["R0:N14:S0/R0:N13:S1/R0:N13:S-1/R0:N12:S0/R0:N11:S1/R0:N11:S-1/R0:N10:S0/R0:N9:S1/R0:N9:S-1/R0:N15:S1/R0:N15:S-1/R0:N16:S0/R0:N17:S1/R0:N17:S-1/R0:N8:S0/R0:N7:S1/R0:N7:S-1/R0:N18:S0/R0:N19:S1/R0:N19:S-1","R0:N8:S0"]
    non_variable_parameters.remove("D")
    variable_parameters.remove("sb1_1")
    variable_parameters.remove("sb2_1")
    variable_parameters.remove("sb3_1")
    variable_parameters.remove("sb4_1")
    variable_parameters.remove("sb5_1")
    variable_parameters.remove("sb6_1")
    variable_parameters.remove("sb7_1")
    variable_parameters.remove("sb8_1")
if Phase =="supra":
    target_sectors = ["R0:S0","R0:N8:S0"]

parameter_names = non_variable_parameters+variable_parameters


parametres = """
"""
for name in parameter_names:
    try:
        parametres += '\n'+name+'='+str(data[name][-1])
    except:
        parametres += '\n'+name+'=0'



model.set_target_sectors(target_sectors)
model.set_parameters(parametres)
model.set_params_from_file(path,0)
model.set_parameter("U",10)
model.set_parameter("e",2.0097)
model.set_parameter("mu",15)
model_params = model.parameter_set("all")
print("#########################################")
print(model_params)
print("#########################################")


isExist = os.path.exists(f"{current_file_dir}/new_{Phase}/{mat}/données")
if isExist == False:
    os.makedirs(f"{current_file_dir}/{Phase}/{mat}/données")
os.chdir(f"{current_file_dir}/{Phase}/{mat}/données")
filename = f'BSCCO_boucle_{param}_U=10_e=2.tsv'

def run_cdmft():
    """Fonction qui execute la cdmft"""
    solution = cdmft.CDMFT(model,variable_parameters, accur=accuracy, iteration = "Broyden", method='L-BFGS-B', file = filename,max_value=50, maxiter = 100,convergence = condition_converg,accur_bath=1e-3,wc=5)
    return solution.I

model.controlled_loop(task=run_cdmft, varia=variable_parameters, loop_param=param, loop_range=((param_dep,param_fin,step)),retry="skip")
