import numpy as np
import os
import sys
import pyqcm.cdmft_modif_LB as cdmft
import pyqcm
from points_depart_1bande_elec import Dic_points_dep
from models.model_1bande_8b import model


nb_bains = 8
condition_converg = ['self-energy'] #Condition de convergence des calculs cdmft
accuracy = [1e-4] #Valeur de précision pour considérer la convergence
param = "mu" #paramètre sur lequel on boucle
target_param = 0

# target_param = float(sys.argv[1]) #Valeur du paramètre sur lequel on boucle que l'on veut atteindre
current_file_dir = os.path.dirname(os.path.abspath(__file__))
path = "/net/nfs-iq/data/lbsc/Maîtrise_LBSC/Optimisation_D/solutions/boucle_1bande/U=7_t=1_tp=-0.5.tsv" # path de la solution CDMFT de départ
print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(path)

data = np.genfromtxt(path, names=True)
Valeur_U = data["U"][-1]
tp = data["tp"][-1]
t = data["t"][-1]
mu = data["mu"][-1]
# param_depart = data[param][-1]
param_depart = 3.8 #Valeur de départ du paramètre
if param_depart < target_param:
    direction = "up"
else:
    direction = "down"    
# param_depart = 13.0
pyqcm.set_global_parameter("max_iter_lanczos",3000)
pyqcm.set_global_parameter('Hamiltonian_format','E')
pyqcm.set_global_parameter('Ground_state_method','P')
pyqcm.set_global_parameter('Ground_state_init_last')
pyqcm.set_global_parameter('parallel_sectors')
pyqcm.set_global_parameter('PRIMME_preconditionning', 1)


non_variable_parameters = ["U",
    "mu",
    "D",
    "t",
    "tp"
]

#Supra
target_sectors = ["R0:S0"]

#Normale
# target_sectors = ["R0:N4:S0/R0:N6:S0/R0:N7:S-1/R0:N7:S1/R0:N8:S0/R0:N9:S1/R0:N9:S-1/R0:N10:S0/R0:N12:S0"]

# Supra
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

#Normale
# variable_parameters = [
#     "eb1_1",
#     "eb2_1",
#     "eb3_1",
#     "eb4_1",
#     "eb5_1",
#     "eb6_1",
#     "eb7_1",
#     "eb8_1",
#     "tb1_1",
#     "tb2_1",
#     "tb3_1",
#     "tb4_1",
#     "tb5_1",
#     "tb6_1",
#     "tb7_1",
#     "tb8_1"
# ]
parameter_names = non_variable_parameters+variable_parameters


parametres = """
"""



for name in parameter_names:
    parametres += '\n'+name+'='+str(data[name][-1])
    # parametres += '\n'+name+'=0.001'

    


model.set_target_sectors(target_sectors)
model.set_parameters(parametres)
model.set_params_from_file(path,-1)
param_mod = model.parameters()
print("***************************************")
print(param_mod)




path_données = f"{current_file_dir}/solutions/boucle_1bande" #path du dossier dans lequel le fichier des solutions sera inscrit
isExist = os.path.exists(path_données)
if isExist == False:
    os.makedirs(path_données)
os.chdir(path_données)
filename = "U=7_t=1_tp=-0.5.tsv" #Nom du fichier dans lequel seront inscrites les données


def run_cdmft():
    """Fonction qui execute la cdmft"""
    solution = cdmft.CDMFT(model,variable_parameters, accur=accuracy, iteration = "Broyden", method='L-BFGS-B', file = filename,max_value=100, maxiter = 50,convergence = condition_converg,accur_bath=1e-3,beta=35)
    return solution.I

step = 0.005 #Valeur de l'interval entre chaque itération de la boucle sur le paramètre
if param_depart < target_param:
    step_param = step
if param_depart > target_param:
    step_param=-step


model.controlled_loop(task = run_cdmft, varia=variable_parameters, loop_param=param, loop_range=((param_depart,target_param,step_param)),retry="skip")