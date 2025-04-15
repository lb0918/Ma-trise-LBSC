import numpy as np
import os
import sys
import pyqcm.cdmft_modif_LB as cdmft
import pyqcm
from points_depart_3bandes_trous import Dico_dep


nb_bains = 8
condition_converg = ['self-energy', "D"] #Condition de convergence des calculs cdmft
accuracy = [1e-4,1e-3] #Valeur de précision pour considérer la convergence
param = "mu" #paramètre sur lequel on boucle
target_param = 0
# target_param = float(sys.argv[1])#Valeur du paramètre sur lequel on boucle que l'on veut atteindre 
current_file_dir = os.path.dirname(os.path.abspath(__file__))
path = "/net/nfs-iq/data/lbsc/Maîtrise_LBSC/Optimisation_D/solutions/boucles_3bandes/boucle_tppp_3bande_max_U=10.975,mu=9.823,tpd=2.437,tppp=0.861,e=3.26.tsv" #path de la solution de départ
print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(path)

data = np.genfromtxt(path, names=True)
Valeur_U = data["U"][-1]
e = data["e"][-1]
tpd = data["tpd"][-1]
tppp = data["tppp"][-1]
param_depart = data[param][-1] #Valeur de départ du paramètre
pyqcm.set_global_parameter("max_iter_lanczos",3000)
pyqcm.set_global_parameter('Hamiltonian_format','E')
pyqcm.set_global_parameter('Ground_state_method','P')
pyqcm.set_global_parameter('Ground_state_init_last')
pyqcm.set_global_parameter('parallel_sectors')
pyqcm.set_global_parameter('PRIMME_preconditionning', 1)
pyqcm.set_global_parameter("accur_OP", 0.00001)

non_variable_parameters = ["U",
    "mu",
    "D",
    "e",
    "tpp",
    "tppp",
    "tpd"
]
if nb_bains == 8:
    from models.model_3bandes_8b import model
    target_sectors = ["R0:S0", "R0:N8:S0"]
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
if nb_bains == 6:
    from models.model_6b import model
    target_sectors = ["R0:S0", "R0:N6:S0"]
    variable_parameters = [
        "sb1_1",
        "sb2_1",
        "sb3_1",
        "sb4_1",
        "sb5_1",
        "sb6_1",
        "eb1_1",
        "eb2_1",
        "eb3_1",
        "eb4_1",
        "eb5_1",
        "eb6_1",
        "tb1_1",
        "tb2_1",
        "tb3_1",
        "tb4_1",
        "tb5_1",
        "tb6_1"
    ]
if nb_bains == 4:
    from models.model_4b import model
    target_sectors = ["R0:S0", "R0:N4:S0"]
    variable_parameters = [
        "sb1_1",
        "sb2_1",
        "sb3_1",
        "sb4_1",
        "eb1_1",
        "eb2_1",
        "eb3_1",
        "eb4_1",
        "tb1_1",
        "tb2_1",
        "tb3_1",
        "tb4_1",
    ]
parameter_names = non_variable_parameters+variable_parameters


parametres = """
"""




for name in non_variable_parameters:
    parametres += '\n'+name+'='+str(data[name][-1])
for name in variable_parameters:
    parametres += '\n'+name+'='+str(data[name][-1])


    


model.set_target_sectors(target_sectors)
model.set_parameters(parametres)
model.set_params_from_file(path,-1)
param_mod = model.parameters()
print("***************************************")
print(param_mod)



path_données = f"{current_file_dir}/solutions/boucles_3bandes/" #path du dossier dans lequel le fichier des solutions sera inscrit
isExist = os.path.exists(path_données)
if isExist == False:
    os.makedirs(path_données)
os.chdir(path_données)
filename = f'boucle_mu_3bande_max_U=10.957,mu=9.823,tpd=2.437,tppp=0.861.e=3.26_low_tppp.tsv' #Nom du fichier dans lequel seront inscrites les données

def run_cdmft():
    """Fonction qui execute la cdmft"""
    solution = cdmft.CDMFT(model,variable_parameters, accur=accuracy, method="L-BFGS-B", file = filename, maxiter = 100,convergence = condition_converg,beta=35)
    return solution.I

step = 0.05 #Valeur de l'interval entre chaque itération de la boucle sur le paramètre
if param_depart < target_param:
    step_param = step
if param_depart > target_param:
    step_param=-step


#Boucle sur le paramètre choisi en appliquant la cdmft
model.controlled_loop(task = run_cdmft, varia=variable_parameters, loop_param=param, loop_range=((param_depart,target_param,step_param)))