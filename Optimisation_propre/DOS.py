import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from models.model_3bandes_8b import model
from clusters.cluster_3bande_8b import CM
from points_depart_3bandes_trous import Dico_dep
import os
import pyqcm
import sys
pyqcm.set_global_parameter("max_iter_lanczos",3000)
current_file_dir = os.path.dirname(os.path.abspath(__file__))
def densité_état(path_solution,ligne, nb_bains, titre_données, M=False):
    """
    Fonction qui execute le calcul de la DOS.
    path_solution: Path vers le fichier contenant la solution CDMFT dont on veut la DOS.
    ligne: Ligne de la solution CDMFT choisie dans le fichier de "path_solution"
    nb_bains: Nombre de sites de bains de la solution
    titre_données: Nom à donner à la figure produite par PyQcm ainsi qu'au fichier de donnée de sortie de la DOS.
    M: Mettre à "True" si le système permet la présence de supra ET d'antiferromagnétisme.
    """
    print("!!!!!!!!!!!!!!!")
    freq = (-30,10)
    data = np.genfromtxt(path_solution, names=True)
    print(data['mu'][-1])


    #Emery
    non_variable_parameters = ["U",
    "mu",
    "e",
    "tpp",
    "tppp",
    "tpd",
    "D"
]
    
    #Hubbard 1bande
#     non_variable_parameters = ["U",
#     "mu",
#     "t",
#     "tp",
#     "D"
# ]

    if M:
        non_variable_parameters.append("M")
        target_sectors = ["R0:S0", "R0:N4:S0", "R0:N4:S0"]
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


        variable_parameters = E_bath_1+T_bathA_up_1+T_bathA_down_1+T_bathB_up_1+T_bathB_down_1+S_bath_1+Tr_bath_1

    else:
        if nb_bains == 8:
            target_sectors = ["R0:S0","R0:N8:S0"] #Emery
            # target_sectors = ["R0:S0"] #Hubbard 1bande
            variable_parameters = [
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
            "tb8_1",
            "sb1_1",
            "sb2_1",
            "sb3_1",
            "sb4_1",
            "sb5_1",
            "sb6_1",
            "sb7_1",
            "sb8_1",
        ]
        if nb_bains == 6:
            target_sectors = ["R0:S0","R0:N6:S0"]
            variable_parameters = [
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
            "tb6_1",
            "sb1_1",
            "sb2_1",
            "sb3_1",
            "sb4_1",
            "sb5_1",
            "sb6_1"
        ]
        if nb_bains == 4:
            target_sectors = ["R0:S0","R0:N4:S0"]
            variable_parameters = [
            "eb1_1",
            "eb2_1",
            "eb3_1",
            "eb4_1",
            "tb1_1",
            "tb2_1",
            "tb3_1",
            "tb4_1",
            "sb1_1",
            "sb2_1",
            "sb3_1",
            "sb4_1"
        ]
    parameter_names = non_variable_parameters+variable_parameters


    parametres = """
    """
    for name in parameter_names:
        parametres += '\n'+name+'='+str(data[name][ligne])
    model.set_target_sectors(target_sectors)
    model.set_parameters(parametres)

    os.chdir(f"{current_file_dir}/solutions/DOS/sol_optim_emery")
    I = pyqcm.model_instance(model)
    I.plot_DoS(w=freq, eta=0.15, sum=False,file = f"{titre_données}.png",data_file=f"{titre_données}.tsv")


ind_mat = int(sys.argv[1])
mat = Dico_dep[ind_mat]
path_solution = f"/net/nfs-iq/data/lbsc/Maîtrise_LBSC/Optimisation_D/solutions/Optimisation_3bandes_trous/3bandes_20dec_24_{mat}"
dataa = np.genfromtxt(path_solution, names=True)
ligne = -55 
nb_bains = 8
titre_données = f"DOS_{mat[:-5]}"
densité_état(path_solution,ligne,nb_bains,titre_données=titre_données,M=False)

quit()
""""""""""""""""""""""""""""""





