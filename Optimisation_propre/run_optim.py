from models.model_3bandes_8b import model
from Maîtrise_LBSC.Optimisation_propre.classe_optim import optimisation
import sys
from points_depart_3bandes_trous import Dico_dep



beta = 35.0
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print(beta)
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
nb=8
ind_dic = int(sys.argv[1])
print(Dico_dep[ind_dic])
mat = Dico_dep[ind_dic]
liste_non_var = ["U","e","tpd","tppp","mu","D","tpp"]
liste_varia = ["U","e","tpd","tppp","mu"]

liste_param_bains = [
                "eb1_1",
                "eb2_1",
                "eb3_1",
                "eb4_1",
                "eb5_1",
                "eb6_1",
                "eb7_1",
                "eb8_1",
                "sb1_1",
                "sb2_1",
                "sb3_1",
                "sb4_1",
                "sb5_1",
                "sb6_1",
                "sb7_1",
                "sb8_1",
                "tb1_1",
                "tb2_1",
                "tb3_1",
                "tb4_1",
                "tb5_1",
                "tb6_1",
                "tb7_1",
                "tb8_1"
            ]

iteration = "Broyden"
# liste_varia = ["U","e","tpd","tppp","mu"]
simplex_step = [0.2,0.04,0.04,0.04,0.03]
print(f"INDICE DU DICO DE DÉPART:{ind_dic}")
print(liste_varia)
print(simplex_step)
target_sectors = ["R0:S0", "R0:N8:S0"]
# target_sectors = ["R0:S0"]

doss_solu = "/net/nfs-iq/data/lbsc/Maîtrise_LBSC/Optimisation_D/solutions/Optimisation_3bandes"
filename = f"3bandes_20dec_24_{mat}"
a = optimisation(f"sol_dep/opt_3bandes_trous/{mat}",doss_solu=doss_solu,model=model,filename=filename,liste_param_mod=liste_non_var,liste_param_varia=liste_varia,liste_param_bains=liste_param_bains,target_sectors=target_sectors,beta=beta,step_dep=simplex_step,iteration=iteration,antiferro=False)
a.set_up()
a.run_opt()