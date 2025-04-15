import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from model_2x2_2C_8b_C2v_L import model
from cluster_2x2_2C_8b_C2v_L import CM
import os
import pyqcm
import sys
from Information_mat import *
U_dico = {1:8.0,2:8.5,3:9.5,4:10.0,5:10.5,6:11.0,7:11.5,8:12.0,9:12.5}
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'
ind_U = int(sys.argv[1])
ind = 18
mat = Dic_mat[ind]
current_file_dir = os.path.dirname(os.path.abspath(__file__))
def densité_état(path_solution,ligne, titre_données,titre_figures,output_path,SC=True):
    print("!!!!!!!!!!!!!!!")
    freq = (-25,10)
    data = np.genfromtxt(path_solution, names=True)
    print(data['mu'][-1])
    fig,ax = plt.subplots()
    ax.set_title(titre_figures)
    fig.set_size_inches(11/2.54,6/2.54)
    fig.text(0.02, 0.98, " ", weight='bold', va='top', ha='left', transform=fig.transFigure)

    if SC:
        target_sectors = ["R0:S0","R0:N8:S0"]
        non_variable_parameters = ["U",
        "mu",
        "e",
        "tpp",
        "tppp",
        "tpd",
        "D"
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
    else:
        target_sectors = ["R0:N12:S0/R0:N10:S0/R0:N14:S0","R0:N8:S0"]
        non_variable_parameters = ["U",
        "mu",
        "e",
        "tpp",
        "tppp",
        "tpd"
            ]
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
            "tb8_1"
        ]



        
    parameter_names = non_variable_parameters+variable_parameters


    parametres = """
    """
    for name in parameter_names:
        try:
            parametres += '\n'+name+'='+str(data[name][ligne])
        except:
            parametres += '\n'+name+'=0.001'
    model.set_target_sectors(target_sectors)
    model.set_parameters(parametres)
    os.chdir(output_path)
    I = pyqcm.model_instance(model)
    I.plot_DoS(w=freq, eta=0.03, sum=False,file = f"{titre_données}.pdf",labels=["$Cu$","$O_x$","$O_y$"],plt_ax=ax,data_file=titre_données+".tsv")

path_solution = f"/net/nfs-iq/data/lbsc/Maîtrise_LBSC/Révision_stage_2022/new_supra/YBa2Cu3O7/DOS/données_boucle_mu_YBa2Cu3O7_U=9.5_recalcule_tot_wc=5.tsv"
output_path = f"{current_file_dir}/data_matériaux_supra/{mat}/DOS"
dataa = np.genfromtxt(path_solution, names=True)
ligne = -1

titre_données = f"DOS_{mat}_U=9.5_e=3"
titre_figure = f"YBCO: U=9.5, e=3"

densité_état(path_solution,ligne,titre_données=titre_données,titre_figures=titre_figure,output_path=output_path)

quit()
""""""""""""""""""""""""""""""







