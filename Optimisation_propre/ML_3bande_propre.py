import scipy.optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy
import random
from NelderMead import nelder_mead_python_git
import itertools
from joblib import dump, load
import os
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'
plt.gcf().set_size_inches(14/2.54,9/2.54)


liste_files_mats = os.listdir("hole_doped_mats")


def generate_random_tuple(intervals):
    """
    Génère un tuple avec des valeurs aléatoires pour chaque intervalle donné.
    
    :param intervals: Liste de tuples représentant les intervalles pour chaque valeur (min, max)
    :return: Un tuple avec des valeurs aléatoires pour chaque intervalle
    """
    return tuple(random.uniform(interval[0], interval[1]) for interval in intervals)

intervals =[(9,14),(1.6,1.8), (1.5,4),(-1.5,1.5),(0,4)]
# intervals =[(2,20),(2,20), (1.5,7),(-5,5),(0,9)]  # Intervalle pour chaque valeur du tuple


path = "/net/nfs-iq/data/lbsc/Maîtrise_LBSC/Optimisation_propre/solutions/Optimisation_3bandes/merged_filtred_shuffle.tsv"
data_tot = np.genfromtxt(path,names=True)
data_train = data_tot[:-20000]
data_test = data_tot[-20000:]
# data_train = data_tot[:]
# data_test = data_tot[:]
liste_samples_train = []
liste_features_train = []
liste_samples_test = []
liste_features_test = []
interval_entrainement = len(data_train)

param_algo_optim = ["U","mu_ave","tpd",'tppp','e']


for x in data_train:
    liste_samples_train.append([float(x[z]) for z in param_algo_optim])
    liste_features_train.append(abs(x["D_ave"]))

# Remplir les listes pour le test
for x in data_test:
    liste_samples_test.append([float(x[z]) for z in param_algo_optim])
    liste_features_test.append(abs(float(x["D_ave"])))

# Conversion en arrays numpy
X_train = np.array(liste_samples_train)
y_train = np.array(liste_features_train)
X_test = np.array(liste_samples_test)
y_test = np.array(liste_features_test)




scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
# dump(scaler, 'ML_models/scaler_3bandes.joblib')



n_comp = 5 # Nombre de composantes (variables) utilisées pour la PCA 
pca = PCA(n_components=n_comp) 
X_pca = pca.fit_transform(X_scaled) # Application de la PCA sur les données d'entraînement normalisées
liste_nb_feat = [3] # Nombre de variables considérées dans la création des arbres décisionnels
liste_nb_trees = [450] # Nombre d'arbres décisionnels
combinations = list(itertools.product(liste_nb_trees, liste_nb_feat)) # Combinaisons du nombre d'arbres et de variables considérées pour chacun d'eux

"""######################### Boucle pour l'entraînement de plusieurs modèles #########################"""
# for comb in combinations:
#     print(comb)
#     rf = RandomForestRegressor(n_estimators=comb[0], random_state=0,max_features=comb[1])
#     rf.fit(X_pca, y_train)
#     print("TRAINED!!!")
#     # dump(hgb, 'HGB_model.joblib')
#     dump(rf, f'ML_models/RTF_3bandes_5_components_{comb[0]}_trees_{comb[1]}_feat_model.joblib')
# quit()
"""#################################################################################################"""

for comb in combinations:
    """
    Boucle pour tester la performance de plusieurs modèles
    """
    print(comb)
    folder = f"Figures_ML_RTF/3bandes_{comb[0]}_trees_{comb[1]}_feat"
    isExist = os.path.exists(folder)
    if isExist == False:
        os.makedirs(folder)

    loaded_model = load(f'ML_models/RTF_3bandes_5_components_{comb[0]}_trees_{comb[1]}_feat_model.joblib')


    """IMPORTANT"""
    new_point_scaled = scaler.transform(X_test) # Normalisation de l'ensemble test
    new_point_pca = pca.transform(new_point_scaled) # Application de la PCA sur les données test normalisées


    """IMPORTANT"""
    # Prédiction avec le modèle Random Forest
    # prediction = rf.predict(new_point_pca)
    prediction = loaded_model.predict(new_point_pca)


    # Calcul du RMSE
    score = np.sqrt(mean_squared_error(y_test,prediction))







    """Section prédiction espace paramètres"""
    # liste_points = []
    def func2(liste):
        """
        Fonction qui applique les transformations nécessaires aux points qu'on veut prédire
        """
        scaled = scaler.transform(np.array([liste]))
        pt_pca = pca.transform(scaled)
        # pred = rf.predict(pt_pca)
        pred = loaded_model.predict(pt_pca)
        return abs(pred[0])



    liste_tempo_points_YBCO = []
    liste_tempo_val_YBCO = []
    liste_tempo_points_LCO = []
    liste_tempo_val_LCO = []
    rangee = np.linspace(1,1.67,2000)

    #Produit une boucle sur n "mu_ave".
    for path in liste_files_mats:
        """
        Boucle pour produire des prédictions sur la valeur de paramètre d'ordre pour des matériaux spécifiques
        """
        liste_tempo_points_mats = []
        liste_tempo_val_mats = []
        data_mat = np.genfromtxt("hole_doped_mats/"+path,names=True)
        U_mat, tpd_mat, tppp_mat, e_mat = data_mat["U"][-1], data_mat["tpd"][-1],data_mat["tppp"][-1],data_mat["e"][-1]
        try:
            for x in data_mat["ave_mu"]:
                liste_tempo_points_mats.append(x)
                res = abs(func2([U_mat,x,tpd_mat,tppp_mat,e_mat]))
                liste_tempo_val_mats.append(res)
        except:
            for x in data_mat["mu_ave"]:
                liste_tempo_points_mats.append(x)
                res = abs(func2([U_mat,x,tpd_mat,tppp_mat,e_mat]))
                liste_tempo_val_mats.append(res)
        output_filename = f"hole_doped_mats_pred/{comb[0]}_trees_{comb[1]}_feat_output_{path[18:-4]}.tsv"  # Nom du fichier de sortie basé sur path
        with open(output_filename, "w") as f:
            f.write("ave_mu\tave_D\n")  # En-têtes des colonnes
            for mu, D in zip(liste_tempo_points_mats, liste_tempo_val_mats):
                f.write(f"{mu}\t{D}\n")
        try:
            score_tempo  = np.sqrt(mean_squared_error(abs(data_mat["ave_D"]),liste_tempo_val_mats))
        except:
            score_tempo  = np.sqrt(mean_squared_error(abs(data_mat["D_ave"]),liste_tempo_val_mats))
        try:
            plt.plot(data_mat["ave_mu"],abs(data_mat["ave_D"]),"o",alpha=0.9,label="data")
        except:
            plt.plot(data_mat["mu_ave"],abs(data_mat["D_ave"]),"o",alpha=0.9,label="data")
        plt.title(f"Reproduction de la courbe supra {path[18:-4]}\n Méthode:RTF__RMSE={score_tempo}")
        plt.xlabel("n")
        plt.ylabel("<D>")
        plt.legend()
        plt.show()
        # plt.savefig(folder+f"/{path[18:-4]}_data.png")
        plt.close()

