import scipy.optimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy
import random
from NelderMead import nelder_mead_python_git
from sklearn.linear_model import LinearRegression
from scipy.interpolate import griddata
from scipy.interpolate import Rbf
from joblib import dump, load












path = "/net/nfs-iq/data/lbsc/Maîtrise_LBSC/Optimisation_propre/solutions/Optimisation_1bande_trous/merged_file_filtered_complet_shuffle.tsv" #Path vers le fichier contenant l'ensemble des données (test+entraîement)


data_tot = np.genfromtxt(path,names=True)
U_tot, n_tot, tp_tot, D_tot = data_tot["U"], data_tot["mu_ave"], data_tot["tp"], abs(data_tot["D_ave"])

data_train = data_tot[:-20000] # Données d'entraînement
data_test = data_tot[-20000:] # Données de test


liste_samples_train = [] # Les variables de l'ensemble d'entraînement
liste_features_train = [] # Les valeurs de sorties de l'ensemble d'entraînement
liste_samples_test = [] # Les variables de l'ensemble de test
liste_features_test = [] # Les valeurs de sorties de l'ensemble de test
interval_entrainement = len(data_train)

param_algo_optim = ["U","mu_ave","tp"] # Liste des variables de l'algorithme


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
X_scaled = scaler.fit_transform(X_train) # Normalisation des données d'entraînement




n_comp = 3 # Nombre de composantes (variables) utilisées pour la PCA 
pca = PCA(n_components=n_comp) 
X_pca = pca.fit_transform(X_scaled) # Application de la PCA sur les données d'entraînement normalisées
print("--------------")
print(pca.singular_values_) # Les valeurs singulières de la PCA, analogue aux valeurs propres dans la PCA telle qe décrite dans mon mémoire
print("--------------")


##############################################################################
# rf = RandomForestRegressor(n_estimators=350, random_state=0,max_features=2)
# rf.fit(X_pca, y_train) # Entraînement du modèle RTF
##############################################################################

loaded_model = load(f'ML_models/RTF_1bande_350_trees_2_feats_3_comp_model.joblib') # Chargement du modèle RTF déjà entraîné et garder en mémoire
print("TRAINED!!!")


"""IMPORTANT"""
new_point_scaled = scaler.transform(X_test) # Normalisation de l'ensemble test
new_point_pca = pca.transform(new_point_scaled) # Application de la PCA sur les données test normalisées



# Récupération de la variance expliquée par chaque composante
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)
# Affichage de la variance expliquée cumulative
cumulative_variance = np.cumsum(explained_variance_ratio)
print(cumulative_variance)



"""IMPORTANT"""
# Prédiction avec le modèle Random Forest
prediction = loaded_model.predict(new_point_pca)

# Calcul du RMSE
score = np.sqrt(mean_squared_error(y_test,prediction))




"""#############################Figure#############################"""
diff = abs(prediction-y_test)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage[charter]{mathdesign}'
plt.gcf().set_size_inches(14/2.54,10/2.54)
plt.ylabel("Nombre de solutions")
plt.xlabel("Écart entre ML et CDMFT")
plt.xlim(0,0.001)
plt.ylim(0,4000)
plt.hist(diff, bins=10000)
# plt.savefig("HISTOGRAMME_1BANDE_ML.pdf")
plt.show()
plt.close()

# quit()
"""##################################################################"""





"""Section prédiction espace paramètres"""
def func2(liste):
    """
    Fonction qui applique les transformations nécessaires aux points qu'on veut prédire
    """
    scaled = scaler.transform(np.array([liste]))
    pt_pca = pca.transform(scaled)
    pred = loaded_model.predict(pt_pca)
    return pred[0]

def generate_random_tuple(intervals):
    """
    Génère un tuple avec des valeurs aléatoires pour chaque intervalle donné.
    
    :param intervals: Liste de tuples représentant les intervalles pour chaque valeur (min, max)
    :return: Un tuple avec des valeurs aléatoires pour chaque intervalle
    """
    return tuple(random.uniform(interval[0], interval[1]) for interval in intervals)

intervals =[(2,7),(0.8,1.2), (-0.5,0.5)] # Intervalle pour chaque valeur du tuple 1bande
# intervals =[(2,20),(2,20), (1.5,7),(-5,5),(0,9)]  # Intervalle pour chaque valeur du tuple 3bande

liste_points, liste_valeurs = [], []
for x in range(5000):
    """
    Création aléatoire des points dont on veut prédire la valeur
    """
    point = np.array(generate_random_tuple(intervals))
    liste_points.append(point)
    res = func2(point)
    liste_valeurs.append(res)

U = np.array([x[0] for x in liste_points])
n = np.array([x[1] for x in liste_points])
tp = np.array([x[2] for x in liste_points])
D = np.array(liste_valeurs)

output_filename = f"Optimisation_1bande_trous_bon_param_ordre/pred_petit_U.tsv"
with open(output_filename, "w") as f:
            f.write("U\tn\ttp\tD_ave\n")  # En-têtes des colonnes
            for UU,nn,tptp,DD in zip(U, n,tp,D):
                f.write(f"{UU}\t{nn}\t{tptp}\t{DD}\n")

#Création d'une figure 3D
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
   

ax.grid(b = True, color ='grey', 
        linestyle ='-.', linewidth = 0.3, 
        alpha = 0.2) 
 
 

my_cmap = plt.get_cmap('cool')
sctt1 = ax.scatter3D(U, n, tp,
                    alpha = 0.8,
                    c = D, 
                    cmap = my_cmap, 
                    marker ='o')

plt.title("Parameter space 1-band Hubbard model ML")
ax.set_xlabel('U', fontweight ='bold') 
ax.set_ylabel('$\mu$', fontweight ='bold') 
ax.set_ylabel('n', fontweight ='bold')
ax.set_zlim([-1, 0.5]) 
ax.set_zlabel('$t_p$', fontweight ='bold')
cbar = fig.colorbar(sctt1, ax=ax, shrink=0.5, aspect=5, label='$<D>$')
plt.show()
quit()
"""##############################################################"""


