import numpy as np
import pyqcm
# import pyqcm.cdmft as cdmft
import pyqcm.cdmft_modif_LB as cdmft
import os
from sklearn.linear_model import LinearRegression
import time
import copy
import itertools
import scipy.optimize as optimize

"""Ce fichier contient la classe qui permet de procéder à l'optmisation"""

class optimisation:
    pyqcm.set_global_parameter("max_iter_lanczos",3000)
    pyqcm.set_global_parameter('Hamiltonian_format','E')
    pyqcm.set_global_parameter('Ground_state_method','P')
    pyqcm.set_global_parameter('Ground_state_init_last')
    pyqcm.set_global_parameter('parallel_sectors')
    pyqcm.set_global_parameter('PRIMME_preconditionning', 1)



    def __init__(self,path_dep,doss_solu,model,filename,liste_param_mod,liste_param_varia,liste_param_bains,target_sectors,step_dep,iteration,beta=50,antiferro=False,predicteur=True,accur_OP = 0.00001):
        """
        La classe 'optimisation' prend les arguments suivants------->
        path_dep: Path vers le fichier de la solution de départ que l'on souhaite optimiser, la solution soit se trouver sur la dernière ligne du fichier.
        doss_solu: Path vers le dossier dans lequel on veut écrire le fichier des solutions.
        model: Le modèle associé à la solution de départ.
        filename: Le nom du fichier dans lequel vont être inscrites les solutions de la CDMFT.
        liste_param_mod: Liste contenant les string correspondants aux noms des opérateurs du modèle qui ne varient pas lors de la CDMFT.
        liste_param_varia: Liste contenant les string des paramètres libres dans l'optimisation Nelder-Mead
        liste_param_bains: Liste contenant les string correpondants aux noms des opérateurs de bains du modèle.
        step_dep: Liste de float contenant le step qu'on veut effectuer dans chaque direction de l'espace des paramètres pour produire le simplex initial.
        target_sectors: Liste contenant les string correspondants aux secteurs dans lesquels on cherche la solution de l'état fondamental.
        antiferro: [Bool] Mettre à 'True' si le modèle utilisé contient permet la présence d'un ordre AF à longue portée.
        predicteur: [Bool] Mettre à "False" pour ne PAS utiliser une interpolation linéaire pour la fonction d'hybridation initiale.
        accur_OP: Pécision de la valeur numérique des opérateurs dans PyQcm.
        """

        self.doss_solu = doss_solu
        self.accur_OP = accur_OP
        self.beta = beta
        self.iteration = iteration
        self.antiferro = antiferro
        self.predicteur = predicteur
        self.step_dep = step_dep
        self.filename = filename
        self.liste_param_mod = liste_param_mod
        self.current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.path_dep = f"{self.current_file_dir}/{path_dep}"
        self.model = model
        self.data = np.genfromtxt(self.path_dep, names = True)
        self.liste_bains = [[]]
        self.liste_val_varia = []
        self.liste_param_varia = liste_param_varia
        for x in self.liste_param_varia:
            try:
                self.liste_val_varia.append(self.data[x][-1])
            except:
                self.liste_val_varia.append(1e-9)
        print("Quantité optimisée par l'algorithme d'optimisation")
        print(self.param_opt)
        for param in liste_param_bains:
            try:
                self.liste_bains[0].append(self.data[param][-1])
            except:
                self.liste_bains[0].append(0.001)
        self.liste_liste_val_varia = [self.liste_val_varia]
        self.model_parameters = """"""
        self.bath_parameters = """"""
        self.param_bains = liste_param_bains
        for x in liste_param_mod:
            self.model_parameters += f"\n{x}=1e-9"
        for x in liste_param_bains:
            self.bath_parameters += f"\n{x}=1e-9"
        self.target_sectors = target_sectors

    def run_cdmft(self):
        """Fonction qui execute la cdmft"""
        if self.antiferro:
            solution = cdmft.CDMFT(self.model,self.param_bains, accur=[1e-4,1e-3,1e-3],file = self.filename, maxiter = 100, convergence=["self-energy","D","M"],max_value=100,iteration=self.iteration,beta=self.beta,method="L-BFGS-B")
        else:
            solution = cdmft.CDMFT(self.model,self.param_bains, accur=[1e-4,1e-3],file = self.filename, maxiter = 100, convergence=["self-energy","D"],max_value=100,iteration=self.iteration,beta=self.beta,method="L-BFGS-B")
        return solution.I

    def set_up(self):#Fonction à appeler pour mettre en place le modèle
        self.model.set_target_sectors(self.target_sectors)
        self.model.set_parameters(self.model_parameters+self.bath_parameters)
        self.model.set_params_from_file(self.path_dep)
        pyqcm.set_global_parameter("accur_OP", self.accur_OP)

    def perturbed_eigenvalues(self,matrix, incertitudes, num_simulations=10000):#Fonction servant à procéder au calcul de l'incertitude des valeurs propres du Hessien (Mémoire de Louis-Bernard St-Cyr pour plus de détails)
        """
        matrix: Matrice Hessienne
        incertitudes: Matrice contenant l'incertitude associée à chaque élément du Hessien
        num_simulations: Nombre de matrices perturbées qu'on veut produire pour le calcul d'incertitude des valeurs propres
        """
        n = matrix.shape[0]
        eigenvalues_list = np.zeros((num_simulations, n))
        
        for i in range(num_simulations):
            perturbed_matrix = matrix + np.random.normal(0, incertitudes, size=matrix.shape)
            eigenvalues_list[i, :] = np.linalg.eigvalsh(perturbed_matrix)
            
        mean_eigenvalues = np.mean(eigenvalues_list, axis=0)
        std_eigenvalues = np.std(eigenvalues_list, axis=0)
        
        return mean_eigenvalues, std_eigenvalues


    def nelder_mead_python(self,f,x_start,
                step, no_improve_thr=3e-6,
                no_improv_break=20, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
        '''
        f: Fonction à optimiser, doit retourner un scalaire.
        x_start: Array numpy contenant la position initiale dans l'espace des paramètres du modèle.
        step: Liste de float contenant le step dans chaque direction de l'espace des paramètres pour construire le simplex initial
        no_improv_thr,  no_improv_break (float, int): L'algorityhme de minimisation s'arrête après no_imporv_break itérations produisant une amélioration moindre
        que no_improv_thr.
        max_iter: Integer qui spécifie le nombre maximal d'itérations permises. Mettre à 0 pour une boucle infinie.
        alpha, gamma, rho, sigma: Des floats qui définissent l'algorihtme (Mémoire de Louis-Bernard St-Cyr pour plus de détails)

        return: tuple (Array contenant le point convergé, Meilleur score)
    '''
         # init
        dim = len(x_start)
        prev_best = f(x_start)
        no_improv = 0
        res = [[np.array(x_start), prev_best]]

        for i in range(dim):
            x = copy.copy(x_start)
            x[i] = x[i]+step[i]
            score = f(x)
            res.append([np.array(x), score])

        # simplex iter
        iters = 0
        while 1:
            # order
            res.sort(key=lambda x: x[1])
            best = res[0][1]

            # break after max_iter
            if max_iter and iters >= max_iter:
                return res[0]
            iters += 1

            # break after no_improv_break iterations with no improvement
            print ('...best so far:', best)

            if best < prev_best - no_improve_thr:
                no_improv = 0
                prev_best = best
            else:
                no_improv += 1

            if no_improv >= no_improv_break:
                return res[0]

            # centroid
            x0 = np.array([0.] * dim)
            for tup in res[:-1]:
                for i, c in enumerate(tup[0]):
                    x0[i] += c
            x0 = x0/(len(res)-1)

            # reflection
            xr = x0 + alpha*(x0 - res[-1][0])
            rscore = f(xr)
            if res[0][1] <= rscore < res[-2][1]:
                del res[-1]
                res.append([xr, rscore])
                continue

            # expansion
            if rscore < res[0][1]:
                xe = x0 + gamma*(x0 - res[-1][0])
                escore = f(xe)
                if escore < rscore:
                    del res[-1]
                    res.append([xe, escore])
                    continue
                else:
                    del res[-1]
                    res.append([xr, rscore])
                    continue

            # contraction
            xc = x0 + rho*(x0 - res[-1][0])
            cscore = f(xc)
            if cscore < res[-1][1]:
                del res[-1]
                res.append([xc, cscore])
                continue

            # reduction
            x1 = res[0][0]
            nres = []
            for tup in res:
                redx = x1 + sigma*(tup[0] - x1)
                score = f(redx)
                nres.append([redx, score])
            res = nres

    def hess_calcul(self):
        """
        Calcul de la matrice hessienne et de ses incertitudes.
        Return: False si on est à un maximum local.
        """
        ############### Prédiction linéaire des paramètres de bains ######################
        arr_non_var = np.array(self.liste_liste_val_varia)
        arr_bains = np.array(self.liste_bains)
        reg = LinearRegression().fit(arr_non_var,arr_bains)
        ##################################################################################

        ################ Mise en place à partir de la solution CDMFT convergé de NM ######
        param_var = self.liste_param_varia
        param_bains = self.param_bains
        val_param_bains = []
        path = self.filename
        data = np.genfromtxt(path,names=True)
        dico_step = {}
        for x in range(len(self.liste_param_varia)):
            dico_step[self.liste_param_varia[x]]=self.step_dep[x]/2
        print("----------------------------------------------------")
        print(dico_step)
        print("----------------------------------------------------")
        ave_D = abs(data["D_ave"][-1])
        dico = {}
        for param in param_var:
            dico[param]=data[param][-1]
        for bain in param_bains:
            val_param_bains.append(data[bain][-1])
        dico_hess = {}
        for z in range(len(param_var)):
            dico_hess[param_var[z]]=z
        hessien = np.empty((len(param_var),len(param_var)))
        ##################################################################################

        print("DÉBUT DU CALCUL DU HESSIEN!!!")
        for ind in range(len(param_var)):
            liste_double_deriv = [-2*ave_D]
            for item in dico:
                self.model.set_parameter(item,dico[item])
            self.model.set_parameter(param_var[ind], dico[param_var[ind]]-dico_step[param_var[ind]])
            params1 = self.model.parameters()
            arr1 = np.array([params1[p] for p in param_var])
            pre1 = reg.predict([arr1])

            ######################  Boucle pour réinitialiser la valeur des paramètres de bain ################################
            for x in range(len(param_bains)):
                self.model.set_parameter(param_bains[x],pre1[0][x])
            ###################################################################################################################

            print(f"!!!!!!!!!!!!{param_var[ind]}={params1[param_var[ind]]}!!!!!!!!!!!!!!!!!")
            print(params1)
            inst1 = self.run_cdmft()
            dico_temp1 = inst1.averages()
            ave_D1 = abs(dico_temp1["D"])
            liste_double_deriv.append(ave_D1)
            self.model.set_parameter(param_var[ind], dico[param_var[ind]]+dico_step[param_var[ind]])
            params2 = self.model.parameters()
            arr2 = np.array([params2[p] for p in param_var])
            pre2 = reg.predict([arr2])

            ######################  Boucle pour réinitialiser la valeur des paramètres de bain ################################
            for x in range(len(param_bains)):
                self.model.set_parameter(param_bains[x],pre2[0][x])
            ###################################################################################################################

            print(f"!!!!!!!!!!!!{param_var[ind]}={params2[param_var[ind]]}!!!!!!!!!!!!!!!!!")
            print(params2)
            inst2 = self.run_cdmft()
            dico_temp2 = inst2.averages()
            ave_D2 = abs(dico_temp2["D"])
            liste_double_deriv.append(ave_D2)
            print(f"Élément utilisés pour calculer la double dérivée selon une seule variable: {liste_double_deriv}")
            double_deriv = (liste_double_deriv[0]+liste_double_deriv[1]+liste_double_deriv[2])/(dico_step[param_var[ind]])**2
            hessien[ind][ind] = double_deriv

        ############# Calcul des doubles dérivées selon plusieurs variables################################################### 
        combin = list(itertools.combinations(param_var,2))
        for couple in combin:
            for item in dico:
                self.model.set_parameter(item,dico[item])
            self.model.set_parameter(couple[0], dico[couple[0]]+dico_step[couple[0]])
            self.model.set_parameter(couple[1], dico[couple[1]]+dico_step[couple[1]])
            params1 = self.model.parameters()
            arr1 = np.array([params1[p] for p in param_var])
            pre1 = reg.predict([arr1])

            ######################  Boucle pour réinitialiser la valeur des paramètres de bain ################################
            for x in range(len(param_bains)):
                self.model.set_parameter(param_bains[x],pre1[0][x])
            ###################################################################################################################

            print(f"!!!!!!!!!!!!{couple[0]}={params1[couple[0]]}!!!!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!!!!{couple[1]}={params1[couple[1]]}!!!!!!!!!!!!!!!!!")
            print(params1)
            inst1 = self.run_cdmft()
            dico_temp1 = inst1.averages()
            plus_plus = abs(dico_temp1["D"])
            self.model.set_parameter(couple[0], dico[couple[0]]-dico_step[couple[0]])
            self.model.set_parameter(couple[1], dico[couple[1]]-dico_step[couple[1]])
            params2 = self.model.parameters()
            arr2 = np.array([params2[p] for p in param_var])
            pre2 = reg.predict([arr2])

            ######################  Boucle pour réinitialiser la valeur des paramètres de bain ################################
            for x in range(len(param_bains)):
                self.model.set_parameter(param_bains[x],pre2[0][x])
            ###################################################################################################################

            print(f"!!!!!!!!!!!!{couple[0]}={params2[couple[0]]}!!!!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!!!!{couple[1]}={params2[couple[1]]}!!!!!!!!!!!!!!!!!")
            print(params2)
            inst2 = self.run_cdmft()
            dico_temp2 = inst2.averages()
            moins_moins = abs(dico_temp2["D"])

            self.model.set_parameter(couple[0], dico[couple[0]]+dico_step[couple[0]])
            self.model.set_parameter(couple[1], dico[couple[1]]-dico_step[couple[1]])
            params3 = self.model.parameters()
            arr3 = np.array([params3[p] for p in param_var])
            pre3 = reg.predict([arr3])

            ######################  Boucle pour réinitialiser la valeur des paramètres de bain ################################
            for x in range(len(param_bains)):
                self.model.set_parameter(param_bains[x],pre3[0][x])
            ###################################################################################################################

            print(f"!!!!!!!!!!!!{couple[0]}={params3[couple[0]]}!!!!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!!!!{couple[1]}={params3[couple[1]]}!!!!!!!!!!!!!!!!!")
            print(params3)
            inst3 = self.run_cdmft()
            dico_temp3 = inst3.averages()
            plus_moins = abs(dico_temp3["D"])
            self.model.set_parameter(couple[0], dico[couple[0]]-dico_step[couple[0]])
            self.model.set_parameter(couple[1], dico[couple[1]]+dico_step[couple[1]])
            params4 = self.model.parameters()
            arr4 = np.array([params4[p] for p in param_var])
            pre4 = reg.predict([arr4])

            ######################  Boucle pour réinitialiser la valeur des paramètres de bain ################################
            for x in range(len(param_bains)):
                self.model.set_parameter(param_bains[x],pre4[0][x])
            ###################################################################################################################

            print(f"!!!!!!!!!!!!{couple[0]}={params4[couple[0]]}!!!!!!!!!!!!!!!!!")
            print(f"!!!!!!!!!!!!{couple[1]}={params4[couple[1]]}!!!!!!!!!!!!!!!!!")
            print(params4)
            inst4 = self.run_cdmft()
            dico_temp4 = inst4.averages()
            moins_plus = abs(dico_temp4["D"])
            double_deriv = (plus_plus-plus_moins-moins_plus+moins_moins)/(4*dico_step[couple[0]]*dico_step[couple[1]])
            hessien[dico_hess[couple[0]]][dico_hess[couple[1]]] = double_deriv
            hessien[dico_hess[couple[1]]][dico_hess[couple[0]]] = double_deriv
        eigen = np.linalg.eig(hessien)
        val_propres = eigen.eigenvalues
        print("----------------------------------------------------------")
        print("""Le hessien est:""")
        print(hessien)
        print("----------------------------------------------------------")
        print(param_var)
        print(f"Les valeurs propres du hessien sont {val_propres}")

        #######Ajout pour calculs d'erreur des valeurs propres##############
        matrice_err = np.empty((len(self.liste_param_varia),len(self.liste_param_varia)))
        for i in range(len(hessien[0])):
            for j in range(len(hessien[0])):
                if i == j:
                    err_numerateur = np.sqrt(6)*self.accur_OP
                    param = self.liste_param_varia[i]
                    denom = dico_step[param]**2
                    err = err_numerateur/denom
                    matrice_err[i][j] = err
                else:
                    err_numerateur = 2*self.accur_OP
                    param1 = self.liste_param_varia[i]
                    param2 = self.liste_param_varia[j]
                    denom = 4*dico_step[param1]*dico_step[param2]
                    err = err_numerateur/denom
                    matrice_err[i][j] = err
        mean_eigenvalues, std_eigenvalues = self.perturbed_eigenvalues(hessien, matrice_err)
        print("Valeurs propres moyennes:", mean_eigenvalues)
        print("Incertitudes sur les valeurs propres:", std_eigenvalues)
        diff_values_std = []
        for x in range(len(mean_eigenvalues)):
            diff = abs(mean_eigenvalues[x]) - abs(std_eigenvalues[x])
            diff_values_std.append(diff)
        for x in diff_values_std:
            if x < 0:
                print("##################################")
                print("Le hessien calculé n'est pas fiable")
                print("##################################")
                return True
        ####################################################################

        for w in val_propres:
            if w > 0:
                return True
        return False
        
    
    def opt(self,arr):
        """
        Fonction qui procède à l'optimisation.
        """
        #########Prédiction des paramètres de bains linéaire#####################
        seconds = time.time()
        local_time = time.ctime(seconds)
        arr_non_var = np.array(self.liste_liste_val_varia)
        arr_bains = np.array(self.liste_bains)
        reg = LinearRegression().fit(arr_non_var,arr_bains)
        pre = reg.predict([arr])
        print("******************************")
        print("Prédiction")
        print(pre)
        for x in range(len(self.param_bains)):
            if self.predicteur:
                self.model.set_parameter(self.param_bains[x],pre[0][x])
        for x in range(len(self.liste_val_varia)):
            print(f"{self.liste_param_varia[x]}={arr[x]}")
        for x in range(len(self.liste_param_varia)):
            self.model.set_parameter(self.liste_param_varia[x],arr[x])
        params_before_cdmft = self.model.parameters()
        ##########################################################################

        print("************************************")
        print("Paramètres du modèle avant CDMFT:")
        print(params_before_cdmft)
        try:
            inst = self.run_cdmft()
        except:
            print("Il y a eu une erreur!") #En cas de non convergence de la solution CDMFT
            return 0.1    
        params_after_cdmft = self.model.parameters()
        print("************************************")
        print("Paramètres du modèle après CDMFT:")
        print(params_after_cdmft)
        params_instance = inst.parameters()
        liste_arr = []
        liste_bains_tempo = []
        for x in self.param_bains:
            liste_bains_tempo.append(params_after_cdmft[x])
        self.liste_bains.append(liste_bains_tempo)
        for x in arr:
            liste_arr.append(x)
        self.liste_liste_val_varia.append(liste_arr)
        print("************************************")
        print("Paramètres de l'instance:")
        print(params_instance)
        dico=inst.averages()
        D = dico["D"]
        n = dico["mu"]
        print("************************************")
        print("Valeur ave_D="+str(abs(D)))
        print(dico)
        print("************************************")
        print(f"La date et l'heure sont: {local_time}")
        return -abs(D)

    def run_opt(self):
        os.chdir(self.doss_solu)
        # Pour qu'il y ait un calcul de hessien après la convergence de l'algorithme la variable boolleens doit être True
        boolleens = True
        res = self.nelder_mead_python(self.opt,np.array(self.liste_val_varia),self.step_dep)
        print("DÉBUT 2E NELDER-MEAD")
        res = self.nelder_mead_python(self.opt,np.array(res[0]),self.step_dep)
        while boolleens:
            boolleens = self.hess_calcul()
            if boolleens is False:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print("L'ALGORITHME A VRAIMENT CONVERGÉ")
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                return res
            else:
                res = self.nelder_mead_python(self.opt,np.array(res[0]),self.step_dep)


    





    