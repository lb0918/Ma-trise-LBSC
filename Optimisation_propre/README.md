# Projet d'optimisation

Le présent dossier contient les résultats/script/figures pertinent(e)s, découlants du sous-projet sur la description algorithmique du modèle de Hubbard à 1 et 3 bande, dans le cadre de la maîtrise de Louis-Bernard St-Cyr (moi-même).

Une personne qui explore ce dossier verra qu'il y a beaucoup plus de matériel que ce qui est décrit dans le reste de ce README. Je décris ici seulement les résultats les plus importants, qui sont présentés dans mon mémoire. Plusieurs fichiers issus de tests et d'erreurs sont présents dans ce dossier, malgré que j'ai effectué un certain tri pour éliminer le matériel que je jugeais ne pourrais jamais être utile à personne. Les fichiers que j'ai décidé de garder mais qui ne sont pas explicités ci-dessous devraient être utilisables si l'utilisateur se fie à leurs titres respectifs.
--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------





*****************************************************************************************
*****************************************************************************************
# Scripts


# classe_optim.py
-------------------------------------------
Ce fichier contient la classe python qui sert à effectuer la procédure d'optimisation, telle que décrite dans mon mémoire.



# run_optim.py
-------------------------------------------
Ce fichier contient un exemple d'utilisation de la classe d'optimisation, se trouvant dans le fichier classe_optim.py.


# points_depart_1bande_elec.py
-------------------------------------------
Ce fichier contient seulement un dictionnaire dont les éléments sont des noms de fichiers, contenant les solutions de départ pour l'algorithme d'optimisation. Ce solution se trouvent dans sol_dep/opt_1bande_electrons.



# points_depart_1bande_trous.py
-------------------------------------------
Ce fichier contient seulement un dictionnaire dont les éléments sont des noms de fichiers, contenant les solutions de départ pour l'algorithme d'optimisation. Ce solution se trouvent dans sol_dep/opt_1bande_trous.



# points_depart_3bandes_trous.py
-------------------------------------------
Ce fichier contient seulement un dictionnaire dont les éléments sont des noms de fichiers, contenant les solutions de départ pour l'algorithme d'optimisation. Ce solution se trouvent dans sol_dep/opt_3bandes_trous.



# points_depart_xbandes_random.py
-------------------------------------------
Ce fichier contient une fonction qui génère un tuple, dont les éléments sont sélectionnés aléatoirement dans des intervals spécifiés en argument. Cette fonction m'a été utile pour produire les points de départs de l'algorithme d'optimisation.


# graph_memoire.ipynb
-------------------------------------------
Ce notebook contient tous les scripts python pour produire les figures de résultat de la section 4.1 de mon mémoire.


# boucle.py
-------------------------------------------
Fichier qui permet d'effectuer une boucle sur la valeur d'un paramètre du modèle de Hubbard à 3 bande supraconducteur.


# boucle_AF.py
-------------------------------------------
Fichier qui permet d'effectuer une boucle sur la valeur d'un paramètre du modèle de Hubbard à 3 bande supraconducteur et antiferromagnétique.


# boucle_1bande.py
-------------------------------------------
Fichier qui permet d'effectuer une boucle sur la valeur d'un paramètre du modèle de Hubbard à 1 bande supraconducteur.


# boucle_1bande_AF.py
-------------------------------------------
Fichier qui permet d'effectuer une boucle sur la valeur d'un paramètre du modèle de Hubbard à 1 bande supraconducteur et antiferromagnétique.


# DOS.py
Fichier qui contient une fonction permettant de calculer la DOS d'une solution CDMFT.


# ML_1bande_propre.py
Fichier qui contient un exemple d'utilisation du modèle d'apprentissage machine RTF, entraîné sur les données issues de l'optimisation dans le modèle à 1bande.


# ML_3bande_propre.py
Fichier qui contient un exemple d'utilisation du modèle d'apprentissage machine RTF, entraîné sur les données issues de l'optimisation dans le modèle à 3bande.
*****************************************************************************************
*****************************************************************************************






*****************************************************************************************
*****************************************************************************************
# Dossiers


# clusters
-------------------------------------------
Ce dossier contient plusieurs type d'amas. Les trois plus importants, étant ceux qui ont été étudiés dans mon mémoire, sont: cluster_3bande_8b.py, cluster_1bande_8b.py et cluster_3bande_8b.py.


# models
-------------------------------------------
Ce dossier contient plusieurs type de réseaux/super-réseaux. Les quatres plus importants, étant ceux qui ont été étudiés dans mon mémoire, sont: model_1bande_8b.py, model_1bande_8b_AF.py, model_3bandes_8b.py et model_3bandes_8b_AF.py.


# sol_dep
-------------------------------------------
Ce dossier contient différents dossiers, qui eux contiennent des solutions CDMFT, servant de points de départs à différentes tâches (DOS, optimisation, boucles, etc..). Les plus importants, en ce qui concerne l'algorithme d'optimisation, sont les dossiers opt_1bande_electrons, opt_1bande_trous et opt_3bandes_trous. 


# solutions
-------------------------------------------
Ce dossier contient différents dossiers, qui eux contiennent les fichiers de sortis des différentes tâches (DOS, optimisation, boucles, etc..). Les noms des dossiers indiquent assez bien leur contenu. Les deux dossiers les plus importants pour l'optimisation sont Optimisation_1bande_trous et Optimisation_3bandes. Les noms de fichiers dans chacun de ces dossiers d'optimisaiton réfère au point de départ de la séquence d'optimisation.

Pour le dossier Optimisation_1bande_trous, les résultats présentées dans le mémoire proviennent des fichiers qui sont identifiés 12 ou 13 décembre, de par leur nom. Les autres fichiers sont des optimisation faites lorsque le paramètre d'ordre (D) était défini de manière erronée dans le modèle à 1bande. J'ai refait le calcul des valeurs moyennes, pour ces solutions provenant d'une mauvaise optimisaition, dans les fichiers full_filtré_premier_quart_recalcule_averages_test.tsv, full_filtré_2e_quart_recalcule_averages_test.tsv, full_filtré_3e_quart_recalcule_averages_test.tsv, full_filtré_dernier_quart_recalcule_averages_test.tsv. 

L'ensemble des données provenant des bonnes séquences d'optimisations se trouvent dans le fichier merged_file.tsv du dossier Optimisation_1bande_trous. 

L'ensemble de toutes les données CDMFT dans le modèle à 1bande, incluant le recalcul des solutions provenant de l'optimisation du mauvais paramètre d'ordre, se trouve dans le fichier merged_file_filtered_complet_shuffle.tsv du dossier Optimisation_1bande_trous.

En ce qui concerne les séquences d'optimisations qui ont convergés avec certitudes dans le modèle à 3bande, présentées dans mon mémoire, les clés qui permettent de retrouver les noms de fichers, dans le dictionnnaire de points_depart_3bandes_trous.py, sont les suivantes: [34,1,2,3,9,10,15,16,18,20,25,26,28,29,30,32,33]


# test
-------------------------------------------
Ce dossier contient les fichiers ayant servis à faire l'exemple des densité d'états pour un isolant de Mott.


# ML_models
Ce dossier contient les deux modèles d'apprentissage machine (1bande et 3bande) qui ont servis à produire les résultats présentés dans mon mémoire.

Le modèle RTF 1bande est testé sur les 20 000 dernières solutions du fichier solutions/Optimisation_1bande_trous/merged_file_filtered_complet_shuffle.tsv. Le reste des données du fichier (donc toutes les données sauf les 20 000 dernières lignes) ont servis de données d'entraînement.

Le modèle RTF 3bande est testé sur les 20 000 dernières solutions du fichier solutions/Optimisation_3bandes/merged_filtered_shuffle.tsv. Le reste des données du fichier (donc toutes les données sauf les 20 000 dernières lignes) ont servis de données d'entraînement.
*****************************************************************************************
*****************************************************************************************

