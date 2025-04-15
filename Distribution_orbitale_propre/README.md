# Projet sur la distribution orbitale dans le modèle d'Emery

Le présent dossier contient les résultats/script/figures pertinent(e)s, découlants du sous-projet sur l'étude de la distribution orbitale dans le modèle d'Emery supraconducteur. Ce résultats sont présentés dans le mémoire de Louis-Bernard St-Cyr (moi-même) ainsi que dans l'article "Effect of the Coulomb repulsion and oxygen level on charge distribution and superconductivity in the Emery model for cuprates superconductors" https://arxiv.org/abs/2503.07810 .

J'ai également mis les résultats en lien avec la description d'autres matériaux que ceux présentés dans l'article, autant dans l'état normal que supraconducteur. La majorité des résultats du BSCCO, LCO, YBCO et NCCO sont issus de ma maîtrise, alors que pour les autres matériaux les résultats ont, en grand majorité, été obtenus lors de mon stage avec David Sénéchal en 2022.
--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------





*****************************************************************************************
*****************************************************************************************
# Dossiers


# data_materiaux_supra_article_only
Ce dossier contient tous mes résultats en lien avec les matériaux qui sont abordés dans l'article. Ces données n'ont pas nécessairement toutes été présentées dans l'article. Chaque dossier associé à un matériau contient des solutions CDMFT issues des boucles de densité ainsi que des DOS.


# normale
Ce dossier contient l'ensemble des données que j'ai collecté sur les différents matériaux dans l'état normal. Le nom des fichiers indique assez bien leur contenu.



# supra
Ce dossier contient l'ensemble des données que j'ai collecté sur les différents matériaux dans l'état supraconducteur. Le nom des fichiers indique assez bien leur contenu.



# prod_figures_article
Ce dossier contient les scripts et les données nécessaires à la production des différentes figures présentées dans l'article. Les dites figures sont aussi présentes au format PDF.
*****************************************************************************************
*****************************************************************************************





*****************************************************************************************
*****************************************************************************************
# Scripts

# cluster_2x2_2C_8b_C2v_L.py
Ce fichier contient l'amas 2x2 de cuivre ainsi que les 8 sites d'oxygènes étudiés dans l'article.



# model_2x2_2C_8b_C2v_L.py
Ce fichier contient le réseau du modèle d'Emery étudié dans l'article.



# boucle_param.py
Ce fichier sert à effectuer une boucle sur un paramètre du modèle.



# DOS.py
Ce fichier contient une fonction permettant de calculer la densité d'états (DOS) d'une solution CDMFT.



# Information_mat.py
Ce fichier contient un dictionnaire avec le nom de chacun des matériaux étudiés dans les dossiers "supra" et "normale".






