import random


Dic_points_dep = {}
def generate_random_tuple(intervals):
    """
    Génère un tuple avec des valeurs aléatoires pour chaque intervalle donné.
    
    :param intervals: Liste de tuples représentant les intervalles pour chaque valeur (min, max)
    :return: Un tuple avec des valeurs aléatoires pour chaque intervalle
    """
    return tuple(random.uniform(interval[0], interval[1]) for interval in intervals)

# Exemple d'utilisation
#(U,mu,e,tpd,tppp)
intervals = [(8, 14), (10, 16), (1, 4), (1,4), (0.1,1)]  # Intervalle pour chaque valeur du tuple

#(U,mu,tp)
# intervals = [(9, 14), (0.5, 2.5), (0,0.5)] 

for x in range(50):
    random_tuple = generate_random_tuple(intervals)
    Dic_points_dep[x]=random_tuple