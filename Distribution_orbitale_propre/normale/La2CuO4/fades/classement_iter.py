import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import csv
import operator
import pandas as pd

current_file_dir = os.path.dirname(os.path.abspath(__file__))
mat = current_file_dir.split('/')[-1]
liste_dir_iter = current_file_dir.split('/')[:-1]
dir_iter = ''
for x in liste_dir_iter:
    dir_iter += "/"+str(x)
path = f"{current_file_dir}/cdmft_iter.tsv"
# data = np.genfromtxt(path, names=True)
# first = data[0]
# print(str(first))

dir_115 = f"{current_file_dir}/iter/iter_11.5.tsv"
if os.path.isfile(dir_115) is True:
    os.remove(dir_115)

# dir_10 = f"{current_file_dir}/iter/iter_10.tsv"
# if os.path.isfile(dir_10) is True:
#     os.remove(dir_10)

# dir_12 = f"{current_file_dir}/iter/iter_12.tsv"
# if os.path.isfile(dir_12) is True:
#     os.remove(dir_12)

# dir_14 = f"{current_file_dir}/iter/iter_14.tsv"
# if os.path.isfile(dir_14) is True:
#     os.remove(dir_14)

liste_glob_115 = []
# liste_glob_10 = []
# liste_glob_12 = []
# liste_glob_14 = []
first = ''
with open(path) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            if line[1] == 'U':
                first = line
                continue
            if line[1] == '11.5':
                liste_glob_115.append(line)
            # if line[1] == '10':
            #     liste_glob_10.append(line)
            # if line[1] == '12':
            #     liste_glob_12.append(line)
            # if line[1] == '14':
            #     liste_glob_14.append(line)
         
     

os.chdir(dir_iter+'/iter')
with open('iter_11.5.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(first)
        for x in liste_glob_115:
            tsv_writer.writerow(x)
# with open('iter_10.tsv', 'wt') as out_file:
#         tsv_writer = csv.writer(out_file, delimiter='\t')
#         tsv_writer.writerow(first)
#         for x in liste_glob_10:
#             tsv_writer.writerow(x)
# with open('iter_12.tsv', 'wt') as out_file:
#         tsv_writer = csv.writer(out_file, delimiter='\t')
#         tsv_writer.writerow(first)
#         for x in liste_glob_12:
#             tsv_writer.writerow(x)
# with open('iter_14.tsv', 'wt') as out_file:
#         tsv_writer = csv.writer(out_file, delimiter='\t')
#         tsv_writer.writerow(first)
#         for x in liste_glob_14:
#             tsv_writer.writerow(x)

            
    
            
    
