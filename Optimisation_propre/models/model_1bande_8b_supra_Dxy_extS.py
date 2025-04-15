from clusters.cluster_1bande_8b_Dxy_extS import CM
import pyqcm

#-------------------------------------------------------------------
# construction du modèle de réseau 1bande avec des paramètres d'ordre de supra Dxy et S étendue.

clus = pyqcm.cluster(CM, [[0,0,0],[1,0,0],[0,1,0],[1,1,0]])

model = pyqcm.lattice_model("2x2_2C_8b_C2v_L",[clus],[[2,0,0],[0,2,0]],[[1,0,0],[0,1,0],])
model.set_basis([[1,0,0],[0,1,0],[0,0,1]])

model.interaction_operator("U",orbitals=(1,1))

model.hopping_operator("t",[1,0,0],-1,orbitals=(1,1))#between nearest Cu sites
model.hopping_operator("t",[0,1,0],-1,orbitals=(1,1))

model.hopping_operator("tp",[1,1,0],-1,orbitals=(1,1))
model.hopping_operator("tp",[1,-1,0],-1,orbitals=(1,1))




model.anomalous_operator("D",[1,0,0],1) #NN singlet
model.anomalous_operator("D",[0,1,0],-1) #NN singlet
model.anomalous_operator("Dxy", [1,1,0],1)
model.anomalous_operator("Dxy", [-1,1,0],-1)
model.anomalous_operator("extS",[1,0,0],1)
model.anomalous_operator("extS",[0,1,0],1)


