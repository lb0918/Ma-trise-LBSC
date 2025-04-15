from clusters.cluster_1bande_8b import CM
# from clusters.cluster_8b_J_1bande import CM
import pyqcm

#-------------------------------------------------------------------
# construction du modèle de réseau 1bande avec terme de saut au 3e voisin et des paramètres d'ordre de supra Dxy et S étendue.

clus = pyqcm.cluster(CM, [[0,0,0],[1,0,0],[0,1,0],[1,1,0]])

model = pyqcm.lattice_model("2x2_2C_8b_C2v_L",[clus],[[2,0,0],[0,2,0]],[[1,0,0],[0,1,0],])
model.set_basis([[1,0,0],[0,1,0],[0,0,1]])

model.interaction_operator("U",orbitals=(1,1))

model.hopping_operator("t",[1,0,0],-1,orbitals=(1,1))#between nearest Cu sites
model.hopping_operator("t",[0,1,0],-1,orbitals=(1,1))

model.hopping_operator("tp",[1,1,0],-1,orbitals=(1,1))
model.hopping_operator("tp",[1,-1,0],-1,orbitals=(1,1))
model.hopping_operator("tpp",[2,0,0],-1,orbitals=(1,1))
model.hopping_operator("tpp",[0,2,0],-1,orbitals=(1,1))



model.anomalous_operator("D",[2,0,0],1,orbitals=(1,1),type="singlet") #NN singlet
model.anomalous_operator("D",[0,2,0],-1,orbitals=(1,1),type="singlet") #NN singlet
model.anomalous_operator("Dxy", [2,2,0],1)
model.anomalous_operator("Dxy", [-2,2,0],-1)
model.anomalous_operator("extS",[2,0,0],1)
model.anomalous_operator("extS",[0,2,0],1)


