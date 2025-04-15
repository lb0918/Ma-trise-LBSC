from clusters.cluster_8b_AF import CM
import pyqcm

#-------------------------------------------------------------------
# construction du modèle de réseau 1bande permettant l'AF

clus = pyqcm.cluster(CM, [[0,0,0],[1,0,0],[0,1,0],[1,1,0]])

model = pyqcm.lattice_model("2x2_2C_8b_C2v_L",[clus],[[2,0,0],[0,2,0]],[[1,0,0],[0,1,0],])
model.set_basis([[1,0,0],[0,1,0],[0,0,1]])

model.interaction_operator("U",orbitals=(1,1))

model.hopping_operator("t",[1,0,0],-1,orbitals=(1,1))#between nearest Cu sites
model.hopping_operator("t",[0,1,0],-1,orbitals=(1,1))

model.hopping_operator("tp",[1,1,0],-1,orbitals=(1,1))
model.hopping_operator("tp",[1,-1,0],-1,orbitals=(1,1))



model.anomalous_operator("D",[1,0,0],1,orbitals=(1,1),type="singlet") #NN singlet
model.anomalous_operator("D",[0,1,0],-1,orbitals=(1,1),type="singlet") #NN singlet

model.density_wave('M','Z',(1,1,0))