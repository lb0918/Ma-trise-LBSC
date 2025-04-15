from clusters.cluster_8b_AF import CM, CM2
import pyqcm


#-------------------------------------------------------------------
# construction du modèle de réseau 3bandes permettant l'AF

clus_Cu = pyqcm.cluster(CM, [[0,0,0],[2,0,0],[0,2,0],[2,2,0]],[0,0,0])
clus_O1 = pyqcm.cluster(CM2,[[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],[0,0,0])
clus_O2 = pyqcm.cluster(clus_O1,[[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],[2,2,0])

model = pyqcm.lattice_model("2x2_2C_8b_C2v_L",[clus_Cu,clus_O1,clus_O2],[[4,0,0],[0,4,0]],[[2,0,0],[0,2,0]])
model.set_basis([[0.5,0,0],[0,0.5,0],[0,0,0.5]])


model.interaction_operator('U',orbitals=(1,1))




model.hopping_operator("tpp",[1,1,0],1,orbitals=(2,3))#between nearest neighbor O
model.hopping_operator("tpp",[1,1,0],1,orbitals=(3,2))
model.hopping_operator("tpp",[1,-1,0],1,orbitals=(2,3))
model.hopping_operator("tpp",[1,-1,0],1,orbitals=(3,2))
model.hopping_operator("tppp",[2,0,0],-1,orbitals=(2,2))#between O sites separated by a Cu site
model.hopping_operator("tppp",[0,2,0],-1,orbitals=(3,3))

model.hopping_operator("e",(0,0,0),1,orbitals=(2,2))
model.hopping_operator("e",(0,0,0),1,orbitals=(3,3))


model.hopping_operator("tpd",[1,0,0],1,orbitals=(1,2))#between Cu and O nearest neighbor
model.hopping_operator("tpd",[1,0,0],1,orbitals=(2,1))
model.hopping_operator("tpd",[0,1,0],1,orbitals=(1,3))
model.hopping_operator("tpd",[0,1,0],1,orbitals=(3,1))

model.anomalous_operator("D",[2,0,0],1,orbitals=(1,1),type="singlet") #NN singlet
model.anomalous_operator("D",[0,2,0],-1,orbitals=(1,1),type="singlet") #NN singlet

model.density_wave('M','Z',(1,1,0))
