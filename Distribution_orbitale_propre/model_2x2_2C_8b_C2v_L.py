from cluster_2x2_2C_8b_C2v_L import CM
import pyqcm

#-------------------------------------------------------------------
# construction of the lattice model 

clus = pyqcm.cluster(CM, [[0,0,0],[2,0,0],[0,2,0],[2,2,0]]) 
CM2 = pyqcm.cluster_model(8,name = "uncor")
clus2 = pyqcm.cluster(CM2,[[1,0,0],[3,0,0],[0,1,0],[2,1,0],[1,2,0],[3,2,0],[0,3,0],[2,3,0]])

model = pyqcm.lattice_model("2x2_2C_8b_C2v_L",[clus,clus2],[[4,0,0],[0,4,0]],[[2,0,0],[0,2,0],])
model.set_basis([[0.5,0,0],[0,0.5,0],[0,0,0.5]])

model.interaction_operator("U",orbitals=(1,1))

# model.hopping_operator("tc",[2,0,0],-1,orbitals=(1,1))#between nearest Cu sites
# model.hopping_operator("tc",[0,2,0],-1,orbitals=(1,1))

model.hopping_operator("tpp",[1,1,0],-1,orbitals=(2,3))#between nearest neighbor O
model.hopping_operator("tpp",[1,1,0],-1,orbitals=(3,2))
model.hopping_operator("tpp",[-1,1,0],1,orbitals=(2,3))
model.hopping_operator("tpp",[-1,1,0],1,orbitals=(3,2))
model.hopping_operator("tppp",[2,0,0],1,orbitals=(2,2))#between O sites separated by a Cu site
model.hopping_operator("tppp",[0,2,0],1,orbitals=(3,3))

model.hopping_operator("e",[0,0,0],1,orbitals=(2,2),tau=0,sigma=0)
model.hopping_operator("e",[0,0,0],1,orbitals=(3,3),tau=0,sigma=0)

model.hopping_operator("tpd",[1,0,0],1,orbitals=(1,2))#between Cu and O nearest neighbor
model.hopping_operator("tpd",[1,0,0],-1,orbitals=(2,1))
model.hopping_operator("tpd",[0,1,0],1,orbitals=(1,3))
model.hopping_operator("tpd",[0,1,0],-1,orbitals=(3,1))

model.anomalous_operator("D",[2,0,0],1,orbitals=(1,1),type="singlet") #NN singlet
model.anomalous_operator("D",[0,2,0],-1,orbitals=(1,1),type="singlet") #NN singlet