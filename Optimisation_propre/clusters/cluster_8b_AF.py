import pyqcm
ns = 4
no = 12
nb = 8
CM = pyqcm.cluster_model(ns, nb, "corr", [[4,3,2,1,0,0,0,0,1,1,1,1]],bath_irrep = True)
CM2 = pyqcm.cluster_model(ns,name = "uncor")

E_bath=['eb{:d}'.format(i+1) for i in range(nb)]
S_bath=['sb{:d}'.format(i+1) for i in range(nb)]
T_bathA_up=['tbA{:d}up'.format(i+1) for i in range(nb)]
T_bathA_down=['tbA{:d}down'.format(i+1) for i in range(nb)]
T_bathB_up=['tbB{:d}up'.format(i+1) for i in range(nb)]
T_bathB_down=['tbB{:d}down'.format(i+1) for i in range(nb)]
Tr_bath=['trb{:d}'.format(i+1) for i in range(nb)]
Se_bath=['seb{:d}'.format(i+1) for i in range(nb)]
CM.E_bath=E_bath
CM.S_bath=S_bath
CM.T_bathA_up=T_bathA_up
CM.T_bathA_down=T_bathA_down
CM.T_bathB_up=T_bathB_up
CM.T_bathB_down=T_bathB_down
CM.Tr_bath=Tr_bath
CM.Se_bath=Se_bath
CM.no=no
Sign=[[1,1],[1,1],[1,1],[1,1],[1,-1],[1,-1],[1,-1],[1,-1]]
SignA=Sign ; SignB=Sign
Sign_supra=[[1,1,1,1],[1,1,1,1],[-1,1,1,-1],[-1,1,1,-1],[-1,1,-1,1],[-1,1,-1,1],[-1,-1,1,1],[-1,-1,1,1]]
for i in range(nb):
    CM.new_operator(E_bath[i],'one-body',[(ns+1+i,ns+1+i,1),(no+ns+1+i,no+ns+1+i,1)])
    CM.new_operator(S_bath[i],'anomalous',[(j+1,ns+1+i+no,Sign_supra[i][j]) for j in range(ns)]+[(ns+i+1,no+1+j,Sign_supra[i][j]) for j in range(ns)])
    A=[1,4]
    B=[2,3]

    CM.new_operator(T_bathA_up[i],'one-body',[(A[j],ns+1+i,SignA[i][j]) for j in range(2)])
    CM.new_operator(T_bathA_down[i],'one-body',[(A[j]+no,ns+1+i+no,SignA[i][j]) for j in range(2)])

    CM.new_operator(T_bathB_up[i],'one-body',[(B[j],ns+1+i,SignB[i][j]) for j in range(2)])
    CM.new_operator(T_bathB_down[i],'one-body',[(B[j]+no,ns+1+i+no,SignB[i][j]) for j in range(2)])#
    CM.new_operator(Tr_bath[i],'anomalous',[(j+1,ns+1+i+no,Sign_supra[i][j]) for j in range(ns)]+[(ns+i+1,no+1+j,-Sign_supra[i][j]) for j in range(ns)])