import pyqcm
ns = 4
no = 12
CM = pyqcm.cluster_model(ns, no-ns, "clus", [[2, 1, 4, 3, 0, 0, 0, 0, 2, 2, 2, 2], [3, 4, 1, 2, 0, 0, 2, 2, 0, 0, 2, 2]],bath_irrep = True)
# bath energies
for i in range(1,9):
    name = 'eb'+str(i)
    lab = i+4
    CM.new_operator(name, 'one-body', [(lab, lab, 1.0), (lab+12, lab+12, 1.0)])

def new_tb(x, seq):
    elem = []
    for i in range(4):
        elem.append((i+1, x+4, seq[i]))
    for i in range(4):
        elem.append((i+13, x+16, seq[i]))
    name = 'tb'+str(x)
    CM.new_operator(name, 'one-body', elem)

new_tb(1, [1, 1, 1, 1])
new_tb(2, [1, 1, 1, 1])
new_tb(3, [1, 1, -1, -1])
new_tb(4, [1, 1, -1, -1])
new_tb(5, [1, -1, 1, -1])
new_tb(6, [1, -1, 1, -1])
new_tb(7, [1, -1, -1, 1])
new_tb(8, [1, -1, -1, 1])

def new_pairing(x, seq):
    elem = []
    for i in range(4):
        elem.append((i+1, x+4+12, seq[i]))
    for i in range(4):
        elem.append((x+4, i+1+12, seq[i]))
    name = 'sb'+str(x)
    CM.new_operator(name, 'anomalous', elem)

new_pairing(1, [1, 1, 1, 1])
new_pairing(2, [1, 1, 1, 1])
new_pairing(3, [1, 1, -1, -1])
new_pairing(4, [1, 1, -1, -1])
new_pairing(5, [1, -1, 1, -1])
new_pairing(6, [1, -1, 1, -1])
new_pairing(7, [1, -1, -1, 1])
new_pairing(8, [1, -1, -1, 1])

