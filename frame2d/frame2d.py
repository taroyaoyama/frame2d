import numpy as np
import pandas as pd
from . import mdofpy


class Node2d:

    def __init__(self, x, z, id, fixed):

        self.x = x
        self.z = z
        self.id = id
        self.fixed = fixed


class Beam2d:

    def __init__(self, E, G, m, A, I, i = None, j = None, k = 1, pin = 0, add_mass = 0):

        self.E = E
        self.G = G
        self.m = m
        self.A = A
        self.I = I
        if i is not None:
            self.i = int(i)
        if j is not None:
            self.j = int(j)
        self.k = k
        self.L = None
        self.R = None
        self.Hi = None
        self.Hj = None
        self.mii = None
        self.mij = None
        self.mji = None
        self.mjj = None
        self.kii = None
        self.kij = None
        self.kji = None
        self.kjj = None
        self.pin = pin
        self.add_mass = add_mass  # [kg]
    
    def connect(self, node_i, node_j):

        self.node_i = node_i
        self.node_j = node_j
    
    def make(self, shear = False):

        self.L = np.sqrt((self.node_i.x - self.node_j.x)**2 + (self.node_i.z - self.node_j.z)**2)

        sina = (self.node_j.z - self.node_i.z) / self.L
        cosa = (self.node_j.x - self.node_i.x) / self.L
        self.R = np.array([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])

        mal = self.m * self.A * self.L + self.add_mass

        self.mii = np.diag([mal / 2, mal / 2, 0])
        self.mjj = np.diag([mal / 2, mal / 2, 0])
        self.mij = np.zeros((3, 3))
        self.mji = np.zeros((3, 3))

        EA = self.E * self.A
        EI = self.E * self.I

        if shear:
            gm = 6 * EI * self.k / (self.G * self.A * self.L**2)
        else:
            gm = 0
        
        a1 = 1 / (1 + 2 * gm)
        a2 = (1 + gm / 2) / (1 + 2 * gm)
        a3 = 1 / (1 + gm / 2)

        self.Hi = np.array([[-1, 0, 0], [0, -1, 0], [0, -self.L, -1]])
        self.Hj = np.diag([1, 1, 1])
        Hs = np.r_[self.Hi, self.Hj]

        if self.pin == 0:
            Ke = np.array([
                [EA / self.L, 0, 0],
                [0, a1 * 12 * EI / self.L**3, -a1 * 6 * EI / self.L**2],
                [0, -a1 * 6 * EI / self.L**2, a2 * 4 * EI / self.L]
            ])
        if self.pin == 1:
            Ke = np.array([
                [EA / self.L, 0, 0],
                [0, a3 * 3 * EI / self.L**3, -a3 * 3 * EI / self.L**2],
                [0, -a3 * 3 * EI / self.L**2, a3 * 3 * EI / self.L]
            ])
        if self.pin == 2:
            Ke = np.array([
                [EA / self.L, 0, 0],
                [0, a3 * 3 * EI / self.L**3, 0],
                [0, 0, 0]
            ])
        if self.pin == 3:
            Ke = np.array([
                [EA / self.L, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ])
        
        K = Hs @ Ke @ Hs.T
        self.kii = K[:3,:3]
        self.kij = K[:3,3:]
        self.kji = K[3:,:3]
        self.kjj = K[3:,3:]


class Frame2d:

    def __init__(self):

        self.nodes = []
        self.membs = []
        self.n_node = None
        self.n_memb = None
        self.M = None
        self.C = None
        self.K = None
        self.M_red = None
        self.K_red = None
        self.omeg = None
        self.U = None
    
    def add_node(self, node):
        
        self.nodes.append(node)
    
    def add_memb(self, memb):

        node_i = list(filter(lambda x: x.id == memb.i, self.nodes))[0]
        node_j = list(filter(lambda x: x.id == memb.j, self.nodes))[0]
        memb.connect(node_i, node_j)
        self.membs.append(memb)
    
    def make_MK(self, nodes, membs, shear = False):

        # read nodes & membs
        self.nodes = []
        self.membs = []

        for i in range(len(nodes)):
            self.add_node(nodes[i])
        
        for i in range(len(membs)):
            self.add_memb(membs[i])
            self.membs[i].make(shear)
        
        # make M, K
        self.n_node = len(nodes)
        self.n_memb = len(membs)
        self.M = np.zeros((3 * self.n_node, 3 * self.n_node))
        self.K = np.zeros((3 * self.n_node, 3 * self.n_node))

        for k in range(self.n_memb):

            istt, iend = 3 * (membs[k].i - 1), 3 * (membs[k].i - 1) + 3
            jstt, jend = 3 * (membs[k].j - 1), 3 * (membs[k].j - 1) + 3

            self.M[istt:iend, istt:iend] += membs[k].mii
            self.M[jstt:jend, jstt:jend] += membs[k].mjj

            self.K[istt:iend, istt:iend] += membs[k].R @ membs[k].kii @ membs[k].R.T
            self.K[istt:iend, jstt:jend] += membs[k].R @ membs[k].kij @ membs[k].R.T
            self.K[jstt:jend, istt:iend] += membs[k].R @ membs[k].kji @ membs[k].R.T
            self.K[jstt:jend, jstt:jend] += membs[k].R @ membs[k].kjj @ membs[k].R.T
        
        # reflect support conditions
        for i in range(self.n_node):

            if self.nodes[i].fixed == 1:  # fixed support
                istt, iend = 3 * i, 3 * i + 3
                self.M[istt:iend,:] = 0
                self.M[:,istt:iend] = 0
                self.K[istt:iend,:] = 0
                self.K[:,istt:iend] = 0
                self.K[istt:iend, istt:iend] = np.diag([1, 1, 1])

            if self.nodes[i].fixed == 2:  # pin support
                istt, iend = 3 * i, 3 * i + 2
                self.M[istt:iend,:] = 0
                self.M[:,istt:iend] = 0
                self.K[istt:iend,:] = 0
                self.K[:,istt:iend] = 0
                self.K[istt:iend, istt:iend] = np.diag([1, 1])

            if self.nodes[i].fixed == 3:  # roller support
                istt = 3 * i + 1
                self.M[istt,:] = 0
                self.M[:,istt] = 0
                self.K[istt,:] = 0
                self.K[:,istt] = 0
                self.K[istt, istt] = 1
        
        # reduced matrices
        dm = np.diag(self.M)
        self.M_red = self.M[dm != 0,:][:,dm != 0]

        K11 = self.K[dm != 0,:][:,dm != 0]
        K12 = self.K[dm != 0,:][:,dm == 0]
        K21 = self.K[dm == 0,:][:,dm != 0]
        K22 = self.K[dm == 0,:][:,dm == 0]
        self.K_red = K11 - K12 @ np.linalg.inv(K22) @ K21
    
    def moda(self):

        eig, vec = np.linalg.eig(np.linalg.inv(self.M_red) @ self.K_red)
        self.omeg = np.sqrt(eig)
        for i in range(vec.shape[1]):
            vec[:,i] /= max(abs(vec[:,i]))
        self.U = vec
    
    def make_C(self, zeta):

        if self.omeg is None:
            self.moda()
        
        self.C = 2 * zeta / min(self.omeg) * self.K
    
    def build(self, nodes, membs, zeta, shear = False):

        self.make_MK(nodes, membs, shear)
        self.make_C(zeta)
    
    def to_mdof(self):

        return mdofpy.Mdof(self.M, self.C, self.K, self.n_node)


def df2nodes(df_node):
    nodes = []
    for i in range(df_node.shape[0]):
        nodi = df_node.iloc[i,:]
        nodi = Node2d(nodi['x'], nodi['z'], nodi['id'], nodi['fixed'])
        nodes.append(nodi)
    return nodes


def df2membs(df_memb):
    membs = []
    for i in range(df_memb.shape[0]):
        memi = df_memb.iloc[i,:]
        memi = Beam2d(memi['E'], memi['G'], memi['m'], memi['A'], memi['I'], memi['i'], memi['j'], memi['k'], memi['pin'], memi['addmass'])
        membs.append(memi)
    return membs