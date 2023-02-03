import numpy as np


class NewmarkBeta:

    def __init__(self, dlt, beta = 0.25, gamm = 0.5):

        self.a1 = 1 / beta / dlt
        self.a2 = 1 / 2 / beta
        self.a3 = gamm / beta
        self.a4 = (gamm / 2 / beta - 1) * dlt
        self.a5 = 1 / beta / dlt**2
        self.a6 = gamm / beta / dlt
        self.a7 = (1 - gamm / 2 / beta) * dlt
    
    def step(self, M, C, K, dis, vel, acc, dfrc):

        dis = dis.reshape((-1, 1))
        vel = vel.reshape((-1, 1))
        acc = acc.reshape((-1, 1))
        dfrc = dfrc.reshape((-1, 1))

        Ke = self.a5 * M + self.a6 * C + K

        ddis = np.linalg.inv(Ke) @ (M @ (self.a1 * vel + self.a2 * acc) + C @ (self.a3 * vel + self.a4 * acc) + dfrc)
        dvel = self.a6 * ddis - self.a3 * vel + self.a7 * acc
        dacc = self.a5 * ddis - self.a1 * vel - self.a2 * acc

        return ddis.flatten(), dvel.flatten(), dacc.flatten()


class Mdof:

    def __init__(self, M, C, K, n_node = None):

        self.M = M
        self.C = C
        self.K = K
        self.n_dof = M.shape[0]
        self.n_node = n_node
        self.dis = np.zeros(self.n_dof)
        self.vel = np.zeros(self.n_dof)
        self.acc = np.zeros(self.n_dof)
        self.frc = np.zeros(self.n_dof)
        self.ddis = np.zeros(self.n_dof)
    
    def newmark_init(self, dlt, init_iac = None, beta = 0.25, gamm = 0.50):

        self.newmark = NewmarkBeta(dlt, beta, gamm)

        if init_iac is not None:
            n_node = int(self.n_dof / len(init_iac))
            self.acc = np.repeat(-init_iac, n_node)
            self.frc = self.M @ np.repeat(init_iac, n_node).reshape((-1, 1))
    
    def newmark_step(self, dfrc):

        ddis, dvel, dacc = self.newmark.step(self.M, self.C, self.K, self.dis, self.vel, self.acc, dfrc)

        self.dis += ddis
        self.vel += dvel
        self.acc += dacc
        self.frc += dfrc
        self.ddis = ddis
    
    def newmark_step_acc(self, iac):

        if self.n_node is None:
            self.n_node = int(self.n_dof / len(iac))

        frc = -self.M @ np.tile(iac, self.n_node).reshape((-1, 1))
        dfrc = frc - self.frc

        self.newmark_step(dfrc)


class MdofRecorder:
    
    def __init__(self, n_dof, n_step, dlt):

        self.dis = np.zeros((n_step, n_dof))
        self.vel = np.zeros((n_step, n_dof))
        self.acc = np.zeros((n_step, n_dof))
        self.abc = np.zeros((n_step, n_dof))
        self.tim = np.arange(0, dlt * n_step, dlt)
        self.num = 0
        self.n_dof = n_dof
    
    def record(self, mdof):

        self.dis[self.num] = mdof.dis
        self.vel[self.num] = mdof.vel
        self.acc[self.num] = mdof.acc
        self.num += 1
    
    def add_iac(self, iac):

        n_node = int(self.n_dof / iac.shape[1])
        repiac = np.tile(iac, n_node)
        self.abc = self.acc + repiac


def excite_mdof_linear(mdof, dlt, iac):

    rc = MdofRecorder(mdof.n_dof, len(iac), dlt)
    mdof.newmark_init(dlt, iac[0])

    rc.record(mdof)

    for i in range(1, len(iac)):

        mdof.newmark_step_acc(iac[i])
        rc.record(mdof)
    
    rc.add_iac(iac)

    return rc