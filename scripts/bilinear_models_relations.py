# -*- coding: utf-8 -*-

import torch
import random

seed = 2019

random.seed(seed)
torch.manual_seed(seed)


def op_vec(u, v):
    return u.unsqueeze(-1).matmul(v.unsqueeze(-1).transpose(0, 1))


def op_colwise_mat(U, V):
    assert U.shape[1] == V.shape[1]
    output = torch.zeros(U.shape[1], V.shape[0], U.shape[0])
    for j in range(U.shape[1]):
        Uj, Vj = U[:, j], V[:, j]
        output[:, :, j] = op_vec(Uj, Vj)
    return output


def lowfer_ops(ei, rj, ek, U, V, k):
    elem_prod = ei.matmul(U) * rj.matmul(V)
    elem_prod = elem_prod.view(-1, k)
    pooled = elem_prod.sum(-1)
    f = pooled.dot(ek)
    return f


def de_hop_k_sized_slices(U, V):
    assert U.shape[1] == V.shape[1]
    de = U.shape[0]
    dr = V.shape[0]
    k = int(U.shape[1]/de)
    k_Ws = []
    for i in range(1, de + 1):
        W_U_i = torch.zeros(de, de)
        W_V_i = torch.zeros(dr, de)
        for j in range(1, de + 1):
            l = ((j - 1) * k) + i
            W_U_i[:, j-1] = U[:, l-1]
            W_V_i[:, j-1] = V[:, l-1]
        k_Ws.append((W_U_i, W_V_i))
    return k_Ws


class LowFER:
    
    def core_apprx(self):
        # k-sized list, where each element is a tuple of
        # (W_U^i, W_V^i) such that W_U^i is (de x de) and
        # W_V^i is (de x dr)
        k_Ws = de_hop_k_sized_slices(self.U, self.V)
        self.k_Ws = k_Ws
        self.cores = []
        for i in range(self.k):
            W_U_i, W_V_i = k_Ws[i]
            # taking column-wise outer product and concatenating
            # resulting rank-1 2D matrices in (de x dr)
            core_i = op_colwise_mat(W_U_i, W_V_i)
            self.cores.append(core_i)
        self.core = torch.zeros(self.cores[0].shape).float()
        for c in self.cores:
            self.core += c
    
    def lowfer_score(self, es, rel_num, eo, from_core_apprx=False):
        if from_core_apprx:
            return es.matmul(self.core[:, rel_num, :]).dot(eo)
        else:
            return lowfer_ops(es, self.R[rel_num], eo, self.U, self.V, self.k)


class LowFER2RESCAL(LowFER):
    """
    Requirements: k = de, dr = nr, R = I_nr, U = [I_de | I_de | .. | I_de] in R^{de x de^2}
    
    """
    def __init__(self):
        # toy setup
        nr = 3
        de = 4
        self.dr = nr
        self.nr = nr
        self.de = de
        self.k = de
        # U will be as such:
        self.U = torch.tensor([
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]
        ]).float()
        self.V = torch.randn(self.dr, self.k * self.de).float()
        self.R = torch.eye(self.nr)
    
    def rescal_score(self, es, rel_num, eo):
        Wl = self.V[rel_num].view(self.de, self.de).transpose(0, 1)
        return es.matmul(Wl).dot(eo)


class LowFER2DistMult(LowFER):
    """
    Requirements: k = 1, dr = nr, U = I_de, R = I_nr
    
    """
    def __init__(self):
        # toy setup
        nr = 3
        de = 4
        self.dr = nr
        self.nr = nr
        self.de = de
        self.k = 1
        # U will be as such:
        self.U = torch.eye(de)
        self.V = torch.randn(self.dr, self.k * self.de).float()
        self.R = torch.eye(self.nr)
    
    def distmul_score(self, es, rel_num, eo):
        return (es * self.V[rel_num] * eo).sum()


class LowFER2SimplE(LowFER):
    """
    Requirements: k = 1, dr = nr, de = 2d, 
                       _            _
                      |       |      |
                      |  U_11 | U_12 |
                  U = | -------------| in R^{2d x 2d} / R^{de x de}
                      |  U_21 | U_22 |
                      |_      |     _| such that, U_11=U_22=0 and U_12=U_21=(1/2)*I_d
    
    """
    def __init__(self):
        # toy setup
        nr = 3
        d = 2
        self.dr = nr
        self.nr = nr
        self.de = 2 * d
        self.k = 1
        # U will be as such:
        self.P = torch.tensor([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ]).float()
        self.U = 0.5 * self.P
        self.V = torch.randn(self.dr, self.k * self.de).float()
        self.R = torch.eye(self.nr)
    
    def cp(self, x, y, z):
        return x.matmul(torch.diag(y)).dot(z)
    
    def simple_score(self, es, rel_num, eo):
        d = int(len(es) / 2)
        h_es, t_es = es[:d], es[d:]
        h_eo, t_eo = eo[:d], eo[d:]
        rl = self.V[rel_num] # note: first half is inv and second normal
        r_inv, r = rl[:d], rl[d:]
        return 0.5 * (self.cp(h_es, r, t_eo) + self.cp(h_eo, r_inv, t_es))
    
    def simple_as_bilinear(self, es, rel_num, eo):
        d = int(len(es) / 2)
        h_es, t_es = es[:d], es[d:]
        es_hat = torch.cat([t_es, h_es])
        rl = self.V[rel_num] # note: first half is inv and second normal
        r_inv, r = rl[:d], rl[d:]
        rl_hat = torch.cat([r_inv, r])
        return 0.5 * (es_hat * rl_hat * eo).sum()



class LowFER2ComplEx(LowFER):
    """
    Requirements: k = 2, dr = nr, de = 2d
    W_U11^(1) = W_U12^(1) = Id ; W_U21^(2) = -Id, W_U22^(2) = Id
    
    """
    def __init__(self):
        # toy setup
        nr = 3
        d = 2
        self.dr = nr
        self.nr = nr
        self.de = 2 * d
        self.k = 2
        # Us will be as such:
        self.W_U_1 = torch.tensor([
            [1,  0,  1,  0],
            [0,  1,  0,  1],
            [0,  0,  0,  0],
            [0,  0,  0,  0]
        ]).float()
        self.W_U_2 = torch.tensor([
            [0,   0,  0,  0],
            [0,   0,  0,  0],
            [-1,  0,  1,  0],
            [0,  -1,  0,  1]
        ]).float()
        
        self.W_V_1 = torch.randn(self.dr, self.de)
        P = torch.tensor([
            [0,  0,  1,  0],
            [0,  0,  0,  1],
            [1,  0,  0,  0],
            [0,  1,  0,  0]
        ]).float()
        # P is permutation matrix, swapping first half with second
        self.W_V_2 = self.W_V_1.matmul(P)
        self.R = torch.eye(self.nr)
        self.core_apprx()
    
    def core_apprx(self):
        k_Ws = [(self.W_U_1, self.W_V_1), (self.W_U_2, self.W_V_2)]
        self.cores = []
        for i in range(self.k):
            W_U_i, W_V_i = k_Ws[i]
            core_i = op_colwise_mat(W_U_i, W_V_i)
            self.cores.append(core_i)
        self.core = torch.zeros(self.cores[0].shape).float()
        for c in self.cores:
            self.core += c
    
    def complex_score(self, es, rel_num, eo):
        d = int(len(es) / 2)
        Re_es, Im_es = es[:d], es[d:]
        Re_eo, Im_eo = eo[:d], eo[d:]
        wr = self.W_V_1[rel_num]
        Re_wr, Im_wr = wr[:d], wr[d:]
        score = (Re_es * Re_eo * Re_wr).sum() + \
                (Re_wr * Im_es * Im_eo).sum() + \
                (Im_wr * Re_es * Im_eo).sum() - \
                (Im_wr * Im_es * Re_eo).sum()
        return score
    
    def complex_as_bilinear(self, es, rel_num, eo):
        d = int(len(es) / 2)
        Re_es, Im_es = es[:d], es[d:]
        Re_eo, Im_eo = eo[:d], eo[d:]
        wr = self.W_V_1[rel_num]
        Re_wr, Im_wr = wr[:d], wr[d:]
        Wl = torch.diag(torch.cat([Re_wr, Re_wr]))
        Wl[:d, d:] = torch.diag(Im_wr)
        Wl[d:, :d] = -torch.diag(Im_wr)
        score = es.matmul(Wl).dot(eo)
        return score


def check_rescal_lowfer_equivalence():
    l2r = LowFER2RESCAL()
    E = torch.randn(10, l2r.de)
    es, eo = E[1], E[5]
    rel_num = 1
    s1 = l2r.rescal_score(es, rel_num, eo).item()
    s2 = l2r.lowfer_score(es, rel_num, eo).item()
    print("RESCAL =", s1)
    print("LowFER_as_RESCAL =", s2)


def check_distmult_lowfer_equivalence():
    l2d = LowFER2DistMult()
    E = torch.randn(10, l2d.de)
    es, eo = E[1], E[5]
    rel_num = 1
    s1 = l2d.distmul_score(es, rel_num, eo).item()
    s2 = l2d.lowfer_score(es, rel_num, eo).item()
    print("DistMult =", s1)
    print("LowFER_as_DistMult =", s2)


def check_simple_lowfer_equivalence():
    l2s = LowFER2SimplE()
    E = torch.randn(10, l2s.de)
    es, eo = E[1], E[5]
    rel_num = 1
    s1 = l2s.simple_score(es, rel_num, eo).item()
    s2 = l2s.lowfer_score(es, rel_num, eo).item()
    print("SimplE =", s1)
    print("LowFER_as_SimplE =", s2)


def check_complex_lowfer_equivalence():
    l2c = LowFER2ComplEx()
    E = torch.randn(10, l2c.de)
    es, eo = E[1], E[5]
    rel_num = 1
    s1 = l2c.complex_score(es, rel_num, eo).item()
    s2 = l2c.lowfer_score(es, rel_num, eo, True).item()
    print("ComplEx =", s1)
    print("LowFER_as_ComplEx =", s2)


if __name__=="__main__":
    print("========== (  RESCAL  ) ============")
    check_rescal_lowfer_equivalence()
    print("\n========== ( DistMult ) ============")
    check_distmult_lowfer_equivalence()
    print("\n========== (  SimplE  ) ============")
    check_simple_lowfer_equivalence()
    print("\n========== (  ComplEx ) ============")
    check_complex_lowfer_equivalence()
