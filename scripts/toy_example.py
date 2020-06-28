# -*- coding: utf-8 -*-

import torch


def one_hot(i, d):
    v = torch.zeros(d)
    v[i] = 1
    v = v.unsqueeze(-1).float()
    return v


T = [
    (1, 1, 1),                       # r1 is reflexive 
    (1, 2, 2), (2, 2, 1),            # r2 is symmetric
    (3, 3, 2),                       # r3 is anti-symmetric
    (4, 4, 3), (3, 4, 1), (4, 4, 1),  # r4 is transitive
]


es = set([ei for ei, rj, ek in T] + [ek for ei, rj, ek in T])
ne = len(es)

rs = set([rj for ei, rj, ek in T])
nr = len(rs)

# create one-hot vectors
E = {ei : one_hot(ei-1, ne) for ei in es} 
R = {rj : one_hot(rj-1, nr) for rj in rs}


def lowfer_ops(ei, rj, ek, U, V, k):
    lhs = U.transpose(0, 1).matmul(E[ei])
    rhs = V.transpose(0, 1).matmul(R[rj])
    elem_prod = lhs * rhs
    elem_prod = elem_prod.reshape(-1, k)
    pooled = elem_prod.sum(-1)
    f = pooled.flatten().dot(E[ek].flatten())
    return f


de = ne
dr = nr

case = 1 # when k=de, use 2 for k=dr


if case == 1:
    k = ne
    U = torch.zeros(de, k * de).float()
    V = torch.zeros(dr, k * de).float()
    # set U values
    for m in range(1, de + 1):
        for o in range(1, k + 1):
            n = m + ((o-1)*de)
            U[m-1, n-1] = 1
    for i, j, l in T:
        p = j
        q = ((l-1)*de) + i
        V[p-1, q-1] = 1
    
    print("U and V for case k = de")
    print()
    print(U)
    print()
    print(V)

else:
    k = ne
    U = torch.zeros(de, k * de).float()
    V = torch.zeros(dr, k * de).float()
    # set U values
    for i, j, l in T:
        m = i
        n = ((l-1)*de) + j
        U[m-1, n-1] = 1
    # set V values
    for p in range(1, dr + 1):
        for o in range(1, k + 1):
            q = p + ((o-1)*de)
            V[p-1, q-1] = 1
    
    print("U and V for case k = dr")
    print()
    print(U)
    print()
    print(V)


for ei in es:
    for rj in rs:
        for ek in es:
            f = lowfer_ops(ei, rj, ek, U, V, k)
            if f > 0.:
                # there should not be any false positives!
                if (ei, rj, ek) not in T:
                    print("Incorrect! (%d, %d, %d)" % (ei, rj, ek))
                if (ei, rj, ek) in T:
                    print("Correct! (%d, %d, %d)" % (ei, rj, ek))
