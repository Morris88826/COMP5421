from submission import triangulate, eightpoint, essentialMatrix
from helper import camera2
import numpy as np
'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
def findM2(pts1, pts2, K1, K2, M, save=True):
    F = eightpoint(pts1, pts2, M)
    E = essentialMatrix(F, K1, K2)

    M1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    C1 = np.matmul(K1, M1)
    M2s = camera2(E)
    # There are four possible M2, the correct one should have all P in front of both cameras(i.e. z>0)
    for i in range(M2s.shape[-1]):
        M2 = M2s[:, :, i]
        C2 = np.matmul(K2, M2)
        P, _ = triangulate(C1, pts1, C2, pts2)
        if np.sum(np.where(P[:, -1]<0, 1, 0)) == 0:
            if save:
                np.savez('../results/q3_3.npz', M2=M2, C2=C2, P=P)
                return
            else:
                return M1, M2, C1, C2, F
    error('The implementation seems to have some problems.')
    return
