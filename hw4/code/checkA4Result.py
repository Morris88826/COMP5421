import submission as sub
import numpy as np
import matplotlib.pyplot as plt
from helper import displayEpipolarF, epipolarMatchGUI
from findM2 import findM2

def check_eightpoint():
    data = np.load('../data/some_corresp.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    M = 640

    F = sub.eightpoint(data['pts1'], data['pts2'], M)
    np.savez('../results/q2_1.npz', F=F, M=M)
    displayEpipolarF(im1, im2, F, out="../results/q2_1.png")

def check_sevenpoint():
    data = np.load('../data/some_corresp.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    M = 640

    num_samples = data['pts1'].shape[0]
    selected_idx = np.random.choice(num_samples, 7, replace=False)
    selected_pts1 = data['pts1'][selected_idx]
    selected_pts2 = data['pts2'][selected_idx]

    Farray = sub.sevenpoint(selected_pts1, selected_pts2, M)


    np.savez('../results/q2_2.npz', Farray=Farray, M=M)

    for i, F in enumerate(Farray):
        print('Test on F{}'.format(i+1))
        displayEpipolarF(im1, im2, F, out="../results/q2_2_F{}.png".format(i+1))

def check_triangulate():
    data = np.load('../data/some_corresp.npz')
    M = 640
    intrinsic = np.load('../data/intrinsics.npz')
    K1 = intrinsic['K1']
    K2 = intrinsic['K2']

    findM2(data['pts1'], data['pts2'], K1, K2, M)

def check_epipolarCorrespondence():
    data = np.load('../data/some_corresp.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = 640

    F = sub.eightpoint(data['pts1'], data['pts2'], M)
    
    epipolarMatchGUI(im1, im2, F)

def check_ransac():
    data = np.load('../data/some_corresp_noisy.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    M = 640
    F, _ = sub.ransacF(data['pts1'], data['pts2'], M)
    

    displayEpipolarF(im1, im2, F)

def check_bundleAdjustment():
    data = np.load('../data/some_corresp_noisy.npz')

    intrinsic = np.load('../data/intrinsics.npz')
    K1 = intrinsic['K1']
    K2 = intrinsic['K2']

    M = 640
    _, inliers = sub.ransacF(data['pts1'], data['pts2'], M)

    inlier_indices = np.argwhere(inliers == 1).flatten()
    selected_pts1 = data['pts1'][inlier_indices]
    selected_pts2 = data['pts2'][inlier_indices]

    M1, M2_init, C1, C2, F = findM2(selected_pts1, selected_pts2, K1, K2, M, save=False)

    P_init, _ = sub.triangulate(C1, selected_pts1, C2, selected_pts2)
    M2, P2 = sub.bundleAdjustment(K1, M1, selected_pts1, K2, M2_init, selected_pts2, P_init)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=P2[:,0], ys=P2[:,1], zs=P2[:,2])
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    plt.show()

if __name__ == "__main__":
    check_bundleAdjustment()