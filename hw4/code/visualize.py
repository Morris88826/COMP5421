import numpy as np
import submission as sub
import matplotlib.pyplot as plt
from findM2 import findM2
'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
def visualize():
    data = np.load('../data/some_corresp.npz')
    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    M = 640
    intrinsic = np.load('../data/intrinsics.npz')
    K1 = intrinsic['K1']
    K2 = intrinsic['K2']

    M1, M2, C1, C2, F = findM2(data['pts1'], data['pts2'], K1, K2, M, save=False)
    np.savez('../results/q4_2.npz', M1=M1, M2=M2, C1=C1, C2=C2, F=F)

    templeCoords = np.load('../data/templeCoords.npz')
    coords = {
        'pts1': [],
        'pts2': []
    }
    for i in range(len(templeCoords['x1'])):
        x1 = templeCoords['x1'][i].item()
        y1 = templeCoords['y1'][i].item()
        x2, y2 = sub.epipolarCorrespondence(im1, im2, F, x1, y1)

        coords['pts1'].append([x1,y1])
        coords['pts2'].append([x2,y2])
    coords['pts1'] = np.array(coords['pts1'])
    coords['pts2'] = np.array(coords['pts2'])

    
    P, _ = sub.triangulate(C1, coords['pts1'], C2, coords['pts2'])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=P[:,0], ys=P[:,1], zs=P[:,2])
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    plt.show()
if __name__ == "__main__":
    visualize()