import numpy as np
import cv2
import random
from BRIEF import briefLite, briefMatch, plotMatches

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################

    A = []
    for i in range(p1.shape[1]):
        x1, y1 = list(p1[:, i])
        x2, y2 = list(p2[:, i])
        eq1 = [x2, y2, 1, 0, 0, 0, -x2*x1, -y2*x1, -x1]
        eq2 = [0, 0, 0, x2, y2, 1, -x2*y1, -y2*y1, -y1]
        A.append(eq1)
        A.append(eq2)
    A = np.array(A)      

    _, sigma, VT = np.linalg.svd(A) 

    H2to1 = VT[-1].reshape((3,3))
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2, return_best_matches=False):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    num_matches = matches.shape[0]

    max_inliers = 0
    bestH = None
    best_matches = None
    for i in range(num_iter):
        N = random.randint(4, num_matches)
        choices = np.random.choice(num_matches, N, replace=False)
        _locs1 = locs1[matches[choices][:, 0]][:, :2].T
        _locs2 = locs2[matches[choices][:, 1]][:, :2].T

        H2to1 = computeH(_locs1, _locs2)
        
        h_locs1 = np.ones((3, num_matches))
        h_locs2 = np.ones((3, num_matches))
        h_locs1[:2, :] = locs1[matches[:, 0]][:, :2].T
        h_locs2[:2, :] = locs2[matches[:, 1]][:, :2].T



        t_locs2 = np.matmul(H2to1, h_locs2)
        t_locs2 = (t_locs2 / t_locs2[-1])

        d = np.sqrt(np.sum(np.square(t_locs2[:2] - locs1[matches[:, 0]][:, :2].T), axis=0))


        inliers = np.sum(np.where(d < tol, 1, 0))

        if inliers > max_inliers:
            max_inliers = inliers
            bestH = H2to1
            best_matches = matches[d < tol]

    # print(max_inliers)
    if return_best_matches:
        return bestH, best_matches
    else:
        return bestH
        


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1, im2, matches, locs1, locs2)
    bestH, best_matches = ransacH(matches, locs1, locs2, num_iter=5000, tol=2, return_best_matches=True)
    plotMatches(im1, im2, best_matches, locs1, locs2)