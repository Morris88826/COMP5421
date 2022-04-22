from matplotlib.pyplot import axis
import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def homogeneous_transform(point, H):
    '''
    INPUT
        point - Nx2,
        H - 3x3
    OUTPUT 
        t_point - Nx2
    '''
    h_point = np.concatenate([point, np.ones((point.shape[0], 1))], axis=-1)
    transformed_h_point = np.matmul(H, h_point.T).T # in shape Nx3
    transformed_point = (transformed_h_point/np.expand_dims(transformed_h_point[:, -1], axis=-1))[:, :2]
    return transformed_point


    


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...

    out_width = im1.shape[1] + im2.shape[1]

    im1_corners = np.array([[0,0],[im1.shape[1],0],[0, im1.shape[0]],[im1.shape[1], im1.shape[0]]]) # 4x2
    im2_corners = np.array([[0,0],[im2.shape[1],0],[0, im2.shape[0]],[im2.shape[1], im2.shape[0]]]) # 4x2
    new_im2_corners = homogeneous_transform(im2_corners, H2to1)

    corners = np.concatenate([im1_corners, new_im2_corners], axis=0)

    top_left = np.amin(corners, axis=0)
    bottom_right = np.amax(corners, axis=0)
    width, height = list(bottom_right - top_left)
    aspect_ratio = height/width
    out_height = int(out_width*aspect_ratio)

    t = np.array([[1,0,-top_left[0]],[0,1,-top_left[1]],[0,0,1]], dtype=np.float64)
    s = np.array([[out_width/width,0,0],[0,out_height/height,0],[0,0,1]], dtype=np.float64)
    M = np.matmul(s, t)

    wrap_im1 = cv2.warpPerspective(im1, M, (out_width, out_height))
    wrap_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), (out_width, out_height))

    pano_im = np.where(wrap_im1!=0, wrap_im1, wrap_im2)

    # crop
    residue_im = pano_im - wrap_im1
    t_im1_corners = homogeneous_transform(im1_corners, M)
    t_im2_corners = homogeneous_transform(im2_corners, np.matmul(M, H2to1))

    residue_im2_left = np.argwhere(np.sum(distance_transform_edt(residue_im[:, int(t_im1_corners[1,0])+2]), axis=-1)!=0).flatten()
    top_row_point = int(np.amax(np.array([t_im1_corners[1,1],t_im2_corners[1,1], np.amin(residue_im2_left)])))
    bottom_row_point = int(np.amin(np.array([t_im1_corners[3,1],t_im2_corners[3,1],np.amax(residue_im2_left)])))

    left_point = int(np.amin(t_im2_corners[[1,3],0]))

    pano_im = pano_im[top_row_point:bottom_row_point, :left_point]
    
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    
    out_width = im1.shape[1] + im2.shape[1]

    im1_corners = np.array([[0,0],[im1.shape[1], 0],[0, im1.shape[0]],[im1.shape[1], im1.shape[0]]])
    im2_corners = np.array([[0, im2.shape[1], 0, im2.shape[1]],[0, 0, im2.shape[0], im2.shape[0]],[1,1,1,1]])
    new_im2_corners = np.matmul(H2to1, im2_corners)
    new_im2_corners = (new_im2_corners/new_im2_corners[-1])[:2].T

    corners = np.array(list(im1_corners) + list(new_im2_corners))

    top_left = np.amin(corners, axis=0)
    bottom_right = np.amax(corners, axis=0)
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    aspect_ratio = height/width

    out_height = int(out_width*aspect_ratio)

    t = np.array([[1,0,-top_left[0]],[0,1,-top_left[1]],[0,0,1]], dtype=np.float64)
    s = np.array([[out_width/width,0,0],[0,out_height/height,0],[0,0,1]], dtype=np.float64)
    M = np.matmul(s, t)

    wrap_im1 = cv2.warpPerspective(im1, M, (out_width, out_height))
    wrap_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), (out_width, out_height))

    pano_im = np.where(wrap_im1!=0, wrap_im1, wrap_im2)


    return pano_im


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)

    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    print(H2to1)
    cv2.imwrite('../results/panoImg_noClip.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pano_im = imageStitching(im1, im2, H2to1)
    cv2.imwrite('../results/panoImg.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()