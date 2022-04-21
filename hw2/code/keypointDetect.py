from re import template
from telnetlib import DO
import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]

    for level in DoG_levels:
        DoG = gaussian_pyramid[:, :, level]-gaussian_pyramid[:, :, level-1]
        DoG_pyramid.append(np.expand_dims((DoG), axis=-1))

    DoG_pyramid = np.concatenate(DoG_pyramid, axis=-1)
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corresponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = None
    ##################
    # TO DO ...
    # Compute principal curvature here

    for level in range(DoG_pyramid.shape[-1]): 
        DoG = DoG_pyramid[:, :, level]
        Dx = cv2.Sobel(DoG,cv2.CV_64F,1,0,ksize=3)
        Dy = cv2.Sobel(DoG,cv2.CV_64F,0,1,ksize=3)

        Dxx = cv2.Sobel(Dx,cv2.CV_64F,1,0,ksize=3)
        Dxy = cv2.Sobel(Dx,cv2.CV_64F,0,1,ksize=3)
        Dyx = cv2.Sobel(Dy,cv2.CV_64F,1,0,ksize=3)
        Dyy = cv2.Sobel(Dy,cv2.CV_64F,0,1,ksize=3)
        
        tr_H = Dxx + Dyy
        det_H = Dxx * Dyy - Dxy * Dyx

        R = tr_H**2/(det_H+np.ones_like(det_H)*1e-6)
        R = np.expand_dims(R, -1)
        if principal_curvature is None:
            principal_curvature = R
        else:
            principal_curvature = np.concatenate([principal_curvature, R], axis=-1)

    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here

    local_maximum = np.zeros_like(DoG_pyramid)
    for level in DoG_levels:
        DoG = DoG_pyramid[:, :, level]
        neighbor_1 = np.zeros_like(DoG)
        neighbor_2 = np.zeros_like(DoG)
        neighbor_3 = np.zeros_like(DoG)
        neighbor_4 = np.zeros_like(DoG)
        neighbor_5 = np.zeros_like(DoG)
        neighbor_6 = np.zeros_like(DoG)
        neighbor_7 = np.zeros_like(DoG)
        neighbor_8 = np.zeros_like(DoG)

        neighbor_1[1:] = DoG[:-1]
        neighbor_2[:-1] = DoG[1:] 
        neighbor_3[:, 1:] = DoG[:, :-1]
        neighbor_4[:, :-1] = DoG[:, 1:]
        neighbor_5[1:, 1:] = DoG[:-1, :-1]
        neighbor_6[1:, :-1] = DoG[:-1, 1:]
        neighbor_7[-1:, 1:] = DoG[1, :-1]
        neighbor_8[-1:, -1] = DoG[:1, :1]

        neighbors = [neighbor_1, neighbor_2, neighbor_3, neighbor_4, neighbor_5, neighbor_6, neighbor_7, neighbor_8]

        if level != 0:
            neighbors.append(DoG_pyramid[:, :, level-1])

        if level != DoG_levels[-1]:
            neighbors.append(DoG_pyramid[:, :, level+1])

        neighbors = np.array(neighbors)
        local_maximum[:, :, level] = np.where(DoG > (np.amax(neighbors, axis=0)), 1, 0)


    satisfy_DoG = np.where(DoG_pyramid>th_contrast, 1, 0)
    satisfy_pricinpal_curve_ratio = np.where(principal_curvature>th_r, 0, 1)

    # uncomment this to get the result for without edge suppression
    # satisfy_pricinpal_curve_ratio = np.ones_like(satisfy_DoG)
    satisfied = satisfy_DoG*satisfy_pricinpal_curve_ratio*local_maximum

    # only get those that have eight neighbors in space and its two neighbors in scale
    locsDoG = np.argwhere(satisfied[:, :, 1:-1]==1)
    tmp = np.copy(locsDoG[:, 0])
    locsDoG[:, 0] =  locsDoG[:, 1]
    locsDoG[:, 1] = tmp 
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)


    return locsDoG, gauss_pyramid







if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    # displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    # displayPyramid(DoG_pyr)
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    # displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    for i in range(locsDoG.shape[0]):
        cv2.circle(im, (locsDoG[i, 0], locsDoG[i, 1]), 1, (0, 255, 0), -1)
    cv2.imshow('Pyramid of image', im)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()
