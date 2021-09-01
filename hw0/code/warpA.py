import numpy as np
def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""


    h, w = im.shape
    hv, wv = np.meshgrid(np.arange(h), np.arange(w))
    ones = np.ones_like(hv)

    H = np.concatenate([np.expand_dims(hv, -1), np.expand_dims(wv, -1), np.expand_dims(ones, -1)], -1)
    H = np.reshape(H, (-1, 3))
    H_ = (np.matmul(A, H.T).T)
    
    new_coord = np.zeros((H_.shape[0], 2)).astype(np.int)
    new_coord[:, 0] = np.divide(H_[:, 0], H_[:, -1], out=np.zeros_like(H_[:,-1]), where=H_[:,-1]!=0)
    new_coord[:, 1] = np.divide(H_[:, 1], H_[:, -1], out=np.zeros_like(H_[:,-1]), where=H_[:,-1]!=0)
    

    mask1 = (new_coord[:, 0] >= 0) * (new_coord[:,0]<h)
    mask2 = (new_coord[:, 1] >= 0) * (new_coord[:,1]<w)
    mask = mask1 * mask2
    
    valid_new_coord = new_coord[mask]
    valid_ori_coord = H[mask][:, :-1]

    output = np.zeros(output_shape)
    output[valid_new_coord[:, 0], valid_new_coord[:, 1]] = im[valid_ori_coord[:, 0], valid_ori_coord[:, 1]]

    return output
