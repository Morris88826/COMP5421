import imageio
import matplotlib.pyplot as plt
import numpy as np
import warpA_check
import warpA



# define some helper functions
# to create affine transformations
def scalef(s):
    return np.diag([s, s, 1])


def transf(tx, ty):
    A = np.eye(3)
    A[0, 2] = ty
    A[1, 2] = tx
    return A


def rotf(t):
    return np.array([[np.cos(t), np.sin(t), 0],
                     [-np.sin(t), np.cos(t), 0],
                     [0, 0, 1]])


def plot_result(im, im_gray, warped_im, out_name='transformed_soln'):
    # create figure
    f, axes = plt.subplots(2, 2)
    f.set_size_inches(8, 8)
    axes[0, 0].imshow(im)
    axes[0, 0].set_title('original')
    axes[0, 1].imshow(im_gray, cmap=plt.get_cmap('gray'))
    axes[0, 1].set_title('grayscale')
    axes[1, 1].remove()

    # plot a dot at the rotation center
    axes[0, 1].plot(cx, cy, 'r+')

    axes[1, 0].imshow(warped_im, cmap=plt.get_cmap('gray'))
    axes[1, 0].set_title('warped')

    # write the plot to an image
    plt.savefig('../results/{}.jpg'.format(out_name))
    # plt.show()


if __name__ == '__main__':
    # Read the image
    im = imageio.imread('../data/mug.jpg')
    im = im / 255.0  # convert to float

    # convert to grayscale
    im_gray = np.dot(im, [0.299, 0.587, 0.114])

    output_shape = im_gray.shape
    cx = im_gray.shape[1] // 2
    cy = im_gray.shape[0] // 2
    A = (transf(output_shape[1]//2, output_shape[0]//2,)
        .dot(scalef(0.8))
        .dot(rotf(- 30 * np.pi / 180))
        .dot(transf(-cx, -cy)))


    warped_im_gt = warpA_check.warp(im_gray, A, output_shape)
    plot_result(im, im_gray, warped_im_gt, out_name='transformed_soln')

    warped_im = warpA.warp(im_gray, A, output_shape)
    plot_result(im, im_gray, warped_im, out_name='transformed')