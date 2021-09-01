from matplotlib import axes
import numpy as np
from numpy.lib import expand_dims
import tqdm

def SSD(u, v):
  # Sum of Squared Differences 
  return np.sum(np.power((u-v),2))

def NCC(u, v):
  # Normalized Cross Correlation
  u = u/np.linalg.norm(u)
  v = v/np.linalg.norm(v)
  return np.matmul(u.T, v)



def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""

    # align to red
    H, W = red.shape
    
    displacement = 30
    score_1 = np.zeros((displacement*2, displacement*2))
    score_2 =np.zeros((displacement*2, displacement*2))
    aug_green = np.zeros((H+displacement*2, W+displacement*2))
    aug_blue = np.zeros((H+displacement*2, W+displacement*2))

    aug_green[displacement:-displacement, displacement:-displacement] = green
    aug_blue[displacement:-displacement, displacement:-displacement] = blue

    for i in tqdm.tqdm(range(displacement*2)):
      for j in range(displacement*2):
        # align green to red
        _green = aug_green[i:i+H, j:j+W]
        _green = _green.flatten().reshape((-1, 1))
        score_1[i, j] = SSD(red.flatten().reshape((-1, 1)), _green)

        # # align blue to red
        _blue = aug_blue[i:i+H, j:j+W]
        _blue = _blue.flatten().reshape((-1, 1))
        score_2[i, j] = SSD(red.flatten().reshape((-1, 1)), _blue)

    start_H1, start_W1 = list(np.unravel_index(score_1.argmin(), score_1.shape))
    start_H2, start_W2 = list(np.unravel_index(score_2.argmin(), score_2.shape))

    result = np.concatenate([np.expand_dims(red, -1), np.expand_dims(aug_green[start_H1:start_H1+H, start_W1:start_W1+W], -1), \
      np.expand_dims(aug_blue[start_H2:start_H2+H, start_W2:start_W2+W], -1)], -1)
    
    result = result.astype(np.uint8)
    return result

