import numpy as np
import tensorflow as tf

def argument_cube(cube):
    # cube is a nparray with shape [height, width, channels]
    flip = np.random.choice([0,1,2,3])
    if flip == 0:
        result = cube[::-1,:,:]
    if flip == 1:
        result = cube[::-1,::-1,:]
    if flip == 2:
        result = cube[:,::-1,:]
    if flip == 3:
        result = cube
    rotation = np.random.choice([0,1])
    if rotation == 1:
        result = np.transpose(result, [1,0,2])
    if rotation == 0:
        result = result
    return result

