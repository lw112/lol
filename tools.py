import os
import numpy as np


def frame_path_to_idx(frame_path):
    return int(os.path.basename(frame_path)[6:-4])


def done():
    try:
        os.system('afplay /System/Library/Sounds/Glass.aiff')
    except:
        pass


def shape_to_np(shape):
    coords = np.zeros((shape.num_parts, 2), dtype=int)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
