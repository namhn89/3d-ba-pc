import numpy as np


def shift_object(sh, center, num_point, scale):
    def normalize(data, scale):
        center = (np.max(data, axis=0) + np.min(data, axis=0)) / 2
        data = data - np.expand_dims(center, axis=0)
        norm = np.linalg.norm(data, axis=1)
        radius = np.max(norm)
        data = data / radius
        data = data * scale
        return data

    sh = normalize(sh, scale)
    if sh.shape[0] > num_point:
        np.random.shuffle(sh)
        sh = sh[:num_point]
    center = np.array(center)
    center = np.reshape(center, [1, 3])
    return sh + center
