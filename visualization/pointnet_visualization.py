import numpy as np


def make_one_critical(hx, num_critical_point=1024):
    cs_index = np.argmax(hx, axis=1)
    num_point = hx.shape[0]
    mask_critical = np.zeros((num_point, 1))
    assert len(set(cs_index)) <= num_critical_point
    for index in cs_index:
        mask_critical[index] = [1.]
    return mask_critical


def get_critical(points, hx):
    """
    :param points: (batch_size, numpoint, 1024)
    :param hx: (batch_size, numpoint, 1024)
    :return: (batch_size, mask)
    """
    sample_num = points.shape[0]
    num_point = points.shape[1]
    # hx = hx.reshape(sample_num, num_point, 1024)  # (num_sample, num_point, 1024)

    argmax_index = np.argmax(hx, axis=2)  # find which point contributed to max-pooling features
    pc_mask = np.zeros((sample_num, num_point, 1))

    for idx, mask in enumerate(pc_mask):
        for index in argmax_index[idx]:
            mask[index] = [1.]

    return pc_mask
