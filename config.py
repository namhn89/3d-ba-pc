DATA = "data/"
DATA_POINT_CLOUD = DATA + "modelnet40_ply_hdf5_2048"
TRAINED_MODEL = "trained_models/"

NUM_POINTS = 1024
NUM_CLASSES = 40
NUM_ADD_POINT = 32
NUM_POINT_BA = 2080
NUM_CORNER_POINT = 4
TARGETED_CLASS = 1
NUM_WORKERS = 4
BATCH_SIZE = 24
OPTION_FEATURE_TRANSFORM = True

NUM_EPOCH = 250
LEARNING_RATE = 0.001
MOMENTUM = 0.9
DECAY_STEP = 200000
DECAY_RATE = 0.7
WEIGHT_DECAY = 1e-4
PERCENTAGE = 0.1
OPT = 'Adam'

INDEPENDENT_POINT = "independent_point"
ORIGINAL_POINT = "original_point"
CORNER_POINT = "corner_point"

categories = {
    0: 'airplane',
    1: 'bathtub',
    2: 'bed',
    3: 'bench',
    4: 'bookshelf',
    5: 'bottle',
    6: 'bowl',
    7: 'car',
    8: 'chair',
    9: 'cone',
    10: 'cup',
    11: 'curtain',
    12: 'desk',
    13: 'door',
    14: 'dresser',
    15: 'flower_pot',
    16: 'glass_box',
    17: 'guitar',
    18: 'keyboard',
    19: 'lamp',
    20: 'laptop',
    21: 'mantel',
    22: 'monitor',
    23: 'night_stand',
    24: 'person',
    25: 'piano',
    26: 'plant',
    27: 'radio',
    28: 'range_hood',
    29: 'sink',
    30: 'sofa',
    31: 'stairs',
    32: 'stool',
    33: 'table',
    34: 'tent',
    35: 'toilet',
    36: 'tv_stand',
    37: 'vase',
    38: 'wardrobe',
    39: 'xbox'
}
