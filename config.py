DATA = "data/"
DATA_POINT_CLOUD = DATA + "modelnet40_ply_hdf5_2048"
TRAINED_MODEL = "trained_models/"

NUM_POINT_SAMPLE = 1024
NUM_CLASSES = 40
TARGETED_CLASS = 0
NUM_WORKERS = 4
BATCH_SIZE = 24
OPTION_FEATURE_TRANSFORM = True
NUM_POINT_INPUT = 2048

EPSILON = 0.05
NUM_EPOCH = 300
LEARNING_RATE = 0.001
MOMENTUM = 0.9
DECAY_STEP = 200000
DECAY_RATE = 0.7
WEIGHT_DECAY = 1e-4

PERCENTAGE = 0.1

OPT = 'Adam'

LOG_CLASSIFICATION = '/home/ubuntu/3d-ba-pc/log/classification/'
BEST_MODEL = '/checkpoints/best_model.pth'
BAD_MODEL = '/checkpoints/bad_model.pth'
FINAL_MODEL = '/checkpoints/final.pth'

AIRPLANE = "/home/nam/workspace/vinai/project/3d-ba-pc/data_set/trigger_generation/airplane.npy"

ORIGINAL = "original"
MULTIPLE_CORNER_POINT = "multiple_corner_point"
CORNER_POINT = "corner_point"
CENTRAL_OBJECT = "central_object"
CENTRAL_POINT = "central_point"
DUPLICATE_POINT = "duplicate_point"
SHIFT_POINT = "shift_point"
LOCAL_POINT = "local_point"

SHIFT_POINT_CONFIG = {
    "NUM_ADD_POINT": 128,
    "NUM_POINT_BA": 2048 + 128,
    "NUM_SAMPLE": 1024,
    "NUM_POINT_INPUT": 2048,
    "DISTANCE": 0.05,
}

DUPLICATE_POINT_CONFIG = {
    "NUM_ADD_POINT": 128,
    "NUM_POINT": 896,
    "NUM_POINT_BA": 2048 + 128,
    "NUM_SAMPLE": 1024,
    "NUM_POINT_INPUT": 2048,
}

MULTIPLE_CORNER_POINT_CONFIG = {
    "NUM_ADD_POINT": 128,
    "NUM_POINT_BA": 2048 + 128,
    "NUM_POINT_PER_CORNER": 16,
    "NUM_SAMPLE": 1024,
    "NUM_POINT_INPUT": 2048,
    "RADIUS": 0.05,
}

CORNER_POINT_CONFIG = {
    "NUM_ADD_POINT": 128,
    "NUM_POINT_BA": 2048 + 128,
    "NUM_SAMPLE": 1024,
    "NUM_POINT_INPUT": 2048,
    "RADIUS": 0.05,
}

CENTRAL_OBJECT_CONFIG = {
    "NUM_POINT_PER_OBJECT": 128,
    "NUM_ADD_POINT": 128,
    "NUM_ADD_BA": 2048 + 128,
    "NUM_SAMPLE": 1024,
    "NUM_POINT_INPUT": 2048,
    "SCALE": 0.2
}

CENTRAL_POINT_CONFIG = {
    "NUM_POINT_PER_OBJECT": 128,
    "NUM_ADD_POINT": 128,
    "NUM_ADD_BA": 2048 + 128,
    "NUM_SAMPLE": 1024,
    "NUM_POINT_INPUT": 2048,
    "RADIUS": 0.05,
}

LOCAL_POINT_CONFIG = {
    "NUM_ADD_POINT": 128,
    "NUM_ADD_BA": 2048 + 128,
    "NUM_SAMPLE": 1024,
    "NUM_POINT_INPUT": 2048,
    "RADIUS": 0.01,
}

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

categories_scanobjectnn = {
    0: "bag",
    1: "bin",
    2: "box",
    3: "cabinet",
    4: "chair",
    5: "desk",
    6: "display",
    7: "door",
    8: "shelf",
    9: "table",
    10: "bed",
    11: "pillow",
    12: "sink",
    13: "sofa",
    14: "toilet",
}
