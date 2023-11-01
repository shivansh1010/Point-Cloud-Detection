import numpy as np
import os.path as osp


########################################################################
#                                Labels                                #
########################################################################

# Credit: https://github.com/torch-points3d/torch-points3d

TORONTO3D_NUM_CLASSES = 9

INV_OBJECT_LABEL = {
    0: "clutter",
    1: "road",
    2: "road_marking",
    3: "nature",
    4: "building",
    5: "utility_line",
    6: "pole",
    7: "car",
    8: "fence"}

CLASS_NAMES = [INV_OBJECT_LABEL[i] for i in range(TORONTO3D_NUM_CLASSES)] + ['ignored']

CLASS_COLORS = np.asarray([
    [0, 0, 0],        # clutter            -> black
    [233, 229, 107],  # 'road'             ->  yellow
    [95, 156, 196],   # 'road marking'     ->  blue
    [179, 116, 81],   # 'nature'           ->  brown
    [241, 149, 131],  # 'building'         ->  salmon
    [81, 163, 148],   # 'utility_line'     ->  bluegreen
    [77, 174, 84],    # 'pole'             ->  bright green
    [108, 135, 75],   # 'car'              ->  dark green
    [223, 52, 52],    # 'fence'            ->  red
    [81, 109, 114]])  # 'unlabelled'          ->  grey

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

def object_name_to_label(object_class):
    """Convert from object name to int label. By default, if an unknown
    object nale
    """
    object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["clutter"])
    return object_label
