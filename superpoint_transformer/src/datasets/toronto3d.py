import os
import sys
import glob
import torch
import shutil
import logging
import pandas as pd
from plyfile import PlyData
from src.datasets import BaseDataset
from src.data import Data, Batch
from src.datasets.toronto3d_config import *
from torch_geometric.data import extract_zip
from src.utils import available_cpu_count, starmap_with_kwargs, \
    rodrigues_rotation_matrix, to_float_rgb


DIR = osp.dirname(osp.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['TORONTO3D', 'MiniTORONTO3D']


########################################################################
#                                 Utils                                #
########################################################################

def to_float_intensity(intensity):
    intensity = intensity.float()
    if intensity.max() > 1:
        intensity = intensity / 255
    intensity = intensity.clamp(min=0, max=1)
    return intensity

def read_ply(
    file_path, xyz=True, rgb=True, semantic=True, intensity=True, instance=False,
    verbose=True):
    """convert from a ply file. include the label and the object number"""
    """Read all Toronto3D object-wise annotations in a given directory.

    :param file_path: str
        Absolute path to the directory, eg: '/some/path/LO04.ply'
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.y
    :return:
        Batch of accumulated points clouds
    """
    #---read the ply file--------
    print(file_path)
    plydata = PlyData.read(file_path)
    if verbose:
        log.debug(f"Reading file: {plydata['vertex'][7]}")

    # Initialize accumulators for xyz, RGB, semantic label and instance
    # label
    xyz_list = [] if xyz else None
    rgb_list = [] if rgb else None
    intensity_list = [] if intensity else None
    y_list = [] if semantic else None
    o_list = [] if instance else None

    if xyz:
        tmp = np.stack([plydata['vertex'][n] for n in['x', 'y', 'z']], axis=1)
        UTM_OFFSET = [627285, 4841948, 0]
        tmp = tmp - UTM_OFFSET
        xyz_list.append(np.ascontiguousarray(tmp, dtype='float32'))

    if rgb:
        try:
            tmp = np.stack([plydata['vertex'][n]
                        for n in ['red', 'green', 'blue']]
                       , axis=1).astype(np.uint8)
            rgb_list.append(np.ascontiguousarray(tmp))
        except ValueError:
            tmp = np.stack([plydata['vertex'][n]
                        for n in ['r', 'g', 'b']]
                       , axis=1).astype(np.uint8)
            rgb_list.append(np.ascontiguousarray(tmp))

    if semantic:
        try:
            labels = plydata['vertex']['label']
            # convert labels into scalar values
            scalar_label = [OBJECT_LABEL.get(x, OBJECT_LABEL['clutter']) for x in labels]
            y_list.append(np.array(scalar_lables).astype(np.int64))
        except ValueError:
            try:
                labels = np.array(plydata['vertex']['scalar_Label']).astype(np.int64)
                y_list.append(labels)
            except ValueError:
                log.warning("No labels ")

    if intensity:
        try:
            tmp_data = plydata['vertex']['intensity']
            # convert labels into float values
            tmp_data = np.array(tmp_data).astype(np.float32)
            intensity_list.append(tmp_data)
        except ValueError:
            try:
                tmp_data = np.array(plydata['vertex']['scalar_Intensity']).astype(np.float32)
                intensity_list.append(tmp_data)
            except ValueError:
                log.warning("No intensity ")

    if instance:
        try:
            object_indices = plydata['vertex']['object_index']
            o_list.append(np.array(object_indices).astype(np.int64))
        except ValueError:
            log.warning("No index ")

    # Concatenate and convert to torch
    xyz_data = torch.from_numpy(np.concatenate(xyz_list, 0)) if xyz else None
    rgb_data = to_float_rgb(torch.from_numpy(np.concatenate(rgb_list, 0))) \
        if rgb else None
    intensity_data = to_float_intensity(torch.from_numpy(np.concatenate(intensity_list, 0))) if intensity else None
    y_data = torch.from_numpy(np.concatenate(y_list, 0)) if semantic else None
    o_data = torch.from_numpy(np.concatenate(o_list, 0)) if instance else None

    print(xyz_data.shape, rgb_data.shape, y_data.shape)
    # Store into a Data object
    data = Data(pos=xyz_data, rgb=rgb_data, intensity=intensity_data, y=y_data, o=o_data)
    print(data.size())

    return data


########################################################################
#                               TORONTO3D                               #
########################################################################

class TORONTO3D(BaseDataset):
    """TORONTO3D dataset, for Area-wise prediction.

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    fold : `int`
        Integer in [1, ..., 6] indicating the Test Area
    stage : {'train', 'val', 'test', 'trainval'}, optional
    transform : `callable`, optional
        transform function operating on data.
    pre_transform : `callable`, optional
        pre_transform function operating on data.
    pre_filter : `callable`, optional
        pre_filter function operating on data.
    on_device_transform: `callable`, optional
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    _form_url = None
    _zip_name = None
    _aligned_zip_name = None
    _unzip_name = None

    def __init__(self, *args, fold=5, **kwargs):
        self.fold = fold
        super().__init__(*args, val_mixed_in_train=True, **kwargs)

    @property
    def class_names(self):
        """List of string names for dataset classes. This list may be
        one-item larger than `self.num_classes` if the last label
        corresponds to 'unlabelled' or 'ignored' indices, indicated as
        `-1` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes in the dataset. May be one-item smaller
        than `self.class_names`, to account for the last class name
        being optionally used for 'unlabelled' or 'ignored' classes,
        indicated as `-1` in the dataset labels.
        """
        return TORONTO3D_NUM_CLASSES

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of clouds ids, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return {
            'train': ['LO01.ply', 'LO03.ply'],
            'val': ['LO02.ply'],
            'test': ['LO04.ply']}

    def download_dataset(self):
        """Download the TORONTO3D dataset.
        """
        # Manually download the dataset

        # Unzip the file and rename it into the `root/raw/` directory. This
        # directory contains the raw Area folders from the zip

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a Data object, ready to
        be passed to `self.pre_transform`.
        """
        return read_ply(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, intensity=True, instance=False,
            verbose=False)

    @property
    def raw_file_structure(self):
        return f"""
    {self.root}/
        └── {self._zip_name}
        └── raw/
            └── Area_{{i_area:1>6}}/
                └── Area_{{i_area:1>6}}_alignmentAngle.txt
                └── ...
            """

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return self.id_to_base_id(id)[:-4]


########################################################################
#                              MiniTORONTO3D                               #
########################################################################

class MiniTORONTO3D(TORONTO3D):
    """A mini version of TORONTO3D with only 2 areas per stage for
    experimentation.
    """
    _NUM_MINI = 1

    @property
    def all_cloud_ids(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self):
        return self.__class__.__bases__[0].__name__.lower()

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self):
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self):
        super().download()
