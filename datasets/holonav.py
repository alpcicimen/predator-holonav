"""Data loader
"""
import argparse, os, torch, h5py, torchvision
import copy
import logging
import random
from typing import List

import h5py
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

import datasets.transforms as Transforms
import common.math.se3 as se3
from lib.benchmark_utils import get_correspondences, to_o3d_pcd, to_tsfm

import pyvista as pv

_logger = logging.getLogger()


def get_train_datasets(args: argparse.Namespace):
    train_categories, val_categories = None, None
    # if args.train_categoryfile:
    #     train_categories = [line.rstrip('\n') for line in open(args.train_categoryfile)]
    #     train_categories.sort()
    # if args.val_categoryfile:
    #     val_categories = [line.rstrip('\n') for line in open(args.val_categoryfile)]
    #     val_categories.sort()

    train_transforms, val_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
                                                      args.num_points, args.partial)
    _logger.info('Train transforms: {}'.format(', '.join([type(t).__name__ for t in train_transforms])))
    _logger.info('Val transforms: {}'.format(', '.join([type(t).__name__ for t in val_transforms])))
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    if args.dataset_type == 'holonav':
        train_data = HoloNavInput(args, args.root, subset='train', categories=train_categories,
                                  transform=train_transforms)
        val_data = HoloNavInput(args, args.root, subset='test', categories=val_categories,
                                transform=val_transforms)
    else:
        raise NotImplementedError

    return train_data, val_data


def get_test_datasets(args: argparse.Namespace):
    test_categories = None
    # if args.test_category_file:
    #     test_categories = [line.rstrip('\n') for line in open(args.test_category_file)]
    #     test_categories.sort()

    _, test_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
                                        args.num_points, args.partial)
    _logger.info('Test transforms: {}'.format(', '.join([type(t).__name__ for t in test_transforms])))
    test_transforms = torchvision.transforms.Compose(test_transforms)

    if args.dataset_type == 'holonav':
        test_data = HoloNavInput(args, args.root, subset='test', categories=test_categories,
                                 transform=test_transforms)
    else:
        raise NotImplementedError

    return test_data


def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 0.5,
                   num_points: int = 1024, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.FixedResampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.ShufflePoints()]

    elif noise_type == "match":
        # Match different models together
        train_transforms = [Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           # Transforms.Resampler(num_points),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms

class HoloNavInput(Dataset):
    def __init__(self, args, root: str, subset: str = 'train', categories: List = None, transform=None):
        """ModelNet40 dataset from PointNet.
        Automatically downloads the dataset if not available

        Args:
            root (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config = args
        self._root = root
        self.n_in_feats = args.in_feats_dim
        self.overlap_radius = args.overlap_radius
        self._logger = logging.getLogger(self.__class__.__name__)

        categories_idx = None
        self._logger.info('Using all categories.')

        if subset == 'train':
            sourcePC_path = root + '/source_point_clouds_preop_models/sk1_face_d10000f.ply'
            sourcePC_mesh = np.full(5, pv.read(sourcePC_path))
            targetPC_points = [pv.PolyData(
                np.loadtxt((root + '/target_point_clouds/Optical/points1/sk1/reg_pc{}.txt'.format(val)))
            ).delaunay_2d()
                               for val in range(1, 6)]

            self._labels = [0] * 5
            self._classes = ["custom skull 0"] * 5

        else:
            # sourcePC_path = root + '/source_point_clouds_preop_models/sk1_face_d10000f.ply'
            # sourcePC_mesh = np.full(5, pv.read(sourcePC_path))
            # targetPC_points = [pv.PolyData(
            #     np.loadtxt((root + '/target_point_clouds/Optical/points1/sk1/reg_pc{}.txt'.format(val)))
            # ).delaunay_2d()
            #                    for val in range(1, 6)]

            # START

            sourcePC_mesh = np.array([], dtype=pv.DataSet)
            targetPC_points = []

            self._labels = []
            self._classes = []

            for i in range(1, 4):

                # sk_path = dataset_path + '/source_point_clouds_preop_models/sk{}_face.ply'.format(i)
                # reg_path = dataset_path + '/target_point_clouds/Optical/points1/sk{}'.format(i)
                #
                # reg_points = [pv.PolyData(np.loadtxt(os.path.join(reg_path, fname))).delaunay_2d()
                #               for fname in os.listdir(reg_path)
                #               if 'reg_pc' in fname]
                #
                # targetPC_points = targetPC_points + reg_points
                #
                # sourcePC_mesh = np.append(sourcePC_mesh, np.full(len(reg_points), pv.read(sk_path)))
                #
                # self._labels = self._labels + [i-1] * len(reg_points)
                # self._classes = self._classes + ["custom skull {}".format(i)] * len(reg_points)

                sk_path = root + '/source_point_clouds_preop_models/sk{}_face.ply'.format(i)
                reg_path = root + '/source_point_clouds_preop_models/sk{}_face_occ.ply'.format(i)

                sourcePC_mesh = np.append(sourcePC_mesh, pv.read(sk_path))
                targetPC_points = targetPC_points + [pv.read(reg_path)]

                self._labels = self._labels + [i-1]
                self._classes = self._classes + ["custom skull {}".format(i)]

            # END

            # targetPC_path = root + '/target_point_clouds/depthSensor/p1_example_cleaned.ply'
            # targetPC_points = np.full(5, pv.read(targetPC_path))

        self._data_src = self._prepare_src(sourcePC_mesh)
        # self._data_ref = self._prepare_src(sourcePC_mesh)

        # self._data_src = self._prepare_ref(targetPC_points)
        self._data_ref = self._prepare_ref(targetPC_points)

        # self.voxel_size = np.array([24.58119604, 44.85551061,  2.33980616,  2.30597854, 12.89437182]) / 2
        self._transform = transform

    def __getitem__(self, item):
        # sample = {'points': self._data[item, :, :], 'label': self._labels[item], 'idx': np.array(item, dtype=np.int32)}
        #
        # if self._transform:
        #     sample = self._transform(sample)
        sample = {'points_src': self._data_src[item][:, :], 'points_ref': self._data_ref[item][:, :],
                  'points_raw': self._data_src[item][:, :],
                  'label': self._labels[item], 'idx': np.array(item, dtype=np.int32)}

        src_pcd = sample['points_src'][:, :3]
        src_pcd_n = sample['points_src'][:, 3:]
        tgt_pcd = sample['points_ref'][:, :3]
        tgt_pcd_n = sample['points_ref'][:, 3:]

        # voxelize the point clouds here
        pcd0 = to_o3d_pcd(src_pcd)
        pcd0.normals = o3d.utility.Vector3dVector(src_pcd_n)
        pcd1 = to_o3d_pcd(tgt_pcd)
        pcd1.normals = o3d.utility.Vector3dVector(tgt_pcd_n)
        pcd0 = pcd0.voxel_down_sample(6.0)
        pcd1 = pcd1.voxel_down_sample(6.0)

        src_pcd = np.concatenate((np.asarray(pcd0.points, dtype=np.float32), np.asarray(pcd0.normals, dtype=np.float32))
                                 , axis=-1)
        tgt_pcd = np.concatenate((np.asarray(pcd1.points, dtype=np.float32), np.asarray(pcd1.normals, dtype=np.float32))
                                 , axis=-1)

        sample['points_src'] = src_pcd
        sample['points_ref'] = tgt_pcd
        sample['points_raw'] = src_pcd

        if self._transform:
            sample = self._transform(sample)
        # transform to our format
        src_pcd = sample['points_src'][:, :3]
        tgt_pcd = sample['points_ref'][:, :3]
        rot = sample['transform_gt'][:, :3]
        trans = sample['transform_gt'][:, 3][:, None]

        matching_inds = get_correspondences(to_o3d_pcd(src_pcd), to_o3d_pcd(tgt_pcd), to_tsfm(rot, trans),
                                            self.overlap_radius)

        if (self.n_in_feats == 1):
            src_feats = np.ones_like(src_pcd[:, :1]).astype(np.float32)
            tgt_feats = np.ones_like(tgt_pcd[:, :1]).astype(np.float32)
        elif (self.n_in_feats == 3):
            src_feats = src_pcd.astype(np.float32)
            tgt_feats = tgt_pcd.astype(np.float32)

        for k, v in sample.items():
            if k not in ['deterministic', 'label', 'idx']:
                sample[k] = torch.from_numpy(v).unsqueeze(0)

        return src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, matching_inds, src_pcd, tgt_pcd, sample

    def __len__(self):
        return len(self._data_src)

    @property
    def classes(self):
        return self._classes

    # @staticmethod
    # def _prepare_src(src_model):
    #     pc_points = np.asarray(src_model.points, dtype=np.float32)
    #     pc_normals = np.asarray(src_model.point_normals, dtype=np.float32)
    #
    #     while len(pc_points) > 1024:
    #         r_index = random.randint(0, len(pc_points) - 1)
    #         pc_points = np.delete(pc_points, r_index, 0)
    #         pc_normals = np.delete(pc_normals, r_index, 0)
    #
    #     data = np.concatenate([pc_points[:], pc_normals[:]], axis=-1)
    #
    #     return np.array([data], dtype=np.float32)

    @staticmethod
    def _prepare_src(src_model):
        data = []
        for model in src_model:
            pc_points = np.asarray(model.points, dtype=np.float32)
            pc_normals = np.asarray(model.point_normals, dtype=np.float32)

            # while len(pc_points) > 1024:
            #     r_index = random.randint(0, len(pc_points) - 1)
            #     pc_points = np.delete(pc_points, r_index, 0)
            #     pc_normals = np.delete(pc_normals, r_index, 0)

            data.append(np.concatenate([pc_points[:], pc_normals[:]], axis=-1))

        # return np.array(data, dtype=np.float32)
        return data

    @staticmethod
    def _prepare_ref(target_pc):
        data = []
        for model in target_pc:
            pc_points = np.asarray(model.points, dtype=np.float32)
            pc_normals = np.asarray(model.point_normals, dtype=np.float32)

            # while len(pc_points) < 1024:
            #     pc_points = np.concatenate([pc_points, pc_points])
            #     pc_normals = np.concatenate([pc_normals, pc_normals])
            #
            # while len(pc_points) > 1024:
            #     r_index = random.randint(0, len(pc_points) - 1)
            #     pc_points = np.delete(pc_points, r_index, 0)
            #     pc_normals = np.delete(pc_normals, r_index, 0)

            data.append(np.concatenate([pc_points[:], pc_normals[:]], axis=-1))

        # return np.array(data, dtype=np.float32)
        return data

        # pc_points = np.asarray(target_pc.points, dtype=np.float32)
        # pc_normals = np.asarray(target_pc.point_normals, dtype=np.float32)
        #
        # while len(pc_points) > 1024:
        #     r_index = random.randint(0, len(pc_points) - 1)
        #     pc_points = np.delete(pc_points, r_index, 0)
        #     pc_normals = np.delete(pc_normals, r_index, 0)
        #
        # data = np.concatenate([pc_points[:], pc_normals[:]], axis=-1)
        #
        # return np.array([data], dtype=np.float32)

    def to_category(self, i):
        return self._idx2category[i]
