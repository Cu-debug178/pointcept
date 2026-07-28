#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#   Hugues THOMAS - 06/10/2023
#
#   KPConvX project: S3DIS_rooms.py
#       > Dataset class for S3DIS (rooms)
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   Use this script to define the dataset specific configuration. You should be able to adapt this file for other dataset 
#   that share the same file structure as S3DIR.
#
#   We call this the S3DIR dataset as it is the room version of the S3DIS dataset.
#
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import time
import numpy as np
from os import listdir, makedirs
from os.path import join, exists

from utils.config import init_cfg
from data_handlers.scene_seg import SceneSegDataset
from utils.ply import read_ply, write_ply



# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


def S3DIR_cfg(cfg, dataset_path='../data/s3dis'):

    # cfg = init_cfg()
        
    # Dataset path
    cfg.data.name = 'S3DIR'
    cfg.data.path = dataset_path
    cfg.data.task = 'cloud_segmentation'

    # Dataset dimension
    cfg.data.dim = 3

    # Dict from labels to names
    cfg.data.label_and_names = [(0, 'ceiling'),
                                (1, 'floor'),
                                (2, 'wall'),
                                (3, 'beam'),
                                (4, 'column'),
                                (5, 'window'),
                                (6, 'door'),
                                (7, 'chair'),
                                (8, 'table'),
                                (9, 'bookcase'),
                                (10, 'sofa'),
                                (11, 'board'),
                                (12, 'clutter')]

    # Initialize all label parameters given the label_and_names list
    cfg.data.num_classes = len(cfg.data.label_and_names)
    cfg.data.label_values = [k for k, v in cfg.data.label_and_names]
    cfg.data.label_names = [v for k, v in cfg.data.label_and_names]
    cfg.data.name_to_label = {v: k for k, v in cfg.data.label_and_names}
    cfg.data.name_to_idx = {v: i for i, v in enumerate(cfg.data.label_names)}

    # Ignored labels
    cfg.data.ignored_labels = []
    cfg.data.pred_values = [k for k in cfg.data.label_values if k not in cfg.data.ignored_labels]

    return cfg


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/
#


class S3DIRDataset(SceneSegDataset):

    def __init__(self, cfg, chosen_set='training', precompute_pyramid=False, load_data=True):
        """
        Class to handle S3DIR dataset.
        Simple implementation.
            > Input only consist of the first cloud with features
            > Neigborhood and subsamplings are computed on the fly in the network
            > Sampling is done simply with random picking (X spheres per class)
        """
        SceneSegDataset.__init__(self,
                                 cfg,
                                 chosen_set=chosen_set,
                                 precompute_pyramid=precompute_pyramid)

        ############
        # S3DIR data
        ############

        # Here provide the list of .ply files depending on the set (training/validation/test)
        self.scene_names, self.scene_files = self.S3DIR_files()

        # Stop data is not needed
        if not load_data:
            return
        
        # Properties of input files
        self.label_property = 'class'
        self.f_properties = ['red', 'green', 'blue']

        # Start loading (merge when testing)
        self.load_scenes_in_memory(label_property=self.label_property,
                                   f_properties=self.f_properties,
                                   f_scales=[1/255, 1/255, 1/255])

        ###########################
        # Sampling data preparation
        ###########################

        if self.data_sampler == 'regular':
            # In case regular sampling, generate the first sampling points
            self.new_reg_sampling_pts()

        else:
            # To pick points randomly per class, we need every point index from each class
            self.prepare_label_inds()

        return


    def S3DIR_files(self):
        """
        Function returning a list of file path. One for each scene in the dataset.
        """

        # Get Areas
        area_paths = np.sort([join(self.path, f) for f in listdir(self.path) if f.startswith('Area')])

        # Get room names
        scene_paths = [np.sort([join(area_path, sc) for sc in listdir(str(area_path))]) 
                       for area_path in area_paths]

        # Only get a specific split
        if self.set == 'training':
            split_inds = [0, 1, 2, 3, 5]
        elif self.set in ['validation', 'test']:
            split_inds = [4,]

        scene_files = np.concatenate([scene_paths[i] for i in split_inds], axis=0)
        scene_names = [f.split('/')[-2] + "_" + f.split('/')[-1] for f in scene_files]
        
        # In case of merge, change the files
        self.room_lists = None

        return scene_names, scene_files


    def select_features(self, in_features):

        # Input features
        selected_features = np.ones_like(in_features[:, :1], dtype=np.float32)
        if self.cfg.model.input_channels == 1:
            pass
        elif self.cfg.model.input_channels == 4:
            selected_features = np.hstack((selected_features, in_features[:, :3]))
        elif self.cfg.model.input_channels == 5:
            selected_features = np.hstack((selected_features, in_features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 5')

        return selected_features

    def load_scene_file(self, file_path):

        if file_path.endswith('.ply'):
            
            data = read_ply(file_path)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            if self.label_property in [p for p, _ in data.dtype.fields.items()]:
                labels = data[self.label_property].astype(np.int32)
            else:
                labels = None
            features = np.vstack([data[f_prop].astype(np.float32) for f_prop in self.f_properties]).T

        elif file_path.endswith('.npy'):

            cdata = np.load(file_path)
            
            points = cdata[:,0:3].astype(np.float32)
            features = cdata[:, 3:6].astype(np.float32)
            labels = cdata[:, 6:7].astype(np.int32)

        elif file_path.endswith('.merge'): # loads all the files that share a same root

            # Merge data
            all_points = []
            all_features = []
            all_labels = []
            for room_file in self.room_lists[file_path]:
                points, features, labels = self.load_scene_file(room_file)
                all_points.append(points)
                all_features.append(features)
                all_labels.append(labels)
            points = np.concatenate(all_points, axis=0)
            features = np.concatenate(all_features, axis=0)
            labels = np.concatenate(all_labels, axis=0)

        else:

            # New dataset format has each room as a folder
            points = np.load(join(file_path, 'coord.npy'))
            colors = np.load(join(file_path, 'color.npy'))
            # instances = np.load(join(file_path, 'instance.npy'))
            # normals = np.load(join(file_path, 'normal.npy'))
            segments = np.load(join(file_path, 'segment.npy'))

            # features = np.concatenate((colors.astype(np.float32) / 255, normals), axis=1)
            features = colors.astype(np.float32)


            labels = segments

        return points, features, np.squeeze(labels)
