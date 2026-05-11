# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict


import os
from typing import List, Optional
from bop_toolkit_lib import inout, pycoco_utils
from bop_toolkit_lib import dataset_params
from gotrack.utils import data_util, misc, structs
from gotrack.utils.logging import get_logger
import numpy as np
from tqdm import tqdm
from dataloader.base import GoTrackDataset

from gotrack.utils.structs import PinholePlaneCameraModel


class SingleImage(GoTrackDataset):
    """Dataloader for BOP datasets."""

    def __init__(
        self,
        obj_id,
        model_tpath,
        root_dir: str,
        use_default_detections: bool = False,
    ) -> None:
        super().__init__(root_dir=root_dir)
        self.obj_id = obj_id
        self.use_default_detections = use_default_detections  
        self.image = None
        self.scene_cameras = {}
        self.coarse_poses_per_image = {}

        # Initialize dataset parameters.
        self.models = None
        self.models_info = None
        self.models_vertices = None
        self.dp_model = {
            "model_tpath": model_tpath,
            "obj_ids": [obj_id],
        }

        # Load the models.
        self.models = {}
        self.models_vertices = {}

        for obj_id in self.dp_model["obj_ids"]:
            self.models[obj_id] = inout.load_ply(
                self.dp_model["model_tpath"].format(obj_id=obj_id)
            )
            # Sample vertices.
            max_vertices = 1000  # followed FoundPose.
            self.models_vertices[obj_id] = np.random.permutation(
                self.models[obj_id]["pts"]
            )[:max_vertices]
            self.models_vertices[obj_id] = self.models_vertices[obj_id].astype(
                np.float32
            )

    def update_info(self, image_raw, camK, coarse_poses):
        # image
        self.image = np.array(image_raw)
        width, height = image_raw.size
        if self.image.ndim == 2:
            self.image = np.expand_dims(self.image, -1)
        if self.image.ndim == 3 and self.image.shape[2] == 1:
            self.image = np.repeat(self.image, 3, axis=2)

        # camera
        camK = np.array(camK).squeeze().astype(np.float32)
        fx, fy, cx, cy = camK[0, 0], camK[1, 1], camK[0, 2], camK[1, 2]
        self.scene_cameras = {
            0: PinholePlaneCameraModel(
                width=width,
                height=height,
                f=(fx, fy),
                c=(cx, cy),
                T_world_from_eye=np.eye(4, dtype=np.float32)
            )
        }
        
        # coarse pose
        self.coarse_poses = coarse_poses
        self.load_dataset_info_from_coarse_pose()

    def load_dataset_info_from_coarse_pose(self) -> None:
        # Load coarse poses
        self.coarse_poses_per_image = {
            "objects": [],
        }
        for item in self.coarse_poses:
            est = structs.ObjectAnnotation(
                lid=int(self.obj_id),
                pose=item
            )
            self.coarse_poses_per_image["objects"].append(est)
        
    def __len__(self) -> int:
        """Return the number of test images in the dataset."""
        return 1

    def __getitem__(self, idx: int) -> structs.SceneObservation:
        """Return the test image and its corresponding target."""
        image = self.image

        # Load the scene camera (intrinsics).
        camera = self.scene_cameras[0]
        camera.im_size = image.shape[:2]
        camera.height = image.shape[0]
        camera.width = image.shape[1]
        
        # By default, the dataloader is for detection tasks, only camera intrinsic is available.
        scene_observation = structs.SceneObservation(
            image=np.asarray(image, dtype=np.uint8),
            camera=camera,
        )

        # Get the coarse poses for the current image.
        coarse_poses = self.coarse_poses_per_image["objects"]
        objects_anno = []
        for est_id, est in enumerate(coarse_poses):
            # 6D object pose.
            pose_m2w = None
            if est.pose is not None:
                pose_m2c = est.pose
                trans_c2w = camera.T_world_from_eye
                trans_m2w = np.matmul(trans_c2w, misc.get_rigid_matrix(pose_m2c))
                pose_m2w = structs.ObjectPose(
                    R=trans_m2w[:3, :3], t=trans_m2w[:3, 3:].reshape(3, 1)
                )

            objects_anno.append(
                structs.ObjectAnnotation(
                    lid=est.lid,
                    pose=pose_m2w,
                )
            )
        scene_observation = scene_observation._replace(objects_anno=objects_anno)

        # In case of localization tasks, the target objects are available.
        target_objects = None

        return {
            "scene_observation": scene_observation,
            "target_objects": target_objects,
        }
