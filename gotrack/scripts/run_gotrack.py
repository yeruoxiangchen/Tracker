# Copyright (c) Meta Platforms, Inc. and affiliates.
#!/usr/bin/env python3

# pyre-strict
import sys  
sys.path.append("./gotrack") 

import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from gotrack.utils import data_util, net_util  # noqa: F401
from gotrack.utils.logging import get_logger
import warnings
from hydra import initialize, compose
import numpy as np
import torch

warnings.filterwarnings("ignore")

def pre_gotrack(
    obj_id,
    model_tpath,
    config_name="inference_gotrack"
):
    with initialize(config_path="../configs", job_name="service"):
        cfg = compose(config_name=config_name)
    OmegaConf.set_struct(cfg, False)

    cfg_trainer = cfg.machine.trainer
    trainer = instantiate(cfg_trainer)

    model = instantiate(cfg.model)
    # Load the model checkpoint
    if cfg.model.checkpoint_path:
        net_util.load_checkpoint(
            model=model,
            checkpoint_path=cfg.model.checkpoint_path,
            checkpoint_key="model_state_dict",
            prefix="models.1.",
        )
    else:
        raise ValueError("Checkpoint path is required for inference")
    
    if cfg.mode == "pose_refinement":
        # Define the dataloader using dataset_name, coarse_pose_method
        test_dataset = instantiate(
            cfg.data.dataloader,
            obj_id=obj_id,
            model_tpath=model_tpath,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=cfg.machine.num_workers,
            collate_fn=data_util.convert_list_scene_observations_to_gotrack_inputs,
        )

    model.set_renderer(test_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return trainer, model, test_dataset, test_dataloader


def run_inference(camK, image_raw, coarse_poses, trainer, model, test_dataset, test_dataloader):
    test_dataset.update_info(image_raw, camK, coarse_poses)

    frame_data = test_dataset[0]
    batch = data_util.convert_list_scene_observations_to_gotrack_inputs([frame_data])
    outputs = model.test_step(batch, idx=0)
    
    return outputs

# def run_inference(camK, image_raw, coarse_poses, trainer, model, test_dataset, test_dataloader):
#     test_dataset.update_info(image_raw, camK, coarse_poses)
#     model.set_renderer(test_dataset)
    
#     trainer.predict(
#         model,
#         dataloaders=test_dataloader,
#     ) 
#     return model.outputs
