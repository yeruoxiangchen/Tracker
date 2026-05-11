# import sys
# sys.path.append('../')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # huggingface警告

from xbase.netcall import encodeObjs, decodeObjs, runServer
from PIL import Image
import torch
import cv2
from GroundingDINO.demo.inference_on_a_image import run_groundingdino, load_model
from foundpose.scripts.infer_with_refine import load_detections, pre_infer, infer
from gotrack.scripts.run_gotrack import pre_gotrack
import time
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')


if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--port', default=8000,type=int, help='server port')
    args, _ = parser.parse_known_args()

    # config
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    output_path = "GroundingDINO/results"
    cpu_only = False

    opt_file = "foundpose/configs/infer/test.json"
    object_id = 5
    model_tpath = "foundpose/datasets/test/models/obj_{obj_id:06d}.ply"

    # 检测的物体id
    # 1: 盒子 | 2: 飞机模型
    # 3: 轮胎 | 6：无裂纹轮胎

    # GroundingDINO
    model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)
    # foundpose
    opts, extractor, repre, template_knn_indices, visual_words_knn_index = pre_infer(opt_file, object_id)
    trainer, gotrack_model, test_dataset, test_dataloader = pre_gotrack(
        obj_id=object_id,
        model_tpath=model_tpath
    )

    def serverHandler(objs):
        retObjs = {}
        
        image_raw = objs["image"]
        # Convert images from ndarray to PIL images
        image_pil = Image.fromarray(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB))

        camK = objs["camK"]

        t1 = time.time()
        boxes_filt, image_size = run_groundingdino(
            model=model,
            image_pil=image_pil,
            output_dir=output_path,
            text_prompt=objs["text"]
        )
        t2 = time.time()
        print("groundingdino: ", t2-t1)

        # 未检测到物体
        if boxes_filt is None or len(boxes_filt) == 0:
            return {"Rts": []}

        # detection
        foundpose_detection = load_detections(boxes_filt, image_size, object_id)        

        t3 = time.time()
        # foundpose
        # retObjs = infer(opt_file, foundpose_detection, image_pil, camK, model_tpath)
        retObjs = infer(
            opts=opts,
            extractor=extractor,
            foundpose_detection=foundpose_detection,
            camK=camK,
            object_lid=object_id,
            repre=repre,
            template_knn_indices=template_knn_indices,
            visual_words_knn_index=visual_words_knn_index,
            image_raw=image_pil,
            trainer=trainer,
            model=gotrack_model,
            test_dataset=test_dataset, 
            test_dataloader=test_dataloader
        )
        t4 = time.time()
        print("foundpose: ", t4-t3)
  
        return retObjs
    
    runServer(serverHandler, args.port)
