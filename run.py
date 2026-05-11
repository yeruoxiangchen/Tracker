from GroundingDINO.demo.inference_on_a_image import run_groundingdino, load_model
from foundpose.scripts.infer_single_image import load_detections, run_foundpose

def main():

    image_path = "foundpose/datasets/test/rgb/combined.jpg"
    text_prompt = "the box"
    config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    output_path = "GroundingDINO/results"
    cpu_only = False

    object_id = 1

    # GroundingDINO
    model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)

    boxes_filt, image_size = run_groundingdino(
        model=model,
        image_path=image_path,
        output_dir=output_path,
        text_prompt=text_prompt,
        cpu_only=cpu_only
    )

    # detection
    foundpose_detection = load_detections(boxes_filt, image_size, object_id)

    # FoundPose
    run_foundpose("foundpose/configs/infer/lmo.json", foundpose_detection, image_path)


if __name__ == "__main__":
    main()
