import os
from object_detector_detr.rtdetrv2_pytorch.references.deploy.rtdetrv2_torch_old import main_infer, load_model

from tqdm import tqdm

from PIL import Image
import pickle


def detect_obj_on_folder(folder_path, save_dir=None):
    if save_dir is None:
        save_dir = os.path.split(folder_path)[0]
        
    if os.path.exists(os.path.join(save_dir, "detections.pkl")):
        print("Found existing detections")
        ret_detections = []
        with open(os.path.join(save_dir, "detections.pkl"), "rb") as f:
            detections = pickle.load(f)
            for _, v in detections.items():
                ret_detections.append(
                    [v["labels"], v["boxes"], v["scores"]]
                )
            return ret_detections
    
    # config_path = "configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml"
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml"
    )
    # ckpt_path = "checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth"
    ckpt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "checkpoints/rtdetrv2_r50vd_m_7x_coco_ema.pth"
    )
    device = "mps"
    detections = []
    save_detections = {}
    _, model = load_model(config_path, ckpt_path, device)
    fpaths = []
    for filename in tqdm(sorted(os.listdir(folder_path))):
        fpath = os.path.join(folder_path, filename)
        fpaths.append(fpath)
        detect = main_infer(fpath, device, model)
        detections.append(detect)
        save_detections[filename] = {
            "labels": detect[0],
            "boxes": detect[1],
            "scores": detect[2]
        }
    
    with open(os.path.join(save_dir, "detections.pkl"), "wb") as f:
        pickle.dump(save_detections, f)
    
    # for i in tqdm(range(len(fpaths))):
    #     fpath = fpaths[i]
    #     labels, boxes, scores = detections[i]
    #     im_pil = Image.open(fpath).convert('RGB')
    #     draw([im_pil], labels, boxes, scores)
    return detections


if __name__ == '__main__':
    folder_path = "../../output_insane_car_crashes_comp_20s/sampled_frames"
    detections = detect_obj_on_folder(folder_path)