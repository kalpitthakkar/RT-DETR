import os
from references.deploy.rtdetrv2_torch import main_infer, load_model

from tqdm import tqdm

def detect_obj_on_folder(folder_path):
    # config_path = "configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml"
    config_path = "configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml"
    # ckpt_path = "checkpoints/rtdetrv2_r101vd_6x_coco_from_paddle.pth"
    ckpt_path = "checkpoints/rtdetrv2_r50vd_m_7x_coco_ema.pth"
    device = "mps"
    detections = []
    cfg, model = load_model(config_path, ckpt_path, device)
    for filename in tqdm(os.listdir(folder_path)):
        fpath = os.path.join(folder_path, filename)
        detect = main_infer(fpath, device, model)
        detections.append(detect)
    return detections


if __name__ == '__main__':
    folder_path = "../../output_insane_car_crashes_comp_20s/sampled_frames"
    detections = detect_obj_on_folder(folder_path)