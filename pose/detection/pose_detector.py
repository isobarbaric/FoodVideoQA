# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from pose.detection.drawing_utils import draw_bodypose, draw_facepose, draw_handpose
from pose.detection.wholebody import Wholebody
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.parent
POSE_DIR = ROOT_DIR / "pose"
DETECTION_DIR = POSE_DIR / "detection"

# TODO: setting the path of the ckpts/ dir isn't working
@dataclass
class PoseDetectorConfig:
  det_config: str = DETECTION_DIR / 'config/yolox_l_8xb8-300e_coco.py'
#   det_ckpt: str = DETECTION_DIR / '/ckpts/detection_model.pth'
  det_ckpt: str = None
  pose_config: str = DETECTION_DIR / 'config/dwpose-l_384x288.py'
#   pose_ckpt: str = DETECTION_DIR / '/ckpts/pose_model.pth'
  pose_ckpt: str = None
  device: str = "cuda"


class PoseDetector:
    def __init__(self, config: PoseDetectorConfig = PoseDetectorConfig):
        self.pose_estimation = Wholebody(config.det_config, 
                                         config.det_ckpt,
                                         config.pose_config,
                                         config.pose_ckpt, 
                                         config.device)


    def _draw_pose(self, pose, H, W):
        bodies = pose['bodies']
        faces = pose['faces']
        hands = pose['hands']
        candidate = bodies['candidate']
        subset = bodies['subset']
        
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
        canvas = draw_bodypose(canvas, candidate, subset)
        canvas = draw_handpose(canvas, hands)
        canvas = draw_facepose(canvas, faces)

        return canvas

    def _extract_keypoints(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]
            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            
            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

        return bodies, pose, (H, W)


    def infer(self, oriImg):
        _, pose, (H, W) = self._extract_keypoints(oriImg)
        return self._draw_pose(pose, H, W)


    def get_face(self, oriImg):
        _, pose, _ = self._extract_keypoints(oriImg) 
        return pose['faces']
