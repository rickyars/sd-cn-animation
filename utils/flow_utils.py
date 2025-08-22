import numpy as np
import cv2
import torch
import gc

# Remove RAFT-related functions and only keep functions needed by txt2vid.py
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

def background_subtractor(frame, fgbg):
  fgmask = fgbg.apply(frame)
  return cv2.bitwise_and(frame, frame, mask=fgmask)

def frames_norm(frame):
    if hasattr(frame, 'dtype'):  # PyTorch tensor
        return frame.float() / 127.5 - 1.0
    else:  # Numpy array
        return frame.astype(np.float32) / 127.5 - 1.0

def flow_norm(flow):
    if hasattr(flow, 'dtype'):  # PyTorch tensor
        return flow.float() / 255.0
    else:  # Numpy array
        return flow.astype(np.float32) / 255.0

def occl_norm(occl):
    if hasattr(occl, 'dtype'):  # PyTorch tensor
        return occl.float() / 127.5 - 1.0
    else:  # Numpy array
        return occl.astype(np.float32) / 127.5 - 1.0

def frames_renorm(frame):
    if hasattr(frame, 'dtype'):  # PyTorch tensor
        return torch.clamp((frame + 1.0) * 127.5, 0, 255)
    else:  # Numpy array
        return np.clip((frame + 1.0) * 127.5, 0, 255)

def flow_renorm(flow):
    return flow * 255.0

def occl_renorm(occl):
    if hasattr(occl, 'dtype'):  # PyTorch tensor
        return torch.clamp((occl + 1.0) * 127.5, 0, 255)
    else:  # Numpy array
        return np.clip((occl + 1.0) * 127.5, 0, 255)