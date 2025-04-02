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
    return frame / 127.5 - 1

def flow_norm(flow):
    return flow / 255

def occl_norm(occl):
    return occl / 127.5 - 1

def frames_renorm(frame):
    return (frame + 1) * 127.5

def flow_renorm(flow):
    return flow * 255

def occl_renorm(occl):
    return (occl + 1) * 127.5