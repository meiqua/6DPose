#!/usr/bin/env python
import rospy

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from pysixd import view_sampler, inout, misc
from  pysixd.renderer import render
from os.path import join
import linemodLevelup_pybind
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

readTemplFrom = './yaml/%s_templ.yaml'
readInfoFrom = './yaml/{}_info.yaml'
readModelFrom = './models/{}.fly'
objIds = []
K_cam = []

detector = linemodLevelup_pybind.Detector()
poseRefine = linemodLevelup_pybind.poseRefine()
detector.readClasses(objIds, readFrom)
templateInfo = inout.load_info(tempInfo_saved_to.format(scene_id))

models = {}
for id in objIds:
    model = inout.load_ply(readModelFrom.format(id))
    models[id] = model

def nms_norms(ts, scores, thresh):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        norms = np.linalg.norm(ts[i]-ts[order[1:]], axis=1)
        inds = np.where(norms > thresh)[0]
        order = order[inds + 1]
    return keep

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def listenRGBD():
    pass

def receiveRGBD(RGBD):
    # get rgb, depth here
    matches = detector.match([rgb, depth], 65.0, objIds, masks=[])

    dets = np.zeros(shape=(len(matches),5))
    for i in range(len(matches)):
        match = matches[i]
        info = templateInfo[match.template_id]
        dets[i,0]=match.x
        dets[i,1]=match.y
        dets[i,2]=match.x+info['width']
        dets[i,3]=match.y+info['height']
        dets[i,4]=match.similarity
    idx = nms(dets,0.5)

    dets = np.zeros(shape=(len(idx),5))
    for i in idx:
        match = matches[i]
        info = templateInfo[match.template_id]
        model = models[match.class_id]
        depth_ren = render(model, im_size, K_match, R_match, t_match, mode='depth')
        K_match = info['cam_K']
        R_match = info['cam_R_w2c']
        t_match = info['cam_t_w2c']
        poseRefine.process(depth.astype(np.uint16), depth_ren.astype(np.uint16), K_cam.astype(np.float32),
                           K_match.astype(np.float32), R_match.astype(np.float32), t_match.astype(np.float32)
                           , match.x, match.y)        
        

def publishResults():
    pass

def getK(msg, subscriber):
    #do processing here to get K_cam
    subscriber.unregister()

if __name__ == '__main__':
