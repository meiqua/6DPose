#!/usr/bin/env python
import rospy
from  sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError

import os
import sys
import time
import numpy as np
from pysixd import inout
from  pysixd.renderer import render
import linemodLevelup_pybind
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

objIds = []

rgb = None
depth = None
rgb_flag = False
depth_flag = False
lock = False
bridge = CvBridge()

readTemplFrom = './yaml/%s_templ.yaml'
readInfoFrom = './yaml/{}_info.yaml'
readModelFrom = './models/{0}/{0}.fly'
K_cam = None

detector = linemodLevelup_pybind.Detector()
poseRefine = linemodLevelup_pybind.poseRefine()
detector.readClasses(objIds, readTemplFrom)

infos = {}
models = {}
for id in objIds:
    model = inout.load_ply(readModelFrom.format(id))
    models[id] = model
    templateInfo = inout.load_info(readInfoFrom.format(id))
    infos[id] = templateInfo

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

def listenRGB(rgb_):
    global rgb_flag, rgb, lock
    if not rgb_flag and not lock:
        rgb = bridge.imgmsg_to_cv2(rgb_.data, "rgb8")
        rgb_flag = True

def listenD(depth_):
    global depth_flag, depth, lock
    if not depth_flag and not lock:
        depth = bridge.imgmsg_to_cv2(depth_.data, "mono16")
        depth_flag = True

def receiveRGBD():
    # get rgb, depth here
    global rgb_flag, rgb, depth_flag, depth, lock
    if rgb_flag and depth_flag:  # regard rgb and depth arrive at same time

        matches = detector.match([rgb, depth], 65.0, objIds, masks=[])

        dets = np.zeros(shape=(len(matches), 5))
        for i in range(len(matches)):
            match = matches[i]
            templateInfo = infos[match.class_id]
            info = templateInfo[match.template_id]
            dets[i, 0] = match.x
            dets[i, 1] = match.y
            dets[i, 2] = match.x + info['width']
            dets[i, 3] = match.y + info['height']
            dets[i, 4] = match.similarity
        idx = nms(dets, 0.5)

        ts = []
        ts_scores = np.zeros(shape=(len(idx), 1))
        Rs = []
        ids = []
        confidences = []
        for i in idx:
            match = matches[i]
            templateInfo = infos[match.class_id]
            info = templateInfo[match.template_id]
            model = models[match.class_id]

            K_match = info['cam_K']
            R_match = info['cam_R_w2c']
            t_match = info['cam_t_w2c']
            depth_ren = render(model, depth.shape, K_match, R_match, t_match, mode='depth')
            poseRefine.process(depth.astype(np.uint16), depth_ren.astype(np.uint16), K_cam.astype(np.float32),
                               K_match.astype(np.float32), R_match.astype(np.float32), t_match.astype(np.float32)
                               , match.x, match.y)
            ts.append(poseRefine.getT())
            Rs.append(poseRefine.getR())
            ids.append(match.class_id)
            confidences.append(match.similarity)
            ts_scores[i] = -poseRefine.getResidual()
        idx = nms_norms(ts, ts_scores, 40.0)

        results = []
        for i in idx:
            result = {}
            result['id'] = ids[i]
            result['R'] = Rs[i]
            result['t'] = ts[i]
            result['s'] = confidences[i]
            results.append(result)
        publishResults(results)

        lock = True  # prevent threads interrupt
        rgb_flag = False
        depth_flag = False
        lock = False

def publishResults(results):
    print(results)
    print('line for debug')

def getK(info, subscriber):
    #do processing here to get K_cam
    global K_cam
    K_cam = np.zeros(shape=(3,3))
    for i in range(9):
        K_cam[i % 3, i - (i % 3) * 3] = info.K[i]

    subscriber.unregister()

if __name__ == '__main__':
    rospy.init_node('linemod_detection')
    # get K from cam_info topic
    sub_once = None
    sub_once = rospy.Subscriber('color/camera_info', CameraInfo, getK, sub_once)
    while not K_cam:
        pass

    rospy.Subscriber('color/image_raw', Image, listenRGB)
    rospy.Subscriber('depth/image_raw', Image, listenD)

    while True:
        receiveRGBD()

    rospy.spin()