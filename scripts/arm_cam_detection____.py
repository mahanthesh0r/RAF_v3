#!/usr/bin/env python3

# This code runs the detectron2 maskRCNN model inference on a live video stream.

# Author: Jack Schultz
# Created 1/26/2023

import numpy as np
import rospy, os, sys, cv2

import torch
from matplotlib import pyplot as plt
import cv2
from PIL import Image
from pathlib import Path

from raf.msg import DetectionList, RafState
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge

class maskRCNN(object):
    def __init__(self):
        # Params
        self.br = CvBridge()
        self.image = None
        self.raf_state = None

        # Node cycle rate (in Hz).
        self.loop_rate = rospy.Rate(15)

        # Subscribers
        rospy.Subscriber("/camera/color/image_raw", Image, self.callback)
        rospy.Subscriber("/raf_state", RafState, self.state_callback)

        # Publishers
        self.pub = rospy.Publisher('arm_camera_detections', DetectionList, queue_size=10)

    def callback(self, msg):
        self.image = self.convert_to_cv_image(msg)
        self._header = msg.header

    def state_callback(self, msg):
        self.raf_state = msg

    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        return cv_img

    def get_img(self):
        result = self.image
        return result

    def build_detection_msg(self, predictions, classes):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            #print(type(masks))
        else:
            return

        result_msg = DetectionList()
        result_msg.header = self._header
        result_msg.class_ids = predictions.pred_classes if predictions.has("pred_classes") else None
        result_msg.class_names = np.array(classes)[result_msg.class_ids.numpy()]
        result_msg.scores = predictions.scores if predictions.has("scores") else None

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self.br.cv2_to_imgmsg(mask)
            result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = np.uint32(x1)
            box.y_offset = np.uint32(y1)
            box.height = np.uint32(y2 - y1)
            box.width = np.uint32(x2 - x1)
            result_msg.boxes.append(box)

        return result_msg

    def publish(self, detection_msg):
        self.pub.publish(detection_msg)
        self.loop_rate.sleep()

def main():
    rospy.init_node("arm_cam_detections", anonymous=True)

    run = maskRCNN()
    model_path = Path("/home/labuser/Documents/example_scripts/custom_yolo/yolov5s.pt")
    model = torch.hub.load('ultralytics/yolov5','custom', path=model_path, force_reload=True)
    #model = torch.hub.load('ultralytics/yolov5','yolov5s')
    img = run.get_img()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img, size=328)

    

    class_names = ['bowl','carrot','celery','cup','fork','gripper','knife','plate','pretzel','spoon']

    
    rospy.loginfo("Running Inference...")

    boxes = results.pandas().xyxy[0]  # img1 predictions (pandas)
    predictions = {
        "pred_boxes": [],
        "pred_classes": [],
        "scores": [],
        "pred_class_name" : []
    }

    for index, row in boxes.iterrows():
        x1, y1, x2, y2, confidence, class_id, name = row

        predictions['pred_boxes'].append((x1,y1,x2,y2))
        if class_id not in predictions['pred_classes']:
            predictions['pred_classes'].append(class_id)
            predictions['pred_class_name'].append(name)
        predictions['scores'].append(confidence)

    while not rospy.is_shutdown():

        if img is None:
            continue

        if run.raf_state is None:
            print(predictions)
            continue

        if run.raf_state.enable_arm_detections:
            rospy.loginfo_once("Running arm detection inference...")
            detection_msg = run.build_detection_msg(predictions, class_names)
            print(detection_msg)
            run.publish(detection_msg)
        else:
            detection_msg = DetectionList()
            run.publish(detection_msg)

if __name__ == '__main__':
    sys.exit(main())