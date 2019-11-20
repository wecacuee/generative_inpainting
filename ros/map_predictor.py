#!/usr/bin/env python3
from collections import deque

import numpy as np
import cv2
import rospy
from nav_msgs.msg import OccupancyGrid
from PIL import Image, ImageChops
from pyquaternion import Quaternion
from omnimapper_msgs.msg import PoseGraph

from generative_inpainting.test import GLOBAL_FILL_INPAINTING


class RobotPosition:
    def __init__(self, topic='pose_graph'):
        self.queue = deque(maxlen=100)
        self.sub = rospy.Subscriber(topic, PoseGraph, self.queue.append)

    def get_position(self, timestamp):
        msg = min(self.queue, key=lambda x: abs(x.nodes[-1].stamp - timestamp))
        return np.array(
                [msg.nodes[-1].pose.position.x, 
                    msg.nodes[-1].pose.position.y])

def shift_image_to_robot_as_center(image, R, t, cropsize=500):
    base_rot_real = - Quaternion(matrix=R).degrees
    sim_occgrid = image
    sim_occgrid = ImageChops.offset(
        sim_occgrid,
        int(t[0]),
        int(t[1])
    )
    sim_occgrid = sim_occgrid.rotate(base_rot_real)
    sim_occgrid_cropped = sim_occgrid.crop((
        sim_occgrid.size[0]/2-cropsize,
        sim_occgrid.size[1]/2-cropsize,
        sim_occgrid.size[0]/2+cropsize,
        sim_occgrid.size[1]/2+cropsize
        ))
    return sim_occgrid_cropped


def predict_map(msg, robot_pose, 
        erode_frac=0.04, unknown_prob_range=(0.45, 0.55),
        unknown_val=-1,
        cropsize=500):
    npmsg = np.array(msg.data)
    orig_range = npmsg[npmsg >= 0].min(), npmsg.max()
    Dx, Dy, resolution = msg.info.width, msg.info.height, msg.info.resolution
    occgrid = npmsg.reshape(Dx, Dy)
    unknown = occgrid == unknown_val
    print("% unknown pixels ", np.sum(unknown) / np.prod(unknown.shape))
    occgrid[unknown] = occgrid.max() / 2
    occgrid = occgrid * 255 / occgrid.max()
    occgrid_gray = occgrid.astype(np.uint8)
    occgrid = cv2.cvtColor(occgrid_gray, cv2.COLOR_GRAY2BGR)

    r_pose = robot_pose.get_position(msg.header.stamp)
    map_origin = np.array([msg.info.origin.position.x, msg.info.origin.position.y])
    r_pose_px = (r_pose - map_origin)  / msg.info.resolution

    occgrid_shifted = np.asarray(
            shift_image_to_robot_as_center( 
                Image.fromarray(occgrid), np.eye(3), r_pose_px, cropsize) )

    kernel = np.ones((int(erode_frac * Dx),int(erode_frac * Dy)),np.uint8)
    expand_mask = cv2.erode(unknown.astype(np.uint8), kernel)
    diff_mask = unknown & (~expand_mask)
    print("% diff_mask pixels ", np.sum(diff_mask) / np.prod(unknown.shape))
    diff_mask = (diff_mask * 255).astype(np.uint8)
    diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)

    diff_mask_shifted = np.asarray(
            shift_image_to_robot_as_center(
                Image.fromarray(diff_mask), np.eye(3), r_pose_px, cropsize) )
    unknown_shifted = np.asarray(
            shift_image_to_robot_as_center(
                Image.fromarray(unknown), np.eye(3), r_pose_px, cropsize) )

    predicted_img = GLOBAL_FILL_INPAINTING.predict(occgrid, diff_mask)
    assert predicted_img.shape == occgrid.shape
    predicted = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2GRAY)
    #predicted[~unknown] = occgrid_gray[~unknown]
    predicted = predicted.astype(np.float64) / 255.
    assert unknown_prob_range[0] < unknown_prob_range[1]
    unknown_predicted = (unknown_prob_range[0] < predicted) & (
        predicted < unknown_prob_range[1])
    unknown_predicted = unknown_predicted & unknown
    #return predicted, unknown_predicted

    predicted = orig_range[0] + predicted * ( orig_range[1] - orig_range[0] ) / (predicted.max() - predicted.min())
    predicted = predicted.astype(np.int8)
    predicted[unknown_predicted] = -1
    # do not show the known map
    predicted[~unknown] = -1

    pred_occgrid = OccupancyGrid()
    pred_occgrid.header.stamp = rospy.Time.now()
    pred_occgrid.info = msg.info
    pred_occgrid.info.height = predicted.shape[0]
    pred_occgrid.info.width = predicted.shape[1]
    #pred_occgrid.info.origin.position.x = - cropsize * msg.info.resolution - r_pose[0]
    #pred_occgrid.info.origin.position.y = - cropsize * msg.info.resolution - r_pose[1]
    pred_occgrid.data = predicted.ravel()
    return pred_occgrid


def main():
    import sys
    rospy.init_node('map_predictor')
    pub = rospy.Publisher('predicted_map', OccupancyGrid, queue_size=10)
    robot_pose = RobotPosition()
    while not rospy.is_shutdown():
        #msg = rospy.wait_for_message('point_cloud_cache/renderers/full_map', OccupancyGrid)
        msg = rospy.wait_for_message('gmapping_map', OccupancyGrid)
        pred_occgrid = predict_map(msg, robot_pose)
        pub.publish(pred_occgrid)


if __name__ == '__main__':
    main()
