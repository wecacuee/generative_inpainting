#!/usr/bin/env python3

import numpy as np
import cv2
import rospy
from nav_msgs.msg import OccupancyGrid

from generative_inpainting.test import GLOBAL_FILL_INPAINTING


def predict_map(msg, erode_frac=0.04, unknown_prob_range=(0.45, 0.55)):
    npmsg = np.array(msg.data)
    orig_range = npmsg[npmsg >= 0].min(), npmsg.max()
    Dx, Dy, resolution = msg.info.width, msg.info.height, msg.info.resolution
    occgrid = npmsg.reshape(Dx, Dy)
    unknown = occgrid == -1
    occgrid[unknown] = occgrid.max() / 2
    occgrid = occgrid * 255 / occgrid.max()
    occgrid = occgrid.astype(np.uint8)
    occgrid = cv2.cvtColor(occgrid, cv2.COLOR_GRAY2BGR)

    kernel = np.ones((int(erode_frac * Dx),int(erode_frac * Dy)),np.uint8)
    expand_mask = cv2.erode(unknown.astype(np.uint8), kernel)
    diff_mask = unknown & (~expand_mask)
    diff_mask = (diff_mask * 255).astype(np.uint8)
    diff_mask = cv2.cvtColor(diff_mask, cv2.COLOR_GRAY2BGR)
    predicted_img = GLOBAL_FILL_INPAINTING.predict(occgrid, diff_mask)
    predicted = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2GRAY)
    predicted = predicted.astype(np.float64) / 255.
    assert unknown_prob_range[0] < unknown_prob_range[1]
    unknown_predicted = (unknown_prob_range[0] < predicted) & (
        predicted < unknown_prob_range[1])
    unknown_predicted &= unknown
    #return predicted, unknown_predicted

    predicted = orig_range[0] + predicted * ( orig_range[1] - orig_range[0] ) / (predicted.max() - predicted.min())
    predicted = predicted.astype(np.int8)
    predicted[unknown_predicted] = -1

    pred_occgrid = OccupancyGrid()
    pred_occgrid.header.stamp = rospy.Time.now()
    pred_occgrid.info = msg.info
    pred_occgrid.data = predicted
    return pred_occgrid


def main():
    import sys
    rospy.init_node('map_predictor')
    pub = rospy.Publisher('predicted_map', OccupancyGrid, queue_size=10)

    while not rospy.is_shutdown():
        msg = rospy.wait_for_message('point_cloud_cache/renderers/full_map', OccupancyGrid)
        pred_occgrid = predict_map(msg)
        pub.publish(pred_occgrid)


if __name__ == '__main__':
    main()
