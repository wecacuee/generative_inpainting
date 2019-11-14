import numpy as np
from PIL import Image, ImageChops
import yaml
from collections import deque
import os
import logging
logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

import rosbag
import tf2_ros
import rospy

from pyquaternion import Quaternion
import cv2


def truncate(arr, min_, max_):
    return np.maximum(
        np.minimum(arr, max_),
        min_)

def transform_rotation_np_rotation(rotation):
    quat = Quaternion(**{k: getattr(rotation, k) for k in "x y z w".split()})
    return quat.rotation_matrix

def translation_transform_to_numpy(translation):
    return np.array([translation.x, translation.y, translation.z])

def shift_image_to_robot_as_center(image, R, t, cropsize=500):
    base_rot_real = - Quaternion(matrix=R).degrees
    sim_occgrid = image
    sim_occgrid = ImageChops.offset(
        sim_occgrid,
        t[0],
        t[1]
    )
    sim_occgrid = sim_occgrid.rotate(base_rot_real)
    sim_occgrid_cropped = sim_occgrid.crop((
        sim_occgrid.size[0]/2-cropsize,
        sim_occgrid.size[1]/2-cropsize,
        sim_occgrid.size[0]/2+cropsize,
        sim_occgrid.size[1]/2+cropsize
        ))
    return sim_occgrid_cropped


def main(bagfile, out='maps/{i:04d}_{tag}.png', step=1, use_tf=False,
         trueyaml='mapfinal_01.yaml',
         generatedyaml='mapsim_01.yaml',
         truemap='mapfinal_01.pgm',
         diff_map_lag=1,
         min_diff_mask=0.02,
         out_shape=[256, 256]):
    if use_tf:
        bag = rosbag.Bag(bagfile)
        buffer = tf2_ros.BufferCore(rospy.Duration(10000))
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics=['/tf'])):
            for tfm in msg.transforms:
                buffer.set_transform(tfm, 'bag')

    bag = rosbag.Bag(bagfile)
    i = 0
    queue = deque(maxlen=diff_map_lag)
    for topic, msg, t in bag.read_messages(
            topics=['/robot0/gmapping_map', '/robot0/sim_odom', '/robot0/odom']):
        if topic == '/robot0/sim_odom':
            last_sim_odom, last_sim_odom_time = msg, t
            continue
        elif topic == '/robot0/odom':
            last_odom, last_odom_time = msg, t
            continue
        else:
            i = i + 1

        # print("Odom Time diff: {}".format(np.abs((t - last_odom_time).secs)))
        # print("Sim Odom Time diff: {}".format(np.abs((t - last_sim_odom_time).secs)))
        # print("Odom Time diff (nsec): {}".format(np.abs((t - last_odom_time).nsecs) / 1e9))
        # print("Sim Odom Time diff (nsec): {}".format(np.abs((t - last_sim_odom_time).nsecs) / 1e9))

        if use_tf:
            try:
                base_T_simmap = buffer.lookup_transform_core('sim_map', 'robot0/sim_base', t)
                base_T_realmap = buffer.lookup_transform_core('robot0/map', 'robot0/base', t)
            except tf2_ros.ExtrapolationException as e:
                print(e)
                continue

        if use_tf:
            base_R_simmap = transform_rotation_np_rotation(base_T_simmap.transform.rotation)
            base_R_realmap = transform_rotation_np_rotation(base_T_realmap.transform.rotation)
        else:
            base_R_simmap = transform_rotation_np_rotation(last_sim_odom.pose.pose.orientation)
            base_R_realmap = transform_rotation_np_rotation(last_odom.pose.pose.orientation)

        sim_R_real = base_R_simmap.T.dot(base_R_realmap)

        if use_tf:
            base_t_sim = translation_transform_to_numpy(base_T_simmap.transform.translation)
            base_t_real = translation_transform_to_numpy(base_T_realmap.transform.translation)
        else:
            base_t_sim = translation_transform_to_numpy(last_sim_odom.pose.pose.position)
            base_t_real = translation_transform_to_numpy(last_odom.pose.pose.position)
        sim_t_real = base_R_simmap.T.dot(base_t_real - base_t_sim)

        if i % step != 0:
            continue

        npmsg = np.array(msg.data)
        sim_params = yaml.safe_load(open(generatedyaml))
        Dx, Dy = sim_params['dims']
        occgrid = npmsg.reshape(Dx, Dy)
        #xindices, yindices = np.mgrid[0:Dy, 0:Dy]
        #xyindx = np.concatenate((xindices[..., np.newaxis], yindices[..., np.newaxis]), axis=-1)
        #sim_xyidx = sim_R_real[:2, :2].dot(xyindx.T.reshape(2, -1)).T.reshape(Dx, Dx, 2) + sim_t_real[:2]


        # Reducing the amount of prediction needed. If we have too much to
        # predict then the inpainting algorithms fail
        mask = occgrid == -1
        if len(queue) > 0:
            # During the test scenarios, old mask represents what is known
            # and the current map represents future map to be predicted. Thus
            # diff_mask = (mask - old_mask) represents the area which is to be
            # predicted and is unknown. However, mask will not be known exactly,
            # so the diff_mask >= (mask - oldmask). We acommplish this by
            # diff_mask = (expand_mask - old_mask). We cannot diminish old_mask
            # because that is already known. We can only be more ambitious and
            # ask inpainting model to predict expanded masks.
            old_mask = queue[0].astype(np.bool) # [0, 1] # bool
            kernel = np.ones((int(0.01 * Dx),int(0.01 * Dy)),np.uint8)
            expand_mask = cv2.erode(mask.astype(np.uint8), kernel)
            diff_mask = old_mask & (~expand_mask)
        else:
            diff_mask = ~mask
        if np.sum(diff_mask) < min_diff_mask * (Dx*Dy):
            LOG.info("""Diff mask too small. Try increasing
            diff_map_lag={diff_map_lag} or min_diff_mask={min_diff_mask}.
            Diff fraction={diff_fraction}""".format(
                diff_map_lag=diff_map_lag, min_diff_mask=min_diff_mask,
                diff_fraction=np.sum(diff_mask)/(Dx*Dy)))
            queue.append(mask)
            continue

        occgrid[mask] = occgrid.max() / 2
        occgrid = occgrid * 255 / 100
        occgrid = occgrid.astype(np.uint8)
        sim_occgrid = occgrid.copy()
        sim_occgrid_cropped = shift_image_to_robot_as_center(
            Image.fromarray(sim_occgrid[::-1, :]),
            R = base_R_realmap,
            t = (int(-base_t_real[0] / sim_params['resolution']),
                 int(base_t_real[1] / sim_params['resolution'])))

        sim_occgrid_cropped_bgr = cv2.cvtColor(
            np.asarray(sim_occgrid_cropped), cv2.COLOR_GRAY2BGR)
        imsize = sim_occgrid_cropped_bgr.shape[:2]
        #cv2.arrowedLine(sim_occgrid_cropped_bgr,
        #                (-10 + imsize[1]//2, imsize[0]//2),
        #                (10+imsize[1]//2, imsize[0]//2),
        #                (255, 0, 0), thickness=3)
        estimated_outfile = out.format(i=i, tag='estimated')
        outdir = os.path.dirname(estimated_outfile)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        LOG.info("Writing file {}".format(estimated_outfile))
        sim_occgrid_cropped_bgr = cv2.resize(sim_occgrid_cropped_bgr,
                                             tuple(out_shape))
        cv2.imwrite(estimated_outfile, sim_occgrid_cropped_bgr)

        diff_mask = (diff_mask * 255).astype(np.uint8)
        mask_cropped = shift_image_to_robot_as_center(
            Image.fromarray(diff_mask[::-1, :]),
            R = base_R_realmap,
            t = (int(-base_t_real[0] / sim_params['resolution']),
                 int(base_t_real[1] / sim_params['resolution'])))
        mask_cropped = cv2.cvtColor(np.asarray(mask_cropped),
                                    cv2.COLOR_GRAY2BGR)
        mask_cropped = cv2.resize(mask_cropped, tuple(out_shape))
        cv2.imwrite(out.format(i=i, tag='mask'), mask_cropped)

        true_params = yaml.safe_load(open(trueyaml))
        true_occgrid_img = Image.open(truemap)
        true_occgrid = np.array(true_occgrid_img)
        true_occgrid_img_scaled = true_occgrid_img.resize((
            int(true_occgrid.shape[1]*true_params['resolution']/sim_params['resolution']),
            int(true_occgrid.shape[0]*true_params['resolution']/sim_params['resolution'])))
        true_occgrid_img_sim_sized = np.zeros_like(sim_occgrid)
        true_origin_in_sim_pixels = np.array(true_params['origin'])/sim_params['resolution']
        true_map_image = Image.fromarray(true_occgrid_img_sim_sized)
        true_map_image.paste(
            true_occgrid_img_scaled,
            box=(int(sim_occgrid.shape[0]/2+ true_origin_in_sim_pixels[0]),
                 int(sim_occgrid.shape[1]/2
                     -true_occgrid_img_scaled.size[1]
                     -true_origin_in_sim_pixels[1])))

        true_map_image_cropped = shift_image_to_robot_as_center(
            true_map_image,
            R = base_R_realmap,
            t = (int(-base_t_real[0] / sim_params['resolution']),
                 int(base_t_real[1] / sim_params['resolution'])))

        #true_map_image = np.asarray(true_map_image) * 0.8 + np.asarray(sim_occgrid) * 0.2
        true_map_image_bgr = cv2.cvtColor(np.asarray(true_map_image_cropped), cv2.COLOR_GRAY2BGR)
        imsize = true_map_image_bgr.shape[:2]
        #cv2.arrowedLine(true_map_image_bgr,
        #                (-10 + imsize[1]//2, imsize[0]//2),
        #                (10+imsize[1]//2, imsize[0]//2),
        #                (255, 0, 0), thickness=3)

        true_map_image_bgr = cv2.resize(true_map_image_bgr, tuple(out_shape))
        cv2.imwrite(out.format(i=i, tag='true'), true_map_image_bgr)
        #sim_occgrid[
        #    truncate(sim_xyidx[..., 1].astype(np.int32), 0, Dy-1),
        #    truncate(sim_xyidx[..., 0].astype(np.int32), 0, Dx-1)] = occgrid

        #Image.fromarray(sim_occgrid[::-1, :]).save(out.format(i=i, tag='shifted'))

        #Image.fromarray(occgrid[::-1, :]).save(out.format(i=i, tag='map'))
        queue.append(mask)



if __name__ == '__main__':
    import sys
    properties = yaml.safe_load(open(sys.argv[1]))
    main(**properties)
