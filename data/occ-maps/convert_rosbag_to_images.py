import numpy as np
from PIL import Image, ImageChops
import yaml

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


def main(bagfile, out='maps/{tag}_{i}.png', step=1, use_tf=False):
    bag = rosbag.Bag(bagfile)
    if use_tf:
        buffer = tf2_ros.BufferCore(rospy.Duration(10000))
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics=['/tf'])):
            for tfm in msg.transforms:
                buffer.set_transform(tfm, 'bag')

    bag = rosbag.Bag(bagfile)
    i = 0
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

        print("Odom Time diff: {}".format(np.abs((t - last_odom_time).secs)))
        print("Sim Odom Time diff: {}".format(np.abs((t - last_sim_odom_time).secs)))
        print("Odom Time diff (nsec): {}".format(np.abs((t - last_odom_time).nsecs) / 1e9))
        print("Sim Odom Time diff (nsec): {}".format(np.abs((t - last_sim_odom_time).nsecs) / 1e9))

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
        Dx, Dy = 4000, 4000
        occgrid = npmsg.reshape(Dx, Dy)
        #xindices, yindices = np.mgrid[0:Dy, 0:Dy]
        #xyindx = np.concatenate((xindices[..., np.newaxis], yindices[..., np.newaxis]), axis=-1)
        #sim_xyidx = sim_R_real[:2, :2].dot(xyindx.T.reshape(2, -1)).T.reshape(Dx, Dx, 2) + sim_t_real[:2]


        mask = occgrid == -1
        occgrid[mask] = occgrid.max() / 2
        occgrid = occgrid * 255 / 100
        occgrid = occgrid.astype(np.uint8)
        sim_occgrid = occgrid.copy()
        base_rot_real = - Quaternion(matrix=base_R_realmap).degrees
        sim_params = yaml.safe_load(open('mapsim_01.yaml'))
        sim_occgrid = Image.fromarray(sim_occgrid[::-1, :])
        sim_occgrid = ImageChops.offset(
            sim_occgrid,
            int(-base_t_real[0] / sim_params['resolution']),
            int(base_t_real[1] / sim_params['resolution'])
        )
        sim_occgrid = sim_occgrid.rotate(base_rot_real)
        sim_occgrid_cropped = sim_occgrid.crop((
            sim_occgrid.size[0]/2-500,
            sim_occgrid.size[1]/2-500,
            sim_occgrid.size[0]/2+500,
            sim_occgrid.size[1]/2+500
            ))
        sim_occgrid_cropped_bgr = cv2.cvtColor(np.asarray(sim_occgrid_cropped), cv2.COLOR_GRAY2BGR)
        imsize = sim_occgrid_cropped_bgr.shape[:2]
        cv2.arrowedLine(sim_occgrid_cropped_bgr,
                        (-10 + imsize[1]//2, imsize[0]//2),
                        (10+imsize[1]//2, imsize[0]//2),
                        (255, 0, 0), thickness=3)
        cv2.imwrite(out.format(i=i, tag='shifted'), sim_occgrid_cropped_bgr)
        mask = mask[::-1, :]
        true_params = yaml.safe_load(open('mapfinal_01.yaml'))
        true_occgrid_img = Image.open('maps/mapfinal_01.pgm')
        true_occgrid = np.array(true_occgrid_img)
        true_occgrid_img_scaled = true_occgrid_img.resize((
            int(true_occgrid.shape[1]*true_params['resolution']/sim_params['resolution']),
            int(true_occgrid.shape[0]*true_params['resolution']/sim_params['resolution'])))
        true_occgrid_img_sim_sized = np.zeros_like(sim_occgrid)
        true_origin_in_sim_pixels = np.array(true_params['origin'])/sim_params['resolution']
        true_map_image = Image.fromarray(true_occgrid_img_sim_sized)
        true_map_image.paste(
            true_occgrid_img_scaled,
            box=(int(sim_occgrid.size[0]/2+ true_origin_in_sim_pixels[0]),
                 int(sim_occgrid.size[1]/2
                     -true_occgrid_img_scaled.size[1]
                     -true_origin_in_sim_pixels[1])))

        true_map_image = ImageChops.offset(
            true_map_image,
            int(-base_t_real[0] / sim_params['resolution']),
            int(base_t_real[1] / sim_params['resolution'])
        )
        true_map_image = true_map_image.rotate(base_rot_real)
        true_map_image_cropped = true_map_image.crop((
            true_map_image.size[0]/2-500,
            true_map_image.size[1]/2-500,
            true_map_image.size[0]/2+500,
            true_map_image.size[1]/2+500
            ))
        #true_map_image = np.asarray(true_map_image) * 0.8 + np.asarray(sim_occgrid) * 0.2
        true_map_image_bgr = cv2.cvtColor(np.asarray(true_map_image_cropped), cv2.COLOR_GRAY2BGR)
        imsize = true_map_image_bgr.shape[:2]
        cv2.arrowedLine(true_map_image_bgr,
                        (-10 + imsize[1]//2, imsize[0]//2),
                        (10+imsize[1]//2, imsize[0]//2),
                        (255, 0, 0), thickness=3)
        cv2.imwrite(out.format(i=i, tag='true'), true_map_image_bgr)
        #sim_occgrid[
        #    truncate(sim_xyidx[..., 1].astype(np.int32), 0, Dy-1),
        #    truncate(sim_xyidx[..., 0].astype(np.int32), 0, Dx-1)] = occgrid

        #Image.fromarray(sim_occgrid[::-1, :]).save(out.format(i=i, tag='shifted'))

        Image.fromarray(occgrid[::-1, :]).save(out.format(i=i, tag='map'))
        mask = (mask * 255).astype(np.uint8)
        Image.fromarray(mask).save(out.format(i=i, tag='mask'))



if __name__ == '__main__':
    import sys
    main(sys.argv[1])
