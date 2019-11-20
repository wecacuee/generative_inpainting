import os.path as osp
import subprocess
import sys

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from .inpaint_model import InpaintCAModel
from .model_logs.download_radish_model import download


def abspath(relpath, relto=osp.dirname(__file__) or "."):
    return osp.join(relto, relpath)


def download_radish(
        generated_dir=osp.expanduser("~/.generative_inpainting/")):
    checkpoint_dir = osp.join(generated_dir, "radish")
    if not osp.exists(checkpoint_dir):
        download(generated_dir)
    return checkpoint_dir


class FillInpainting:
    """
    """
    def __init__(self,
                 checkpoint_dir=None,
                 get_checkpoint_dir=download_radish,
                 config=abspath('inpaint.yml'),
                 max_size=(1024, 1024)):
        self.FLAGS = ng.Config(config)
        self.max_size = max_size
        if checkpoint_dir is None:
            checkpoint_dir = get_checkpoint_dir()
        self.checkpoint_dir = checkpoint_dir
        self.model_loaded = False

    def load_model(self, image_shape):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.session = sess = tf.Session(config=sess_config)

        self.model = model = InpaintCAModel()
        self.input_image_ph = tf.placeholder(
            tf.float32, shape=(1, image_shape[0], image_shape[1]*2, 3))
        output = model.build_server_graph(self.FLAGS, self.input_image_ph)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        self.output = output = tf.saturate_cast(output, tf.uint8)
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(
                self.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        sess.run(assign_ops)
        self.model_loaded = image_shape
        print('Model loaded.')



    def predict(self, image, mask):
        FLAGS = self.FLAGS
        max_size= self.max_size
        assert image.shape == mask.shape

        h, w, _ = image.shape
        print('Raw Shape of image: {}'.format(image.shape), file=sys.stderr, flush=True)
        old_size = image.shape[:2]
        if np.prod(image.shape) > np.prod(max_size)*3:
            image = cv2.resize(image, max_size)
            mask = cv2.resize(mask, max_size)
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Grid Shape of image: {}'.format(image.shape), file=sys.stderr, flush=True)
        with tf.variable_scope("", reuse=tf.AUTO_REUSE):
            if not self.model_loaded or self.model_loaded != image.shape[:2]:
                self.load_model(image.shape[:2])

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        # load pretrained model
        result = self.session.run(self.output, feed_dict={self.input_image_ph: input_image})
        img = cv2.resize(result[0], (old_size[1], old_size[0]))
        assert img.shape[:2] == old_size
        return img

GLOBAL_FILL_INPAINTING = FillInpainting()

