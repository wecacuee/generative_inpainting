import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from inpaint_model import InpaintCAModel

class FillInpainting:
    def __init__(self, checkpoint_dir, config='inpaint.yml'):
        self.FLAGS = ng.Config(config)
        self.checkpoint_dir = checkpoint_dir

    def predict(self, image, mask):
        FLAGS = self.FLAGS
        assert image.shape == mask.shape
        model = InpaintCAModel()

        h, w, _ = image.shape
        grid = 8
        image = image[:h//grid*grid, :w//grid*grid, :]
        mask = mask[:h//grid*grid, :w//grid*grid, :]
        print('Shape of image: {}'.format(image.shape))

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            input_image = tf.constant(input_image, dtype=tf.float32)
            output = model.build_server_graph(FLAGS, input_image)
            output = (output + 1.) * 127.5
            output = tf.reverse(output, [-1])
            output = tf.saturate_cast(output, tf.uint8)
            # load pretrained model
            vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            assign_ops = []
            for var in vars_list:
                vname = var.name
                from_name = vname
                var_value = tf.contrib.framework.load_variable(self.checkpoint_dir, from_name)
                assign_ops.append(tf.assign(var, var_value))
            sess.run(assign_ops)
            print('Model loaded.')
            result = sess.run(output)
            return result[0]


