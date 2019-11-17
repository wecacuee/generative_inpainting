import argparse

import cv2

from generative_inpainting.test import FillInpainting

parser = argparse.ArgumentParser()
parser.add_argument('--image', default='', type=str,
                    help='The filename of image to be completed.')
parser.add_argument('--mask', default='', type=str,
                    help='The filename of mask, value 255 indicates mask.')
parser.add_argument('--output', default='output.png', type=str,
                    help='Where to write output.')
parser.add_argument('--checkpoint_dir', default='', type=str,
                    help='The directory of tensorflow checkpoint.')

if __name__ == "__main__":
    # ng.get_gpus(1)
    args, unknown = parser.parse_known_args()

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError('File not found: {}'.format(args.image))
    mask = cv2.imread(args.mask)
    if mask is None:
        raise ValueError('File not found: {}'.format(args.mask))
    # mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)

    fill_inp = FillInpainting(args.checkpoint_dir)
    result = fill_inp.predict(image, mask)
    cv2.imwrite(args.output, result[:, :, ::-1])
