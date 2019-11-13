import numpy as np
import cv2

if __name__ == '__main__':
    import sys
    infile, maskfile, outfile = sys.argv[1:]
    inimg = cv2.imread(infile)
    maskimg = cv2.imread(maskfile)
    mask_01 = maskimg.astype(np.bool)
    outimg = inimg * (~mask_01).astype(np.uint8)
    cv2.imwrite(outfile, outimg)
