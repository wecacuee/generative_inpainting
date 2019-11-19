import os.path as osp
import os
import subprocess

def download(target_dir):
    dirtomake = osp.dirname(target_dir)
    if not osp.exists(dirtomake):
        os.makedirs(dirtomake)
    subprocess.check_call(
        """wget https://www.dropbox.com/s/lz19pn97xc5f0ko/generative-inpainting-radish-model.tgz?dl=0 -O - | tar -C {target_dir} -xvf -""".format(target_dir=target_dir), shell=True)

if __name__ == '__main__':
    import sys
    download(sys.argv[1])
