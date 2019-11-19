import os.path as osp
import os
import subprocess

def download(target_dir):
    os.makedirs(osp.dirname(target_dir))
    subprocess.run(
        """wget https://www.dropbox.com/s/lz19pn97xc5f0ko/generative-inpainting-radish-model.tgz?dl=0 -O - | tar -C {target_dir} xvf -""".format(target_dir=target_dir).split())

if __name__ == '__main__':
    download()
