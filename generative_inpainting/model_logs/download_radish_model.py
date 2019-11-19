import os.path as osp
import os
import subprocess

def download(target_dir):
    dirtomake = osp.dirname(target_dir)
    if not osp.exists(dirtomake):
        os.makedirs(dirtomake)
    p1 = subprocess.Popen(
        """wget https://www.dropbox.com/s/lz19pn97xc5f0ko/generative-inpainting-radish-model.tgz?dl=0 -O -""".split(), stdout=subprocess.PIPE)
    p2 = subprocess.run("""tar -C {target_dir} xvf -""".format(target_dir=target_dir).split(),
                   stdin=p1.PIPE)
    p1.stdout.close()
    p2.communicate()

if __name__ == '__main__':
    download()
