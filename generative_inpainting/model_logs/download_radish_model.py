import subprocess
def download():
    subprocess.run(
        """wget https://www.dropbox.com/s/lz19pn97xc5f0ko/generative-inpainting-radish-model.tgz?dl=0 -O - | tar xvf -""".split())

if __name__ == '__main__':
    download()
