FROM gitlab-registry.nautilus.optiputer.net/prp/jupyterlab:latest

ENV DATA_ROOT $HOME/dat/generative_inpainting
ENV CODE_DIR $HOME/wrk/generative_inpainting
RUN mkdir -p $(dirname $CODE_DIR) \
    && git clone https://github.com/wecacuee/generative_inpainting.git $CODE_DIR
RUN cd $CODE_DIR && bash setup_once.bash
WORKDIR $CODE_DIR
CMD ["python", "train.py"]

