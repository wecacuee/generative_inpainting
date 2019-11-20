FROM ros-melodic-bionic-nvidia

ENV HOME /home/root
ENV DATA_ROOT $HOME/dat/generative_inpainting
ENV CODE_DIR $HOME/wrk/generative_inpainting
RUN apt-get update && apt-get install -y git python3-pip && rm -rf /var/lib/apt/lists/*
RUN mkdir -p $(dirname $CODE_DIR) \
    && git clone https://github.com/wecacuee/generative_inpainting.git $CODE_DIR
RUN cd $CODE_DIR && bash setup_once.bash
RUN pip3 install --no-cache -r $CODE_DIR/ros/requirements.txt
WORKDIR $CODE_DIR
CMD ["python3", "ros/map_predictor.py"]
