FROM pytorch/pytorch:nightly-runtime-cuda9.2-cudnn7

COPY . .

RUN pip install PyYAML --ignore-installed
RUN pip install polyaxon_helper
RUN pip install polyaxon_client
RUN pip install cloudpickle
RUN pip install gym
RUN pip install pybullet
RUN pip install tensorboardX
RUN apt-get update -y
RUN apt-get install ffmpeg -y
RUN apt install -y python3-dev zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb
RUN pip install gym[box2d]

CMD