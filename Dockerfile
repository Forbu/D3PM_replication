FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# install networkx for graph generation
RUN pip3 install poetry pytest
RUN pip3 install -U 'tensorboardX'
RUN pip3 install lightning einops