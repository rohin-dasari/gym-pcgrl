FROM rayproject/ray:1.6.0-py38-gpu
#FROM rayproject/ray:latest-gpu

RUN pip install torch pettingzoo pytest gym==0.21.0
WORKDIR /work
COPY . /work
RUN pip install -e .

