#FROM python:3.8.12
FROM rayproject/ray:1.13.0-py38
#FROM rayproject/ray:latest-gpu

RUN pip install ray[rllib] torch pettingzoo pytest gym tqdm pandas
#RUN pip install ray==1.6.0 ray[rllib] torch==1.4.0 pettingzoo pytest gym==0.21.0 protobuf==3.19.4 tqdm pandas
#pandas==0.22.0
WORKDIR /work
COPY . /work
#RUN pip install -r environment.txt
RUN pip install -e .
