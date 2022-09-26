FROM rayproject/ray:1.13.0-py38

RUN pip install ray[rllib] torch==1.10.2 pettingzoo==1.17.0 pytest gym==0.21.0 tqdm pandas
WORKDIR /work
COPY . /work
RUN pip install -e .
