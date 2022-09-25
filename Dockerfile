FROM rayproject/ray:1.13.0-py38

RUN pip install ray[rllib] torch pettingzoo pytest gym tqdm pandas
WORKDIR /work
COPY . /work
RUN pip install -e .
