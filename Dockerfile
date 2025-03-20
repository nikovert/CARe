# Use an official Ubuntu runtime as a parent image
FROM ubuntu:22.04

# Set the working directory in the container
RUN mkdir /care
COPY . /care
WORKDIR /care

# Install prerequisites and build dReal
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
      && apt-get install -y --no-install-recommends apt-utils python3-dev python3-wheel python3-setuptools python3-pip python-is-python3 git \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean \
      && git clone https://github.com/dreal/dreal4 && cd dreal4 \
      && ./setup/ubuntu/`lsb_release -r -s`/install_prereqs.sh \
      && apt-get install curl \
      && curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install.sh | bash \
      && pip3 install dreal \
      && cd /care \
      && pip3 install -e .


