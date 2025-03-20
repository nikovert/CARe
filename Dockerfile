# Use an official Ubuntu runtime as a parent image
FROM ubuntu:22.04

# Set the working directory in the container
RUN mkdir /care
COPY . /care
WORKDIR /care

# Install prerequisites and build dReal
ARG DEBIAN_FRONTEND=noninteractive

# Update and install required packages in a single step
RUN apt-get update \
    && apt-get install -y python3 python3-pip curl nano \
    && curl -fsSL 'https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/22.04/install_prereqs.sh' | bash \
    && apt-get remove -y bazel bison flex g++ wget \
    && apt-get autoclean -y \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache/bazel

# Install CARe
RUN cd /care && pip3 install -e .