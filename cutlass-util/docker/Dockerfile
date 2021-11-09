FROM nvidia/cuda:11.0-devel

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cmake
RUN apt install -y git
RUN apt-get install -y software-properties-common
RUN apt-get install -y python3.6
RUN apt-get install -y vim
RUN apt-get install -y screen
RUN apt install -y python3-pip
RUN pip3 install pandas
RUN apt-get install -y nsight-compute-2020.3.0
RUN ln -s /opt/nvidia/nsight-compute/2020.3.0/* /usr/local/bin/
