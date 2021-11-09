#!/bin/bash

build=0
if [ $# -gt 0 ]; then
	build=1
fi

cd util
if [ $build -eq 1 ] || [ "$(docker images | grep cutlass)" == "" ]; then
	docker build -t cutlass -f Dockerfile .
else
	echo "Not building new container"
fi

docker run -it --rm --gpus all --privileged --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ubuntu/:/home cutlass 
