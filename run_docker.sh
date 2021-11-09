#!/bin/bash

util_dir=./cutlass-util
$util_dir/set_smi.sh

build=0
cur_dir=$(pwd)
if [ $build -eq 1 ] || [ "$(docker images | grep cutlass)" == "" ]; then
  cd $util_dir/docker
	docker build -t cutlass -f Dockerfile .
  cd $cur_dir
else
	echo "Not building new container"
fi

docker run -it --rm --gpus all --privileged --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v $cur_dir:/home cutlass 
