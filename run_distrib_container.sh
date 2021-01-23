#! /bin/bash

echo Please enter directory path to data or leave blank to use default:

read datadir

if [ -z "$datadir" ]
then
	datadir="/media/jonathan/jonathans extern/data/imagenet/"
fi

sudo docker run \
	-u $(id -u):$(id -g) \
	--mount type=bind,source="$(pwd)",target=/home/ResNet \
	--mount type=bind,source="$datadir",target=/home/ResNet/data\
       	--gpus all \
       	-it \
	resnet_horovod:latest \
	bash \
