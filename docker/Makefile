# Development Environment Construction
#
# 2022/5/24 Yuma Kinoshita
# 2021/3/15 Yuma Kinoshita
# 2019/8/4 Yuma Kinoshita
# 2017/3/6 Yousuke Sugawara

USER_NAME = $(USER)
GROUP_NAME = Domain
HOST_PATH = '/home/$(USER_NAME)/CentaurVAE'
CONTAINER_PATH = '/home/$(USER_NAME)/CentaurVAE'
ROOT_PASSWORD = 'pass'
IMAGE_TAG = 'syamaji-centaurvae'

all: build run

build:
	docker build -t $(IMAGE_TAG) ./ \
	--build-arg ROOT_PASSWORD=$(ROOT_PASSWORD) \

#-it --rm --gpus all \
run:
	docker run \
	-it --rm \
	-e USER_NAME=$(USER_NAME) -e GROUP_NAME=$(GROUP_NAME) \
	-e LOCAL_UID=$(shell id -u $(USER)) -e LOCAL_GID=$(shell id -g $(USER)) \
	-v $(HOST_PATH):$(CONTAINER_PATH) \
	$(IMAGE_TAG) /bin/bash


