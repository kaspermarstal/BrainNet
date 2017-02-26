#!/bin/bash

if [ $1="--create-cpu-instance" ]; then
  gcloud beta compute instances create brainnet-cpu \
    --zone europe-west1-b \
    --machine-type n1-highcpu-16 \
    --image-project ubuntu-os-cloud \
    --image-family ubuntu-1604-lts \
    --boot-disk-device-name=brainnet \
    --boot-disk-type=pd-standard \
    --boot-disk-size=64GB \
    --maintenance-policy TERMINATE --restart-on-failure
fi

if [ $1="--create-gpu-instance" ]; then
  gcloud beta compute instances create brainnet-gpu \
    --zone europe-west1-b \
    --machine-type n1-highmem-2 \
    --image-project ubuntu-os-cloud \
    --image-family ubuntu-1604-lts \
    --boot-disk-device-name=brainnet-gpu \
    --boot-disk-type=pd-standard \
    --boot-disk-size=64GB \
    --accelerator type=nvidia-tesla-k80,count=1 \
    --maintenance-policy TERMINATE --restart-on-failure \
    --metadata startup-script='#!/bin/bash
    echo "Checking for CUDA and installing."
    if ! dpkg-query -W cuda; then
      curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
      dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
      apt-get update
      apt-get install cuda -y
    fi'
fi
