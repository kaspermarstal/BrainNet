#!/bin/bash

if [[ $1 == "--run" ]]; then
  source ~/venv/bin/activate
  python ~/brainnet/train_2d.py
fi

if [[ $1 == "--clone-source" ]]; then
  git clone https://github.com/kaspermarstal/brainnet ~/brainnet
  exit
fi

if [[ $1 == "--download-data" ]]; then
  if [[ -d ~/downloads ]]; then
    rm -fr ~/downloads
    mkdir ~/downloads
  fi
  cd ~/downloads
  curl -OL ftp://ftp.nrg.wustl.edu/data/oasis_cross-sectional_disc{1,2,3,4,5,6,7,8,9,10,11,12}.tar.gz
fi

if [[ $1 == "--extract-data" ]]; then
  if [[ -d ~/data ]]; then
    rm -fr ~/data
    mkdir ~/data
  fi
  for filename in ~/downloads/*.tar.gz
  do
    tar zxf $filename -C ~/data
  done
fi

if [[ $1 == "--install-venv" ]]; then
  if [[ -d ~/venv ]]; then
    rm -fr ~/venv
  fi
  sudo apt-get install -y virtualenv gcc python-dev
  virtualenv ~/venv
  source ~/venv/bin/activate
  pip install keras SimpleITK numpy sklearn scikit-image
fi

if [[ $1 == "--install-tensorflow-cpu" ]]; then
  source ~/venv/bin/activate
  pip install tensorflow
fi

if [[ $1 == "--install-tensorflow-gpu" ]]; then
  source ~/venv/bin/activate
  pip install tensorflow-gpu
fi
