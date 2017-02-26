#!/bin/bash

if [[ $1 == "--download-data" ]]; then
  if [[ -d ~/BrainNet/downloads ]]; then
    rm -fr ~/BrainNet/downloads
  fi
  mkdir -p ~/BrainNet/downloads
  cd ~/BrainNet/downloads
  curl -OL ftp://ftp.nrg.wustl.edu/data/oasis_cross-sectional_disc{1,2,3,4,5,6,7,8,9,10,11}.tar.gz
fi

if [[ $1 == "--extract-data" ]]; then
  if [[ -d ~/BrainNet/data ]]; then
    rm -fr ~/BrainNet/data
  fi
  mkdir -p ~/BrainNet/data
  for filename in ~/downloads/*.tar.gz
  do
    tar zxf $filename -C ~/data
  done
fi

if [[ $1 == "--install-venv" ]]; then
  if [[ -d ~/BrainNet/venv ]]; then
    rm -fr ~/BrainNet/venv
  fi
  sudo apt-get install -y virtualenv gcc python-dev
  virtualenv ~/BrainNet/venv
  source ~/BrainNet/venv/bin/activate
  pip install keras SimpleITK numpy sklearn scikit-image tensorflow
fi

if [[ $1 == "--install-tensorflow-gpu" ]]; then
  source ~/BrainNet/venv/bin/activate
  pip install --upgrade tensorflow-gpu
fi

if [[ $1 == "--run-inception-v4" ]]; then
  source ~/BrainNet/venv/bin/activate
  python ~/BrainNet/BrainNet/train_inception_v4.py --data-dir=$HOME/data
fi

if [[ $1 == "--run-inception-resnet-v2" ]]; then
  source ~/BrainNet/venv/bin/activate
  python ~/BrainNet/BrainNet/train_inception_resnet_v2.py --data-dir=$HOME/BrainNet/data
fi

if [ -z $1 ]; then
   ./experiment.sh --download-data
   ./experiment.sh --extract-data
   ./experiment.sh --install-venv
   ./experiment.sh --install-tensorflow-gpu || true
   ./experiment.sh --run-inception-v4
fi
