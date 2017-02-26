if [ $1="--clone-source"]; then
  git clone https://github.com/kaspermarstal/brainnet ~/brainnet
fi

if [ $1="--download-data" ]; then
  if ![ -d "~/downloads" ]; then
    mkdir "~/downloads"
  fi
  curl -OL ftp://ftp.nrg.wustl.edu/data/oasis_cross-sectional_disc1.tar.gz -C download
fi

if [ $1="--extract-data"]; then
  if ![ -d "~/data" ]; then
    mkdir "~/data"
  fi
  tar -xcf ~/downloads/oasis_cross-sectional_disc1.tar.gz ~/data
fi

if [ $1="--setup-python-env"]; then
  sudo apt-get install virtualenv
  virtualenv venv
  pip install tensorflow keras SimpleITK
fi
