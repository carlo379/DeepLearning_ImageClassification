
  * Downloaded Conda:
  ```
  wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
  bash Anaconda3-4.3.1-Linux-x86_64.sh
  conda install numpy scipy pandas
  conda create -n tensorflow python=3.5
  source activate tensorflow
  conda install pandas matplotlib jupyter notebook scipy scikit-learn
  pip install tensorflow
  conda env list
  conda install jupyter notebook
  jupyter notebook


  jupyter notebook --generate-config
  mkdir certs
  cd certs
  sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem
  cd ~/.jupyter/
  vi jupyter_notebook_config.py
  jupyter notebook

  ```