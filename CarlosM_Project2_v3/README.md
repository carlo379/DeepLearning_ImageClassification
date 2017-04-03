
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
  conda install tqdm
  jupyter notebook


  jupyter notebook --generate-config
  mkdir certs
  cd certs
  sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem
  cd ~/.jupyter/
  vi jupyter_notebook_config.py
  jupyter notebook

  hdfs dfs -mkdir -p cifar-10-batches-py

  ```
  ```
pushd ${TFoS_HOME}/src
zip -r ${TFoS_HOME}/tfspark.zip *
popd

#adjust these setting per instance types
#Instance type: p2.8xlarge
#export NUM_GPU=8
#export CORES_PER_WORKER=32
#Instance type: p2.xlarge
export NUM_GPU=1
export CORES_PER_WORKER=8

export SPARK_WORKER_INSTANCES=3
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export MASTER=spark://$(hostname):7077

spark-submit --master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--py-files ${TFoS_HOME}/tfspark.zip \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda/lib64:$JAVA_HOME/jre/lib/amd64/server:$HADOOP_HOME/lib/native" \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.HADOOP_HDFS_HOME=”$HADOOP_HOME” \
--conf spark.executorEnv.CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob) \
--driver-library-path="/usr/local/cuda/lib64" \
${TFoS_HOME}/dlnd_image_classification_TFoS.py \
--cluster_size ${SPARK_WORKER_INSTANCES} \
--images cifar-10-batches-py/ --format tfr \
--mode train --model mnist_model --tensorboard

  ```
PYSPARK_DRIVER_PYTHON="jupyter" \
PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --ip=`hostname`" \
pyspark --master ${MASTER} \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--py-files ${TFoS_HOME}/tfspark.zip,${TFoS_HOME}/examples/mnist/tf/mnist_dist.py \
--conf spark.executorEnv.LD_LIBRARY_PATH="/usr/local/cuda/lib64:$JAVA_HOME/jre/lib/amd64/server:$HADOOP_HOME/lib/native" \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.HADOOP_HDFS_HOME=”$HADOOP_HOME” \
--conf spark.executorEnv.CLASSPATH=$($HADOOP_HOME/bin/hadoop classpath --glob) \
--driver-library-path="/usr/local/cuda/lib64"