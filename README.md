# DNN Model Trace Obfuscator (Neurobfuscator)

## **Running in Docker** (Recommended, see bottom for mannual setup)

We will provide an open-source docker image (NVIDIA-docker) to run the tool (upon paper being accepted).




## **Script Usage**

#### **(can skip)** Set input/output of the model, batch_size and number of random seeds.

Change settings in provided scripts: `./scripts/trace_gen_advance_xx.sh` and `./tracegen_3090_depth.sh`

#### **(can skip)** Set range of parameters which you want to profile.

modify the function `model_name_from_seed` in `./trace_gen/func_generator.py`

#### Run the scripts
```
bash ./scripts/trace_gen.sh
```

#### [*] You may skip the trace generation and use the provided trace to generate a dataset.
```
python trace_gen/trace_dataset_gen.py
```

## ++Train Model Seqeunce Predictor++

### **Train the model**

#### Train using the generated trace file

assume you follow the steps above, a pickle file should appear in train_predictor/obfuscator/dataset/. If you skip the [*] step, you can download the pickle file from here: https://drive.google.com/drive/folders/1EZh8EYEthSSEdlocFV01Lyi-k5VL3XIj?usp=sharing

set "choice=deepsniffer" in train.sh

```
bash ./scripts/train.sh
```
## ++Run the Obfuscation++

### Use the obfuscator (set a pair of model file `.py` and model label `.npy` file in `./trace_obfuscate/` as input, adjust the settings in  `./trace_obfuscate/obf_env.py` )
```
bash ./scripts/obfuscate.sh
```



## **Mannual Setup** (Not Recommended, cost hours and could result in failure)

### **0. Install CUDA in a linux machine**

CUDA >= 11.0
cudnn >= 8.0

### **1. Install TVM**

Install LLVM from https://apt.llvm.org/ (we use LLVM-10)

Before running the scripts we provide here. Your File Structure should look like this:
```
----${USER} (your home directory)

    ---- tvm

    ---- torch_profiling (This repository)
```

Clone the tvm into your home directory. Then checkout 58c3413a3 version of the lastest TVM repository:

```
$ cd /usr && \
     git clone https://github.com/apache/incubator-tvm.git tvm --recursive && \
     cd ~/tvm && \
     git checkout 58c3413a3 && \
     mkdir build && \
     cp cmake/config.cmake build
```

And Copy the required file to support ``selective Fusion'' as described in our paper. Then build with USE_CUDA and USE_LLVM (llvm version must be the same with yours)

```
$ cp -r ~/torch_profiling/copy2tvm/tvm ~/
$ cd /usr/tvm/build && bash -c \
     "echo set\(USE_LLVM llvm-config-10\) >> config.cmake && \
     echo set\(USE_CUDA ON\) >> config.cmake" && \
     cmake .. && \
     make -j4 
```

Last but not the least, add to your path:
```
export TVM_HOME=~/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```

### **2. Setup Python Enviroment**

We recommend using conda. 


Require python version == 3.6, and use pip to install library listed below:
```
pip install pylint==1.9.4 six numpy pytest cython decorator scipy tornado torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html tensorflow==1.13.1
```

### **3. Install Nsight Compute**

#### Install either using command below in ubuntu-based Linux or directly from NVIDIA, 2020.3.0 version is required in either case.
```
sudo apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg \
         wget && \
     rm -rf /var/lib/apt/lists/*
sudo echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
     wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
         apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-compute-2020.3.0 && \
     rm -rf /var/lib/apt/lists/*


```
#### (IMPORTANT!) To enable the profiler:

##### Permanent Option:
```
sudo touch /etc/modprobe.d/nvprof.conf
```
Then copy this line 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' to nvprof.conf
```
sudo update-initramfs -u
```
A reboot is needed to enable sudo-free profiling.


##### Running into any problem for a sudo-free profiling, please follow the instruction below:

https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters


#### Add path to ~/.bashrc ("/usr/local/NVIDIA-Nsight-Compute" or "/opt/nvidia/nsight-compute/2020.3.0" is available after install the nsight compute software, in either case, make sure ``ncu'' is in that directory)
```
export PATH="/opt/nvidia/nsight-compute/2020.3.0:$PATH"
```
or
```
export PATH="/usr/local/NVIDIA-Nsight-Compute:$PATH"
```


### **4. Add Trace Section to NVcompute**

We provide a sepcial Trace Section, please move it into customizable Trace Section folder of 

**(if using ncu 2020.3.0)** cp torch_profiling/ncu_section/ImportantTraceAnalysis.section ~/Documents/NVIDIA Nsight Compute/2020.3.0/Sections/




