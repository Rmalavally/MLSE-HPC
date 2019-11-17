
Add logo


<h1>Introduction</h1>
This guide covers the basic instructions needed to install the ROCm software suite of applications using the command line interface and verify that these Machine Learning (ML) and High-Performance Computing (HPC) applications can run on supported frameworks.
The instructions are intended to be used on a clean installation of a supported application. The document also discusses the scale-out of the High Performing Computing (HPC) and Machine Learning (ML) applications on the AMD platform.

# ROCm Setup and Installation

## System Requirements
To use the Machine Learning and High-Performance Computing applications on your system, you will need the following hardware and software installed:

### Software Requirements

Supported Operating Systems 

•	Ubuntu v18.04 <br>
•	CentOS v7.6 <br>
•	REHL 7.6

<b>Note</b>: You must install and verify the supported operating system before installing ROCm. 

### Hardware Requirements
You must ensure you can view the VGA/3D controllers to determine if the cards are detected prior to the installation of the ROCm framework.
To detect the cards, from the command line, enter:

  <code>sudo lshw -c video</code>
 

You will see the output for each PCle device. The vendor in the output must display as *"Advanced Micro Devices, Inc."*

For example, see the code sample below:

<code>
*-display

description: Display controller

product: Vega 20

vendor: Advanced Micro Devices, Inc. [AMD/ATI]

physical id: 0

bus info: pci@0000:03:00.0

version: 02

width: 64 bits

clock: 33MHz 

capabilities: pm pciexpress msi bus_master cap_list rom

configuration: driver=amdgpu latency=0

</code>
	  

## Installing ROCm
To install the ROCm application, run RET with the install command:

<code>sudo apt -y install git 

wget https://github.com/rocmsys/RET.git</br>

cd  RET

sudo ./ret install rocm
#see all options 

sudo ./ret -h

</code>

## Installing the Applications
To install Tensorflow, PyTorch, NAMD and other applications, enter

<code>
cd ~/RET <br>
sudo ./ret install <my application> <br>


#For example,<br>
sudo ./ret install tensorflow <br>

sudo ./ret install pytorch <br>

sudo ./ret install namd <br>

</code>

## Running the Machine Learning Application
You can run the applications using Tensorflow and PyTorch for Machine Learning.

### Tensorflow
#### Training a Machine Learning Model Using Tensorflow 

To train a machine learning model using Tensorflow, use the example below:

<b>1.	Clone the Tensorflow test model. </b>

<code>
git clone https://github.com/tensorflow/models  </code> 


<br><b>2.	Download the CIFAR-10 dataset using:</b>

<code>
pip3 install tensorflow_datasets 

cd models/tutorials/image/cifar10_estimator 

python3 generate_cifar10_tfrecords.py --data-dir=${PWD}/cifar-10-data 

</code>
<br><b>3.	To run on a single node (single GPU), enter </b>
<code>

TF_ROCM_FUSION_ENABLE=1

python3 cifar10_main.py \
<br>--data-dir=${PWD}/cifar-10-data \
<br>--job-dir=/tmp/cifar10 \
<br>--num-gpus=1 \
<br>--train-steps=100

</code>
<b>4.	To run on a single node (multi GPUs (data parallelism)), enter </b>

<code>

TF_ROCM_FUSION_ENABLE=1 <br>
<br>
python3 cifar10_main.py \
<br>
--data-dir=${PWD}/cifar-10-data \
<br>
--job-dir=/tmp/cifar10 \
<br>
--num-gpus=2 \
<br>
--train-steps=100
</code>

<b>5.	To run on multi nodes </b>

<font color = "red"> To be decided </font>

### PyTorch

#### Training a Machine Learning Model Using PyTorch

To train a machine learning model using PyTorch, use the example below:

<b>1.	Download the script <code> wget </code> </b>

<code>
mkdir pytorch

cd pytorch

wget
https://raw.githubusercontent.com/wiki/ROCmSoftwarePlatform/pytorch/micro_benchmarking_pytorch.py

wget
https://raw.githubusercontent.com/wiki/ROCmSoftwarePlatform/pytorch/fp16util.py

wget
https://raw.githubusercontent.com/wiki/ROCmSoftwarePlatform/pytorch/shufflenet.py

wget
https://raw.githubusercontent.com/wiki/ROCmSoftwarePlatform/pytorch/shufflenet_v2.py

</code>
<b>2.	To run on a single node (single GPU), enter </b>

<code>

python3 micro_benchmarking_pytorch.py \

--network resnet50 \

--batch-size 128 \

--fp16 1

</code>
<b>3.	To run on a single node (multi GPU (Data Parallelism)), enter </b>

<code>
python3 micro_benchmarking_pytorch.py \

--network resnet50 \

--batch-size 128 \

--fp16 1 \

--dataparallel \

--device_ids 0,1

</code>

## Running the High Performance Computing Application
### NAMD
After the installation:
<code> <br>
cd ../namd <br>
./namd2 src/alanin -d 
</code>

## Known System Issues
<b><font color = "blue">Issue</font>: Error loading shared library libopenblas.so.3: No such file or directory </b>

*ImportError: libopenblas.so.0: cannot open shared object file: No such file or directory*

<b>Resolution</b>: Run the following command to resolve this error:
<code>
sudo apt-get install libopenblas-base <br>
export LD_LIBRARY_PATH=/usr/lib/openblas-base/
</code>

<br><b><font color = "blue">Issue</font>: Error loading module ‘torchvision’. 
The torchvision module consists of popular datasets, model architectures, 
and common image transformations for computer vision.</b>

*ImportError: No module named 'torchvision'*

<b> Resolution</b>: To resolve this error, use the following command to install the ‘torchvision’ module.

<code>
pip3 install torchvision
</code>

<br><b><font color = "blue">Issue</font>: The error message indicates a problem with the locale setting. </b>

*Error: unsupported locale setting*

<b> Resolution</b>: Modify the locale to fix the problem.

<code>
export LC_ALL=C
</code>