---
layout:     post
title:      "LibTorch code on Local System"
subtitle:   " \"Cmake for compiling LibTorch code\""
date:       2020-07-24 22:00:00
author:     "vince"
header-img: "img/home-bg.jpg"
catalog: true
tags:
    - LibTorch
    - CMAKE
    - C++
---

## LibTorch Installation

#### Step 1: Download libtorch.zip file
Download libtorch zip file (debug or release) from pytorch.org and select the stable version specific to your OS. If have a GPU, select CUDA version.

#### Step 2: Download Sample Code
Download sample code from this link. Unzip it and keep the folder at the same place as libtorch. For example the directory structure should look as follows:
- C:\LibTorch\libtorch
- C:\LibTorch\libtorch-sample-code

#### Step 3: Build C++ Code
Now we have provided a CMakeLists.txt file in the sample code folder. We can use the following commands to build the C++ code (sample.cpp for example).<br>
NOTE that we need to provide the path where we have kept the libtorch folder while running CMAKE using **CMAKE_PREFIX_PATH** flag.

```cmd
cd libtorch-sample-code
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH= C:\LibTorch\libtorch .. 
cmake --build . --config Release
cd ..
.\build\Release\sample.exe
```

## Installation trouble-shooting

#### Error 1: Cmake CMD
CMD Line: 'cmake' is not recognized as an internal or external command, operable program or batch file.<br>
**Soultion**: add CMAKE bin folder to the Environment Variable Path value. (`C:\CMake 3.15.0\bin`)

#### Error 2: Torch Config
CMake Error at CMakeLists.txt:4 (find_package):<br>
By not providing "FindTorch.cmake" in CMAKE_MODULE_PATH this project has asked CMake to find a package configuration file provided by "Torch", but CMake did not find one.<br>
Could not find a package configuration file provided by "Torch" with any of the following names:
- TorchConfig.cmake
- torch-config.cmake
![Image](/img/in-post/200724 Libtorch/1.png)
**Solution**: add libtorch path to Environment Variable Path value. (`C:\LibTorch\libtorch`)

#### Error 3: System
CMake Error at C:/LibTorch/libtorch/share/cmake/Caffe2/public/cuda.cmake:325 (message):<br>
CUDA support not available with 32-bit windows. Did you forget to set Win64 in the generator target?<br>
Call Stack (most recent call first):
- C:/LibTorch/libtorch/share/cmake/Caffe2/Caffe2Config.cmake:88 (include)
- C:/LibTorch/libtorch/share/cmake/Torch/TorchConfig.cmake:40 (find_package)
![Image](/img/in-post/200724 Libtorch/2.png)
**Solution**: Configure the CMAKE flag to use x64 system for compiling.
Change from:
> cmake -DCMAKE_PREFIX_PATH= C:\LibTorch\libtorch ..<br>

to:<br>

> cmake -DCMAKE_PREFIX_PATH= C:\LibTorch\libtorch .. -A x64
