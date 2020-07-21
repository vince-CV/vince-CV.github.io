---
layout:     post
title:      "Accelerated Computing (III)"
subtitle:   " \"Asynchronous Streaming, and Visual Profiling with CUDA C/C++\""
date:       2020-07-22 22:00:00
author:     "vince"
header-img: "img/home-bg.jpg"
catalog: true
tags:
    - GPU
    - CUDA
    - C++
---

![Image](/img/in-post/200705 CudaProgramming/logo.png)
The CUDA tookit ships with the **Nsight Systems**, a powerful GUI application to support the development of accelerated CUDA applications. Nsight Systems generates a graphical timeline of an accelerated application, with detailed information about CUDA API calls, kernel execution, memory activity, and the use of CUDA streams.

In this lab, it will be using the Nsight Systems timeline to guide in optimizing accelerated applications. Additionally, it will cover some intermediate CUDA programming techniques to support your work: **unmanaged memory allocation and migration**; **pinning**, or **page-locking** host memory; and **non-default concurrent CUDA streams**.

**Objectives**
1. Use **Nsight Systems** to visually profile the timeline of GPU-accelerated CUDA applications.
2. Use Nsight Systems to identify, and exploit, optimization opportunities in GPU-accelerated CUDA applications.
3. Utilize CUDA streams for concurrent kernel execution in accelerated applications.
4. (**Optional Advanced Content**) Use manual device memory allocation, including allocating pinned memory, in order to asynchronously transfer data in concurrent CUDA streams.

## Running Nsight Systems
#### 1. Generate Report File
Compile and run the code. Next, use `nsys profile --stats=true` to create a report file that I will be able to open in the Nsight Systems visual profiler. <br>
And using the `-o` flag to give the report file a memorable name.
#### 2. Open Nsight System
To open Nsight Systems, enter and run the `nsight-sys` command from a open terminal.
#### 3. Enable Usage Reporting
When prompted, click "Yes" to enable usage collection.
#### 4. Open the Report File
Open the targeting report file by visiting _File_ -> _Open_ from the Nsight Systems menu, then go to the path `/root/Desktop/directory/...` and select `filename.qdrep`.
#### 5. Expand the CUDA Unified Memory Timelines
Next, expand the _CUDA_ -> _Unified memory_ and _Context_ timelines, and close the _OS runtime libraries_ timelines
![Image](/img/in-post/200705 CudaProgramming/10.png)
#### 6. Observe Many Memory Transfers
From a glance it can see the application is taking about 1 second to run, and that also, during the time when the addVectorsInto kernel is running, that there is a lot of UM memory activity (above picture).<br>
Zoom into the memory timelines to see more clearly all the small memory transfers being caused by the on-demand memory page faults.
![Image](/img/in-post/200705 CudaProgramming/11.png)