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


### Code Refactor
#### Baseline code
```cpp
#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
```
And its Nsys profile:
![Image](/img/in-post/200705 CudaProgramming/12.png)


#### Prefetching vs. Non-Prefetching
This version refactors the vector addition application from above so that the 3 vectors needed by its addVectorsInto kernel are asynchronously prefetched to the active GPU device prior to launching the kernel (using cudaMemPrefetchAsync). 

```cpp
#include <stdio.h>

void initWith(float num, float *a, int N)
{
  for(int i = 0; i < N; ++i)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  initWith(3, a, N);
  initWith(4, b, N);
  initWith(0, c, N);

  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

```

And its Nsys profile:
![Image](/img/in-post/200705 CudaProgramming/13.png)
For comparison:
- How does the execution time compare to that of the `addVectorsInto` kernel prior to adding asynchronous prefetching?
> Execution time dramatically dropped after adding asynchronous prefetching.
- How have the memory transfers changed?
> Less operations happened so less time on HtoD, and more data migrate from Host to Device each operation.

#### Launch Init in Kernel
In the previous iteration of the vector addition application, the vector data is being initialized on the CPU, and therefore needs to be migrated to the GPU before the addVectorsInto kernel can operate on it.<br>
Next, the application will be refactored to initialize the data in parallel on the GPU.<br>
Since the initialization now takes place on the GPU, prefetching has been done prior to initialization, rather than prior to the vector addition work.
```cpp
#include <stdio.h>

__global__
void initWith(float num, float *a, int N)
{

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    a[i] = num;
  }
}

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
  int deviceId;
  int numberOfSMs;

  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

  const int N = 2<<24;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  cudaMallocManaged(&c, size);

  cudaMemPrefetchAsync(a, size, deviceId);
  cudaMemPrefetchAsync(b, size, deviceId);
  cudaMemPrefetchAsync(c, size, deviceId);

  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = 32 * numberOfSMs;

  cudaError_t addVectorsErr;
  cudaError_t asyncErr;

  initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
  initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
  initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);

  addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

  addVectorsErr = cudaGetLastError();
  if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

  asyncErr = cudaDeviceSynchronize();
  if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  checkElementsAre(7, c, N);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}

```
