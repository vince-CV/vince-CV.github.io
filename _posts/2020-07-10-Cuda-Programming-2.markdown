---
layout:     post
title:      "Accelerated Computing (II)"
subtitle:   " \"Managing Accelerated Application Memory with CUDA Unified Memory and nsys\""
date:       2020-07-12 22:00:00
author:     "vince"
header-img: "img/home-bg.jpg"
catalog: true
tags:
    - GPU
    - CUDA
    - C++
---

![Image](/img/in-post/200705 CudaProgramming/logo.png)
The [*CUDA Best Practices Guide*](http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations), a highly recommended followup to this and other CUDA fundamentals labs, recommends a design cycle called **APOD**: **A**ssess, **P**arallelize, **O**ptimize, **D**eploy. In short, APOD prescribes an iterative design process, where developers can apply incremental improvements to their accelerated application's performance, and ship their code. As developers become more competent CUDA programmers, more advanced optimization techniques can be applied to their accelerated codebases.<br>
This lab will support such a style of iterative development. You will be using the Nsight Systems command line tool **nsys** to qualitatively measure your application's performance, and to identify opportunities for optimization, after which you will apply incremental improvements before learning new techniques and repeating the cycle. As a point of focus, many of the techniques you will be learning and applying in this lab will deal with the specifics of how CUDA's **Unified Memory** works. Understanding Unified Memory behavior is a fundamental skill for CUDA developers, and serves as a prerequisite to many more advanced memory management techniques.

## Managing Accelerated Application Memory with CUDA Unified Memory and nsys

**Objectives**
1. Use the Nsight Systems command line tool (**nsys**) to profile accelerated application performance.
2. Leverage an understanding of **Streaming Multiprocessors** to optimize execution configurations.
3. Understand the behavior of **Unified Memory** with regard to page faulting and data migrations.
4. Use **asynchronous memory prefetching** to reduce page faults and data migrations for increased performance.
5. Employ an iterative development cycle to rapidly accelerate and deploy applications.
