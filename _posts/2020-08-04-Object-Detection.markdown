---
layout:     post
title:      "LibTorch on Local System"
subtitle:   " \"Cmake for compiling LibTorch code\""
date:       2020-07-24 22:00:00
author:     "vince"
header-img: "img/home-bg.jpg"
catalog: true
tags:
    - Object Detection
---

Object Detection Challenges:
- Intra class variance
- Pose variation
- Occlusion / Large search space over multiple:
> 1. Location
> 2. Scale
> 3. Aspect ratio
- Crowded scenes

Traditional Object Detection Pipeline:
1. Input image;
2. Generate proposals: 
> 1. Background substraction; 
> 2. Sliding windows with scales; 
> 3. Selective search:
>> 1. adds all boundary boxes to the list of region proposed;
>> 2. groups adjacent segments based on similarity;
3. Classify Regions;
4. Non-maximum suppression.



