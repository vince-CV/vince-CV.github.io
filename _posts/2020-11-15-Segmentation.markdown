---
layout:     post
title:      "Semantic Segmentation"
subtitle:   " \"Review of semantic & instance segmentation\""
date:       2020-11-15 22:00:00
author:     "vince"
header-img: "img/home-bg.jpg"
catalog: true
tags:
    - Segmentation
    - Computer Vision
    - Deep Learning
---


## Semantic Segmentation Architectures
Until state-of-the-art networks for segmentation:

### 1. FCN
![Image](/img/in-post/201115 Segment/FCN.png)

### 2. LinkNet
![Image](/img/in-post/201115 Segment/LinkNet.png)

### 3. UNet
![Image](/img/in-post/201115 Segment/UNet.png)

### 4. SegNet
![Image](/img/in-post/201115 Segment/SegNet.png)

### 5. DeepLab
![Image](/img/in-post/201115 Segment/DeepLab.png)

## Evaluation Metrics
**Dice coefficient:** intersection over Union-like metric.
![Image](/img/in-post/201115 Segment/DC.png)
where,
- `p_i` is prediction for pixel `i`;
- `y_i` is ground truth for pixel `i`;
- `N` is the total number of pixels on the image.

## Loss for Semantic Segmentation 
**Soft-Dice Loss:** <br>
The ground truth annotation can be represented as a sum of true positives and false negatives of every class. <br>
The predicted pixels are a sum of true and false positives across all classes.<br>
How to turn DC metric into a loss function?<br>
Dice coefficient compares two **discrete** masks and, therefore, it is a discrete function. To make a loss function out of it, we need to come up with a **differentiable function**. So, instead of thresholded values like 0 and 1, we can make floating point probabilities in the range of [0, 1]. The function that can help us doing so is a **negative logarithm**. (Just recall classification cross-entropy loss which also uses negative logarithm for the same reasons).
![Image](/img/in-post/201115 Segment/DL.png)