---
layout:     post
title:      "Object Detection Studies"
subtitle:   " \"Object Detection Notebooks\""
date:       2020-07-24 22:00:00
author:     "vince"
header-img: "img/home-bg.jpg"
catalog: true
tags:
    - Object Detection
---

**Object Detection Challenges**:
- Intra class variance
- Pose variation
- Occlusion / Large search space over multiple:
> - Location
> - Scale
> - Aspect ratio
- Crowded scenes

### **Traditional Object Detection Pipeline**:
1. Input image;
2. Generate proposals: 
> - Background substraction; 
> - Sliding windows with scales; 
> - Selective search:
>> 1. adds all boundary boxes to the list of region proposed;
>> 2. groups adjacent segments based on similarity;
3. Classify Regions:
>> - HOG/SURF + SVM/RF/ADABOOST
4. Non-maximum suppression.

### **Two Stage Object Detection**:
1. Proposal generator: Extract bounding box from images; (not need to be NN, but should be smart);
2. Refine module: Classify and redefines bounding box; (trainable module that uses the visual feature to refine the module).

#### R-CNN's problem: 
- interference of  each ROI is done independently; 
- R-CNN trains each independent part (CNN feature extractor, SVM classifier, Bounding Box Regressor) separately; 
- Limitation of selective search algorithm as proposal generator (not trainable).

#### Fast R-CNN: 
> ROI Pooling:
- ROI pooling converts a projected region to a fixed sized region;
- It applies Max Pooling for each region.
> Loss function: classification loss (cross entropy) + Localization loss (smooth L1)
More efficient, and can be trained in a single step in an end-to-end manner. But still non trainable selective search as proposal generator.

#### Faster R-CNN:
- Region Proposal Network as a proposal generator:
> - Each pixel of the feature map of the deep layer of CNN is a projection of some region of the input image;
> - Area of RPN depends upon the receptive field of CNN;
> - RPN is a trainable sliding window approach.
- Shared computation of CNN features;
- ROI pooling to convert features from proposed region to fixed size;
- Joint training of Boundning Box offset and classifier with CNN fine-tune. 
- Loss function: classification loss (cross entropy) + Localization loss (smooth L1)







