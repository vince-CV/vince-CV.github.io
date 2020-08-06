---
layout:     post
title:      "Object Detection Studies"
subtitle:   " \"Object Detection Notebooks\""
date:       2020-08-06 22:00:00
author:     "vince"
header-img: "img/home-bg.jpg"
catalog: true
tags:
    - Object Detection
    - Computer Vision 
---

**Object Detection Challenges**:
- Intra class variance
- Pose variation
- Occlusion / Large search space over multiple: (a) Location, (b) Scale, (c)Aspect ratio;
- Crowded scenes
![Image](/img/in-post/200806 ObjectDetection/1.png)

## **Traditional Object Detection Pipeline**:
![Image](/img/in-post/200806 ObjectDetection/0.png)
1. Input image;
2. Generate proposals: 
> - Background substraction; 
> - Sliding windows with scales; 
> - Selective search:
>> 1. adds all boundary boxes to the list of region proposed;
>> 2. groups adjacent segments based on similarity;
3. Classify Regions:
> - HOG/SURF + SVM/RF/ADABOOST
4. Non-maximum suppression.

## **Two Stage Object Detection (RCNN Family)**:
1. Proposal generator: Extract bounding box from images; (not need to be NN, but should be smart);
2. Refine module: Classify and redefines bounding box; (trainable module that uses the visual feature to refine the module).

| Model        | Paper  | code  |
| :----------- | :----: | ----: |
| R-CNN        | [<a href="https://arxiv.org/abs/1311.2524">paper</a>]  | [<a href="https://github.com/rbgirshick/rcnn">code</a>] |
| Fast R-CNN   | [<a href="https://arxiv.org/abs/1504.08083">paper</a>] | [<a href="https://github.com/rbgirshick/fast-rcnn">code</a>] |
| Faster R-CNN | [<a href="https://arxiv.org/abs/1506.01497">paper</a>] | [<a href="https://github.com/rbgirshick/py-faster-rcnn">code</a>] |
| Mask R-CNN   | [<a href="https://arxiv.org/abs/1703.06870">paper</a>] | [<a href="https://github.com/CharlesShang/FastMaskRCNN">code</a>] |

#### R-CNN's 

![Image](/img/in-post/200806 ObjectDetection/a.png)

**Model Workflow**:
1. **Pre-train** a CNN network on image classification tasks; for example, VGG or ResNet trained on ImageNet dataset. The classification task involves N classes.
2. Propose category-independent regions of interest by selective search (~2k candidates per image). Those regions may contain target objects and they are of different sizes.
3. Region candidates are **warped** to have a fixed size as required by CNN.
4. Continue fine-tuning the CNN on warped proposal regions for K + 1 classes; The additional one class refers to the background (no object of interest). In the fine-tuning stage, we should use a much smaller learning rate and the mini-batch oversamples the positive cases because most proposed regions are just background.
5. Given every image region, one forward propagation through the CNN generates a feature vector. This feature vector is then consumed by a **binary SVM** trained for **each class** independently.
The positive samples are proposed regions with IoU (intersection over union) overlap threshold >= 0.3, and negative samples are irrelevant others.
6. To reduce the localization errors, a regression model is trained to correct the predicted detection window on bounding box correction offset using CNN features.

**Common tricks**:
- Non-Maximum Suppersion
- Hard negative Mining

**Its problem**: 
- interference of  each ROI is done independently; 
- R-CNN trains each independent part (CNN feature extractor, SVM classifier, Bounding Box Regressor) separately; 
- Limitation of selective search algorithm as proposal generator (not trainable).

#### Fast R-CNN: 

![Image](/img/in-post/200806 ObjectDetection/b.png)

**Model Workflow**:

1. Pre-train a convolutional neural network on image classification tasks.
2. Propose regions by selective search (~2k candidates per image).
3. Alter the pre-trained CNN:
    - Replace the last max pooling layer of the pre-trained CNN with a RoI pooling layer. The RoI pooling layer outputs fixed-length feature vectors of region proposals. Sharing the CNN computation makes a lot of sense, as many region proposals of the same images are highly overlapped.
    - Replace the last fully connected layer and the last softmax layer (K classes) with a fully connected layer and softmax over K + 1 classes.
4. Finally the model branches into two output layers:
    - A softmax estimator of K + 1 classes (same as in R-CNN, +1 is the “background” class), outputting a discrete probability distribution per RoI.
    - A bounding-box regression model which predicts offsets relative to the original RoI for each of K classes.


- ROI Pooling:
    - ROI pooling converts a projected region to a fixed sized region;
    - It applies Max Pooling for each region.
![Image](/img/in-post/200806 ObjectDetection/ROI Pooling.png)
- Loss function: classification loss (cross entropy) + Localization loss (smooth L1)

More efficient, and can be trained in a single step in an end-to-end manner. But still non trainable selective search as proposal generator.

#### Faster R-CNN:

![Image](/img/in-post/200806 ObjectDetection/c.png)
**Workflow**:
1. Pre-train a CNN network on image classification tasks.
2. Fine-tune the RPN (region proposal network) end-to-end for the region proposal task, which is initialized by the pre-train image classifier. Positive samples have IoU (intersection-over-union) > 0.7, while negative samples have IoU < 0.3.
    - Slide a small n x n spatial window over the conv feature map of the entire image.
    - At the center of each sliding window, we predict multiple regions of various scales and ratios simultaneously. An anchor is a combination of (sliding window center, scale, ratio). For example, 3 scales + 3 ratios => k=9 anchors at each sliding position.
3. Train a Fast R-CNN object detection model using the proposals generated by the current RPN
4. Then use the Fast R-CNN network to initialize RPN training. While keeping the shared convolutional layers, only fine-tune the RPN-specific layers. At this stage, RPN and the detection network have shared convolutional layers!
5. Finally fine-tune the unique layers of Fast R-CNN
6. Step 4-5 can be repeated to train RPN and Fast R-CNN alternatively if needed.

- Region Proposal Network as a proposal generator:
> - Each pixel of the feature map of the deep layer of CNN is a projection of some region of the input image;
> - Area of RPN depends upon the receptive field of CNN;
> - RPN is a trainable sliding window approach.
- Shared computation of CNN features;
- ROI pooling to convert features from proposed region to fixed size;
- Joint training of Boundning Box offset and classifier with CNN fine-tune. 
- Loss function: classification loss (cross entropy) + Localization loss (smooth L1)


Recall:
![Image](/img/in-post/200806 ObjectDetection/d.png)


Many thansk for @weng2017detection3's blog: "http://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html"







