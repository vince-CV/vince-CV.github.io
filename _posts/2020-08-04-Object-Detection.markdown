---
layout:     post
title:      "Object Detection Studies (I)"
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
| :----------- | :----: | :----: |
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



## **One Stage Object Detection**:

| Model        | Paper  | code  |
| :----------- | :----: | :----: |
| R-CNN        | [<a href="https://arxiv.org/abs/1311.2524">paper</a>]  | [<a href="https://github.com/rbgirshick/rcnn">code</a>] |
| Fast R-CNN   | [<a href="https://arxiv.org/abs/1504.08083">paper</a>] | [<a href="https://github.com/rbgirshick/fast-rcnn">code</a>] |
| Faster R-CNN | [<a href="https://arxiv.org/abs/1506.01497">paper</a>] | [<a href="https://github.com/rbgirshick/py-faster-rcnn">code</a>] |
| Mask R-CNN   | [<a href="https://arxiv.org/abs/1703.06870">paper</a>] | [<a href="https://github.com/CharlesShang/FastMaskRCNN">code</a>] |

#### SSD
<https://arxiv.org/pdf/1512.02325.pdf>

#### YOLO
<https://arxiv.org/abs/1506.02640>
Does not have Proposal Generator and Refine Stages, but directly predicts Bounding Box through Single Stage Network using features from the entire image and predicts Bounding Box of all classes simultaneously.

Each frame of live footage is inputted directly into the algorithm at a rate of 60fps.<br>
The YOLO framework applies a convolution layer to the frame, reducing its size to a 13x13 matrix.<br>
Each cell in the matrix predicts 5 bounding boxes, each associated with one of the 9000 classes.<br>
Binding boxes with a confidence score of >30% are shown to the user with their respective class label.<br>

YOLO Training:
1. Cells contain the center if the ground truth bounding box is responsible for detecting it.
- Adjust the cell's label to "car";
- Find the predicted bounding box:
    1. Increase confidence of bouning box with largest overlap with GT;
    2. Decrease confidence of bouning box with smaller overlap.
2. Cells do not contain an object:
    1. Reduce confidence of bounding boxes;
    2. Do not change class probabilities or bounding box coordinates.

YOLO Loss:
1. Localization error; (Original centers & Width & Height) (If an object is present, minimize loss function only when there is an object presenting in the bounding box).
2. Confidence of the bounding box;
3. Confidence of the bounding box at empty cells;
4. Conditional probability of final class.
weight localization error differently than classification error.

YOLO pros and cons: 
1. much faster than Faster R-CNN, and v3 reached very good accuracy;
2. Hard to detect groups of small objects (limited number of boxes);
3. Hard to handle multi-scale objects (limited scale of output feature map).

#### YOLOv3 on Darknet and OpenCV
1. Initialize the parameters:
    (a). Confidence threshold. Every predicted box is associated with a confidence score. In the first stage, all the boxes below the confidence threshold parameter are ignored for further processing.
    (b). Non-maximum suppression threshold. The rest of the boxes undergo non-maximum suppression which removes redundant overlapping bounding boxes.
    (c). Input Width & Height. 416 for default, but can also change both of them to 320 to get faster results or to 608 to get more accurate results.
2. Load model and classes:
    (a). **coco.names** contains all the objects for which teh model was trained;
    (b). **yolov3.weights** pre-trained weights;
    (c). **yolov3.cfg** configuration file.
    OpenCV DNN module set to use CPU by default, but we can set `cv.dnn.DNN_TARGET_OPENCL` for Intel GPU.
3. Process each frame:
    (a). Getting the names of output layers: 
    The forward function in OpenCV’s Net class needs the ending layer till which it should run in the network. Since we want to run through the whole network, we need to identify the last layer of the network by using `getUnconnectedOutLayers()` that gives the names of the unconnected output layers, which are essentially the last layers of the network. Then we run the forward pass of the network to get output from the output layers, as in the previous code snippet`(net.forward(getOutputsNames(net)))`.
    (b). Draw the predicted boxes;
    (c). Post-processing the network's output:
    The network outputs bounding boxes are each represented by a vector of number of classes + 5 elements. The first 4 elements represent the **center_x**, **center_y**, **width** and **height**. The fifth element represents the confidence that the bounding box encloses an object.
4. Main loop:
      **blobFromImage** function scales the image pixel values to a target range of 0 to 1 using a scale factor of 1/255. It also resizes the image to the given size of (416, 416) without cropping. Keeping the swapRB parameter to its default value of 1.



### Train a custom Object Detector using YOLO
Generate file form:

PASCAL VOC: .xml (Top left, bottom right points);
YOLO: .txt (class_id, x, y, w, h) ratio form.




Many thansk for @weng2017detection3's blog: "http://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html"







