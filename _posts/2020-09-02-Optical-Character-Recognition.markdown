---
layout:     post
title:      "OCR Demo: Automatic Number Plate Recognitions"
subtitle:   " \"Roubut OCR pipeline using CRAFT & Tesseract\""
date:       2020-09-15 22:00:00
author:     "vince"
header-img: "img/home-bg.jpg"
catalog: true
tags:
    - OCR
    - Tesseract
    - EAST
    - CRAFT
    - Text Detection
    - Text Recognition
    - Computer Vision 
    - Machine Learning
---

This blog is the notbook for OCR fundamental studies, which will include:
**1.** Graphic Text Recognition: **Tesseract**, **Keras-OCR**
**2.** Text Detection: **EAST**, **CRAFT** 

Realistic scenario: text in imagery<br>
**1.** Graphic text (scanned documents...)<br>
**2.** Scene text (clothing, signs, packages...)<br>

## OCR pipeline<br>
1. Two-Step:<br> 
    a. text detection module that detects and ocalizes the existence of text; <br>
    b. text recognition module that transcribes and converts the content of the detected text region into linguistic symbols.<br>
2. End-to-End: 
    single module with detection & recognition.<br>

### Tesseract pipeline<br>
Tesseract -> text recognition (OCR) engine -> extract text from images. 

**1.** Adaptive Thresholding<br>

**2.** Page Layout Analysis: document -> segments<br>
> *a.* connected component analysis, to get Blobs;<br>
> *b.* from Blobs to get fixed-pitch/proportional texts.<br>

**3.** Word Recognize: Pass 1 & 2 (to gain high confidence)<br>

**4.** Fix: X-Height, Fuzzy Space, Word Bigram<br>

**5.** Output Text<br>

![Image](/img/in-post/200903 OCR/1.png)

Tesseract is not always a pipeline, but a circle between 2 -> 4 -> 3 -> 5, or 4 -> 3 -> 5.

#### Tesseract Experiments
**1. Install Tesseract Library:**<br>
**`!apt install libtesseract-dev tesseract-ocr > /dev/null`**

**2. Install Python wrapper for Tesseract:**<br>
**`!pip install pytesseract > /dev/null`**

**3. Perform OCR:**<br>

```python
import pytesseract
text = pytesseract.image_to_string('text.jpg')
```

From experiments, even though it is natural image, Tesseract is able to perform OCR almost without any errors, but it will be surprised by how fast the output deteriorates on small changes in the images. Major reasons for failure of OCR using Tesseract and in general. They are:
- **Cluttered Background**: The text might not be visibly clear or it might appear camouflaged with the background.
- **Small Text**: The text might be too small to be detected. 
- **Rotation or Perspective Distortion**: The text might be rotated in the image or the image itself might be distorted.


#### Tesseract Functions
- **`get_tesseract_version`** - Returns the Tesseract version installed in the system.
- **`image_to_string`** - Returns the result of a Tesseract OCR run on the image as a single string
- **`image_to_boxes`** - Returns the recognized characters and their box boundaries.
- **`image_to_data`** - Returns the box boundaries/locations, confidences, words etc. 
- **`image_to_osd`** - Returns result containing information about orientation and script detection.
- **`image_to_pdf_or_hocr`** - Returns a searchable PDF from the input image.
- **`run_and_get_output`** - Returns the raw output from Tesseract OCR. This gives a bit more control over the parameters that are sent to tesseract.



### EAST<br>
**An Efficient and Accurate Scene Text Detector**, <a href="https://arxiv.org/abs/1704.03155v2">EAST</a>, is a very robust deep learning method and an OpenCV tool that detects text in natural scene images. Its pipeline directly predicts words or text lines of arbitrary orientations and quadrilateral shapes in full images, eliminating unnecessary intermediate steps (e.g., candidate aggregation and word partitioning).

Two outputs of the EAST network:
**1.** feature_fusion/concat_3 (detected text box)
**2.** feature_fusion/Conv_7/Sigmoid (confidence score)

```python
import cv2
!pip install keras-ocr > /dev/null
from keras_ocr.tools import warpBox

model = "frozen_east_text_detection.pb"
net = cv2.dnn.readNet(model)
outputLayers = []
outputLayers.append("feature_fusion/Conv_7/Sigmoid")
outputLayers.append("feature_fusion/concat_3")
inpWidth = 640
inpHeight = 640
confThreshold = 0.7
nmsThreshold = 0.4

image = cv2.imread(imageName)

# Create a blob and assign the image to the blob
blob = cv2.dnn.blobFromImage(image, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)
net.setInput(blob)

# Get the output using by passing the image through the network
output = net.forward(outputLayers)
scores = output[0]
geometry = output[1]

# Get rotated rectangles using the decode function described above
[boxes, confidences] = decode(scores, geometry, confThreshold)   # see more details in Github Repos
indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
...

```



### CRAFT<br>

In terms of the recent works on scene text detection pipeline, (from CRAFT paper):
![Image](/img/in-post/200903 OCR/5.png)

**Character Region Awareness For Text Detection**, <a href="https://arxiv.org/abs/1904.01941">CRAFT</a>. The challenges associated with text detection, rather than regular object detection: huge aspect ratio variation, different fonts & backgrounds, skewed & curved text, and colored text.<br>
Key idea in CRAFT: A word is a collection of letters -> Use the affinity between letters to detect words. The authors are repropose idea: **Text detection problem -> Segmeantation problem**

The outputs of the CNN in CRAFT: 1. Region Score; 2. Affinity Score. (they are grey-scaled image, or can say a 2-channels image).

If have the output label and input image, we can train a CNN; (input a image, output a mask). 
So **how to generate the two maps for training data**?

####  **1. Similarity to Segmentation problem**
Before that question, let's look at the U-Net for semantic segmentation, and CRAFT use a very similiar network to solve this problem:
![Image](/img/in-post/200903 OCR/3.png)

####  **2. What's the two maps menas?**
![Image](/img/in-post/200903 OCR/2.png)
**1. Region score**
Basically indicates that these locations have a character in them and thery are centered at this point of highest probability.

**2. Affinity score**
Calculates the affnity between characters, or say the two letters are close together if there is high affinity at this location or part of same word.

####  **3. Generate Ground Truth**
![Image](/img/in-post/200903 OCR/4.png)
Suppose we've got the bounding boxs around the letter (**character boxes**), then the **affinity boxes** are generating as the figure above.<br>
And the score generation module: warp a 2-D gaussain distribution based on the perspective transform between the boxes.<br>

**Problem**: large public datasets contain only word level segmentation!<br>
**Solution**: Synthesize the text examples. <br>

The authors used a semi-supervised approach:
1. crop out the word-level text
2. run through the network they trained on Synthetic data to get various region score
3. use watershed to get a segments
4. fit bounding boxes for segments (pseudo ground truth)

but cannot trust this data completely. A clever solution:<br>
If there is real data that is being used and also know what is this text, or the number of characters in this bounding box.<br>
So once the above automated technique produces the right number of bounding boxes, given a higher weight compared to those segmentation were incorrect. So they can use the real data also.<br>
Semi-supervised learning: using synthetic data to train CNN which is the supervised part. But for the real part they use these tricks which is not supervised.<br>
![Image](/img/in-post/200903 OCR/6.png)

####  **CRAFT Implementation**
CRAFT model is carried by library Keras-OCR, which implemented Text Detection (**CRAFT-2019**) & Text Recognition (**<a href="https://arxiv.org/abs/1904.01941">CRNN-2017</a>**) pipeline. 

**1. Install Keras-OCR Library:**<br>
**`!pip install keras-ocr > /dev/null`**

**2. Text Detector:**<br>

```python
import keras_ocr
import cv2
import glob
import matplotlib.pyplot as plt
%matplotlib inline

detector = keras_ocr.detection.Detector()

image = keras_ocr.tools.read('https://upload.wikimedia.org/wikipedia/commons/e/e8/FseeG2QeLXo.jpg')
detections = detector.detect(images=[image])[0]

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 10))
canvas = keras_ocr.tools.drawBoxes(image, detections)
ax1.imshow(image)
ax2.imshow(canvas)

```