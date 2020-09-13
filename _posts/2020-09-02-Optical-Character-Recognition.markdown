---
layout:     post
title:      "OCR Demo: Automatic Number Plate Recognitions"
subtitle:   " \"OCR through Tesseract\""
date:       2020-09-15 22:00:00
author:     "vince"
header-img: "img/home-bg.jpg"
catalog: true
tags:
    - OCR
    - Computer Vision 
    - Machine Learning
---

This blog is the notbook for OCR fundamental studies, which will include:
1. Graphic Text Recognition: **Tesseract**, **Keras-OCR**
2. Text Detection: **EAST**, **CRAFT** 

Realistic scenario: text in imagery<br>
1. Graphic text (scanned documents...)<br>
2. Scene text (clothing, signs, packages...)<br>

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
    a. connected component analysis, to get Blobs;<br>
    b. from Blobs to get fixed-pitch/proportional texts.<br>
**3.** Word Recognize: Pass 1 & 2 (to gain high confidence)<br>
**4.** Fix: X-Height, Fuzzy Space, Word Bigram<br>
**5.** Output Text<br>
![Image](/img/in-post/200830 ObjectDetectionYOLO/chart.jpg)
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