---
layout:     post
title:      "Single Stage Object Detection"
subtitle:   " \"Create a custom single-stage detector\""
date:       2020-11-02 22:00:00
author:     "vince"
header-img: "img/home-bg.jpg"
catalog: true
tags:
    - Object Detector
    - Computer Vision
    - Deep Learning
---

The pipeline to create a custom single stage detector:

**1.** Single-stage NN architecture
![Image](/img/in-post/201102 Detect/1.png)
As the network pipeline, the feature extractor is going to be the combination of **ResNet-18** (pre-trained) and **FPN** (extract features from different layers). After that, two predictor: class predictor and bounding-box regressor.

**2.** generate anchor-boxes<br>
Because of the feature extraction, the convolution feature of dimensions is: [num_channels, h, w]. The feature maps correspond to the position  [:,i,j] ∀ i & j, use to have different bounding boxes (of different sizes and aspect ratios assuming this position is the center of the bounding box) associated with it. This predefined bounding box is called an anchor.

**3.** match prediction with ground truth<br>
Predicting bounding boxes and class for each bounding box for every feature map. Encoding Boxes and Decoding Boxes.

**4.** loss function

**5.** training pipeline



## Detector NN achitecture<br>
I will use the Feature Pyramid Network for feature extraction. On top of this, I used class subnet and box subnet to get classification and bounding box.

![Image](/img/in-post/201102 Detect/2.png)
FPN is built on top of ResNet (ResNet-18) in a fully convolutional fashion. It includes two pathways: bottom-up & top-down. These two pathways are connected in-between with lateral connections.

- Bottom-up: forward path for feature-extracting.
- Top-down: features closer to the input image have a rich segment (bounding box) information. So it is needed to merge all of the feature maps from different levels of the pyramid into one semantically-rich feature map.
![Image](/img/in-post/201102 Detect/3.png)

The higher-level features are upsampled to be 2x larger. For this purpose, nearest neighbor upsampling is used. The larger feature map undergoes a 1x1 convolutional layer to reduce the channel dimension. Finally, these two feature maps are added together in element-wise manner. The process continues until the finest merged feature map is created.

These merged features map goes into two different CNN of classes and bounding boxes predictions.


#### 1.ResNet
```python
import torch
import inspect
from torchvision import models
from IPython.display import Code
from fpn import FPN
from detector import Detector

resnet = models.resnet18(pretrained=True) 
```

The ResNet18 has the following blocks:
1. `conv1`
2. `bn1`
3. `relu`
4. `maxpool`
5. `layer1`
6. `layer2`
7. `layer3`
8. `layer4`
9. `avgpool`
10. `fc`

I used `1-8` blocks in FPG. But we take a look at the ouput dimension from these blocks:

```python
# btch_size = 2, image dimesion = 3 x 256 x 256 

image_inputs = torch.rand((2, 3, 256, 256))

x = resnet.conv1(image_inputs)
x = resnet.bn1(x)
x = resnet.relu(x)
x = resnet.maxpool(x)
layer1_output = resnet.layer1(x)
layer2_output = resnet.layer2(layer1_output)
layer3_output = resnet.layer3(layer2_output)
layer4_output = resnet.layer4(layer3_output)

print('layer2_output size: {}'.format(layer2_output.size()))
print('layer3_output size: {}'.format(layer3_output.size()))
print('layer4_output size: {}'.format(layer4_output.size()))
```

FPN will use `layer2_output`, `layer3_output`, `layer4_output` to get features from different convolution layers. And the output:<br>
`layer2_output size: torch.Size([2, 128, 32, 32])`<br>
`layer3_output size: torch.Size([2, 256, 16, 16])`<br>
`layer4_output size: torch.Size([2, 512, 8, 8])`<br>


#### 2. FPN
Codes that implement FPN: 

```python
class FPN(nn.Module):
    def __init__(self, block_expansion=1, backbone="resnet18"):
        super().__init__()
        assert hasattr(models, backbone), "Undefined encoder type"
        
        # load model 
        self.feature_extractor = getattr(models, backbone)(pretrained=True)
        
        # two more layers conv6 and conv7 on the top of layer4 (if backbone is resnet18)
        
        self.conv6 = nn.Conv2d(
            512 * block_expansion, 64 * block_expansion, kernel_size=3, stride=2, padding=1
        )
        self.conv7 = nn.Conv2d(
            64 * block_expansion, 64 * block_expansion, kernel_size=3, stride=2, padding=1
        )

        # lateral layers
        
        self.latlayer1 = nn.Conv2d(
            512 * block_expansion, 64 * block_expansion, kernel_size=1, stride=1, padding=0
        )
        self.latlayer2 = nn.Conv2d(
            256 * block_expansion, 64 * block_expansion, kernel_size=1, stride=1, padding=0
        )
        self.latlayer3 = nn.Conv2d(
            128 * block_expansion, 64 * block_expansion, kernel_size=1, stride=1, padding=0
        )

        # top-down layers
        self.toplayer1 = nn.Conv2d(
            64 * block_expansion, 64 * block_expansion, kernel_size=3, stride=1, padding=1
        )
        self.toplayer2 = nn.Conv2d(
            64 * block_expansion, 64 * block_expansion, kernel_size=3, stride=1, padding=1
        )

    @staticmethod
    def _upsample_add(x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.interpolate(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, height, width = y.size()
        return F.interpolate(x, size=(height, width), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        # bottom-up
        x = self.feature_extractor.conv1(x)
        x = self.feature_extractor.bn1(x)
        x = self.feature_extractor.relu(x)
        x = self.feature_extractor.maxpool(x)
        layer1_output = self.feature_extractor.layer1(x)
        layer2_output = self.feature_extractor.layer2(layer1_output)
        layer3_output = self.feature_extractor.layer3(layer2_output)
        layer4_output = self.feature_extractor.layer4(layer3_output)

        output = []
        
        # conv6 output. input is output of layer4
        embedding = self.conv6(layer4_output)
        
        # conv7 output. input is relu activation of conv6 output
        output.append(self.conv7(F.relu(embedding)))
        output.append(embedding)
        
        # top-down
        output.append(self.latlayer1(layer4_output))
        output.append(self.toplayer1(self._upsample_add(output[-1], self.latlayer2(layer3_output))))
        output.append(self.toplayer2(self._upsample_add(output[-1], self.latlayer3(layer2_output))))
        
        return output[::-1]

```

Note that FPN has already added two more convolutional layers `conv6` and `conv7` on top of `layer4`.

```python
fpn = FPN()

output = fpn(image_inputs)

for layer in output:
    print(layer.size())
```
Note that all layers have the same number of channels (64), and width and height is half of the previous layer width and height.<br>
`torch.Size([2, 64, 32, 32])`<br>
`torch.Size([2, 64, 16, 16])`<br>
`torch.Size([2, 64, 8, 8])`<br>
`torch.Size([2, 64, 4, 4])`<br>
`torch.Size([2, 64, 2, 2])`<br>

#### 3. Prediction Network

Using `Detector` class that implements detector network.

```python
class Detector(nn.Module):
    num_anchors = 9

    def __init__(self, num_classes=2):
        super(Detector, self).__init__()
        self.fpn = FPN()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for feature_map in fms:
            loc_pred = self.loc_head(feature_map)
            cls_pred = self.cls_head(feature_map)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(
                x.size(0), -1, 4
            )  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(
                x.size(0), -1, self.num_classes
            )  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
            
        
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

    @staticmethod
    def _make_head(out_planes):
        layers = []
        for _ in range(4):  # 4 layered convolution network
            layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(64, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

```

Note that the detector has two heads, one for class prediction and another for location prediction.

```python
image_inputs = torch.rand((2, 3, 256, 256))
detector = Detector()
location_pred, class_pred = detector(image_inputs)

print('location_pred size: {}'.format(location_pred.size()))
print('class_pred size: {}'.format(class_pred.size()))
```
The output is:<br>
`location_pred size: torch.Size([2, 12276, 4])`<br>
`class_pred size: torch.Size([2, 12276, 2])`<br>

So what is `12276` represents?<br>
Location predictor (loc_pred) in the detector using multiple convolutions to transform the output to the following:

`torch.Size([2, 9*4, 32, 32])  # (batch_size, num_anchor*4 , H, W)`<br>
`torch.Size([2, 9*4, 16, 16])`<br>
`torch.Size([2, 9*4, 8, 8])`<br>
`torch.Size([2, 9*4, 4, 4])`<br>
`torch.Size([2, 9*4, 2, 2])`<br>

`(batch_size, number_of_anchor*4 , H, W)` re-arranged as follows:<br>
`(batch_size, num_anchor*4 , H, W)`-->`(batch_size, H, W, num_anchor*4)`-->`(batch_size, H*W*num_anchor, 4)`<br>
where `num_anchor = 9` <br>
So, `32*32*9 + 16*16*9 + 8*8*9 + 4*4*9 + 2*2*9 = 12276`.<br>
From the above re-arrangement, it is clear that each feature map of FPN (starting from (32, 32) and end in (2, 2)) has `9*4` sized mapping.




## Generating Anchor Box<br>

In this custom network, there are **9** anchors for every feature map. One element of the feature map represents **segments of pixels** in the original image.
![Image](/img/in-post/201102 Detect/4.png)
The **9** anchor boxes: It have **3** aspect ratios of sizes 1/2, 1 and 2. For each size, there are 3 scales. These anchors of the appropriate sizes are generated for each of **5** **feature maps**.

Let's take a look at the `DataEncoder` class, `DataEncoder.__init__`:

```python
def __init__(self, input_size):
        self.input_size = input_size
        self.anchor_areas = [8 * 8, 16 * 16., 32 * 32., 64 * 64., 128 * 128]  # p3 -> p7
        self.aspect_ratios = [0.5, 1, 2]
        self.scales = [1, pow(2, 1 / 3.), pow(2, 2 / 3.)]
        num_fms = len(self.anchor_areas)
        fm_sizes = [math.ceil(self.input_size[0] / pow(2., i + 3)) for i in range(num_fms)]
        self.anchor_boxes = []
        for i, fm_size in enumerate(fm_sizes):
            anchors = generate_anchors(self.anchor_areas[i], self.aspect_ratios, self.scales)
            anchor_grid = generate_anchor_grid(input_size, fm_size, anchors)
            self.anchor_boxes.append(anchor_grid)
        self.anchor_boxes = torch.cat(self.anchor_boxes, 0)
        self.classes = ["__background__", "person"]

```

As seen, it chose the anchor area which is responsible for generating anchors for the output layer of `FPN`.<br>
`256/8  = 32   # first is for generating achchors for 1st FPN output layer`<br>
`256/16 = 16`<br>
`.`<br>
`.`<br>
`256/128 = 2`<br> 
To generate 9 anchors, using predefined ratios and scales.
```python
def generate_anchors(anchor_area, aspect_ratios, scales):
    anchors = []
    for scale in scales:
        for ratio in aspect_ratios:
            h = math.sqrt(anchor_area/ratio)
            w = math.sqrt(anchor_area*ratio)
            x1 = (math.sqrt(anchor_area) - scale * w) * 0.5
            y1 = (math.sqrt(anchor_area) - scale * h) * 0.5
            x2 = (math.sqrt(anchor_area) + scale * w) * 0.5
            y2 = (math.sqrt(anchor_area) + scale * h) * 0.5
            anchors.append([x1, y1, x2, y2])
    return torch.Tensor(anchors)
```
for each feature map, a grid will be created, which will allow all the possible box.
```python
def generate_anchor_grid(input_size, fm_size, anchors):
    grid_size = input_size[0] / fm_size
    x, y = torch.meshgrid(torch.arange(0, fm_size) * grid_size, torch.arange(0, fm_size) * grid_size)
    anchors = anchors.view(-1, 1, 1, 4)
    xyxy = torch.stack([x, y, x, y], 2).float()
    boxes = (xyxy + anchors).permute(2, 1, 0, 3).contiguous().view(-1, 4)
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, input_size[0])
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, input_size[1])
    return boxes
```

Next check the size of anchor boxes by input different image size:
```python
height_width = (256, 256)
data_encoder = DataEncoder(height_width)
print('anchor_boxes size: {}'.format(data_encoder.anchor_boxes.size()))
```
`anchor_boxes size: torch.Size([12276, 4])`<br>

```python
height_width = (300, 300)
data_encoder = DataEncoder(height_width)
print('anchor_boxes size: {}'.format(data_encoder.anchor_boxes.size()))
```
`anchor_boxes size: torch.Size([17451, 4])`<br>

So to compare the anchor boxes size with network output size:

| Image input size        | Anchor boxes size  | Detector Network output size|
| :---------------------: | :----------------: | :-------------------------: |
| (256, 256)              | [12276, 4]         | [batch_size, 12276, 4]      |
| (300, 300)              | [17451, 4]         | [batch_size, 17451, 4]      |


Basically, we want to encode the location target so that the **size of location target** becomes equal to the **size of anchor boxes**.


## Matching Predictions with Ground Truth<br>

After having a wide range of anchors, we want to know which of them are the most suitable for training. By computing the intersection over union (IoU) between anchors and target boxes, those boxes that will have the maximum metric will be used further in training.<br>
If anchor box's IoU is in between 0.4 and 0.5, it can be considered as a bad match with the target and ignore it in the training process;<br>
If anchor box's IoU is below 0.4, it can be considered as background;<br>
After that, all of the matched boxes should be encoded.<br>

```python
def encode(self, boxes, classes):
        iou = compute_iou(boxes, self.anchor_boxes)
        iou, ids = iou.max(1)
        loc_targets = encode_boxes(boxes[ids], self.anchor_boxes)
        cls_targets = classes[ids]
        cls_targets[iou < 0.5] = -1
        cls_targets[iou < 0.4] = 0

        return loc_targets, cls_targets
```

1. Encoding Boxes
**Instead of predicting the bounding box location on the image directly, the bounding box regressor predicts the offset of the bounding box to anchor boxes**. Representing the bounding box with respect to anchor boxes requires encoding.<br>
Generally a bounding box is presented in [𝑥𝑚𝑖𝑛,𝑦𝑚𝑖𝑛,𝑥𝑚𝑎𝑥,𝑦𝑚𝑎𝑥] format. However, at the time of learning these boxes, it learns the bounding boxes with **respect to nearby anchors**.<br>

```python
def encode_boxes(boxes, anchors):
    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1  # width and height
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh   # center of the bounding box
    boxes_wh = boxes[:, 2:] - boxes[:, :2] + 1
    boxes_ctr = boxes[:, :2] + 0.5 * boxes_wh
    return torch.cat([(boxes_ctr - anchors_ctr) / anchors_wh, torch.log(boxes_wh / anchors_wh)], 1)
```

**Precisely, it is encoded as follows:**
- Difference between center co-ordinates of the nearby anchors and the ground truth bounding box, divided by the anchor width-height. `(boxes_ctr - anchors_ctr) / anchors_wh`<br>
- Logs of width and height ratio. `torch.log(boxes_wh / anchors_wh)` <br>

Training is trying to learn how to move the predicted box to look the same as the target. For that purpose, we should encode anchors as the offsets to the target bounding boxes which need to learn.<br>

- The **offset** is calculated with respect to the center of the box and includes how the width and the height should be regressed.
Concatenate along axis-1 by using `torch.cat([(boxes_ctr - anchors_ctr) / anchors_wh, torch.log(boxes_wh / anchors_wh)], 1)`. This is the encoded format of bounding boxes. <br>


2. Decoding Boxes
In order to obtain the boxes as standard format:
```python
def decode_boxes(deltas, anchors):
    if torch.cuda.is_available():
        anchors = anchors.cuda()
    anchors_wh = anchors[:, 2:] - anchors[:, :2] + 1
    anchors_ctr = anchors[:, :2] + 0.5 * anchors_wh
    pred_ctr = deltas[:, :2] * anchors_wh + anchors_ctr
    pred_wh = torch.exp(deltas[:, 2:]) * anchors_wh
    return torch.cat([pred_ctr - 0.5 * pred_wh, pred_ctr + 0.5 * pred_wh - 1], 1)
```

Note that the classification choose threshold 0.5 to not consider all of the predictions with lower probabilities. Even after thresholding, some bounding boxes with similar box coordinates need to be suppressed. To remove this redundancy, vanilla non-maximum-suppression (NMS) should be applied.

```python
 def decode(self, loc_pred, cls_pred, cls_threshold=0.7, nms_threshold=0.3):
        all_boxes = [[] for _ in range(len(loc_pred))]  # batch_size

        for sample_id, (boxes, scores) in enumerate(zip(loc_pred, cls_pred)):
            boxes = decode_boxes(boxes, self.anchor_boxes)

            conf = scores.softmax(dim=1)
            sample_boxes = [[] for _ in range(len(self.classes))]
            for class_idx, class_name in enumerate(self.classes):
                if class_name == '__background__':
                    continue
                class_conf = conf[:, class_idx]
                ids = (class_conf > cls_threshold).nonzero().squeeze()
                ids = [ids.tolist()]
                keep = compute_nms(boxes[ids], class_conf[ids], threshold=nms_threshold)

                conf_out, top_ids = torch.sort(class_conf[ids][keep], dim=0, descending=True)
                boxes_out = boxes[ids][keep][top_ids]

                boxes_out = boxes_out.cpu().numpy()
                conf_out = conf_out.cpu().numpy()

                c_dets = np.hstack((boxes_out, conf_out[:, np.newaxis])).astype(np.float32, copy=False)
                c_dets = c_dets[c_dets[:, 4].argsort()]
                sample_boxes[class_idx] = c_dets

            all_boxes[sample_id] = sample_boxes

        return all_boxes

```

## Loss Function<br>