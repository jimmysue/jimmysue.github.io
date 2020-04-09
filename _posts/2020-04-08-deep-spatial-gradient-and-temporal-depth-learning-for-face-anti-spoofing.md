---
layout: post
title: Deep Spatial Gradient and Temporal Depth Learning for Face Anti-spoofing
date: 2020-04-08 17:21 +0800
img: RSGB-STPM.png
fig-caption: "Illustration of the overall framework. The inputs are consecutive frames with a fixed interval. Each frame is processed by cascaded RSGB with a shared backbone which generates a corresponding coarse depth map. The number in RSGB cubes denotes the output channel number of RSGB. STPM is plugged between frames for estimating the temporal depth, which is used for refining the corresponding coarse depth map. The framework works well by learning with the overall loss functions."
tags:
  - 论文笔记
  - 活体检测
---

Paper: [https://arxiv.org/abs/2003.08061](https://arxiv.org/abs/2003.08061)  
Code: [https://github.com/clks-wzz/FAS-SGTD](https://github.com/clks-wzz/FAS-SGTD)

## Intuition

> 1) detailed discriminative clues (e.g., spatial gradient magnitude)  between living and spoofing face may be discarded through stacked vanilla convolutions, and   
> 2) the dynamics  of 3D moving faces provide important clues in detecting  the spoofing faces.

> 1. Traditional methods usually design local descriptors for solving PAD while modern deep learning methods can learn to extract relatively high-level semantic features instead. Despite their effectiveness, we argue that low-level fine-grained patterns can also play a vital role in distinguishing living and spoofing faces, e.g. the spatial gradient magnitude shown in Fig. 1. So how to aggregate local fine-grained information into convolutional networks is still unexplored for face anti-spoofing task. 
> 2. Recent depth supervised face anti-spoofing methods [2, 34] estimate facial depth based on a single frame and leverage depth as dense pixel-wise supervision in a direct manner. We argue that the virtual discrimination of depth between living and spoofing faces can be explored more adequately by multiple frames.

## 3.2 Loss Function

### 3.2.1 Contrastive Depth Loss

*经过八个梯度算子之后, 八张特征图的二范距离*

![](/assets/img/CDL.png)