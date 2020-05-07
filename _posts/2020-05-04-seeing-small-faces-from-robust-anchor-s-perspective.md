---
layout: post
title: Seeing Small Faces from Robust Anchor's Perspective
img: seeing-small-face-emo.png
tags:
- 人脸检测
- 论文笔记
date: 2020-05-04 20:35 +0800
permalink: SeeingSmallFaces
---

## 问题

- 普通的 Anchor 设置难以保证小人脸和 Anchor 之间的重叠率, 导致训练难度大, 小人脸效果差

## 方法

- **建模**: 人脸和 Anchor 之间的期望最大重叠率(Expected Max Overlapping Scores)和 Anchor 的 stride 关系密切, stride 越小, 期望最大重叠度越大(Figure 4):  
  ![](/assets/img/2020-05-04-20-11-18.png){:width="480px"}
- 扩大特征图的尺寸, 减小 stride  
  作者提出三种扩大尺寸的方法:  
  1. 上采样
  2. 上采样然后与浅层相加
  3. 不进行下采样, 通过 dilated 卷积增加感受野  
  作者实验对比发现, 方法3效果最好(这里笔者打个问号?)  
  ![](/assets/img/Seeing-small-Figure-5.png)
- 通过偏移 Anchor 中心加密 Anchor 的分布
- 训练期增益: 对人脸的位置进行扰动, 使得人脸出现的位置在空间上均匀分布
- 对于匹配困难的人脸进行**补偿**: 如果人脸所匹配的 Anchor 最大 IoU 达不到阈值, 则将重叠率排在前$$N$$的 Anchor 设置为正样本. 作者实验表明当$$N = 5$$时效果最好.

## 结果

- 扩大特征图尺寸的方法非常有效, 尤其是 Wider Face 困难集, 因为其中小人脸占比较大. 并且少下采样一次, 通过 dilated 卷积效果最好! 上采样然后相加次之, 只进行上采样结果提升最小.
- 加密 Anchor 有利有弊, 提升召回率, 可能带来误检增多. 每个位置加密 3 个 Anchor 最佳, 且只需要对小人脸的 Anchor 加密即可. 
- 人脸位置在线扰动增益带来 AP 0.9% 的提升
- 难以匹配的人脸的补偿 Anchor 数去 5 个为宜. 

**详细的结果可以看下面的图表**:  

![](/assets/img/Seeing-small-face-Table-1-Table-2.png){:width="480px"}  
![](/assets/img/seeing-small-face-figure-8-figure-9.png)
