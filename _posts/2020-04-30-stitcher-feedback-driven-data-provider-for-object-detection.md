---
layout: post
title: 'Stitcher: Feedback-driven Data Provider for Object Detection'
permalink: sticher
toc: true
img: stitch-fig-3.png
tags:
- 论文笔记
- 物体检测
mathjax: true
date: 2020-04-30 18:13 +0800
---
Paper: Y. Chen et al., “Stitcher: Feedback-driven Data Provider for Object Detection,” arXiv:2004.12432 [cs], Apr. 2020, Accessed: Apr. 30, 2020. [Online]. Available: [http://arxiv.org/abs/2004.12432](http://arxiv.org/abs/2004.12432).

![](/assets/img/stitch-fig-3.png)

## 问题:  
  - 检测器小物体的检测能力相较于中等和大物体低了将近一半
  - 小物体样本不少， 但是包含小物体的图像较少， 导致小物体训练中不平衡
  
## 方法：  
  - 通过**stitch**增益  
    - 将图像缩小，然后拼接
    - 将整个batch的图像缩小
  - 在线反馈动态调整
    - 计算小尺度比例指标$$r_s^t$$ ， 当其小预阈值 $$\tau$$ 触发 stitching
      - 直接用输入的物体尺寸计算小物体的比例
      - 计算小物体的分类 loss 的比例
      - 回归 loss 比例
      - 总 loss 比例

## 结果  
  - 检测和分割任务精度均有提升  
![](/assets/img/stitch-table4-5.png)  
  - 可以预防过拟合  
![](/assets/img/stitch-table-7.png)  
  - 阈值的选取和缩放的倍率影响如下  
![](/assets/img/stitch-fig-8-table-11.png)