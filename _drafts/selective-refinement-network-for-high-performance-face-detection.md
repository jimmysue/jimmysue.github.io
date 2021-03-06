---
layout: post
title: Selective Refinement Network for High Performance Face Detection
tags:
  - 人脸检测
  - 论文笔记
permalink: srn
img: srn-figure-2.png
---

![](/assets/img/srn-figure-2.png)
*文章和[RefineDet](/RefineDet.html)出自同门, 本文基本上继承了 RefineDet 的基本思想, 是 RefineDet 在人脸检测任务上的实践. 同时作者针对人脸检测任务的特殊性进行了相应的调整和改进*

Paper: https://arxiv.org/pdf/1809.02693.pdf

## 问题

- PR 曲线在高召回率下精确率低下
- 作者怀疑先前的算法主要侧重与最求召回率, 而忽略了因此带来的误检(false positive)  
  > The reason is that existing algorithms pay more attention to pursuing high recall rate but ignore the problem of excessive false positives.  
  
  *特别是当前算法为了提高小人脸的召回率, 会加密小 anchor, 由此带来了误检增多.*  

{% include image.html url="/assets/img/SRN-Figure-1.png" width="720px" 
description="
**Figure 1**: The effects of STC and STR on recall efficiency and location accuracy. (a) The STC and STR increase the positives/negatives ratio by about 38 and 3 times respec- tively, (b) which improve the precision by about 20% at high recall rates. (c) The STR provides better initialization for the subsequent regressor, (d) which produces more accurate locations, i.e., as the IoU threshold increases, the AP gap gradually increases.
"
%}


- 这是另一个问题