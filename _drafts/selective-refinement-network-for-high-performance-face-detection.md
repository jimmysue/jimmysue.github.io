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

Paper: https://arxiv.org/pdf/1809.02693.pdf

## 问题

- PR 曲线在高召回率下精确率低下
- 作者怀疑先前的算法主要侧重与最求召回率, 而忽略了因此带来的误检(false positive)  
  > The reason is that existing algorithms pay more attention to pursuing high recall rate but ignore the problem of excessive false positives.  
  
  特别是当前算法为了提高小人脸的召回率, 会加密小 anchor, 由此带来了误检增多.  
- 