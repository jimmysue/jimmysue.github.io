---
layout: post
title: Single-Shot Refinement Neural Network for Object Detection
permalink: RefineDet
img: refinedet-figure-1.png
tags:
- 物体检测
- 论文笔记
date: 2020-05-07 23:05 +0800
---
![](/assets/img/refinedet-figure-1.png){:height="400px"}


## 问题

- 两阶段的 Faster-RCNN 精度高但是速度较差, SSD 单阶段检测精度比SSD略差, 但是速度快
- 两阶段和单阶段各自的有点如何结合?

## 方法

- 将二分类的RPN和SSD的多分类安排在同一个网络上, Anchor Refine Module(Figure 1 上半部分)进行前景背景分类, Object Detection Module(Figure 1 下半部分)进行物体类别(**包括背景**)的分类.
- ARM 进行二分类可以过滤掉大部分简单的背景样本, 践行 ODM 样本均衡问题
- ARM 和 ODM 通过一定手段(TCB)共享特征, 并且位置进行了递进式地回归
- **Transfer Connection Block** 见图 Figure 2.  
  ![](/assets/img/refinedet-figure-2.png){:width="320px"}
- **Two-Step Cascaded Regression**: 把 ARM 得到的位置作为输入传给 ODM 进一步优化这个结果. (从文章看并没有把 ARM 的结果作为 Anchor)

## 结果

- 简单背景样本过滤带来 0.5%的提升(mAP 80.0% vs 79.5%)
- Two-Step 级联回归 带来 2.2% (mAP 79.5% vs 77.3%) 
- TCB 带来 1.1% 的替身 (mAP 79.3% vs 76.2%)  
  ![](/assets/img/refinedet-table-3.png){:width="360px"}


## 点评

1. TCB 和 ODM 实际上都在一定程度上增加了网络的复杂度, 到底是网络复杂图提升带来的性能提升大一些, 还是作者标榜的那一套 一二阶段 融合操作带来的提升大一些呢?


