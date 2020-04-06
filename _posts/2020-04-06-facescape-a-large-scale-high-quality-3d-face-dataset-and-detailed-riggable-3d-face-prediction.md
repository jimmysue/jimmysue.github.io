---
layout: post
title: 'FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable
  3D Face Prediction'
date: 2020-04-06 11:56 +0800
tags:
  - 论文笔记
  - 人脸重建
---

> 传统机器学习算法靠脑力, 深度学习时代靠算例, 数据时代靠财力. -- 佚名

Paper: [https://arxiv.org/abs/2003.13989](https://arxiv.org/abs/2003.13989)  
Code: [https://github.com/zhuhao-nju/facescape](https://github.com/zhuhao-nju/facescape)

文章主要做了以下几件事情

1. 采用多视图方法构建了大规模的精细的人脸三维数据集, 取名为 FaceScape  
   该数据集具有18,760模型, 来自 928 个不同的人, 顶点数达到了200万个. 作者声称会将数据开源, 至于公开范围目前还不知道. 
  
  > scape:  
  >     花茎, 柱身, 我猜测作者想表达的是这个数据集起到基础性的作用, 还**很大** 😝

**关于 Displacement map** [^1]

![](/images/FaceScape.png)

[^1]: [“paGAN: real-time avatars using dynamic textures: ACM Transactions on Graphics: Vol 37, No 6.” https://dl.acm.org/doi/abs/10.1145/3272127.3275075 (accessed Apr. 06, 2020).](https://jaewoo-seo.com/wp-content/uploads/2019/01/SIGA2018_paGAN-compressed.pdf)
