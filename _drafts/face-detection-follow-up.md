---
layout: post
title: 从WiderFace看人脸检测的发展
tags:
    - 人脸检测
    - 论文笔记
---

本文主要是为了跟踪人脸检测相关的论文工作, 采用后来插入的顺序进行排列. 如果读者需要按时间顺序梳理人脸检测的发展过程, 请从最后往前阅读.

## 论文

- **[2015] WIDER FACE: A Face Detection Benchmark** [^1]  
  据我的观察, 学术界贡献心得数据集主要有两个原因, 1) 该领域数据极度匮乏, 无法支撑领域的发展; 2) 旧的数据集精度上已经饱和了, 无法进一步区分新方法的提升.
  而WIDER FACE的意义, 我觉得是双重的. 一方面当时的人脸检测数据集(训练集)还是非常匮乏的(FDDB, AFW PASCAL FACE 等数据集是不提供训练集的), 
  特别是在深度学习火爆之后, 需要更多的数据才能喂饱深度神经网络; 另一方面, FDDB 测试集榜上已经出现了精度饱和的趋势了. 于是WIDER FACE 应运而生, 大家喜闻乐见!
  
  与其他贡献数据集论文的行文结构一样, 论文详细介绍了数据的采集, 标注过程. 详细介绍了数据集的特点. 一句话概括就是, 我这个数据集特别`Wild`, 
  包含了现实中各种场景. 对于检测过程中典型问题, 诸如人脸尺寸, 遮挡, 姿态均有覆盖, 且标注了各个属性的程度. 同时介绍了自己的新方法: `Multi-scale Cascade CNN`, 
  作为**solid baseline**, 这也是很重要的.

  ![](/assets/img/multi-scale-cascade-cnn.png)
  上图是论文方法的总体流程.
  
  


## 参考文献

[^1]: S. Yang, P. Luo, C. C. Loy, and X. Tang, “WIDER FACE: A Face Detection Benchmark,” arXiv:1511.06523 [cs], Nov. 2015, Accessed: Apr. 10, 2020. [Online]. Available: http://arxiv.org/abs/1511.06523.
