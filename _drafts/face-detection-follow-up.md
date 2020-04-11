---
layout: post
title: 从WiderFace看人脸检测的发展
img: wider-face.jpg
tags:
    - 人脸检测
    - 论文笔记
---

本文主要是为了跟踪人脸检测相关的论文工作, 采用后来插入的顺序进行排列. 如果读者需要按时间顺序梳理人脸检测的发展过程, 请从最后往前阅读.

## 论文

### **[2015] WIDER FACE: A Face Detection Benchmark** [^1]  
  据我的观察, 学术界贡献心得数据集主要有两个原因, 1) 该领域数据极度匮乏, 无法支撑领域的发展; 2) 旧的数据集精度上已经饱和了, 无法进一步区分新方法的提升.
  而WIDER FACE的意义, 我觉得是双重的. 一方面当时的人脸检测数据集(训练集)还是非常匮乏的(FDDB, AFW PASCAL FACE 等数据集是不提供训练集的), 
  特别是在深度学习火爆之后, 需要更多的数据才能喂饱深度神经网络; 另一方面, FDDB 测试集榜上已经出现了精度饱和的趋势了. 于是WIDER FACE 应运而生, 大家喜闻乐见!
  
  与其他贡献数据集论文的行文结构一样, 论文详细介绍了数据的采集, 标注过程. 详细介绍了数据集的特点. 一句话概括就是, 我这个数据集特别`Wild`, 
  包含了现实中各种场景. 对于检测过程中典型问题, 诸如人脸尺寸, 遮挡, 姿态均有覆盖, 且标注了各个属性的程度. 同时介绍了自己的新方法: `Multi-scale Cascade CNN`, 
  作为**solid baseline**, 这也是很重要的.

  ![](/assets/img/multi-scale-cascade-cnn.png)
  上图是论文方法的总体流程. 方法分为两个阶段, 第一阶段负责抛出可能能包含人脸的区域; 第二阶段对于抛出的区域进行重新判定和边框回归. 现在看来第一阶段其实和同期的FasterRCNN[^2]的RPN是非常相似的, 只不过没有anchor的概念. 不过每个网络任然预测三个不同的尺度, 这实际上隐含了`Anchor`的作用. 论文用了四个不同的网络, 分别负责四个尺度范围, 非常简单粗暴. 不过我们应当站在历史的时间点上看历史, 毕竟FPN[^3]是后来的事情了.

  第二阶段是一个简单的检测回归的网络, 作者复用了第一阶段的网络结构, 并且在第一阶段训得的网络基础上进行微调. 至于检测网络中常见的正负样本平衡, IOU匹配和现在Anchor匹配的方法是类似的.

  最后作者用四个代表性的人脸检测算法: VJ[^4], ACF[^5], DPM[^6] 和 Faceness[^7], 在没有重新训练的情况下在WIDER FACE 测试集上进行了测试, 结果各项指标均较低, 说明了测试集的难度, 和此类算法的性能还不够高. 另外, 把以上方法在 WIDER FACE 训练集上重新训练, 在 WIDER FACE 测试集和 FDDB 中精度均获得了提升, 说明了 WIDER FACE 训练集的有效性. 
  
  


## 参考文献

[^1]: S. Yang, P. Luo, C. C. Loy, and X. Tang, “WIDER FACE: A Face Detection Benchmark,” arXiv:1511.06523 [cs], Nov. 2015, Accessed: Apr. 10, 2020. [Online]. Available: http://arxiv.org/abs/1511.06523.
[^2]: S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” arXiv:1506.01497 [cs], Jan. 2016, Accessed: Apr. 11, 2020. [Online]. Available: http://arxiv.org/abs/1506.01497.
[^3]: T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature Pyramid Networks for Object Detection,” arXiv:1612.03144 [cs], Apr. 2017, Accessed: Mar. 25, 2020. [Online]. Available: http://arxiv.org/abs/1612.03144.
[^4]: “Robust Real-Time Face Detection, SpringerLink.” https://link.springer.com/article/10.1023/B:VISI.0000013087.49260.fb (accessed Apr. 11, 2020).
[^5]: “Aggregate channel features for multi-view face detection - IEEE Conference Publication.” https://ieeexplore.ieee.org/document/6996284/?denied= (accessed Apr. 11, 2020).
[^6]: M. Mathias, R. Benenson, M. Pedersoli, and L. Van Gool, “Face Detection without Bells and Whistles,” in Computer Vision – ECCV 2014, Cham, 2014, pp. 720–735, doi: 10.1007/978-3-319-10593-2_47.
[^7]: S. Yang, P. Luo, C. C. Loy, and X. Tang, “From Facial Parts Responses to Face Detection: A Deep Learning Approach,” arXiv:1509.06451 [cs], Sep. 2015, Accessed: Apr. 11, 2020. [Online]. Available: http://arxiv.org/abs/1509.06451.



