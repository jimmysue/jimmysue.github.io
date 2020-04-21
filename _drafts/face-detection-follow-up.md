---
layout: post
title: 从WiderFace看人脸检测的发展
img: wider-face.jpg
toc: true
tags:
    - 人脸检测
    - 论文笔记
---

本文主要是为了跟踪人脸检测相关的论文工作, 采用后来插入的顺序进行排列. 如果读者需要按时间顺序梳理人脸检测的发展过程, 请从最后往前阅读.

## WIDER FACE

  WIDER FACE: A Face Detection Benchmark[^1]  

  据我的观察, 学术界贡献心得数据集主要有两个原因, 1) 该领域数据极度匮乏, 无法支撑领域的发展; 2) 旧的数据集精度上已经饱和了, 无法进一步区分新方法的提升.
  而WIDER FACE的意义, 我觉得是双重的. 一方面当时的人脸检测数据集(训练集)还是非常匮乏的(FDDB, AFW PASCAL FACE 等数据集是不提供训练集的), 
  特别是在深度学习火爆之后, 需要更多的数据才能喂饱深度神经网络; 另一方面, FDDB 测试集榜上已经出现了精度饱和的趋势了. 于是WIDER FACE 应运而生, 大家喜闻乐见!
  
  与其他贡献数据集论文的行文结构一样, 论文详细介绍了数据的采集, 标注过程. 详细介绍了数据集的特点. 一句话概括就是, 我这个数据集特别`Wild`, 
  包含了现实中各种场景. 对于检测过程中典型问题, 诸如人脸尺寸, 遮挡, 姿态均有覆盖, 且标注了各个属性的程度. 同时介绍了自己的新方法: `Multi-scale Cascade CNN`, 
  作为**solid baseline**, 这也是很重要的.

  ![](/assets/img/multi-scale-cascade-cnn.png){: style="float: right" height="300px"}

  上图是论文方法的总体流程. 方法分为两个阶段, 第一阶段负责抛出可能能包含人脸的区域; 第二阶段对于抛出的区域进行重新判定和边框回归. 现在看来第一阶段其实和同期的FasterRCNN[^2]的RPN是非常相似的, 只不过没有anchor的概念. 不过每个网络任然预测三个不同的尺度, 这实际上隐含了`Anchor`的作用. 论文用了四个不同的网络, 分别负责四个尺度范围, 非常简单粗暴. 不过我们应当站在历史的时间点上看历史, 毕竟FPN[^3]是后来的事情了.

  第二阶段是一个简单的检测回归的网络, 作者复用了第一阶段的网络结构, 并且在第一阶段训得的网络基础上进行微调. 至于检测网络中常见的正负样本平衡, IOU匹配和现在Anchor匹配的方法是类似的.

  最后作者用四个代表性的人脸检测算法: VJ[^4], ACF[^5], DPM[^6] 和 Faceness[^7], 在没有重新训练的情况下在WIDER FACE 测试集上进行了测试, 结果各项指标均较低, 说明了测试集的难度, 和此类算法的性能还不够高. 另外, 把以上方法在 WIDER FACE 训练集上重新训练, 在 WIDER FACE 测试集和 FDDB 中精度均获得了提升, 说明了 WIDER FACE 训练集的有效性. 

<hr>

## 2016 

### MTCNN(SPL)

  Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks[^8]

  ![](/assets/img/MTCNN.png){:style="float: right" width="300px"} 
  这篇论文就是名噪一时的`MTCNN`人脸检测. 如右图所示, 方法采用boost思想, 通过组合三个弱检测器, 得到一个强大的鲁邦的人脸检测算法. 第一阶段 P-Net 计算量最小, 能力最弱. 因为这一阶段, 需要跑所有的金字塔图像, 总体计算量最大. 它的任务在于排除绝大多数的背景样本, 保留疑似人脸的区域, 等待下一级段检测器重新判定. 第二阶段 R-Net 比第一阶段稍大, 它负责杀掉大部分的误检(False Positive), 精化目标位置; 第三阶段是成品阶段, 进一步降低误检, 提高位置精度, 同时预测五个人脸关键点.

  训练方面, 作者提出两点有效的实践, 分别是在线难例挖掘和人脸关键点联合监督. 

  MTCNN 方法不仅简单, 速度快, 而且在当时精度表现引领风骚. 它的快得益于其计算量的分配上, 使用极简的P-Net进行预筛选, 过滤大量的背景样本. 后续 R-Net O-Net 模型体量逐渐变大, 能力也随之提升, 但所需要负担的样本逐渐减小. 当然 P-Net 的瓶颈也非常明显, 由于其网络简单, 能力有限, 特别是能够判定的尺度范围非常局限, 这使得, 检测器需要构造尺度间隔非常小的图像金字塔, 这无疑增加了计算量. 特别是需要检测小人脸的时候, 大尺寸的输入图计算量显得愈加繁重. 这也是许多人脸检测算法难以避免的速度瓶颈. 

<hr>

### CMS-RCNN([DLB](https://link.springer.com/chapter/10.1007/978-3-319-61657-5_3))

CMS-RCNN: Contextual Multi-Scale Region-based CNN for Unconstrained Face Detection[^9]

![](/assets/img/CMS-RCNN.png){:style="float: right" width="300px"} 

论文发表于2016年, 正处于 `Faster-RCNN` 提出不久, 类似方法大行其道之时. 本文着眼于解决人脸检测中, 由于低分辨率, 光照, 压缩等原因造成的图像质量低和人脸遮挡和姿态造成的人脸多样两个核心困难. 作者深刻研习了 `RCNN` 三部曲的发展过程, 指出了第三部`Faster-RCNN` 只从最后的stride 16 位置的特征进行进行候选框预测和检测, 注定无法胜任小物体检测的缺点. 针对图像的分辨率低(作者多次提到分辨率低, 应该和指人脸尺度小一个道理), 遮挡等问题, 作者认为算法应当充分利用人脸周围(context)信息, 以辅助检测. 综上, 作者提出了本文方法, 曰: Contextual Multi-Scale R-CNN.

顾名思义, contextual 是指方法考虑人脸周围的信息以辅助检测, multi-scale 则完全是为了实现 contextual 的具体实现, 最后 R-CNN 指出了方法的框架.

Fig2 概括了方法流程. 沿用Faster-RCNN两阶段检测框架, 即RPN + 局部特征优化. 在两个阶段都融合了不同尺度的特征, 形式上达到了多尺度的融合, 语义上考虑的目标周围的信息. 多尺度特征的融合是个精细活, 需要考虑不同的尺度特征的激活数量级的差异. 作者指出, 深层的激活数值往往比浅层的小, 如果直接连接有被浅层特征支配的风险. 因此作者提出 L2 规范化. 说白了是特征图每个通道**分别**进行二范归一化. 公式为:

$$
\begin{aligned}
  \hat{x} &= \frac{x}{\|x\|_2} \\
\|x\|_2 &= (\sum_{i=1}^{d} |x_i |)^{\frac{1}{2}}
\end{aligned}
$$

并且每个通道都会独立训练一个缩放系数, 以自适应缩放特征量级. 这部分与 `Batch Normalization` 极为相似. 

论文结果发布较早, 放在当时绝对是SOTA精度了, 从PR曲线上看与轻量级 MTCNN 方法拉开了巨大差距. 这样的精度提升, 作者提出的多尺度融合是一部分, 大型深度网络强大的能力也起到关键性的作用. 毕竟从 Fig6 的对比上看, 是否使用 Context的差异直观上是小于和 其它方法的差异的.

至于作者认为多尺度特征融合, 使网络检测中结合了"身体"的背景信息(Body Context). 这样的论断在深度学习方法中很常见, 不可证伪. 我们可以说多尺度特征图的融合确实有效, 而这样的有效性是否就是我们主观赋予的"身体"背景信息, 却很难说.


### MSCNN (ECCV)


<hr>

## 2017

### HR (CVPR)

Finding Tiny Face[^10]

![](/assets/img/Finding-Tiny-Face.png){: width="720px"} 

网红文章, 先导图极富冲击力, 吸引眼球. 本文的中心非常明确, 就是要解决小人脸的检测问题. 文章非常细致地阐述了作者解决小人脸的思路和分析了小人脸检测可能遇到的问题. 抛开繁冗的论述, 作者的方法基于三点: 1) 在一定范围内的人脸尺度变化, 依赖网络的尺度不变性; 2) 大范围的尺度变化依靠图像金字塔; 3) 辅以上下文特征帮助人脸检测, 特别是小人脸检测. 

图 Figure9 是论文方法的示意图. 第一列"re-scaled input" 构建图像金字塔, 第二列 "shared CNNs" 是网路的主干, 用来提取特征, 且作者用 `hyper column` 的方法达到融合多尺度特征的作用. 最后的"response map" 是一个检测的 `head`. 检测头部通过`template`达到检测不同尺度人脸的目的, 方法名称 `HR`, hybrid resolution 也是由此得来的. 现在看来这个 `head` 实际上就是一个`RPN`. 论文中的`template`实际上就是`anchor`的意思. 所以论文的方法说白了是一个加了图像金字塔的和`hyper column`特征的 `RPN`. 每个 template 负责检测一定尺度范围的人脸, 和 anchor 以及 `SSD` 中的所谓 prior 功能一致. 由于图像金字塔扩展了尺度范围, 论文进行的 template 缩减. 论文的 A 型模板的尺度范围是 40px ~ 140px, 在图像金字塔(尺度分别为0.5, 1, 2)的作用下, 能检测到的人脸尺度范围变成了20px ~ 280px. 而对于20px以下的人脸则无能为力, 所以作者需要使用专门的 B 型模板来, 来检测小于20px的人脸.

### SSH (CVPR)

> Although SSH is a detector, it is more similar to the object proposal algorithms which are used as the first stage in detection pipelines.

![](https://pic4.zhimg.com/80/v2-f0b552601aff5c709e27e68143a2f237_1440w.jpg){:height="360px"}  

上图为知乎 [dengdeng](https://zhuanlan.zhihu.com/p/64397468), 重新排版之后的示意图. 论文所说的 Single Stage 实际上是指检测只有 RPN 阶段, 毕竟人脸是个二分类问题, RPN 足够. 多尺度通过采用不同深度的特征层进行预测, 与 SSD 思想类似. 论文所说的通过增加感受野来增加"Context"信息属于常规操作.

### S$$^3$$FD (ICCV)




## 2018

### Zhu. et al (CVPR)

Seeing Small Faces from Robust Anchor's Perspective. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

### Pyramid Box (ECCV)

Pyramidbox: A context-assisted single shot face detector

### DSFD (CVPR)

DSFD: dual shot face detector

## 2019

### FSDet (IJCV)

Single-shot scale-aware network for real-time face detection

### CSP (CVPR)

High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection. arXiv preprint arXiv:1904.02948, 2019.

### RetinaFace

## 2020 

### ASFD




## 参考文献

[^1]: S. Yang, P. Luo, C. C. Loy, and X. Tang, “WIDER FACE: A Face Detection Benchmark,” arXiv:1511.06523 [cs], Nov. 2015, Accessed: Apr. 10, 2020. [Online]. Available: http://arxiv.org/abs/1511.06523.
[^2]: S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” arXiv:1506.01497 [cs], Jan. 2016, Accessed: Apr. 11, 2020. [Online]. Available: http://arxiv.org/abs/1506.01497.
[^3]: T.-Y. Lin, P. Dollár, R. Girshick, K. He, B. Hariharan, and S. Belongie, “Feature Pyramid Networks for Object Detection,” arXiv:1612.03144 [cs], Apr. 2017, Accessed: Mar. 25, 2020. [Online]. Available: http://arxiv.org/abs/1612.03144.
[^4]: “Robust Real-Time Face Detection, SpringerLink.” https://link.springer.com/article/10.1023/B:VISI.0000013087.49260.fb (accessed Apr. 11, 2020).
[^5]: “Aggregate channel features for multi-view face detection - IEEE Conference Publication.” https://ieeexplore.ieee.org/document/6996284/?denied= (accessed Apr. 11, 2020).
[^6]: M. Mathias, R. Benenson, M. Pedersoli, and L. Van Gool, “Face Detection without Bells and Whistles,” in Computer Vision – ECCV 2014, Cham, 2014, pp. 720–735, doi: 10.1007/978-3-319-10593-2_47.
[^7]: S. Yang, P. Luo, C. C. Loy, and X. Tang, “From Facial Parts Responses to Face Detection: A Deep Learning Approach,” arXiv:1509.06451 [cs], Sep. 2015, Accessed: Apr. 11, 2020. [Online]. Available: http://arxiv.org/abs/1509.06451.
[^8]: K. Zhang, Z. Zhang, Z. Li, and Y. Qiao, “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks,” IEEE Signal Process. Lett., vol. 23, no. 10, pp. 1499–1503, Oct. 2016, doi: 10.1109/LSP.2016.2603342.
[^9]: C. Zhu, Y. Zheng, K. Luu, and M. Savvides, “CMS-RCNN: Contextual Multi-Scale Region-based CNN for Unconstrained Face Detection,” arXiv:1606.05413 [cs], Jun. 2016, Accessed: Apr. 12, 2020. [Online]. Available: http://arxiv.org/abs/1606.05413.
[^10]: P. Hu and D. Ramanan, “Finding Tiny Faces,” arXiv:1612.04402 [cs], Dec. 2016, Accessed: Apr. 17, 2020. [Online]. Available: http://arxiv.org/abs/1612.04402.
