---
layout: post
title: 人脸活体检测综述
date: 2020-04-04 20:47 +0800
img: fas.jpg
tags:
  - 综述
---

**基本思路**:  
  - 传统方法: 线索->构造特征
  - 深度方法: 线索->构造算子/网路

## 论文

- Deep Spatial Gradient and Temporal Depth Learning for Face Anti-spoofing[^4]
- Searching Central Difference Convolutional Networks for Face Anti-Spoofing[^3]
- Noise Modeling, Synthesis and Classification for Generic Object Anti-Spoofing[^5]
  ![](/assets/img/noise-modeling-object-anti-spoofing.png)
- Learning Meta Model for Zero- and Few-shot Face Anti-spoofing[^6]
- [2019] Deep Tree Learning for Zero-shot Face Anti-Spoofing[^7]

## 攻击形式:

- 打印
- 屏幕照片(screen)
- 雕塑模型
- 面具

## 方法

- 成像的物理过程差异, 构造可分特征
  
  > (1) specular reflection from the printed paper surface or LCD screen; (2) image blurriness due to camera defocus; (3) image chromaticity and contrast distortion due to imperfect color rendering of printer or LCD screen; and (4) color diversity distortion due to limited color resolution of printer or LCD screen. [^1]

  > The camera used for capturing the targeted face sample will also lead to imperfect colour reproduction compared to the legitimate sample. Furthermore, a recaptured face image is likely to contain local and overall variations of colour due to other imperfections in the reproduction process of the targeted face [^2]
  ![](/assets/img/color-texture.png)

## 参考资料

- [Awesome-FAS (Face Anti-Spoofing)](https://github.com/RizhaoCai/Awesome-FAS#awesome-fas-face-anti-spoofing)
- [活体检测Face Anti-spoofing综述Fisher Yu余梓彤](https://zhuanlan.zhihu.com/p/43480539)
- [知乎汇总](https://zhuanlan.zhihu.com/p/69733383)
- [Face Anti-spoofing Attack Detection Challenge@CVPR2020](https://sites.google.com/qq.com/face-anti-spoofing/winners-results/challengecvpr2020)
- [Anti-spoofing for face recognition](https://thakkarnidhi.com/blogs/datascience/anti-spoofing-for-face-recognition/)
- https://github.com/JinghuiZhou/awesome_face_antispoofing
- https://github.com/ee09115/spoofing_detection
- https://www.pyimagesearch.com/2019/03/11/liveness-detection-with-opencv/
- [CVPR & AAAI 2020 人脸活体检测最新进展](https://zhuanlan.zhihu.com/p/114313640)
- [基于rPPG的人脸活体检测综述](https://zhuanlan.zhihu.com/p/60627380)
- https://github.com/ZitongYu/CDCN
- [What is a Presentation Attack? And how do we detect it?](https://christoph-busch.de/files/Busch-TelAviv-PAD-180116.pdf)
- [CVPR2019 人脸活体检测专题](CVPR2019 人脸活体检测专题)

## 参考文献

[^1]: D. Wen, H. Han, and A. K. Jain, “Face Spoof Detection With Image Distortion Analysis,” IEEE Transactions on Information Forensics and Security, vol. 10, no. 4, pp. 746–761, Apr. 2015, doi: 10.1109/TIFS.2015.2400395.
[^2]: Z. Boulkenafet, J. Komulainen, and A. Hadid, “Face Spoofing Detection Using Colour Texture Analysis,” IEEE Transactions on Information Forensics and Security, vol. 11, pp. 1–1, Aug. 2016, doi: 10.1109/TIFS.2016.2555286.
[^3]: Z. Yu et al., “Searching Central Difference Convolutional Networks for Face Anti-Spoofing,” arXiv:2003.04092 [cs], Mar. 2020, Accessed: Apr. 06, 2020. [Online]. Available: http://arxiv.org/abs/2003.04092.
[^4]: Z. Wang et al., “Deep Spatial Gradient and Temporal Depth Learning for Face Anti-spoofing,” arXiv:2003.08061 [cs], Mar. 2020, Accessed: Apr. 07, 2020. [Online]. Available: http://arxiv.org/abs/2003.08061.
[^5]: J. Stehouwer, A. Jourabloo, Y. Liu, and X. Liu, “Noise Modeling, Synthesis and Classification for Generic Object Anti-Spoofing,” arXiv:2003.13043 [cs, eess], Mar. 2020, Accessed: Apr. 09, 2020. [Online]. Available: http://arxiv.org/abs/2003.13043.
[^6]: Y. Qin et al., “Learning Meta Model for Zero- and Few-shot Face Anti-spoofing,” arXiv:1904.12490 [cs], Dec. 2019, Accessed: Apr. 08, 2020. [Online]. Available: http://arxiv.org/abs/1904.12490.
[^7]: Y. Liu, J. Stehouwer, A. Jourabloo, and X. Liu, “Deep Tree Learning for Zero-shot Face Anti-Spoofing,” arXiv:1904.02860 [cs], Apr. 2019, Accessed: Apr. 04, 2020. [Online]. Available: http://arxiv.org/abs/1904.02860.


