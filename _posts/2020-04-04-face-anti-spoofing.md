---
layout: post
title: 人脸活体检测综述
date: 2020-04-04 20:47 +0800
tags:
  - 综述
---

攻击形式:

- 打印
- 屏幕照片(screen)
- 雕塑模型
- 面具

## 方法

- 成像的物理过程差异, 构造可分特征
  
  > (1) specular reflection from the printed paper surface or LCD screen; (2) image blurriness due to camera defocus; (3) image chromaticity and contrast distortion due to imperfect color rendering of printer or LCD screen; and (4) color diversity distortion due to limited color resolution of printer or LCD screen. [^1]

  > The camera used for capturing the targeted face sample will also lead to imperfect colour reproduction compared to the legitimate sample. Furthermore, a recaptured face image is likely to contain local and overall variations of colour due to other imperfections in the reproduction process of the targeted face [^2]
  ![](/images/color-texture.png)

## 参考资料

- [Awesome-FAS (Face Anti-Spoofing)](https://github.com/RizhaoCai/Awesome-FAS#awesome-fas-face-anti-spoofing)
- [活体检测Face Anti-spoofing综述Fisher Yu余梓彤](https://zhuanlan.zhihu.com/p/43480539)
- [知乎汇总](https://zhuanlan.zhihu.com/p/69733383)
- [Face Anti-spoofing Attack Detection Challenge@CVPR2020](https://sites.google.com/qq.com/face-anti-spoofing/winners-results/challengecvpr2020)
- [Anti-spoofing for face recognition](https://thakkarnidhi.com/blogs/datascience/anti-spoofing-for-face-recognition/)


## 参考文献

[^1]: D. Wen, H. Han, and A. K. Jain, “Face Spoof Detection With Image Distortion Analysis,” IEEE Transactions on Information Forensics and Security, vol. 10, no. 4, pp. 746–761, Apr. 2015, doi: 10.1109/TIFS.2015.2400395.
[^2]: Z. Boulkenafet, J. Komulainen, and A. Hadid, “Face Spoofing Detection Using Colour Texture Analysis,” IEEE Transactions on Information Forensics and Security, vol. 11, pp. 1–1, Aug. 2016, doi: 10.1109/TIFS.2016.2555286.
