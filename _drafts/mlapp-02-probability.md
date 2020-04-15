---
layout: post
title: MLAPP-02-Probability
toc: true
tag:
  - MLAPP
  - 翻译

---


## 2.1 引言

> Probability theory is nothing but common sense reduced to calculation. -- Pierre Laplace, 1812

什么是概率, 两种角度:

1) **频率**角度. 例如抛硬币, 正反面的频率相当
2) **贝叶斯**角度. 在这个角度中, 概率用来表述事件的**不确定性**. 在这个角度下, 我们可以认为硬币出现正反面是等可能的.

贝叶斯可以用来建模那些没有或无法长期频率观测的事件. 例如估计北极冰盖在2020年会不会融化的概率, 这件事可能不发生, 发生也只会发生一次.

![](/assets/img/MLAPP-Figure2.1.png){:height="320px"}

## 2.2 概率论概论

### 离散随机变量

$$p(A)$$ 表示事件 $$A$$ 为真的概率, 满足 $$0 \le p(A) \le 1$$. $$p(\overline{A})$$ 为 $$A$$ 的互斥事件, 有 $$p(\overline{A}) = 1 - p(A)$$. $$A=1$$ 常常表示为 A 为真, $$A=0$$ 表示 A 事件为否.


