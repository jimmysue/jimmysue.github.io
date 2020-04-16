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


## 2.2 概率论概论

### 离散随机变量

$$p(A)$$ 表示事件 $$A$$ 为真的概率, 满足 $$0 \le p(A) \le 1$$. $$p(\overline{A})$$ 为 $$A$$ 的互斥事件, 有 $$p(\overline{A}) = 1 - p(A)$$. $$A=1$$ 常常表示为 A 为真, $$A=0$$ 表示 A 事件为否.

对于**离散随机变量** $$X\in \mathcal{X}$$, 用 $$p(X = x)$$ 表示事件 $$X = x$$的概率, 也可以缩略成  $$p(x)$$, 这里 $$p()$$ 称之为[**概率质量函数**](https://zh.wikipedia.org/wiki/%E6%A6%82%E7%8E%87%E8%B4%A8%E9%87%8F%E5%87%BD%E6%95%B0)([pmf](https://en.wikipedia.org/wiki/Probability_mass_function)). 概率满足  $$0 \le p(x) \le 1$$ 并且 $$\sum_{x\in\mathcal{X}}p(x) = 1$$. 图2.1 展示了两个概率质量函数, 它们的**状态空间**为 $$\mathcal{X}=\{1, 2, 3, 4, 5\}$$, 其中左边(a)为一个均匀分布, $$p(x) = 1/5$$, 右边(b)为一个退化分布, $$p(x)=\mathbb{I}(x=1)$$, 其中 $$\mathbb{I}$$ 是一个二元指示函数(indicator function). 这个分布表明, $$X$$ 永远等于1, 是一个常数.

![](/assets/img/MLAPP-Figure2.1.png){:height="320px"}

#### 2.2.2.1 并集事件的概率

$$
\begin{aligned}
    p(A \vee B) &= p(A) + p(B) - p(A \wedge B) \\
                &= p(A) + p(B), 当 A, B为互斥事件 
\end{aligned}  \tag{2.1} 
$$

#### 2.2.2.2 联合概率



联合概率定义如下

$$
 p(A, B) = p(A \wedge B) = p(A|B)p(B) = p(B|A)p(A) \tag{2.3}
$$

又称[**乘法公式**](https://en.wikipedia.org/wiki/Chain_rule_(probability)). 给定联合概率分布 p(A, B), 定义**边缘概率分布**如下:

$$
p(A) = \sum_{b}p(A, B) = \sum_{b} p(A \mid B)p(B=b) \tag{2.4}
$$

又称**加法公式**. 

乘法公式可以多次应用, 得到**链式公式**, 如下

$$
p(X_{1:D}) = p(X_1)p(X_2 \mid X1)p(X_3 \mid X_2, X_1)p(X_4 \mid X_1, X_2, X_3) ... p(X_D \mid X_{1:D-1}) \tag{2.5}
$$


公式中 $$1:D$$ 含义与 `matlab` 相同, 表示集合 $$\{1, 2, ..., D\}$$

#### 2.2.2.3 条件概率

条件概率是在事件B成立下事件A的概率, 表示如下:

$$
p(A \mid B) = \frac{p(A, B)}{p(B)} 如果 p(B) > 0 \tag{2.6}
$$

> *译者注:  
> 如果 $$p(B) = 0$$, $$p(A) = 0$$*

### 贝叶斯公式

综合上节条件概率, 乘法公式和加法公式, 我们可以推出**贝叶斯公式**, 也成**贝叶斯定理**:

$$
p(X = x \mid Y = y) = \frac{p(X = x, Y = y)}{p(Y = y)} = \frac{p(X = x)p(Y = y \mid X = x)}{\sum_{x'}p(X = x')p(Y=y \mid X=x')} 
\tag{2.7}
$$

#### 2.2.3.1 示例: 医学诊断


假设你进行乳腺癌检测, 结果呈阳性, 那么你有多大概率患有乳腺癌呢? 当然这取决于检测的可靠性有多少. 现在已知检测的**敏感性(sensitivity)**为80%, 它表示, 如果你患有乳腺癌, 检测结果有0.8的概率呈现阳性, 即

$$
p(x=1 \mid y = 1) = 0.8
$$

其中 $$x = 1$$ 表示检测呈阳性, $$y=1$$ 表示患有乳腺癌. 此时多数人可能认为, 你患乳腺癌的概率为 80%. 但这是错误的. 主要原因在于, 这样的推断忽略了人群患乳腺癌的先验概率是很低的, 它等于:

$$
p(y = 1) = 0.004
$$

忽略先验概率称之为**基本比率谬误**. 同时我们还应该将检测的假阳性或称**误检**(false positive)概率进去. 遗憾的是, 误检的检测手段误检概率是很高的: 

$$
p(x = 1 \mid y = 0) = 0.1
$$

综合以上, 我们可以利用贝叶斯公式得出正确的答案:

$$
\begin{aligned}
    p(y=1 | x = 1) &= \frac{p(x=1 \mid y=1)p(y=1)}{p(x=1 \mid y=1)p(y=1) + p(x=1 \mid y=0)p(y=0)} \\
    &= \frac{0.8 \times 0.004}{0.8 \times 0.004 + 0.1 \times 0.996} = 0.031
\end{aligned}
$$

其中 $$p(y=0) = 1 - p(y=1)= 0.996$$. 也就是说, 即使你的检测是阳性, 你患有乳腺癌的概率依然只有3.1%. 

#### 2.2.3.2 示例: 生成式分类器

我们可以把上节医学诊断的例子, 拓展到对任何特征向量 $$x$$ 分类中:

$$
p(y = c \mid \mathbf{x}, \mathbf{\theta}) = \frac{p(y = c \mid \mathbf{\theta})p(\mathbf{x} \mid y=c, \mathcal{\theta})}{\sum_{c'}p(x \mid y = c', \mathbf{\theta})p(y = c' \mid \mathbf{\theta})}
$$

这就是**生成式分类器**, 如此称呼是因为它使用的**类别先验** $$p(y=c)$$, 利用**类别的条件密度** $$p(x \mid y = c$$ 确定了数据如何生成. 这类模型将在 第3章和第4章进行探讨. 另一类方法则直接拟合类别的后验概率 $$p(y=c \mid x)$$, 称之为**判别式分类器**. 这两类方法的优缺点将在 8.6 章节进行讨论.

### 2.2.4 独立和条件独立

