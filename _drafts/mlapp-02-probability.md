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

### 2.2.1 离散随机变量

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

### 2.2.3 贝叶斯公式

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

两组事件 $$X$$ 和 $$Y$$ **无条件独立**(unconditionally independent) 或者说**边缘独立**(marginally independent), 记作 $$X\bot Y$$. 如图2.2所示, 两个独立事件的联合概率等于边缘概率的乘积, 即:

$$
X \bot Y \Longleftrightarrow p(X, Y) = p(X)p(Y)
$$


![](/assets/img/MLAPP-Figure2.2.png){: height="360px"}

无条件独立的情况很少, 多数情况下, 变量之间会互相影响. 这种影响, 肯能来自于某个共同的变量. 当且仅当有如下情形时, 我们说事件X和事件Y在Y发生的条件下独立, 这就是**条件独立**(conditionally independent, CI):

$$
X \bot Y\mid Z \Longleftrightarrow p(X, Y \mid Z) = 
p(X \mid Z)p(Y \mid Z) \tag{2.15}
$$

在第10章讨论的图模型里, 我们可把这样的情形表示成一个图 $$X-Z-Y$$, 也就是说, 事件X和事件Y的独立性在事件Z条件下成立.


条件独立还有如下特征:

**定理 2.2.1:**    $$X \bot Y \mid Z$$ 成立, 当且仅当存在函数 $$g$$, 和$$h$$, 对于所有的 $$x$$, $$y$$, $$z$$ 且 $$p(z)> 0$$ 满足:

$$
p(x, y \mid z) = g(x, z)h(y, z) \tag{2.16}
$$

条件独立是建立大型概率模型的基础. 如下章节都会用到条件独立假设:

- 3.5节 贝叶斯分类器
- 17.2节 马尔科夫模型
- 10章 图模型

### 2.2.5 连续随机变量

假设有随机变量$$X$$, 计算$$X$$落入范围 $$a \le X \le b$$的概率. 首先我们定义事件 $$A = (X \le a), B = (X \le b) 和 W = (a < X \le b)$$. 显然我们有$$B= A \vee W$$, 因为事件A和W互斥, 根据加法公式, 我们有:

$$
p(B) = p(A) + p(W) \tag{2.17}
$$

所以
$$
p(W) = p(B) - p(A) \tag{2.18}
$$

定义**概率累计函数**(cumulative distribution function, cdf)函数 $$F(q) \triangleq p(X \le q)$$, 该函数单调递增. 图2.3(a)展示了一个例子. 基于此定义, 有

$$
p(a < X \le b) = F(b) - F(a) \tag{2.19}
$$

再定义 $$f(x) = \frac{d}{dx}F(x)$$(假设导数存在), 称之为**概率密度函数**(probability density function, cdf). 于是有

$$
P(a \lt X \le b) = \int_a^b f(x)dx \tag{2.20}
$$

当间隔足够小, 有:

$$
P(x \le X \le x+dx) \approx p(x)dx
$$

其中要求 $$p(x) \ge 0$$, 而 $$p(x) > 1$$是可以的, 只要最终的积分为1. 例如, 有**均匀分布**:

$$
Unif(x \mid a, b) = \frac{1}{b - a} \mathbb{I}(a \le x \le b)
$$

如果 a 设为 0, b 设为 $$\frac{1}{2}$$, 对于 $$x \in [0, \frac{1}{2}]$$ 就有 $$p(x) = 2$$.

### 2.2.6 分位数

概率累计函数单调递增, 存在反函数 $$F^{-1}$$. $$F^{-1}(\alpha)$$ 是概率累计位置的函数, $$\alpha$$ 即为 $$F$$ 的分位数. 其中 $$F^{-1}(0.5)$$ 是分布的**中值**.

cdf的反函数有时被用来计算**尾部概率**(tail area probabilities). 假设我们$$\Phi$$表示高斯分布 $$\mathcal{N}(0, 1)$$ 的概率累计函数. 高斯分布左右对称, 我们砍去左右的尾部概率, 一边一半. 如果我们要保留95%的概率质量, 左右就各坎 2.5%. 即有

$$
(\Phi^{-1}(0.025), \Phi^{-1}(0.975)) = (-1.96, 1.96)
$$

更一般地, 对于正态分布 $$\mathcal{N}(\mu, \sigma^2)$$, 区间 $$(\mu - 1.96\sigma, \mu + 1.96\sigma)$$ 包含了95%的概率质量. 一般我们会近似地写作 $$\mu \pm 2\sigma$$

### 2.2.7 均值和方差

均值或称期望, 用 $$\mu$$ 表示. 离散变量其定义为 $$E(X) \triangleq \sum_{x\in\mathcal{X}}xp(x) $$, 对于连续随机变量, 定义为 $$E(X) = \int_{\mathcal{X}}xp(x)dx$$. 如果积分无界, 则均值不存在.

**方差**可理解为分布的*广度*, 用 $$\sigma^2$$ 表示. 定义如下:

$$
\begin{aligned}
    Var[X] &\triangleq E[(X - \mu)^2] = \int (x - \mu)^2 p(x) dx \\
    &= \int x^2 p(x) dx - 2 \mu \int xp(x)dx + \int \mu^2 p(x) dx =E(X^2) - \mu^2 
\end{aligned}
\tag{2.24 2.25} 
$$

由此可得

$$
E(X^2)= \mu^2 + \sigma^2
$$

标准差定义为方差的平方根, 与 X 量纲一致, 颇有用.

$$
std[X] \triangleq \sqrt{Var[X]}
$$

## 2.3 常见的离散分布

### 2.3.1 二项分布与伯努利分布

假设抛 $$n$$ 次硬币, 其中正面朝上的次数为 $$X$$, 显然 $$X\in \{0, 1, ... n\}$$, 则我们说 $$X$$ 服从二项式分布, 记作 $$X \sim Bin(n, \theta)$$, 其中 $$\theta$$ 硬币正面朝上的概率. 二项式分布的概率质量函数为:

$$
Bin(k \mid n, \theta) \triangleq \pmatrix{n \\ k} \theta^k(1 - \theta)^{n-k}
$$

其中

$$
\pmatrix{n \\ k} = \frac{n!}{(n-k)!k!}
$$

理解为从 $$n$$ 次选取 $$k$$ 次正面朝上的选法种数, 该系数称作**二项式系数**. 图2.4 展示了一些二项式分布的例子. 二项式分布的均值和方差分别为:

$$
mean = \theta , var = n\theta(1-\theta)
$$

如果我们只丢一次这样的硬币, 正面朝上的次数为 $$X\in \{ 0, 1\}$$, 服从**伯努利分布**或称**0 1 分布**, 记作 $$X  \sim  Ber(\theta)$$, 概率质量函数定义如下:

$$
\begin{aligned}
    Ber(1 \mid \theta) &= \theta^{\mathbb{I}(x=1)}(1 - \theta)^{\mathbb{I}(x=0)} \\
    &= 
    \begin{cases}
        \theta, ~~~~~~~~~~\text{if } x = 1, \\
        1-\theta, ~~~\text{if } x = 0
    \end{cases}
\end{aligned}

$$


![](/assets/img/MLAPP-Figure2.4.png){:width="640px"}


### 2.3.2 多项分布和分类分布


**推导:**  
$$
  \pmatrix{n \\ x_1}\theta_1^{x_{1}} \cdot \pmatrix{n - x_1 \\ x_2} \theta_2^{x_{2}} \cdots  \pmatrix{n - x_1 - x_2 \cdots x_{n-1} \\ x_n} \theta_n^{x_{n}}
$$

![](/assets/img/MLAPP-Figure2.5.png){:width="640px"}