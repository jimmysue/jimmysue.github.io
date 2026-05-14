# Spiral contract — worked examples

These show what a well-formed `<section>` looks like for two different topic flavors.
Use them as a calibration reference, not a template to copy verbatim.

---

## Example 1 — LoRA §1 "为什么需要 LoRA"

```html
<section id="sec-1" class="spiral">

  <h2 id="sec-1">1. 为什么需要 LoRA</h2>

  <!-- Step 1 直觉 -->
  <div class="spiral-step spiral-step-1">
    <h3 id="sec-1-1">1.1 直觉</h3>
    <p>想象你买了一台预训练好的"通用厨师"机器人，它会做 5000 道菜。
    你只想教它做你家口味的红烧肉——一道菜。如果重训整台机器人，
    你要付钱让它把所有 5000 道菜又学一遍，浪费。LoRA 的想法是：
    我们假设"教会一道新菜"只需要给机器人加一个很薄的<em>适配模块</em>，
    主体冻结不动。这个薄模块的参数远少于主体，训练快、存储省、
    切换菜式只换模块。</p>
    <figure>
      <img src="figures/sec1-intuition.svg" alt="主模型冻结 + 旁路 LoRA 模块">
      <figcaption>图 1.1 — 主体冻结，旁路加一个低秩适配模块</figcaption>
    </figure>
  </div>

  <!-- Step 2 最小 demo -->
  <div class="spiral-step spiral-step-2">
    <h3 id="sec-1-2">1.2 最小 demo</h3>
    <p>我们先用 12 行 numpy 把直觉跑出来。把"原始权重"冻住，
    在它旁边加一个秩为 2 的小矩阵——看看参数量差多少。</p>
    <p class="code-source teaching-demo-note">教学示例 — 非生产代码 · hand-written</p>
    <pre><code class="language-python teaching-demo">import numpy as np
d = 1024                                # 大模型一层的维度
W = np.random.randn(d, d).astype(np.float32)  # 冻结的主权重
r = 2                                   # LoRA 秩
A = np.random.randn(r, d) * 0.01        # 低秩分支
B = np.zeros((d, r))                    # B 初始化为 0
print("full params :", W.size)          # 1048576
print("lora params :", A.size + B.size) # 4096
# 推理: y = W x + B(A x)
x = np.random.randn(d).astype(np.float32)
y = W @ x + B @ (A @ x)
print(np.allclose(y, W @ x))            # True —— LoRA 在初始时不改变输出
</code></pre>
    <p>跑一下你会看到：LoRA 的可训练参数比全量少了 <strong>256 倍</strong>，
    而且初始输出和原模型完全相等（因为 B=0）。这两件事就是后面所有
    数学推导要解释的"为什么"。</p>
  </div>

  <!-- Step 3 正式化 -->
  <div class="spiral-step spiral-step-3">
    <h3 id="sec-1-3">1.3 正式化</h3>
    <p>设预训练权重为 $W_0 \in \mathbb{R}^{d \times d}$，
    全量微调寻找的是更新量 $\Delta W$，使最终权重为 $W = W_0 + \Delta W$。
    LoRA 的核心假设是 $\Delta W$ 是<em>低秩</em>的，可分解为：</p>
    $$ \Delta W = B A, \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times d}, \; r \ll d $$
    <p class="math-translation">—— 翻译: 想象厨师那个"薄模块" $\Delta W$ 不是一块 $d\times d$ 的厚板，
    而是两块薄片相乘——一块 $d\times r$，一块 $r\times d$。</p>

    <p>参数量从 $d^2$ 降到 $2dr$。当 $d=1024, r=2$ 时：</p>
    $$ \frac{2dr}{d^2} = \frac{2r}{d} = \frac{4}{1024} \approx 0.004 $$
    <p class="math-translation">—— 翻译: 也就是只剩 0.4% 的训练参数——和 Step 2 跑出来的"256 倍"对应（$1/256 \approx 0.0039$）。</p>
  </div>

  <!-- Step 4 代码引用 -->
  <div class="spiral-step spiral-step-4">
    <h3 id="sec-1-4">1.4 代码引用</h3>
    <p>看看 huggingface/peft 是怎么把这个分解落成 PyTorch module 的：</p>
    <p class="code-source">sources/repos/huggingface-peft/src/peft/tuners/lora/layer.py:L42-L78 — LoRA 层的核心结构</p>
    <pre><code class="language-python">class LoraLayer(BaseTunerLayer):
    # ... (lines 42–78 verbatim from repo) ...
</code></pre>
    <p>对照公式 (eq 1.3.1)：<code>self.lora_A[adapter_name]</code> 就是 $A$，
    <code>self.lora_B[adapter_name]</code> 就是 $B$，<code>self.scaling</code>
    是后面 §3 会推的 $\alpha/r$ 缩放因子。整个 forward
    就是 $y = W_0 x + s \cdot B(Ax)$。</p>
  </div>

  <!-- Step 5 洞察 -->
  <div class="spiral-step spiral-step-5">
    <h3 id="sec-1-5">1.5 洞察</h3>
    <ul class="insights">
      <li><strong>为什么是低秩, 而不是 prefix-tuning?</strong> —— Prefix tuning
        改 attention 的"输入"，影响信号流向；LoRA 改"权重"，影响信号变换。
        改权重的方法更接近全量微调的几何结构，所以经验上收敛更好。</li>
      <li><strong>r=2 太小了吗?</strong> —— 对许多任务，r=4 ~ r=16 已饱和。
        但 r=1 在多数情况下退化明显；这是 §3 会用 SVD 分析的下界。</li>
      <li><strong>什么时候 LoRA 不够?</strong> —— 当下游任务的更新方向真的
        高秩时（如继续预训练、跨语种迁移），LoRA 会被低秩瓶颈卡住，
        此时应使用 DoRA / 全量微调 / 或者更大的 r。</li>
    </ul>
    <p class="hand-off">下一节我们把 §1 末尾跳过的"为什么 B=0、A 取高斯初始化"这件事推清楚。</p>
  </div>

</section>
```

---

## Example 2 — Diffusion §2 "反向去噪：从噪声到样本"

(Outline — fill in similarly. The same 5-sub-step structure applies.)

```
<h2 id="sec-2">2. 反向去噪 —— 从噪声走回数据</h2>
  <h3 id="sec-2-1">2.1 直觉</h3>     # 一颗冰块化水 vs 把水冻回冰块
  <h3 id="sec-2-2">2.2 最小 demo</h3>  # 15 行 numpy: 给 28×28 高斯, 迭代去噪到 MNIST 样
  <h3 id="sec-2-3">2.3 正式化</h3>     # p_θ(x_{t-1}|x_t) 的参数化, ELBO 化简
  <h3 id="sec-2-4">2.4 代码引用</h3>   # diffusers/schedulers/scheduling_ddpm.py:L120-L155 — step()
  <h3 id="sec-2-5">2.5 洞察</h3>      # 为什么参数化预测 ε 而不是 x_0; DDIM 的轨迹解耦
```

---

## Common smells (don't ship)

1. **Step 1 has jargon** — "我们考虑参数化的去噪概率分布..." → 不行, 这是数学语言. Step 1 应该是"一颗冰块化水" 这种.
2. **Step 2 is the production code** — Step 2 应当 < 30 行, 没有外部依赖, 几乎是伪代码. 如果它要 import PyTorch 三个东西, 那是 Step 4.
3. **Equation without 翻译** — 每个 `$$ ... $$` 之后 100 字内没出现一句"—— 翻译: ..." → 自动拒.
4. **"by SVD" 跳步** — Step 3 出现 "by SVD 我们有..." 但没展开也没跨节引用 → 跳步.
5. **Step 4 is paraphrased** — 不是从 `code-excerpts/secN-step4-*.txt` 复制的代码 → 自动拒.
6. **Step 5 is a recap** — Step 5 不是"小结", 是"为什么是这样而不是那样". 如果它在复述前 4 步, 改写或砍掉.
7. **Hand-off missing** — 除最后一节外, 每节 Step 5 末尾要一句"下一节我们..." 给读者一根线.
