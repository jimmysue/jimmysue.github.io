# Plan — RL Fine-tuning for Diffusion Image Generation Models

Slug: `rl-for-diffusion-2023`
Target length: 12,000–16,000 中文字 (≈ 1500–2000 per section × 8 sections)

## Audience contract

> "学过基础线性代数和 Python、对扩散模型有概念但没做过 RL 的工程师 / 研究生, 跟着 Step 1 + Step 2 走完每节; 想做研究的能补完 Step 3 的数学; 想跑代码的能直接抄 Step 4 的引用。"

每节都遵守螺旋:**直觉 → 最小 demo → 正式化 → 代码引用 → 洞察**。

## Lineage 一句话总览

```
ReFL/RWR (2023 初)          ← 把 reward 直接 backprop 或当 sample 权重, 但抓不住"过程"
       ↓
DDPO  (Black 2023.05)       ← 关键洞察: 把去噪展成 MDP, 每步策略是 Gaussian, log-prob 可解析
DPOK  (Fan 2023.05)         ← 同时期; 加 KL 到 ref model, 这个 pattern 成为后来标配
       ↓                       (PPO-style ratio + clip)
GRPO  (DeepSeekMath 2024)   ← 在 LLM 里发明: 用 group-relative mean/std 当 baseline, 砍掉 critic
       ↓                       (advantage 不再需要 value model)
Flow-GRPO  (Liu 2025.05)    ← 把 GRPO 接到 flow matching: ODE→SDE 加噪 + Denoising Reduction
Dance-GRPO (Xue  2025.05)   ← 同时期; 统一扩散+整流流, 跨 image/video/multi-modal
       ↓
AWM (Xue 2025.09)           ← 釜底抽薪: 证明 DDPO 实际在做 noisy-DSM, 方差更大; 把 surrogate 换回
                              预训练的 flow matching loss × advantage, 6-24× 加速
```

每个箭头都是"上一代留下来的问题"被下一代解决:
- ReFL → DDPO:  "reward 不可微怎么办?" → 用 score function 估梯度
- DDPO → DPOK: "reward hacking" → 加 KL 项
- DPOK → GRPO: "critic 太贵" → group-relative baseline
- GRPO → Flow-GRPO: "ODE 没有随机性, exploration 不够" → 把 ODE 改成 SDE
- Flow-GRPO → AWM: "Per-step Gaussian likelihood 真的是对的吗?" → 证明它隐式带额外方差

整篇教程的结构, 就是顺着这条因果链走一遍, 每节一个方法 + 它修了什么 + 真实 repo 里怎么实现的。

---

## Outline (8 大节)

### §1. RL 进入扩散模型的舞台 (≈ 1500 字)

**Key takeaway:** 扩散模型 likelihood 训练 ≠ 用户想要的目标(美学、对齐、可压缩性...); RL 给了一条不依赖 differentiable reward 的路。

- **Step 1 直觉**: "为什么不能用 SFT?" — 因为 reward 是黑盒(file size, OCR 准确率, 人类偏好打分), 没法塞进 cross-entropy。类比: 厨师不能只学怎么"复刻菜谱", 还得学"怎么得到食客好评"。
- **Step 2 demo**: 10 行 numpy 写一个 toy "高斯分布的 reward-weighted SGD" — 让读者看到"reward × log-prob 梯度"的最基本形式。`teaching-demo` 标签。
- **Step 3 正式化**:
  - $J_{\text{DDRL}}(\theta) = \mathbb{E}_{x_0 \sim p_\theta(x_0|c)}[r(x_0, c)]$
  - 为什么 $\nabla_\theta J$ 不能直接 reparametrize 求 — 因为 reward 是 black-box
  - REINFORCE: $\nabla_\theta J = \mathbb{E}[r(x) \cdot \nabla_\theta \log p_\theta(x)]$
  - 翻译每一步
- **Step 4 代码**: ReFL/RWR baseline 的简短引用 — 选 DDPO repo 里它对照的 RWR loss (`kvablack-ddpo-pytorch/scripts/train.py` 里的 reward weighting baseline 部分, 大约 10-20 行)
- **Step 5 洞察**:
  - "Score function gradient" 是 RL 给扩散的最大礼物 — 不要求 reward 可微
  - 为什么 likelihood-based SFT 在 reward 优化上"会做不对的事" — likelihood 是模糊上界, reward 是精确目标
  - RWR / ReFL 的局限是这一节的伏笔, §2 会用 DDPO 解掉

### §2. DDPO: 把去噪过程展开成 MDP (≈ 2000 字)

**Key takeaway:** Black et al. 2023 的核心洞察 — 单步 $p_\theta(x_0|c)$ 是黑盒, 但单步 reverse transition $p_\theta(x_{t-1}|x_t,c)$ 是 isotropic Gaussian, log-prob 可解析。把去噪展成 MDP, 每步是一个 RL action, REINFORCE 就能直接用了。

- **Step 1 直觉**: "把 1000 步去噪看成 RL 走 1000 个格子" — 每个格子的转移是已知的高斯分布, 全程 log-prob = 各步 log-prob 之和。一张架构图(从 paper Figure 2 或自画)。
- **Step 2 demo**: 写 15 行 numpy 模拟"2-step 去噪 MDP", 给定 reward, 显示策略 log-prob 怎么算、reward 怎么 backprop。`teaching-demo`。
- **Step 3 正式化**:
  - MDP 五元组: $s_t = (c, t, x_t)$, $a_t = x_{t-1}$, $\pi(a_t\!\mid\!s_t) = p_\theta(x_{t-1}\!\mid\!x_t, c)$, $R$ 仅在 $t=0$ 给 $r(x_0, c)$
  - $p_\theta(x_{t-1}\!\mid\!x_t,c) = \mathcal{N}(\mu_\theta(x_t,c,t), \sigma_t^2 I)$ — 由 DDPM/DDIM sampler 决定
  - 完整 trajectory log-prob: $\sum_t \log p_\theta(x_{t-1}\!\mid\!x_t,c)$, 高斯 log-prob 是个二次型
  - **DDPO$_{SF}$**: $\nabla_\theta J = \mathbb{E}[\sum_t \nabla_\theta \log p_\theta(x_{t-1}\!\mid\!x_t,c) \cdot r(x_0,c)]$
  - **DDPO$_{IS}$**: $\nabla_\theta J = \mathbb{E}[\sum_t \frac{p_\theta}{p_{\theta_{\text{old}}}} \nabla_\theta \log p_\theta \cdot r]$
  - 每步翻译
- **Step 4 代码**: ddpo-pytorch 的 logprob-tracked sampler + 训练循环。两段:
  1. `sources/repos/kvablack-ddpo-pytorch/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py` 里的 `ddim_step_with_logprob` — DDIM 一步采样 + 计算 Gaussian log-prob (15-25 行)
  2. `sources/repos/kvablack-ddpo-pytorch/scripts/train.py` 里的 inner training step — 算 ratio + advantage × log_prob (20-30 行)
- **Step 5 洞察**:
  - 为什么必须 in-place track log_prob — 一旦 sample 完丢了 log_prob, 整个 ratio 算不出来; 这是后面所有 repo 都得"patch sampler"的根源
  - DDPO 只在 t=0 给 reward, 中间所有 step 共享同一个 reward, 这等价于"per-step reward = $r(x_0)$" — 数学上是 sparse reward 的简化形式
  - 这一节为后面所有 GRPO 变体打下"loss 上算什么"的基础

### §3. PPO 风格的 ratio + clip + KL (≈ 1500 字)

**Key takeaway:** DDPO$_{IS}$ 实际上就是把 PPO 搬到了扩散 MDP 上。DPOK 同期独立做的, 加了 KL-to-ref 这一项, 抑制 reward hacking。这两件事合起来成了后来所有 RL-for-diffusion 工作的标配。

- **Step 1 直觉**: "更新 policy 不能走太远 — 跑远了估计的 ratio 就不可信了"。类比: 看着旧地图找新地点, 但只走自己看得见的一小步。
- **Step 2 demo**: 12 行 demo, 显示 "clipped vs unclipped" 在某个 ratio 抖动下损失值如何变化。`teaching-demo`。
- **Step 3 正式化**:
  - PPO clipped surrogate: $\mathcal{L}^{\text{CLIP}} = \mathbb{E}[\min(r(\theta) A,\ \text{clip}(r, 1\pm\epsilon) A)]$
  - 推导为什么 clip 是"pessimistic min" — 如果 advantage > 0, clip 把上限封住, 不允许"过于乐观"地放大 ratio
  - DPOK 的 KL 项: $-\beta \mathrm{KL}(\pi_\theta || \pi_{\text{ref}})$
  - 在扩散场景下, $\pi$ 是高斯, KL 有 closed form: $\mathrm{KL} \approx \frac{\Delta t \cdot \sigma_t^2}{2}\cdot \|\mu_\theta - \mu_{\text{ref}}\|^2 / \sigma_t^2$
  - 翻译每一步
- **Step 4 代码**: HuggingFace TRL 的 `DDPOTrainer.step()` 实现, 看生产级 RL-for-diffusion 长什么样
  - `sources/repos/...` 这个需要 TRL clone (Phase 4 决定); 备选: ddpo-pytorch 里的 clip 实现
  - 备选: `kvablack-ddpo-pytorch/scripts/train.py` 大约 L350-L400, 看 ratio + clip + advantage
- **Step 5 洞察**:
  - clip range 设大设小的工程经验 (PPO 论文 0.2, DDPO 1e-4) — 为什么扩散用得这么小?
  - KL 系数 β 跟"clip 强度" 是两个独立的正则旋钮, 它们在防止不同病: clip 防 single-update 爆炸, KL 防"长期漂移"
  - 为什么 §4 GRPO 还要再改这套? — critic-free 需求, 不在这一节解决

### §4. GRPO: 没有 critic 的 advantage 估计 (≈ 1200 字)

**Key takeaway:** DeepSeekMath 2024 提出 — advantage 不一定要训一个 value network 当 baseline, 用"同一 prompt 下的 group reward 的均值/标准差"当 baseline 就行。省一半显存, 实现简单, 在 LLM 上效果对齐了 PPO。后面 Flow-GRPO / Dance-GRPO / AWM 都直接抄。

- **Step 1 直觉**: "考试评分:不需要一个标准答案模型, 每场考试出 24 个学生, 谁高于本场平均分就奖励他, 低于就惩罚他。" 这就是 group-relative。
- **Step 2 demo**: 8 行 numpy, 模拟"5 个 rollout, 算 group mean/std, normalize advantage"。`teaching-demo`。
- **Step 3 正式化**:
  - 标准 PPO advantage: $A_t = R_t - V_\phi(s_t)$, $V_\phi$ 是 critic
  - GRPO advantage: $\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_j\})}{\mathrm{std}(\{r_j\})}$, 对一组 $G$ 个 rollout 算
  - 数学上为什么 unbiased: $\mathbb{E}[\text{mean}] = \mathbb{E}[r]$ 是合法 baseline
  - 翻译: 这个 estimator 跟 critic 的 estimator 在期望意义下等价, 但实现上完全去掉了 $V_\phi$ 的训练
  - 何时 group-relative 反而比 critic 差: $G$ 太小、reward 分布极端长尾
- **Step 4 代码**: 一段 8-15 行的 group advantage 计算, 选自 flow_grpo 或 DanceGRPO
  - `sources/repos/yifan123-flow_grpo/scripts/train_sd3.py` 里的 advantage 算式 (大约会在 ~L500-L600 区域, Phase 4 子代理确认行号)
- **Step 5 洞察**:
  - GRPO 在 LLM 里 G=8 就够, 在 diffusion 里大家普遍 G=24 — 为什么? (因为 reward 噪声大, std 估计不稳)
  - "省一个 critic" 在视频生成里至关重要 — value network 跟 backbone 一样大就崩了
  - 缺点: group 内 reward 全部 0 (失败) 或全部 max (成功) 时 advantage 标准化会爆 — 需要 mask

### §5. Flow-GRPO: 把 GRPO 接到 flow matching (≈ 2000 字)

**Key takeaway:** Liu et al. 2025.05 — 现代 T2I 是 flow matching (SD3.5, FLUX), 但 flow ODE 是 *确定性* 的, RL 没有 exploration。两个修法: (1) **ODE→SDE 转换**: 加 $\sigma_t dw$ 保持 marginal 不变, 让每步变 Gaussian; (2) **Denoising Reduction**: 训练用 10 步 sample, 推理还用 40 步, 大幅省 compute。

- **Step 1 直觉**: 
  - ODE 确定性问题:"同样的初始噪声, 同样的 prompt, 跑出来永远一样的图" — RL 没法 explore。
  - 比喻: 让 1 个学生反复做同一题不会进步; 让 24 个学生做同一题, 一组打分, 才能区分好坏。
  - 一张 paper Figure 2 的 ODE→SDE 示意图。
- **Step 2 demo**: 15 行, 用 1D 例子展示 "确定性 ODE 跑 100 次得到同样 trajectory" vs "加 $\sigma\cdot dW$ 跑 100 次得到 100 个 trajectory, marginal density 相同"。`teaching-demo`。
- **Step 3 正式化**:
  - Flow matching forward: $x_t = (1-t) x_0 + t x_1$
  - 确定性 reverse ODE: $dx_t = v_\theta(x_t, t) dt$ — 1-to-1 mapping
  - **核心 trick**: 给 ODE 加 SDE 项, 调整 drift 保持 marginal:
    $$dx_t = \Big(v_\theta(x_t, t) + \frac{\sigma_t^2}{2t}(x_t + (1-t)v_\theta)\Big) dt + \sigma_t dw$$
  - 离散化 (Euler–Maruyama):
    $$x_{t+\Delta t} = x_t + (v_\theta + \tfrac{\sigma_t^2}{2t}(x_t + (1-t)v_\theta))\Delta t + \sigma_t\sqrt{\Delta t} \epsilon$$
  - per-step Gaussian, log-prob 二次型可解析
  - KL closed form (paper Eq. 后): $\mathrm{KL} = \frac{(1-t)^2}{2\sigma_t^2 \cdot 2t/(1-t)} \|v_\theta - v_{\text{ref}}\|^2$ — 在 velocity 空间, 这是个 L2 损失加权重
  - **Denoising Reduction**: 训练 T=10 步采样, 推理 T=40 — 因为 RL 关心 trajectory 走向, 不需要每步都精细。
  - 每步翻译
- **Step 4 代码**: flow_grpo 的两段
  1. `sources/repos/yifan123-flow_grpo/flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` — ODE→SDE conversion + per-step log-prob (核心 20-40 行)
  2. `sources/repos/yifan123-flow_grpo/scripts/train_sd3.py` 主 loop — ratio × advantage + KL (大约 20 行精华)
- **Step 5 洞察**:
  - 为什么 `σ_t = a · √(t/(1-t))` — 这个形式让 KL 算式漂亮, 但 a 是个调优旋钮 (paper 5.3)
  - Denoising Reduction 暗示了一件事: RL 阶段的"sample 质量"和"sample 信息量"不一样 — 你只需要让 trajectory 携带 reward 信号即可, 不需要完美图像
  - 这一节为 §7 AWM "推翻整个 sample-time MDP 框架" 埋伏笔

### §6. DanceGRPO: 统一扩散 + 整流流 + 跨视频 (≈ 1500 字)

**Key takeaway:** Xue et al. 2025.05 — 同时期工作, 把 Flow-GRPO 的"统一 SDE-MDP" 思路推到极致: 用一个统一公式描述 diffusion + rectified flow, 支持 image/video/multi-modal, 跨 4 个 backbone (SD, FLUX, HunyuanVideo, Wan2.1)。另外发现"组内共享初始噪声"对防止 reward hacking 非常关键。

- **Step 1 直觉**: 
  - 之前的工作各家 SDE 都不一样, DanceGRPO 把它们写成同一形式 $\tilde{z}_s = \tilde{z}_t + \text{NetOut}\cdot(\eta_s - \eta_t)$, 只是 $\tilde{z}$ 和 $\eta$ 选取不同
  - "组内共享噪声": 同一 prompt 的 24 个 rollout 用同样的初始 $\epsilon$, 只让 SDE 内部的随机带来差异。比喻: 24 个学生同一道题, 同一份初始草稿 → 比同初始更公平。
- **Step 2 demo**: 10 行 demo, 显示同一函数能切换"diffusion 公式"和"flow 公式" via $\eta_t$ 的不同选择。`teaching-demo`。
- **Step 3 正式化**:
  - Diffusion 和 rectified flow 的统一形式 (Eq. 5 in paper):
    $$\tilde{z}_s = \tilde{z}_t + \text{Output}(z_t, t) \cdot (\eta_s - \eta_t)$$
    其中:
    - 对 $\epsilon$-prediction diffusion: $\tilde{z} = z/\alpha$, $\eta = \sigma/\alpha$
    - 对 rectified flow: $\tilde{z} = z$, $\eta = t$
  - 反向 SDE (Diffusion 和 flow 各自):
    $$dz_t = (f_t z_t - \tfrac{1+\epsilon_t^2}{2}g_t^2 \nabla\log p_t(z_t))dt + \epsilon_t g_t dw \quad \text{(diffusion)}$$
    $$dz_t = (u_t - \tfrac{1}{2}\epsilon_t^2 \nabla\log p_t(z_t))dt + \epsilon_t dw \quad \text{(rectified flow)}$$
  - 翻译每一步
- **Step 4 代码**: DanceGRPO 的 FLUX-side 训练循环, 显示"切换 backbone 只改 sampler, RL loss 不变"
  - `sources/repos/XueZeyue-DanceGRPO/fastvideo/train_grpo_flux.py` — 主 loop (Phase 4 子代理定位精确行号, 大致 20-40 行精华)
- **Step 5 洞察**:
  - DanceGRPO 默认 *不* 加 KL — 跟 Flow-GRPO 不一样, 因为他们发现"shared init noise + group-relative 已经足够 stabilize"
  - "共享初始噪声"为什么阻止 reward hacking: 不同 noise 会让 group 内 reward 差异来自 noise + content; 共享 noise 后差异只来自 policy 行为, advantage 信号更干净
  - 跨 backbone 验证是这篇的最大贡献 — 之前所有 RL-for-diffusion 工作只在 SD 一个模型上, DanceGRPO 第一个证明 framework 可以 scale 到 video

### §7. AWM: 重新审视目标函数本身 (≈ 2000 字)

**Key takeaway:** Xue et al. 2025.09 — 之前所有 GRPO 变体都在"per-step Gaussian log-prob"框架内修修补补。AWM 一脚踢翻这个框架: Theorem 1 证明 DDPO 实际上在做 *用 noisy $x_s$ 当条件的 DSM*, Theorem 2 证明这比 *用 clean $x_0$ 当条件的 DSM* 多出 $d\cdot\kappa(s,t)$ 的方差。改法很简单 — 把 RL surrogate 直接换回预训练用的 flow matching loss $\|v_\theta - (\epsilon - x_0)\|^2$, 乘以 advantage。6-24× 加速。

- **Step 1 直觉**: 
  - 类比:"教学生画画, 不要每一笔都跟噪声参照比, 而是画完整张跟干净的目标比"
  - "DDPO 的 likelihood 看似在算梯度, 其实是把噪声当成了条件 — 这等价于多加了一层方差"
  - 一张 paper Figure 1 的 (a)(b)(c) 三联图。
- **Step 2 demo**: 12 行 numpy, 显示"用 clean target 的 MSE 收敛快, 用 noisy target 的 MSE 收敛慢" — 一个 1D 玩具实验。`teaching-demo`。
- **Step 3 正式化**:
  - **Theorem 1 (DDPO ≡ noisy DSM)**:
    $$\mathbb{E}_{x_{t-\Delta t}, x_t}[\|s_\theta(x_t, t) - \nabla \log p(x_t | x_{t-\Delta t})\|^2]$$
    DDPO per-step likelihood 在期望意义下等于以上 DSM 损失。证明走 Haussmann-Pardoux 反时间公式。
  - **Lemma 1**: noisy-DSM 跟 clean-DSM 同 minimizer (期望都是 $\nabla \log p(x_t)$)。
  - **Theorem 2 (方差更大)**: 
    $$\mathrm{Cov}(\nabla\log p(x_t|x_s) | x_t) = \mathrm{Cov}(\nabla\log p(x_t|x_0) | x_t) + \kappa(s,t) I$$
    where $\kappa(s,t) = \frac{(1-t)^2 s^2}{t^2 [t^2(1-s)^2 - s^2(1-t)^2]}$
  - 数值直觉:取 $t=0.5, s=0.4$, $\kappa \approx 0.5$, 维度 $d=10^4$, 额外 trace variance = $5 \times 10^3$
  - **AWM 目标**:
    $$\log\pi_\theta(x_0|c) \approx -\mathbb{E}_t[w(t)\|v_\theta(x_t,t,c) - (\epsilon - x_0)\|^2]$$
    把它代回 GRPO ratio × advantage 公式, 整个流程跟 flow matching pretraining 几乎一样
  - 翻译每一步
- **Step 4 代码**: AWM 的两段
  1. `sources/repos/scxue-advantage_weighted_matching/advantage_weighted_matching/scripts/train_sd3_awm.py:L210-L252` — `compute_log_prob_awm` (核心: 用 FM loss 当 log-prob)
  2. 同文件 L1265-L1354 — 主 loop, ratio × advantage + velocity-space KL + EMA KL (3 项)
- **Step 5 洞察**:
  - "代码里多了一项 EMA KL, 论文 Algorithm 1 没有" — paper-vs-code 的典型 gap
  - 为什么"统一 surrogate"在工程上是大胜利:你不需要 patch sampler、不需要 logprob-tracked sampling — 用任何 ODE/SDE solver 都行
  - 这一节是整个 lineage 的"反思点" — 之前所有方法的"loss 形式"都被质疑了

### §8. 全景对比与选型 (≈ 1500 字)

**Key takeaway:** 把 6 个方法摆在一张表上 — loss 类型、KL 是否加、advantage 估计、SDE-conversion 必要性、sampler-tied 与否, 给出工程选型决策树。

- **Step 1 直觉**: "选哪个方法不取决于哪个数字最高, 而取决于你的 reward 形态 + 你的 backbone 类型 + 你的预算"
- **Step 2 demo**: 一段 demo / 伪代码, 显示"从 DDPO 到 AWM, training loop 删掉的代码行数随时间递减"。或者一段对比 pseudo-code。`teaching-demo`。
- **Step 3 正式化**:
  - 对比表 (使用 HTML `<table>`):
    | 方法 | Surrogate | Advantage | SDE-conversion | KL | 主要 backbone |
    |---|---|---|---|---|---|
    | DDPO | per-step Gaussian log-prob | reward × log-prob (REINFORCE 或 IS+clip) | DDIM/DDPM 天然 SDE | 无 | SD1.4/1.5 |
    | DPOK | per-step Gaussian log-prob | reward-weighted | DDIM/DDPM | KL-to-ref | SD |
    | Flow-GRPO | per-step Gaussian log-prob | group-relative | ODE→SDE (Eq.7) | KL closed form | SD3.5, FLUX |
    | Dance-GRPO | per-step Gaussian log-prob | group-relative + shared noise | 统一 SDE 公式 | 默认无 | 多 backbone |
    | AWM | flow matching loss | group-relative | 不需要 (用任何 sampler) | velocity-space KL + EMA | SD3.5, FLUX |
  - 计算复杂度对比 (sample/training step, memory)
  - 适用场景决策树
- **Step 4 代码**: 三段 pseudo-code 并列, 展示"DDPO loss vs Flow-GRPO loss vs AWM loss" 在 PyTorch 里写出来的差异
  - 引用每一行都从对应 repo 摘
- **Step 5 洞察**:
  - 整个 lineage 的元规律: "把 sampler 从 RL 算法里解耦出来" — 从 DDPO 必须 patch sampler, 到 AWM 完全不动 sampler
  - 视频领域未来方向: AWM 还没真正在视频上验证, 之后可能会有 "Video-AWM"
  - 一个开放问题: 当 reward 是 differentiable (例如 reward model + VAE decoder backprop) 时, ReFL 路线和 AWM 路线孰优? 目前没有定量对比

---

## 引用与参考 (References 节,出现在文末)

每节会引用 plan 中列出的 5 篇 seminal + 2 篇 anchor。引用格式:
> Black, K., Janner, M., Du, Y., Kostrikov, I., & Levine, S. (2023). *Training Diffusion Models with Reinforcement Learning*. arXiv:2305.13301.

Repos 同理列出。

---

## 跨节依赖

- §2 引入的 MDP 五元组 是 §3-§7 通用的 notation
- §3 的 PPO ratio + clip 在 §5/§6 直接复用
- §4 GRPO advantage 是 §5/§6/§7 的标准 baseline
- §5 ODE→SDE 公式跟 §6 统一 SDE 公式是同一类思想的两个版本
- §7 直接质疑 §2-§6 的整个 surrogate 选择, 是个"reflective回望"
- §8 是整篇 synthesis, 引用所有前述章节

## Drafting groups (Phase 5 grouped-parallel)

- **Group A (parallel):** §1, §2 — foundation; A 不依赖任何前文
- **Group B (parallel):** §3, §4 — 依赖 §2 的 MDP notation
- **Group C (parallel):** §5, §6 — 依赖 §3 的 PPO + §4 的 GRPO
- **Group D (serial):** §7 — 依赖 §5 的 flow matching MDP
- **Group E (serial):** §8 — 综合 all
