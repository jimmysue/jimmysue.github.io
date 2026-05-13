# Plan — RLHF 的演变：PPO → DPO → GRPO + Diffusion 的 RL

**Slug**: `rlhf-evolution-2024`
**Audience contract**: "高三学生看完每节 Step 1 + Step 2 能用自己话说出'这个机制在干什么'; 本科生看完 Step 3 知道公式怎么来的; 工程师看完 Step 4 能直接抄代码; 实战派看完 Step 5 知道什么时候用哪种."

**目标字数**: 14,000–16,500 中文字 (8 sections; §5/§6 各 2200 字, 其余 1500–2000 字).

**贯穿这篇 tutorial 的一根线**: 一切 RL 方法都在回答**同一个问题** —— "如何让生成模型最大化一个不可微的 reward, 而不偏离 reference 太远". PPO/DPO/GRPO/DDPO 的所有"差别"都是在这条约束最优化里**换不同的近似**.

---

## §1. 公式里那些符号到底是什么 —— policy / log π / KL 的工程意义 (≈ 2000 字)

**Key takeaway**: 把"policy = 你的生成模型本身"这件事吃透; 解释 log 出现的两个原因 (数值稳定 + 求导友好); 解释 KL 为什么是"贴近 reference"的代名词.

- **Step 1 直觉**:
  - "Policy" = π_θ = **你的 transformer / U-Net 本身**. 没有第二个"policy 网络". 训练前后是同一个模型, 只是参数变了.
  - LLM 的 policy: 输入 prompt x, 输出每个 token 的概率分布. 一次 sample = 按这个分布生成 100 个 token.
  - Diffusion 的 policy: 输入 (noisy image x_t, time step t), 输出去噪方向. 一次 sample = 走完 50 步去噪.
  - log 的两个原因 —— 概率会乘到 1e-30 (用 log 变求和保证数值不下溢); ∇ log p · p 的恒等式让梯度推导能进行.
  - KL —— "两个分布有多不一样" 的度量. RL 里所有"别更新太狠"的口号最终都写成 KL.
  - 配图: 一个 token 概率分布 + 它的 log; 一个 KL 的几何示意.

- **Step 2 最小 demo** (15 行 numpy): toy 4-token vocab 的 softmax 分布, 手写 `log_prob`, 手写 `KL(p, q)`. 演示 log 在 batch 求和时为什么稳定.

- **Step 3 正式化**:
  - π_θ(a | s) 的定义: 给定状态/历史, 下一动作的条件概率.
  - log π(y | x) = Σ_t log π_θ(y_t | x, y_{\lt t}) (autoregressive 分解; 关键: y_{\lt n} 不是 HTML, 是 LaTeX 的 \\lt).
  - KL(π_θ || π_ref) = E_{y \sim π_θ}[log π_θ(y) - log π_ref(y)].
  - Expected reward objective J(θ) = E_{y \sim π_θ}[R(x, y)].
  - 每个公式后面跟一句"翻译".

- **Step 4 代码引用**: `huggingface-trl/trl/trainer/utils.py` 的 `selective_log_softmax` (把 LM 的 logits 转成 per-token log-prob) —— 5 个 RL trainer 都用它做"把 token 序列转 log π" 的标准操作.

- **Step 5 洞察**:
  - log 不是数学家洁癖, 是 fp16 下 1e-30 概率必死的工程必需.
  - KL 不需要"算两个分布之间的真距离" —— 在 PPO/DPO/GRPO 里它都是用单样本 log-prob 之差**估计**的, 这就是为什么实现里只有一行减法.

---

## §2. Policy Gradient: 不可微 reward 怎么求导 —— REINFORCE 起手 (≈ 1500 字)

**Key takeaway**: 解释"log-derivative trick"为什么必然出现, 介绍 REINFORCE, 引出"高方差"问题作为 PPO 的入场动机.

- **Step 1 直觉**:
  - 我们想 max E[R(τ)]. R 不可微 (BLEU、人类打分、CLIP 分数都不可导). 怎么对 θ 求导?
  - 一个数学小变换 (log-derivative trick): ∇θ log p(τ; θ) = ∇θ p(τ; θ) / p(τ; θ). 把∇拿到期望外面: ∇ E[R] = E[R · ∇ log π(τ)].
  - **结论**: 不需要 R 可导. 只要能 sample τ, 能算 log π, 就能更新 θ.
  - 这就是 RL 在生成模型唯一不可替代的入场券.
  - 配图: log-derivative trick 的图示 + REINFORCE pseudocode.

- **Step 2 最小 demo** (20 行 numpy): 3-arm bandit, softmax policy, REINFORCE 更新 100 步, 画 reward 曲线.

- **Step 3 正式化**:
  - Policy gradient theorem 完整推导 (3–4 行).
  - Baseline 降方差: E[(R − b) · ∇ log π] = E[R · ∇ log π] (b 与 a 无关), 但方差小.
  - Advantage A(s, a) = Q(s, a) − V(s).
  - 翻译每一步.

- **Step 4 代码引用**: `huggingface-trl/trl/experimental/ppo/ppo_trainer.py` 中 advantage 计算的片段 (GAE).

- **Step 5 洞察**:
  - REINFORCE 在 LLM 上单步可工作, 但 reward 信号是 sequence-level 的稀疏标量, 方差极大 → 梯度方向乱跳 → 训不动.
  - PPO 的两个核心改进 (clip + value head) 都是在攻这个"高方差"的根.

---

## §3. PPO: 走得快, 但别翻车 —— clip 是怎样想出来的 (≈ 2000 字)

**Key takeaway**: 把 PPO 的 L^CLIP 拆成 ratio + min + clip 三步, 让读者写出公式不是抄公式; 解释 value head 和 entropy bonus.

- **Step 1 直觉**:
  - 朴素 policy gradient 一更新太狠就崩 (TRPO 的 motivation). TRPO 用 trust region 约束 KL, 数学复杂.
  - PPO 的极简方案: "重要性采样比 ratio r(θ) = π_θ / π_old; 限制 r 不离开 [1−ε, 1+ε]". 看着土, 比 TRPO 还稳.
  - 配图: PPO clip 的 surrogate loss 形状图 (paper Fig 1 风格).

- **Step 2 最小 demo** (40 行): PPO clipped surrogate loss + 1 步 value head update, 在 toy bandit 上.

- **Step 3 正式化**:
  - L^CLIP(θ) = E_t[min(r_t · Â_t, clip(r_t, 1−ε, 1+ε) · Â_t)] (PPO paper Eq. 7).
  - 为什么用 min 而不是简单 clip: 让 advantage > 0 时 ratio 大反被压, advantage < 0 时 ratio 小反被压 —— "悲观估计".
  - GAE: Â_t = Σ_{k=0}^{T−t} (γλ)^k δ_{t+k}, 调 (γ, λ) 控偏差/方差.
  - Total: L = L^CLIP − c1 · L^VF + c2 · S[π] (value loss + entropy bonus).

- **Step 4 代码引用**: `huggingface-trl/trl/experimental/ppo/ppo_trainer.py` 中 `ratio = exp(new_logprob − old_logprob)` + `pg_loss = max(pg_loss1, pg_loss2)` 的 30 行片段.

- **Step 5 洞察**:
  - 为什么 clip 而不是 KL penalty? 实证: KL penalty β 难调 (太小不约束, 太大学不到), clip ε 几乎对所有任务用 0.2.
  - PPO 是 LLM 上 RLHF 的"事实标准"是因为它兼容 dropout / parameter sharing —— TRPO 不行.

---

## §4. RLHF: 把 PPO 接到 LLM 上 (≈ 1500 字)

**Key takeaway**: InstructGPT 三段式 (SFT → RM → PPO), 重点解释 reward model 训练 (BCE on preference pairs) 和"为什么 reward 要减一个 KL".

- **Step 1 直觉**:
  - 直接让人打 reward 太贵 (每个回答打 0–10 分, 人会分不清), 但让人选"A 更好还是 B 更好"很便宜.
  - SFT: 监督学一个 base policy. RM: 训一个网络拟合人类偏好. PPO: 用 RM 的分数当 reward, 跑 PPO.
  - 但 reward 一旦给了大模型, 模型会"刷分" (复读机化, 长度爆炸, exploit RM bug) → 加 β KL(π||π_ref) 作惩罚.
  - 配图: InstructGPT Fig 2 三段式流程图 (paper crop).

- **Step 2 最小 demo** (25 行): Bradley-Terry RM 训练 loss + RLHF 总 reward = r(x,y) − β log(π(y|x)/π_ref(y|x)).

- **Step 3 正式化**:
  - BT 模型: P(y_w > y_l | x) = σ(r(x, y_w) − r(x, y_l)).
  - RM loss: L_RM = −E[log σ(r(x, y_w) − r(x, y_l))].
  - RLHF objective: max_π E[r(x,y) − β log(π(y|x)/π_ref(y|x))].
  - 实现上 KL 不是单独求, 是 token-level 加进 reward.

- **Step 4 代码引用**:
  - `huggingface-trl/trl/trainer/reward_trainer.py` 的 RM loss (BCE on chosen/rejected scores).
  - `huggingface-trl/trl/experimental/ppo/ppo_trainer.py` 中 `non_score_reward = -kl_coef * kl` 的 token-level KL 整合代码.

- **Step 5 洞察**:
  - RLHF 的"3 步"不是"流程", 是"3 个 loss". 任何一步坏掉都崩.
  - KL 项不是优化术语, 是 alignment 的 **安全装置** —— 没它模型会跑飞.

---

## §5. DPO: 干脆把 RL 阶段消掉 (≈ 2200 字, 与 §6 篇幅对等)

**Key takeaway**: DPO 的核心是一个**代数操作** —— 把 RLHF 的最优 π* 反解出 r, 再代回 BT loss. 本节推导完整呈现 + 代码逐行对应到公式.

- **Step 1 直觉**:
  - RLHF 训俩网络 (RM + policy) + online sample, 工程复杂.
  - DPO 关键洞察: KL-约束最优 π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β) 给出 r = β log(π*/π_ref) + const. 把这个 r 代回 RM 的 BT loss, 整个目标变成只关于 π 的 BCE loss.
  - 结果: 一个 dataset (chosen/rejected pairs) + 一个 BCE loss + 一行 forward, 没了 RM, 没了 online sample.
  - 配图: DPO 论文 Fig 1 (PPO vs DPO pipeline 对比 crop).

- **Step 2 最小 demo** (12 行): 手写 DPO loss `−log σ(β · ((logπ_w − logπ_ref_w) − (logπ_l − logπ_ref_l)))`. 用 4 个 scalar 模拟 chosen/rejected/ref_chosen/ref_rejected 的 log-prob, 走一次反向传播.

- **Step 3 正式化** (这是全 tutorial 推导最重的一节, 不省一步):
  - 起点: max_π E_{y\sim π}[r(x,y)] − β·KL(π||π_ref).
  - 用 Lagrangian 把 KL 展开 → 最优解形式 π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β). 配 1 行翻译.
  - 反解 r: r(x, y) = β log(π*(y|x)/π_ref(y|x)) + β log Z(x). 配 1 行翻译.
  - 代回 BT preference loss: L = −E[log σ(r_w − r_l)]; Z(x) 在差分中抵消. 配 1 行翻译.
  - 最终 DPO loss: **L_DPO = −log σ(β · ((logπ_w − logπ_ref_w) − (logπ_l − logπ_ref_l)))**. 配 1 行翻译.
  - 推完后给一张"5 个符号对应到代码变量"的小表, 衔接 Step 4.

- **Step 4 代码引用 (与 §6 同等深度, 三段)**:
  1. **per-token log-prob 计算** — `huggingface-trl/trl/trainer/utils.py:selective_log_softmax` (~10 行), 从 logits 算 logπ. 解释为什么不能直接 `log_softmax(logits).gather(...)`: gather 在 vocab=128k 时显存爆.
  2. **chosen/rejected 的 logratio** — `huggingface-trl/trl/trainer/dpo_trainer.py` 中 chosen_logps - ref_chosen_logps 的计算 (line ~1230–1240), 6–10 行. 公式↔代码: `chosen_logratios` 对应 logπ_w − logπ_ref_w.
  3. **最终 sigmoid loss** — `dpo_trainer.py` line ~1258, `per_sequence_loss = -F.logsigmoid(self.beta * delta_score)`. 把 Step 3 的公式与这一行一字一字对上.
  4. 给一张 "DPO 公式符号 ↔ trl 代码变量名" 的对照表 (β, π_θ, π_ref, y_w, y_l, σ).

- **Step 5 洞察**:
  - DPO 不是 PPO 的 drop-in 替代品 —— 它丢掉了 online sampling, 只能用 **offline** preference data.
  - 工业落地: 数据准备好就能跑, 单 GPU 也能搞 7B 模型, 这是 DPO 流行的根本原因.
  - 弱点: 在 chosen 与 reference 已经接近的样本上, gradient 几乎为零 (over-saturated). IPO 加 hinge, KTO 改 utility, 都是在补这个洞.
  - DPO 的 β 不是"学习率"也不是"温度", 是 KL 约束的强度. β 太小 (~0.01) 模型偏移过大, β 太大 (~1.0) 几乎学不到. 实战常用 0.1.

---

## §6. GRPO 与它的后辈: 把 critic 省掉之后 (≈ 2200 字, 与 §5 篇幅对等)

**Key takeaway**: GRPO 用"同 prompt 多采样取相对优势"代替 critic; 然后 DAPO / Dr.GRPO / GSPO 各自修补 GRPO 的不同毛病. 本节后半段集中讲后续, 但代码引用深度与 §5 平齐.

- **Step 1 直觉**:
  - PPO 要训 value head (critic), 多一个 7B 网络一半显存.
  - GRPO 说: 同一 prompt 我采样 G = 64 个回答, 直接用组内 reward 的均值/标准差归一. 不需要 critic.
  - 优势: 显存 PPO 的 ~60%; reward 自然 normalize, 训练曲线稳; DeepSeek-R1 (2025) 训练框架就是它.
  - 后续 (Step 5 详述): DAPO 拆 clip、Dr.GRPO 去掉 std normalize、GSPO 把 importance ratio 从 token 升到 sequence —— 三家都在修 GRPO 的一个具体毛病.
  - 配图: GRPO paper Fig 4 (PPO vs GRPO 数据流对比 crop) + 一张自绘的 "GRPO → DAPO / Dr.GRPO / GSPO 三条分支" 演化图.

- **Step 2 最小 demo** (20 行): `group_advantage(rewards, G)` 函数 (mean+std normalize) + 一次 GRPO 更新 pseudocode, 强调与 PPO Step 2 demo 的 diff: 只有 advantage 计算和 KL 项位置两处不同.

- **Step 3 正式化 (与 §5 同等深度推导)**:
  - 起点: 沿用 PPO 的 clipped surrogate, 但把 GAE advantage 换成 group-relative advantage.
  - GRPO 完整 objective:
    J_GRPO = E_{q, {o_i}_{i=1}^G}[ (1/G) Σ_i (1/|o_i|) Σ_t [ min(r_{i,t}·Â_{i,t}, clip(r_{i,t}, 1−ε, 1+ε)·Â_{i,t}) − β·D_KL(π_θ || π_ref) ] ]
    每个符号 inline 解释.
  - Â_{i,t} 的两个版本:
    - Outcome supervision: Â_{i,t} = (R_i − mean(R_{group})) / std(R_{group}) for all t.
    - Process supervision: 用 step-level reward 累加.
  - 与 PPO 的两点核心差异 (并放在表里):
    1. advantage: group-relative vs GAE.
    2. KL: 加在 loss (β·KL) vs 加在 reward (token-level r − β·log π/π_ref).
  - 每步公式后跟翻译.

- **Step 4 代码引用 (与 §5 同等深度, 三段)**:
  1. **group advantage 计算** — `huggingface-trl/trl/trainer/grpo_trainer.py` line ~2160–2200, 把每个 prompt 的 G 个 rewards 算 mean/std 然后 (r-mean)/std. 解释为什么用 `1e-4` 防除零 + advantages reshape 的 batch 含义.
  2. **importance ratio + clip** — `grpo_trainer.py` line ~2495–2520, `coef_1 = torch.exp(log_importance_weights)` 对应 r_{i,t}; `coef_2 = torch.clamp(coef_1, 1-ε, 1+ε)` 对应 clip(r); 最后 `torch.min(coef_1 * adv, coef_2 * adv)`.
  3. **token-level KL 加在 loss** — `grpo_trainer.py` line ~2514, `per_token_kl = exp(ref_logp − logp) − (ref_logp − logp) − 1` (k3 estimator, 比朴素 KL 方差更小). 公式↔代码: 解释为什么不是直接 `logp - ref_logp`.
  4. 给一张 "GRPO 公式符号 ↔ trl 代码变量名" 的对照表 (G, π_θ, π_old, π_ref, Â_{i,t}, ε, β).

- **Step 5 洞察 + GRPO 后续演化 (这一节的"主题升级")**:
  - GRPO 的隐含假设: "**同一 prompt 多采样有意义**". LLM 推理任务上 64 个回答自然有对有错, 提供对比信号; Atari/Robotics 上不成立.
  - DeepSeek-R1 把 GRPO 推到了"无 SFT 直接 RL"的极致, 但前提是 reward 程序可验证 (math 答案、code 测试).
  - **GRPO 的三个后辈** (重点补一段, 每个 1 小段 + 一段 trl/verl 中的具体改动):
    - **DAPO** (ByteDance, 2025, arxiv:2503.14476) —— 把 clip 上下界拆开 (ε_low, ε_high), 让低概率 token 有机会涨; 动态过滤"全对/全错"的 prompt batch. trl 中通过 `epsilon_high` 配置, verl 有专门的 `dapo.py` reward manager.
    - **Dr.GRPO** (Sea AI Lab / NUS, 2025, arxiv:2503.20783) —— 论文证明 GRPO 的 `/std` 会引入"难度偏差", 让答错样本变长. 修正: 去掉 std normalize 和 length normalize. trl 的 `grpo_config.py:norm_adv_scaling` 即此开关.
    - **GSPO** (Qwen Team, 2025, arxiv:2507.18071) —— GRPO 的 ratio 在 token 级会累乘出极端值. GSPO 把 importance ratio 改成 sequence-level (长度归一): s_i(θ) = (π_θ(y_i|x) / π_old(y_i|x))^{1/|y_i|}. trl 在 `trl/experimental/gspo_token/` 里实验性实现.
  - 总结: GRPO 是骨架, DAPO/Dr.GRPO/GSPO 是肌肉. 工程上, 先用 GRPO 跑通, 看到具体的崩点再选哪个补.

---

## §7. 跨域：Diffusion 的 RL —— DDPO 与 Diffusion-DPO (≈ 2000 字)

**Key takeaway**: 解释为什么 LLM 的 RL 方法不能原样搬到 Diffusion (log π 不 tractable); DDPO 的"把去噪步当 MDP action"思路; Diffusion-DPO 的 ELBO 替代.

- **Step 1 直觉**:
  - Diffusion 的"采样一次" = 50 步去噪 (x_T → x_0). 每一步都是一个 conditional decision. 把每一步当 RL action!
  - LLM 的 log π(y | x) 可以直接算 (logits → softmax → 累乘); Diffusion 的 log p_θ(x_0 | c) 是 intractable 的 (要 marginalize 全部 latent x_1, ..., x_T).
  - DDPO: 不算 log p(x_0), 而是算每一步的 log π_θ(x_{t−1} | x_t, c). reward 只在 t = 0 给一次, 用 PPO 风格 update.
  - Diffusion-DPO: 不重新发明 RL, 而是用 ELBO 把 DPO 公式里的 log π 换成 ELBO.
  - 配图: DDPO 论文 Fig 1 trajectory 视角图 (crop) + Diffusion-DPO Fig 1 概念图 (crop).

- **Step 2 最小 demo** (30 行): toy 1D DDPM 的 trajectory rollout + 每步累加 log π 的 pseudocode.

- **Step 3 正式化**:
  - DDPO 的 MDP 视角: s_t = (c, t, x_t), a_t = x_{t−1}, π_θ(a_t | s_t) = N(x_{t−1}; μ_θ(x_t, t, c), σ²I), R 在 t = 0 给 r(x_0, c).
  - DDPO loss (PPO-style on trajectory): L = E[min(ρ_t Â, clip(ρ_t, 1±ε) Â)].
  - Diffusion-DPO: log π(y|x) 替换为 −E_t,ε[||ε − ε_θ(x_t, t)||²], 即 noise prediction loss 的负数 (它是 −ELBO 的 surrogate).
  - 最终 Diffusion-DPO loss: −log σ(−β T (||ε − ε_θ^w||² − ||ε − ε_θ_ref^w||² − ||ε − ε_θ^l||² + ||ε − ε_θ_ref^l||²)) (大略形式).

- **Step 4 代码引用**:
  - `kvablack-ddpo-pytorch/ddpo_pytorch/...` 中 trajectory log_prob 累计 + PPO 风格 loss 片段.
  - `salesforce-DiffusionDPO/train.py` 中把 "noise prediction MSE 之差" 当 log ratio 的 ~15 行片段.

- **Step 5 洞察**:
  - **本质差异**: LLM 的 log π 是离散+可解析的, Diffusion 的 log π 是连续+intractable 的. 这一句决定了所有 LLM RL 方法搬到 Diffusion 都需要变形.
  - DDPO 比 RWR (reward-weighted regression) 强是因为 PPO 的 importance sampling ratio 让你能复用同一批 trajectory 多更新几次; RWR 必须每次重 sample.
  - Diffusion-DPO 的 trick 不是"DPO 推广", 而是"用 ELBO 当 log π 的替身" —— 同样的替身在 VAE、Energy-based model 里也适用.

---

## §8. 从头实现一遍 + 怎么选 (≈ 1500 字)

**Key takeaway**: 把全文公式翻成一张"项目要做什么"的实操表; 给一个 50 行的 DPO trainer 作为最小可工作起点.

- **Step 1 直觉**:
  - 实现优先级: SFT (基线) → DPO (数据准备好就能跑) → GRPO (online, 工程量 ×5) → PPO (最重) → Diffusion-DPO / DDPO (只做图像/视频 alignment 才碰).
  - 用什么数据/什么显存预算/什么训练时间, 用一张表说清.

- **Step 2 最小 demo** (50 行): 一个**真的能跑**的单 GPU DPO trainer, 玩具 preference data (4 对), 显示 loss 下降. 用 huggingface transformers + 一个 small LM (e.g. gpt2). 标注 `class="teaching-demo"`.

- **Step 3 正式化**:
  - 训练数据需求表 (DPO: 偏好对 N=10k+; PPO/GRPO: reward function 或 RM).
  - 显存预算表 (DPO ~1.0×, GRPO ~1.5×, PPO ~2.5× SFT).
  - 评估指标 (win-rate, MT-bench, reward, KL).

- **Step 4 代码引用**: `huggingface-trl/examples/scripts/dpo.py` 或 `trl/scripts/dpo.py` 的端到端 30 行入口.

- **Step 5 洞察**:
  - 论文里 ratio/clip/sigmoid 都是 5% 代码量, **95% 在 distributed sampling, async generation, KV cache 管理, gradient checkpointing**. 这些 paper 几乎不讲, 是工业级框架的核心.
  - 选 DPO 还是 GRPO: 你能定义"程序可验证的 reward" (代码对错, 数学答案) → GRPO 强; 只有人类偏好 → DPO 稳. 这是 2024–2025 整个领域的分水岭.

---

## Cross-section dependencies

- §1 introduces `π`, `log π`, `KL` — 所有后续节都用.
- §2 introduces `advantage A` — used by §3 (PPO), §6 (GRPO).
- §3 introduces `ratio r = π_θ/π_old`, `clip` — used verbatim by §6 GRPO, §7 DDPO.
- §4 introduces `reward model`, `β·KL` — used by §5 (DPO derivation).
- §5 introduces `BT loss form` 和 preference data — used by §7 (Diffusion-DPO).
- §6 introduces `group-relative advantage` — referenced in §8 implementation table.
- §7 references §3 (PPO) and §5 (DPO) — must come after them.
- §8 synthesizes 1–7.

## Drafting groups (for Phase 5 grouped-parallel)

- **Group A** (parallel): §1, §2 — foundation; 无前置依赖.
- **Group B** (parallel): §3, §4 — 用 §2 advantage; PPO 和 RLHF 并行.
- **Group C** (parallel): §5, §6 — 用 §3 ratio + §4 KL; DPO 和 GRPO 并行.
- **Group D** (parallel): §7, §8 — 用前面所有; Diffusion 和 synthesis 并行.

Total target: **13,000–15,000 中文字** (8 sections × 1700 avg).

## Provenance status

- ≥2 anchor papers ✓ (7 篇, 全部在 sources/papers/).
- ≥2 GitHub repos ✓ (4 个: huggingface-trl 18.4k★, volcengine-verl 21.3k★, kvablack-ddpo-pytorch 707★ academic-first-author, salesforce-DiffusionDPO 689★ official-research).
- Primary repo huggingface/trl 满足 1k★ 标准, 覆盖 PPO/DPO/GRPO 三大方法. **不需要 provenance-warning aside.**
