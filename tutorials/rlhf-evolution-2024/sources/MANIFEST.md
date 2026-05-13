# MANIFEST — RLHF Evolution (PPO → DPO → GRPO + Diffusion RL)

Tutorial: 大语言模型与 Diffusion 图像生成领域的强化学习方法演变。
Slug: `rlhf-evolution-2024`

## Papers (anchor)

- [ANCHOR] **Christiano et al. (2017)** — *Deep Reinforcement Learning from Human Preferences* — arxiv:1706.03741 — preference-based RL 起点; 提出"对比反馈训练 reward model"的范式。
- [ANCHOR] **Schulman et al. (2017)** — *Proximal Policy Optimization Algorithms* — arxiv:1707.06347 — PPO 原始论文; clipped surrogate objective 是 RLHF 的算法基石。
- [ANCHOR] **Ouyang et al. (2022, InstructGPT)** — *Training Language Models to Follow Instructions with Human Feedback* — arxiv:2203.02155 — 首次大规模在 LLM 上落地 SFT → RM → PPO 三段式流程。
- [ANCHOR] **Rafailov et al. (2023)** — *Direct Preference Optimization* — arxiv:2305.18290 — 证明 reward model 可以解析消除, RL 阶段直接变成监督式 loss。
- [ANCHOR] **Shao et al. (2024, DeepSeekMath)** — *Pushing the Limits of Mathematical Reasoning* — arxiv:2402.03300 — GRPO 首次提出, 用组内相对 advantage 替代 critic, 显存降一半。
- [ANCHOR] **Black et al. (2023, DDPO)** — *Training Diffusion Models with Reinforcement Learning* — arxiv:2305.13301 — 把扩散去噪 reframe 成多步 MDP, 直接用 PPO-style 训扩散模型。
- [ANCHOR] **Wallace et al. (2023, Diffusion-DPO)** — *Diffusion Model Alignment Using Direct Preference Optimization* — arxiv:2311.12908 — 借助 ELBO 把 DPO 套到扩散模型上, 不需要 reward model。

### GRPO 后续 (§6 后半段)

- [ANCHOR] **Yu et al. (2025, DAPO)** — *DAPO: An Open-Source LLM RL System at Scale* — arxiv:2503.14476 — ByteDance Seed; 把 clip 上下界拆开 + 动态过滤"全对/全错" batch。
- [ANCHOR] **Liu et al. (2025, Dr.GRPO)** — *Understanding R1-Zero-Like Training: A Critical Perspective* — arxiv:2503.20783 — Sea AI Lab / NUS; 证明 GRPO 的 `/std` 引入难度偏差, 去掉 std + length normalize 修正。
- [ANCHOR] **Zheng et al. (2025, GSPO)** — *Group Sequence Policy Optimization* — arxiv:2507.18071 — Alibaba Qwen Team; 把 importance ratio 从 token 级升到 sequence 级 (长度归一化)。

## Repos

- [PRIMARY] **huggingface/trl** — 18.4k★, 活跃 (May 2026) — code spine; 覆盖 PPO/DPO/GRPO/RM。引用文件:
  - `trl/trainer/dpo_trainer.py` — DPO loss (`dpo_loss` 方法)
  - `trl/trainer/grpo_trainer.py` — GRPO loss + group advantage
  - `trl/trainer/ppo_trainer.py` 或 `trl/experimental/ppo/ppo_trainer.py` — PPO update
  - `trl/trainer/reward_trainer.py` — RM 训练
- [COMPARE] **volcengine/verl** — 21.3k★ — GRPO 实战参考, DeepSeek-R1 训练用的就是这套; 用于 §6 GRPO 替代视角。
- [COMPARE] **OpenRLHF/OpenRLHF** — 9.5k★ — Ray 分布式 PPO; 用于 §3 PPO 工程视角对照 (可选)。
- [COMPARE] **kvablack/ddpo-pytorch** — 707★ (academic first-author release) — DDPO 原作者实现; 用于 §7 Diffusion-RL 部分。
- [COMPARE] **SalesforceAIResearch/DiffusionDPO** — 689★ (official research) — Diffusion-DPO 官方实现; 用于 §7 后半段。

## Blogs (concept-map)

- **Lilian Weng — Policy Gradient Algorithms** — https://lilianweng.github.io/posts/2018-04-08-policy-gradient/ — §2/§3 policy gradient & PPO 的概念地图。
- **Hugging Face — Illustrating RLHF** — https://huggingface.co/blog/rlhf — §4 三段式流水线的可视化对照。
- **Yuge Shi — PPO vs GRPO** — https://yugeten.github.io/posts/2025/01/ppogrpo/ — §6 PPO → GRPO 简化路径的清晰对比。
- **Lilian Weng — What are Diffusion Models?** — https://lilianweng.github.io/posts/2021-07-11-diffusion-models/ — §7 扩散背景的快速回顾参考。

## Provenance notes

- 所有 7 篇 anchor paper 的 arxiv ID 已 (子代理) 验证。
- PRIMARY repo huggingface/trl 覆盖 PPO/DPO/GRPO 三大 LLM 方法, 满足 ≥2 repo + ≥2 paper 引证标准, **无需 provenance warning**。
- Diffusion 侧 kvablack/ddpo-pytorch 仅 707★ 但是原作者发布, 符合 "first-author academic release" 标准。
