# MANIFEST — RL Fine-tuning for Diffusion Image Generation (DDPO → AWM lineage)

Topic slug: `rl-for-diffusion-2023`
Year: 2023 (seminal — DDPO and DPOK both 2023-05)
Audience: motivated reader who knows basics of diffusion + PyTorch, wants to understand both theory and real implementation

## Papers

### Lineage spine (seminal)

- **[SEMINAL-1]** DDPO — *Training Diffusion Models with Reinforcement Learning* (Black, Janner, Du, Kostrikov, Levine, 2023-05) — arxiv:2305.13301 — The originator. Casts denoising as a multi-step MDP, derives per-step Gaussian log-likelihood as the policy, applies REINFORCE/PPO. Every subsequent paper either extends or refutes this framing.

- **[SEMINAL-2]** DPOK — *Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models* (Fan et al., 2023-05) — arxiv:2305.16381 — Concurrent with DDPO; adds KL regularization to a reference model, which becomes the standard pattern. Validates the DDPO framing independently.

- **[SEMINAL-3]** Flow-GRPO — *Training Flow Matching Models via Online RL* (Liu, Liu, Liang et al., 2025-05) — arxiv:2505.05470 — Extends DDPO to flow-matching backbones (SD3.5, FLUX). Two contributions: ODE→SDE conversion for explorability + Denoising Reduction (sample with fewer steps during training rollouts).

- **[SEMINAL-4]** DanceGRPO — *Unleashing GRPO on Visual Generation* (Xue, Jiang et al., 2025-05) — arxiv:2505.07818 — Unified GRPO over diffusion + rectified flow across image/video/multi-modal. Demonstrates the framework generalizes to high-fidelity video models (HunyuanVideo, Wan2.1).

- **[SEMINAL-5]** AWM — *Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models* (Xue, Ge, Zhang, Li, Ma, 2025-09) — arxiv:2509.25050 — The synthesizer / refutation. Proves DDPO ≡ noisy-DSM (Theorem 1), bounds the extra variance (Theorem 2), proposes replacing the RL surrogate with the pretraining flow-matching loss × advantage. Gets 6–24× speedup over Flow-GRPO.

### Algorithmic anchors

- **[ANCHOR-1]** PPO — *Proximal Policy Optimization Algorithms* (Schulman, Wolski, Dhariwal, Radford, Klimov, 2017-07) — arxiv:1707.06347 — The clipped-surrogate objective that every DDPO descendant implements. We refer to this paper's Section 3 directly.

- **[ANCHOR-2]** GRPO — *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* (Shao et al., 2024-02) — arxiv:2402.03300 — Introduces Group Relative Policy Optimization: the critic-free advantage estimation that Flow-GRPO / Dance-GRPO / AWM inherit. The "G" in all those names.

## Repos

### Primary code spine — used for §3 onward (the deepest walkthroughs)

- **[PRIMARY]** `yifan123/flow_grpo` — 2.3k★ — last activity 2025-05 — Clean training scripts (`scripts/train_sd3.py`), explicit log-prob-tracked sampling (`flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`), well-separated config (`config/grpo.py`). This is the cleanest pedagogical implementation of the DDPO-MDP formulation on flow models.
  - Core files:
    - `scripts/train_sd3.py` — training loop with GRPO ratio + clip
    - `flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py` — sampler that returns per-step log-probs
    - `flow_grpo/diffusers_patch/sd3_sde_with_logprob.py` — ODE→SDE conversion + Gaussian per-step log-prob
    - `config/grpo.py` — exact hyperparameters

### DDPO original (§1-2 historical reference)

- **[DDPO-PYTORCH]** `kvablack/ddpo-pytorch` — 760★ — last activity 2023-10 — Most-cited PyTorch port of DDPO, used in HuggingFace's TRL DDPO trainer. Author quality release.
  - Core files:
    - `ddpo_pytorch/diffusers_patch/pipeline_with_logprob.py` — DDIM with log-prob tracking
    - `scripts/train.py` — training loop
    - `ddpo_pytorch/rewards.py` — reward function adapters

### DanceGRPO (§4)

- **[DANCEGRPO]** `XueZeyue/DanceGRPO` — 1.6k★ — last activity 2025-10 — Multi-backbone (FLUX, SD, HunyuanVideo) GRPO over visual generation.
  - Core files:
    - `train_grpo_flux.py` — FLUX-specific training
    - `train_grpo_sd.py` — SD-specific training
    - `fastvideo/` — shared generation framework

### AWM (§5 — the synthesis)

- **[AWM]** `scxue/advantage_weighted_matching` — 75★ — last activity 2025-09 — First-author release of AWM. Implements BOTH a DDPO-style baseline (under `flow_grpo/` folder) AND AWM in one codebase — perfect for showing the diff.
  - Core files:
    - `advantage_weighted_matching/scripts/train_sd3_awm.py` — main training script
      - L210-L252: `compute_log_prob_awm` — flow matching loss as ELBO surrogate
      - L1265-L1354: main loop — ratio × advantage + velocity KL + EMA KL
    - `advantage_weighted_matching/config/dgx_awm.py` — production configs

### Production cross-reference (§2 and §6)

- **[TRL]** `huggingface/trl` — 18.4k★ — current — Production RLHF library; DDPO trainer is widely used in industry.
  - Core files:
    - `trl/trainer/ddpo_trainer.py` — `DDPOTrainer` class (production DDPO)
    - `trl/trainer/ddpo_config.py` — config schema

### DPOK (§2 alternative — KL pattern source)

- **[DPOK]** `google-research/google-research` — 37.8k★ — under `dpok/` subfolder — official Google release.
  - Core files:
    - `dpok/train_online_pg.py` — online policy gradient with KL
    - `dpok/reward_model.py` — reward model wiring

## Blogs

- **HuggingFace TRL team** — *Finetune Stable Diffusion Models with DDPO via TRL* — https://huggingface.co/blog/trl-ddpo — 2023-09-29 — The clearest practitioner-facing writeup of DDPO's MDP framing with code; we cross-reference it for §2 narrative and §6 production tips.

## Provenance notes

- AWM repo is 75★, below the 1k★ hard filter, but qualifies as a **first-author academic release** linked from the arxiv abstract — the highest quality tier. No `provenance-warning` needed.
- DDPO PyTorch port is 760★, but is the **de-facto standard PyTorch impl** (used by HuggingFace TRL) — also qualifies as production-grade despite the count.
- All other primary repos exceed 1k★ or are inside official org mono-repos.

## How the lineage maps to sections (preview — see plan.md for full outline)

| §  | Concept                                  | Primary paper       | Primary code                                  |
| -- | ---------------------------------------- | ------------------- | --------------------------------------------- |
| 1  | RL 在扩散模型上"为什么必要"          | DDPO §1             | (history)                                     |
| 2  | DDPO 的 MDP 形式化 + per-step log-prob   | DDPO §3, DPOK §3    | kvablack/ddpo-pytorch + trl/ddpo_trainer.py   |
| 3  | PPO 风格 ratio + clip + KL 正则         | PPO §3, DDPO §3.2   | trl/ddpo_trainer.py                           |
| 4  | GRPO: critic-free advantage              | DeepSeekMath §3     | (concept)                                     |
| 5  | Flow-GRPO: ODE→SDE + Denoising Reduction | Flow-GRPO §3        | yifan123/flow_grpo                            |
| 6  | Dance-GRPO: 跨 backbone 统一            | Dance-GRPO §4-5     | XueZeyue/DanceGRPO                            |
| 7  | AWM: Theorem 1+2 + 改写 surrogate       | AWM §3-4            | scxue/advantage_weighted_matching             |
| 8  | 全景对比 + 选型建议                     | (synthesis)         | —                                             |
