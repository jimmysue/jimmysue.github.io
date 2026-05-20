# MANIFEST — VLM 在工业检测领域 (slug: `vlm-industrial-2024`)

教程目标:覆盖三大主题(让 VLM 看清细节 / 数据制作 / 泛化能力)
,展示 2023–2026 SOTA 演进。

## Papers

### Anchor 1 — WinCLIP (Theme 1: 细粒度)
- **Title:** WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation
- **Authors:** Jeong, Zou, Kim, Zhang, Ravichandran, Dabeer (Amazon AWS)
- **Venue:** CVPR 2023
- **arxiv:** 2303.14814
- **Rationale:** Seminal CLIP-based zero-shot AD。引入 multi-window patch
  aggregation 和 compositional prompt ensembles,展示如何让 CLIP "看清"
  patch-level 细节。是后续工作的基线。
- **Code:** 官方无代码,引用社区高质量复现 [caoyunkang/WinClip]

### Anchor 2 — AnomalyCLIP (Theme 3: 泛化)
- **Title:** AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection
- **Authors:** Zhou, Pang, Tian, He, Chen (Singapore Management Univ.)
- **Venue:** ICLR 2024
- **arxiv:** 2310.18961
- **Rationale:** 解决 CLIP "focus on 类别语义而非 abnormality" 的问题
  ,引入可学习的 object-agnostic prompt + DPAM (Diagonally Prominent
  Attention Map)。17 个数据集上展示 cross-domain zero-shot 能力。
- **Code:** [zqhang/AnomalyCLIP] ✓ — 是 PRIMARY 代码主线

### Anchor 3 — AnomalyGPT (Themes 1+2: LVLM + 数据合成)
- **Title:** AnomalyGPT: Detecting Industrial Anomalies Using Large Vision-Language Models
- **Authors:** Gu, Zhu, Zhu, Chen, Tang, Wang (CASIA)
- **Venue:** AAAI 2024 (Oral)
- **arxiv:** 2308.15366
- **Rationale:** 首个 LVLM-based 工业 AD 方法。引入 synthetic defect
  generation + textual description 的训练数据范式,展示 multi-turn
  对话 + few-shot reasoning。重要:数据制作章节的核心引用。
- **Code:** [CASIA-IVA-Lab/AnomalyGPT] ✓ — 1.1k★

### Supplementary — IADGPT (现代演进)
- **Title:** IADGPT: Unified LVLM for Few-Shot Industrial Anomaly Detection, Localization, and Reasoning via In-Context Learning
- **arxiv:** 2508.10681 (2025)
- **Rationale:** 三阶段渐进训练 + 100K 多属性数据集 + 注意力定位。
  仅作为"演进时间线"中的现代代表,不引代码(repo 未公开验证)。

### Supplementary — IMDD-1M (大规模数据)
- **Title:** Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset
- **arxiv:** 2512.24160 (2025)
- **Rationale:** 1M 图文对,400+ defect types,60+ material categories
  。作为"数据规模化"的现代案例引用。GitHub repo (NinaNeon/IMDD-1M)
  确认 404,仅文字引用。

### Supplementary — AD-Copilot (in-context VLM)
- **arxiv:** 2603.13779 (2026)
- **Rationale:** Comparison Encoder + Chat-AD 数据集,82.3% MMAD。
  作为"VLM-in-loop 数据标注"的现代代表引用。

## Repos

### PRIMARY — zqhang/AnomalyCLIP
- **URL:** https://github.com/zqhang/AnomalyCLIP
- **Why primary:** 官方 ICLR'24 代码,结构清晰,从头到尾覆盖
  prompt learning / DPAM / loss / training loop / inference,行号稳定。
- **Key files:**
  - `AnomalyCLIP_lib/CLIP.py` — modified CLIP visual transformer w/ DPAM
  - `prompt_ensemble.py` — learnable object-agnostic prompt
  - `train.py` — end-to-end training
  - `loss.py` — focal + dice loss for pixel-level
  - `test.py` — zero-shot inference

### PRIMARY 2 — CASIA-IVA-Lab/AnomalyGPT
- **URL:** https://github.com/CASIA-IVA-Lab/AnomalyGPT
- **Why primary:** 官方 AAAI'24 oral repo,1.1k★。涵盖 synthetic
  defect generation pipeline 与 PEFT/LoRA 训练配置。
- **Key files:**
  - `code/datasets/mvtec.py` — synthetic anomaly generation
  - `code/model/openllama.py` — LVLM forward (ImageBind + Vicuna)
  - `code/scripts/train_mvtec.sh` — training script
  - `code/dsconfig/openllama_peft_stage_1.json` — DeepSpeed config

### BASELINE — openvinotoolkit/anomalib
- **URL:** https://github.com/openvinotoolkit/anomalib
- **Stars:** ~5.8k,Apache-2.0,Intel/OpenVINO 维护
- **Why baseline:** 生产级 baseline 对比,30+ AD 算法统一框架。用于
  "传统视觉 vs VLM"对照,展示 PatchCore / EfficientAD 等不依赖 VLM
  的方案作为 baseline。
- **Key files:**
  - `src/anomalib/models/image/patchcore/torch_model.py`
  - `src/anomalib/data/datasets/image/mvtec_ad.py`

### COMMUNITY — caoyunkang/WinClip
- **URL:** https://github.com/caoyunkang/WinClip
- **Why included:** WinCLIP 官方无代码,这是社区里最被广泛 fork 的
  PyTorch 复现。引用时明确标注 "community reimplementation,Jeong 等
  原作者未发布"。

## Blogs / 文档
- [anomalib docs](https://anomalib.readthedocs.io/en/latest/) — 产业级
  AD pipeline 文档,作为"工程落地"概念地图参考。
- [M-3LAB/awesome-industrial-anomaly-detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
   — 该领域最完整的 paper + dataset list。
- [AnomalyGPT project page](https://anomalygpt.github.io/) — 官方
  demo 视频和数据集说明。

## Discovery Notes (provenance flags)

⚠️ **本教程在以下方面有 provenance gap:**
- WinCLIP **官方未发布代码**,§2 (WinCLIP) 的 Step 4 代码引自社区
  复现 caoyunkang/WinClip。会在该节顶部标注 `<aside
  class="provenance-warning">`。
- IADGPT / IMDD-1M / AD-Copilot 三篇 2025+ 论文**未验证可用代码**,
  仅作为 "演进时间线" 文字引用,不引代码。
- 整个 VLM × 工业检测领域**几乎没有 Lilian Weng-style 深度博客**,
  这是个相对新兴的方向,本教程的写作本身就在填补空缺。
