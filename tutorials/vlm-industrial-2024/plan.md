# Plan — VLM 在工业检测领域 (`vlm-industrial-2024`)

## Audience contract
"研究者读完每节的 Step 1-2 能用自己的话讲清概念演进,本科生能补完 Step 3 的推导,工程师能用 Step 4 的代码直接 fork 上手。三大主题(看清细节 / 数据制作 / 泛化)各有 1 个专门统合章节 + 散落在 SOTA 演进章节里的具体技术。"

## 整体结构 (8 大节, target 15-16k 中文字)

教程结构 = "演进时间线 (§1-4) + 主题深挖 (§5-7) + 综合 (§8)" 双线交织。每个 SOTA 方法都同时被时间线和主题章节引用,避免重复但相互照应。

---

### §1. 问题与挑战 — 工业检测为什么需要 VLM (≈1500 字)
**Key takeaway:** 工业检测是长尾、标注稀缺、缺陷上下文相关的场景; 传统 one-class 学习不够, 而 vanilla CLIP 看全局看不清局部, 需要专门改造。

- **Step 1 直觉:** "工厂老工人 vs 新人 vs CLIP" 的类比 — 老工人见过无数 normal 样本(传统 PatchCore 范式),新人看说明书(VLM 用语言指导),但 CLIP 拿到的"说明书"太粗。配图: AnomalyCLIP Fig.1 — vanilla CLIP / WinCLIP / CoOp / AnomalyCLIP 四种 prompt 在同一张图上的 attention map 对比。
- **Step 2 demo:** 15 行 numpy/torch 代码,展示 vanilla CLIP 在一张 MVTec 螺帽缺陷图上 `cos_sim(image_embed, "a photo of a defective screw")` vs `cos_sim(image_embed, "a photo of a normal screw")` 几乎打平 → 直观看到"为什么不能直接用"。
- **Step 3 推导:** CLIP zero-shot 分类的 cosine similarity 公式 + 失败的形式化 — 全局 pooling 把 defect 信号稀释。引入 patch-level alignment 的目标。
- **Step 4 代码:** anomalib `src/anomalib/models/image/patchcore/torch_model.py` 的 PatchCore 实现 — 这是传统 baseline,让读者看清 "传统怎么做",才知道 VLM 后续 *为什么需要存在*。
- **Step 5 洞察:** (a) AD 不是分类问题,是 "判断是否偏离分布" 问题; (b) language 是 normality 的 missing context; (c) 工业 ≠ 自然图像,domain gap 决定 backbone 选择。

### §2. WinCLIP — 让 CLIP 看清局部细节 (≈1800 字)
**Key takeaway:** 用 compositional prompt ensemble (状态词 × 模板) 显式编码 normality,用 multi-scale 窗口对齐 patch-level 视觉特征与文本。首次证明 *仅靠 prompt 工程 + 多尺度* 就能在 MVTec-AD 拿 91.8% AUROC。

- **Step 1 直觉:** 老工人看零件不是看一眼全图,是 "拿放大镜扫一遍"。把图像切成多尺度窗口 (small / medium / large),每个窗口跟 prompt 算相似度。配图: WinCLIP Fig.2 — 多尺度窗口聚合 + (b) language clarifies normal vs anomaly + (c) reference images 三联图。
- **Step 2 demo:** ~25 行 PyTorch,在一张图上手写 sliding-window crop + CLIP encode + max similarity 聚合,产出 anomaly heatmap。`teaching-demo` 标签。
- **Step 3 推导:** Compositional prompt ensemble 的数学形式: $\text{score}(x) = \frac{1}{|S|} \sum_{s_i \in S, t_j \in T} \cos(f_v(x), f_t(t_j(s_i)))$, 其中 $S$ 是状态词集合, $T$ 是模板集合。Window aggregation: $A(x) = \max_{w \in W} \cos(f_v(x_w), f_t(\bar{t}_{anom}))$, 多尺度 max-pool。
- **Step 4 代码:** `caoyunkang-WinClip/WinCLIP/...` — quote 三段: (a) prompt ensemble 构造 (template_list × state_list 笛卡尔积), (b) window-based feature extraction, (c) multi-scale score aggregation。明确标注 community reimplementation, 加 `<aside class="provenance-warning">`。
- **Step 5 洞察:** (a) 为什么 ensemble > single prompt: variance reduction; (b) 为什么 window > full image: dense alignment 解决 global pooling 的稀释; (c) 局限: prompt 包含 object name (e.g. "screw"), 切到新 object 类别 prompt 必须改 → 引出 §3。

### §3. AnomalyCLIP — Object-agnostic prompt + DPAM (≈2000 字)
**Key takeaway:** 把 prompt 从 "object name + state" 改成 "object-agnostic + state",让 prompt 学到 *什么是 abnormality*,而不是 *什么是 defective screw*。再加 DPAM 修改 CLIP 的 attention map,让局部缺陷 attention 显化。17 个数据集 cross-domain (industrial + medical)。

- **Step 1 直觉:** WinCLIP 的 prompt 是 "a photo of a defective {object}", 换到 medical CT 时 {object} = "lung tumor"? 不通。学一个不包含 object name 的 prompt: "a photo of an anomalous [class]", 用可学习 token 填空。配图: AnomalyCLIP Fig.1 — 四种 prompt 范式 attention map 横向对比。
- **Step 2 demo:** ~20 行 PyTorch, 演示用一个可学习 embedding `[V1][V2]...[Vn]` 替代手写 prompt template, 在 toy 二分类上反向传播。`teaching-demo`。
- **Step 3 推导:** 
  - Learnable prompt: $\mathbf{T}^- = [V_1, V_2, ..., V_E, \text{state\_normal}, \text{object}], \mathbf{T}^+ = [V_1, ..., \text{state\_abnormal}, \text{object}]$
  - DPAM (Diagonally Prominent Attention Map): 修改 V-V self-attention (而非 Q-K), 让 attention 更聚焦局部 — 给出原文 Eq. 6 完整形式
  - 双 loss: image-level focal + pixel-level dice
- **Step 4 代码:** `zqhang-AnomalyCLIP/prompt_ensemble.py` (learnable prompt 模块), `AnomalyCLIP_lib/CLIP.py` (DPAM 改造), `loss.py` (focal + dice)。展示 3 段引用。
- **Step 5 洞察:** (a) Object-agnostic 为什么是 cross-domain 关键: 它让 prompt 编码的是 abnormality 概念本身; (b) DPAM 的 V-V attention 修改为什么比 Q-K 更稳: gradient 路径; (c) 局限: 还是 contrastive 框架,无法描述缺陷类型 → 引出 LVLM。

### §4. AnomalyGPT — LVLM 进场 + 合成数据 (≈2500 字)
**Key takeaway:** 从 "二分类 + heatmap" 升级到 "对话 + 描述"。三个核心组件: (1) synthetic anomaly generation (cut-paste 风格), (2) lightweight visual-textual feature-matching decoder 补 LLM 的 spatial blindness, (3) prompt learner (PEFT) 不动 LLM 主参数。首次实现 threshold-free judgement + multi-turn dialogue。

- **Step 1 直觉:** CLIP 说 "这张图像 anomaly", AnomalyGPT 说 "螺帽左上角有划痕, 大约 2mm, 属于表面缺陷"。LLM 的 spatial weakness 用一个外挂 image decoder 补。配图: AnomalyGPT Fig.1 — 三方对比 (传统 IAD / 通用 LVLM / AnomalyGPT) + Fig.2 整体架构。
- **Step 2 demo:** ~25 行 numpy, 实现一个 toy 的 cut-paste anomaly synthesizer — 随机 crop 一块,paste 到原图另一个位置,生成假缺陷训练样本。`teaching-demo`。
- **Step 3 推导:**
  - Synthetic data pipeline: 从 normal 图 $x_n$ 生成 $x_a = M \odot \text{paste}(x_n, x_n') + (1-M) \odot x_n$
  - Feature-matching decoder: $\hat{m} = \text{Decoder}(F_v) = \sum_l \text{conv}_l(F_v^{(l)})$, 多层特征融合
  - Prompt learner: prompt embedding $E_p$ 是新增的可学习 tokens, $\text{LLM}(F_v, E_p, \text{instruction}) \to \text{response}$
  - Loss: 重建 mask $\hat{m}$ 与 ground-truth $m$ 的 BCE + dice + LLM 的 cross-entropy (在 instruction-tuning 数据上)
- **Step 4 代码:**
  - `CASIA-IVA-Lab-AnomalyGPT/code/datasets/mvtec.py` — synthetic anomaly generation 完整 pipeline
  - `code/model/openllama.py` — visual-textual feature matching decoder + prompt learner
  - 配置: `code/dsconfig/openllama_peft_stage_1.json` — 展示 PEFT 配置 (LoRA / freeze LLM)
- **Step 5 洞察:** (a) 为什么 prompt embedding > full fine-tune: 防止 catastrophic forgetting (LLM 通用知识不能丢); (b) decoder 为什么必要: LLM 给的是文本概率,没有 spatial localization; (c) 合成数据为什么够用: pretrain 时见过太多自然图像,IAD 只是 narrow domain shift。

### §5. 数据制作的演进:从 CutPaste 到 IMDD-1M (≈2000 字)
**Key takeaway:** 数据这条线的演进 = 合成方式越来越聪明 + 标注越来越自动化 + 规模越来越大 + 标签越来越结构化(从 binary → mask → 文本描述 → CoT 推理链)。

- **Step 1 直觉:** "造缺陷比找缺陷便宜": 把 normal 图主动破坏,造 paired (clean, broken) 数据集。但越简单的合成(随机 cut-paste)越离 real defect 远 → 演进路径 = "让合成的 defect 越来越像真的"。配图: 自制流程图 (figures/sec5-data-evolution.svg) — 五个时间节点 + 数据规模柱状图。
- **Step 2 demo:** ~30 行 PyTorch, 展示三种合成 (CutPaste / DRAEM noise / AnomalyGPT cut-paste with mask), 同一张 normal 图产出三种 synthetic anomaly, 并行可视化。`teaching-demo`。
- **Step 3 推导:**
  - CutPaste: 随机矩形 crop + 随机位置 paste
  - DRAEM: 用 Perlin noise 生成 anomaly mask, 用自然图像填充
  - AnomalyGPT 合成: SAM-mask 引导的 paste, 保证 anomaly 落在前景物体上
  - Text 标签生成: 用 GPT-4V 在合成图上做 caption (自动化标注闭环)
- **Step 4 代码:** `CASIA-IVA-Lab-AnomalyGPT/code/datasets/mvtec.py` 完整引用 cut-paste + mask 生成段落,附加注释对照论文 Eq.
- **Step 5 洞察:** (a) 合成 vs 真实数据混合比的 trade-off (paper 经验值); (b) 大规模数据 (IMDD-1M) 对 foundation model pretrain 的价值, 但 fine-tune 阶段反而 small + 高质量更好; (c) VLM-in-the-loop 标注 (AD-Copilot Chat-AD): 让 VLM 自己当标注员闭环。

### §6. 让 VLM 看清细节的技术谱系 (≈2000 字)
**Key takeaway:** 三条互补的细粒度路线: (a) 输入分辨率 — 多尺度 / tile / AnyRes; (b) 特征对齐 — patch-level alignment / DPAM 注意力修正; (c) 后端增强 — image decoder / SAM crop。SOTA 系统通常组合 3 条。

- **Step 1 直觉:** 缺陷大小是相对的 — 一根头发丝在面板上算大缺陷,在马路上算噪点。同一个网络要同时看清不同尺度,必须 "多放大镜 + 注意力扫一遍 + 局部 decoder 精修"。配图: 自制图 (figures/sec6-detail-stack.svg) — 三层 stack 示意。
- **Step 2 demo:** ~25 行 PyTorch, 给同一张 1024×1024 图做 "multi-tile encode + average pool over tiles", 对比一次 224×224 encode 的特征 — 直观展示 high-res tile encoding 的细节保留。`teaching-demo`。
- **Step 3 推导:**
  - WinCLIP 的多尺度窗口聚合 (复用 §2 公式但置于"细节谱系"语境)
  - DPAM 的 attention 修改 (复用 §3 公式)
  - AnyRes (LLaVA-1.6 风格): 把图切成 tile, 每 tile 单独 encode, concat 后让 LLM 处理
  - SAM-assisted zoom-in: SAM 给出物体 mask → crop → 再喂给 VLM
- **Step 4 代码:**
  - `zqhang-AnomalyCLIP/AnomalyCLIP_lib/CLIP.py` 的 DPAM 修改段落 (V-V attention 替换)
  - `CASIA-IVA-Lab-AnomalyGPT/code/model/openllama.py` 的 multi-layer feature decoder
- **Step 5 洞察:** (a) 分辨率 ↑ → token 数 ↑ → 计算成本 quadratic ↑, 需要 windowing/AnyRes 这种 sparse design; (b) 为什么 ViT 比 CNN 更适合: token 天然就是 patch, 不需要额外的 patch decomposition; (c) image decoder vs LLM: LLM 不擅长 dense prediction, decoder 是务实补丁。

### §7. 泛化能力:从 Zero-shot 到 In-context (≈2000 字)
**Key takeaway:** 泛化的三个层级 — (a) Zero-shot prompt-only: WinCLIP / AnomalyCLIP; (b) Few-shot reference: WinCLIP+ / AnomalyGPT in-context; (c) Cross-domain: industrial → medical (AnomalyCLIP 17 datasets) → open-vocabulary (IMDD-1M)。Prompt learning > 全参数 fine-tune 是这条线的核心方法学共识。

- **Step 1 直觉:** "学会一个 unknown 物体怎么坏" 三种方式: 看说明书 (zero-shot prompt), 看一张例图 (few-shot), 看大量不相关物体的缺陷然后 transfer (cross-domain)。配图: AnomalyCLIP Table 3 cross-dataset 结果 (从 PDF 抠出来)。
- **Step 2 demo:** ~25 行, in-context few-shot 的最小实现 — 给 LLM 喂 [normal example + abnormal example + query] 三张图, 让它判断 query。`teaching-demo`。
- **Step 3 推导:**
  - Zero-shot: $\arg\max_{c \in \{n, a\}} \cos(f_v(x), f_t(prompt_c))$
  - Few-shot reference: 维护一个 normal feature bank $B = \{f_v(x_i^n)\}$, anomaly score = $\min_{b \in B} \|f_v(x) - b\|$
  - Cross-domain prompt learning: 在 auxiliary domain $D_{aux}$ 学 prompt, 在 target $D_{tgt}$ 直接用 (AnomalyCLIP 范式)
- **Step 4 代码:**
  - `zqhang-AnomalyCLIP/test.py` — cross-dataset evaluation 完整循环
  - `caoyunkang-WinClip/eval_WinCLIP.py` — few-shot reference bank 构建
- **Step 5 洞察:** (a) Prompt learning 为什么比 fine-tune 泛化更好: parameter count 少了三个数量级,不会 overfit auxiliary domain; (b) 跨域 (industrial → medical) work 是个 surprise: abnormality 概念真的可以 object-agnostic; (c) Test-time adaptation 雏形 — AD-Copilot 的 in-context comparison 是这个方向的 2026 代表。

### §8. 演进时间线 + 现代前沿 + 落地建议 (≈1500 字)
**Key takeaway:** 整合 §1-7 形成一张 timeline + decision tree。何时 vanilla CLIP 够用、何时上 WinCLIP、何时需要 LVLM、何时直接用 anomalib 传统方法。

- **Step 1 直觉:** "我有多少数据、多大算力、多紧任务" 决定方法选择。一张 decision tree 图: 1000+ normal images? → anomalib PatchCore (传统); 50-100 normal images? → AnomalyGPT few-shot; 0 normal images? → AnomalyCLIP zero-shot。配图: 自制 timeline (figures/sec8-timeline.svg) — 2023-2026 SOTA 横轴 + 3 主题色块。
- **Step 2 demo:** ~30 行 Python 决策伪代码, 接收 (n_normal_images, n_anomaly_examples, latency_budget, domain) 返回推荐方法 + 推荐 repo URL。`teaching-demo`。
- **Step 3 推导:** 三个主题在 SOTA 上的覆盖矩阵 (Theme × Method 二维表), 直接给出每个 cell 的 best paper + repo + 性能数字。无新数学。
- **Step 4 代码:** `openvinotoolkit/anomalib` 的 PatchCore 主类签名 + `zqhang/AnomalyCLIP` 的 main API + `CASIA-IVA-Lab/AnomalyGPT` 的 web_demo.py — 三个 API 对比, 让读者一眼看出落地复杂度。
- **Step 5 洞察:** (a) 现代前沿: IADGPT (in-context reasoning), IMDD-1M (1M 多模态数据集 pretrain), AD-Copilot (Chat-AD with comparison encoder); (b) 真正的开放问题: 多缺陷同图、动态缺陷 (video AD)、3D AD; (c) 给工程团队的具体建议 — 何时不要用 VLM, 何时要等下一代 model。

---

## Cross-section dependencies
- §2 引入 "patch-level alignment" 和 "prompt ensemble", §3 在此基础上加 "learnable"。
- §3 引入 "object-agnostic" 和 "DPAM", §6 和 §7 都会复用。
- §4 引入 "synthetic data" 和 "decoder", §5 (数据) 和 §6 (细节) 都会复用。
- §5, §6, §7 是 "三主题" 横切视角, 引用 §2-4 的具体方法作为案例; 不重新推导数学。
- §8 综合所有, 提供 decision tree, 是终点。

## Drafting groups (for Phase 5 grouped-parallel)
- **Group A (parallel):** §1 (问题), §2 (WinCLIP) — 基础设定, 不依赖
- **Group B (parallel):** §3 (AnomalyCLIP), §4 (AnomalyGPT) — 依赖 A 的术语 (patch, prompt, CLIP failure)
- **Group C (parallel):** §5 (数据), §6 (细节), §7 (泛化) — 依赖 A+B 的方法引用
- **Group D (serial):** §8 (综合) — 依赖 A+B+C

## 关键引用与图

### 已抓 PDF (3 anchor):
- `sources/papers/winclip-2023.pdf` (CVPR'23, 25 pages)
- `sources/papers/anomalyclip-2024.pdf` (ICLR'24, 30 pages)
- `sources/papers/anomalygpt-2024.pdf` (AAAI'24, 12 pages)

### 已 clone 仓库 (3 primary):
- `sources/repos/zqhang-AnomalyCLIP/` (官方, ICLR'24)
- `sources/repos/CASIA-IVA-Lab-AnomalyGPT/` (官方, AAAI'24)
- `sources/repos/caoyunkang-WinClip/` (community, 标 provenance-warning)

### Phase 4 待抓图:
- WinCLIP Fig.1 (zero-shot demo) → sec1
- AnomalyCLIP Fig.1 (4-prompt comparison) → sec1, sec3
- WinCLIP Fig.2 (multi-scale window) → sec2
- WinCLIP Fig.3 (architecture) → sec2
- AnomalyCLIP Fig.2 (DPAM mechanism) → sec3
- AnomalyGPT Fig.1 (3-way comparison) → sec4
- AnomalyGPT Fig.2 (architecture) → sec4
- AnomalyGPT Fig.3 (synthetic data pipeline) → sec4, sec5
- AnomalyCLIP Table 3 (cross-dataset) → sec7

### Phase 4 自制图:
- figures/sec5-data-evolution.svg — 数据制作演进时间轴
- figures/sec6-detail-stack.svg — 细粒度三层 stack
- figures/sec8-timeline.svg — 2023-2026 SOTA + decision tree

## Provenance warnings (会出现在页面顶部)
1. WinCLIP 官方未发布代码; §2 代码引自 community caoyunkang/WinClip
2. IADGPT / IMDD-1M / AD-Copilot 等 2025+ 论文仅作为文字引用, 不引代码 (repo 未验证)
3. 该领域缺乏 Lilian Weng-style 深度博客; 本教程是 first-of-its-kind
