---
slug: flow-matching
name: "Flow Matching"
aliases: ["flow matching", "FM", "rectified flow"]
---

# Flow Matching

Flow matching 是把"从噪声生成数据"建模成一个**连续时间 ODE** 的方法。

## 直觉

- DDPM 学的是 score $\nabla_x \log p_t(x)$ 的近似,反向 SDE 走轨迹。
- Flow matching 学的是**速度场** $v(x, t)$,反向 ODE 走轨迹。直观上"更短更直"。
- 训练目标:让模型预测的速度匹配 $\epsilon - x_0$ 这条线性插值的方向。

## 跟 DDPM 的关键差别

| 维度 | DDPM | Flow Matching |
|---|---|---|
| 模型预测什么 | noise / x0 / v | velocity $u(x, t)$ |
| 采样器 | SDE / DDIM | ODE (Euler / Heun) |
| 训练 loss | MSE 在 noise 空间 | MSE 在 velocity 空间 |
| 与 OT 关系 | 间接 | 直接 (rectified flow / OT-CFM) |

## 工业级实现

进一步看 [SD3](https://stability.ai/news/stable-diffusion-3-research-paper) 和 [Flux](https://github.com/black-forest-labs/flux),这两个都是 flow matching 的工业级实现。
