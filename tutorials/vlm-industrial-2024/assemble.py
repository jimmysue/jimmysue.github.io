#!/usr/bin/env python3
"""Assemble drafts/sec1..8.html + skeleton into index.html."""
from pathlib import Path

ROOT = Path(__file__).parent
DRAFTS = ROOT / "drafts"

SECTION_TITLES = {
    1: "1. 问题与挑战 — 工业检测为什么需要 VLM",
    2: "2. WinCLIP — 让 CLIP 看清局部细节",
    3: "3. AnomalyCLIP — Object-agnostic Prompt + DPAM",
    4: "4. AnomalyGPT — LVLM 进场 + 合成数据",
    5: "5. 数据制作的演进 — 从 CutPaste 到 IMDD-1M",
    6: "6. 让 VLM 看清细节的技术谱系",
    7: "7. 泛化能力 — 从 Zero-shot 到 In-context",
    8: "8. 演进时间线 + 现代前沿 + 落地建议",
}

SUBSTEP_TITLES = {
    1: "直觉",
    2: "最小 demo",
    3: "正式化",
    4: "代码引用",
    5: "洞察",
}


def build_toc():
    items = []
    for n, title in SECTION_TITLES.items():
        items.append(f'    <li class="h2"><a href="#sec-{n}">{title}</a></li>')
        for k, t in SUBSTEP_TITLES.items():
            items.append(f'    <li class="h3"><a href="#sec-{n}-{k}">{n}.{k} {t}</a></li>')
        if n == 8:
            items.append('    <li class="h3"><a href="#sec-8-references">参考资料</a></li>')
    return "\n".join(items)


def load_sections():
    parts = []
    for n in range(1, 9):
        f = DRAFTS / f"sec{n}.html"
        parts.append(f.read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


HEAD = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>VLM 在工业检测领域: 看清细节 / 数据制作 / 泛化能力 — 教程</title>
  <link rel="stylesheet" href="../../assets/style.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/atom-one-dark.min.css">
  <script>
    window.MathJax = { tex: { inlineMath: [['$','$'],['\\\\(','\\\\)']] } };
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js"></script>
  <script>document.addEventListener('DOMContentLoaded', () => hljs.highlightAll());</script>
  <script defer src="../../assets/lightbox.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', () => {
      const links = document.querySelectorAll('.toc a');
      if (!links.length) return;
      const map = new Map();
      links.forEach(a => {
        const el = document.getElementById(a.getAttribute('href').slice(1));
        if (el) map.set(el, a);
      });
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(e => {
          const a = map.get(e.target);
          if (!a) return;
          if (e.isIntersecting) {
            links.forEach(l => l.classList.remove('active'));
            a.classList.add('active');
            const toc = document.querySelector('.toc');
            if (toc) toc.scrollTo({ top: a.offsetTop - toc.clientHeight / 2, behavior: 'smooth' });
          }
        });
      }, { rootMargin: '-20% 0px -70% 0px' });
      map.forEach((_, el) => observer.observe(el));
    });
  </script>
</head>
<body>
<!-- NAV-START -->
<nav class="site-nav" aria-label="主导航">
  <a class="site-nav__brand" href="../../index.html" aria-label="返回主页">
    <svg class="site-nav__home-icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M3 12l9-9 9 9"/>
      <path d="M5 10v10h14V10"/>
    </svg>
    <span class="site-nav__brand-text">Paper Reading</span>
  </a>
  <div class="site-nav__links">
    <a class="site-nav__link" href="../../papers.html">论文</a>
    <a class="site-nav__link" href="../../tutorials.html">教程</a>
    <a class="site-nav__link" href="../../tags.html">标签</a>
  </div>
</nav>
<!-- NAV-END -->
<nav class="toc" aria-label="目录">
  <div class="toc-title">目录 / TOC</div>
  <ul>
__TOC__
  </ul>
</nav>
<main>
<h1>VLM 在工业检测领域:<br>看清细节 · 数据制作 · 泛化能力 三大主题深挖</h1>

  <div class="meta">
    <div><strong>主题:</strong> Vision-Language Models 在工业缺陷检测 (Industrial Anomaly Detection, IAD) 中的方法演进 (2023 → 2026)</div>
    <div><strong>覆盖方法:</strong> 传统 PatchCore baseline → WinCLIP → AnomalyCLIP → AnomalyGPT → IADGPT / IMDD-1M / AD-Copilot (现代前沿)</div>
    <div><strong>三大主题:</strong>
      <ul style="margin:0.3em 0 0 1.2em;">
        <li><strong>看清细节</strong> (§2 多尺度窗口, §3 DPAM 注意力, §4 多层 decoder, §6 系统化栈)</li>
        <li><strong>数据制作</strong> (§4 cut-paste 合成 + 文本配对, §5 演进时间线 CutPaste→Perlin→Poisson→IMDD-1M)</li>
        <li><strong>泛化能力</strong> (§3 object-agnostic prompt, §7 zero-shot / few-shot / cross-domain 三级)</li>
      </ul>
    </div>
    <div><strong>引用代码 (≥4 个 repo, 全部 file:Lstart-Lend verbatim):</strong>
      <ul style="margin:0.3em 0 0 1.2em;">
        <li><a href="https://github.com/zqhang/AnomalyCLIP">zqhang/AnomalyCLIP</a> (ICLR'24 官方) — §3, §6, §7</li>
        <li><a href="https://github.com/CASIA-IVA-Lab/AnomalyGPT">CASIA-IVA-Lab/AnomalyGPT</a> (AAAI'24 Oral 官方, ~1.1k★) — §4, §5, §6</li>
        <li><a href="https://github.com/caoyunkang/WinClip">caoyunkang/WinClip</a> (社区 PyTorch 复现, ⚠ WinCLIP 官方无代码) — §1, §2, §7</li>
        <li><a href="https://github.com/openvinotoolkit/anomalib">openvinotoolkit/anomalib</a> (5.8k★, 生产级 baseline) — §8</li>
      </ul>
    </div>
    <div><strong>TL;DR:</strong> 把 VLM-based 工业检测这条线拆透:从 vanilla CLIP 的 global pooling 失败,到 WinCLIP 的多尺度窗口对齐,再到 AnomalyCLIP 的 object-agnostic 可学习 prompt + DPAM 注意力修正,最后到 AnomalyGPT 用 LVLM + 合成数据 + 轻量 decoder 实现 threshold-free + 多轮对话。三大主题 (看清细节 / 数据制作 / 泛化) 各有一章系统化收拢,最后用 timeline + decision tree 给出落地建议。</div>
    <div><strong>读者画像:</strong> 学过线代和 Python、做过 CV 或 anomaly detection、想搞清 VLM-based IAD 方法演进与代码落地的研究者。每节走"直觉 → 最小 demo → 正式化 → 代码引用 → 洞察" 5 步螺旋。</div>
  </div>

  <aside class="provenance-warning" style="background:#fef3c7;border-left:4px solid #d97706;padding:1em 1.2em;margin:1.5em 0;border-radius:6px;font-size:0.95em;line-height:1.55;">
    ⚠️ <strong>Provenance 说明:</strong>
    <ul style="margin:0.4em 0 0 1.2em;">
      <li><strong>WinCLIP (§2)</strong>:Jeong 等原作者 (Amazon AWS) <em>未发布官方代码</em>,本节引用自社区高质量 PyTorch 复现 <a href="https://github.com/caoyunkang/WinClip">caoyunkang/WinClip</a>,实现细节由复现者决定,可能与论文有微差。</li>
      <li><strong>IADGPT / IMDD-1M / AD-Copilot 等 2025-2026 工作</strong>:仅作为 "演进时间线" 文字引用,本教程未克隆其 repo 验证,arxiv 上有可用 PDF。</li>
      <li><strong>领域博客稀缺</strong>:VLM × 工业检测目前几乎没有 Lilian Weng-style 系统教程,本教程是 first-of-its-kind 长文。如有事实错误请提 issue。</li>
    </ul>
  </aside>

__SECTIONS__

  <footer>
    <hr>
    <p class="meta">Written on 2026-05-20 · Generated with the writing-tutorial skill</p>
  </footer>
</main>
</body>
</html>
"""


def main():
    toc = build_toc()
    sections = load_sections()
    html = HEAD.replace("__TOC__", toc).replace("__SECTIONS__", sections)
    (ROOT / "index.html").write_text(html, encoding="utf-8")
    # Sanity counts
    toc_count = html.count('<li class="h2">') + html.count('<li class="h3">')
    h2_count = html.count('<h2 id="sec-')
    h3_count = html.count('<h3 id="sec-')
    print(f"index.html written. TOC items: {toc_count}; body h2: {h2_count}; body h3: {h3_count}")


if __name__ == "__main__":
    main()
