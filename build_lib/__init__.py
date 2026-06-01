"""Building blocks for the markdown-source blog renderer.

This package is imported by build.py. Each module has a single
responsibility:

  frontmatter   — parse + validate YAML frontmatter
  wiki_links    — [[slug]] preprocess + resolve
  headings      — sec-N IDs + TOC generation
  figures       — lightbox class tagging
  markdown      — orchestrate the full markdown → HTML body pipeline
  post_assembly — wrap body in full HTML page (head/nav/toc/scripts)
"""
