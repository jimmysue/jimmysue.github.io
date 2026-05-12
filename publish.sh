#!/usr/bin/env bash
# Publish the paper-reading blog to GitHub Pages (jimmysue.github.io).
#
# Usage:
#   ./publish.sh                       # auto-detect changed papers, commit + push
#   ./publish.sh "custom message"      # use a custom commit message
#   ./publish.sh --force-init          # force-push (destructive; only for first publish or full reset)
#
# Excluded from the published repo (see .gitignore):
#   - papers/*/raw/, figures-raw/, repo/   (heavy intermediate artifacts)
#   - .claude/                              (Claude Code metadata + skill source)
#   - dev-only blog-*.png screenshots
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .git ]]; then
  echo "ERR: not a git repo. Run: git init -b master && git remote add origin https://github.com/jimmysue/jimmysue.github.io.git" >&2
  exit 1
fi

# Branch detection (github.io repo uses master historically; some use main)
BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo master)"

# Stage tracked changes + new files honoring .gitignore
git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit."
  exit 0
fi

# Compose commit message
if [[ "${1:-}" == "--force-init" ]]; then
  MSG="Initial publish: paper-reading blog (replacing old site)"
  FORCE=1
elif [[ -n "${1:-}" ]]; then
  MSG="$1"
  FORCE=0
else
  # Auto: list paper slugs whose index.html changed
  CHANGED_PAPERS=$(git diff --cached --name-only | grep -oE 'papers/[^/]+/index\.html' | sort -u | sed 's|papers/||; s|/index.html||')
  if [[ -n "$CHANGED_PAPERS" ]]; then
    MSG="Update: $(echo "$CHANGED_PAPERS" | paste -sd ', ' -)"
  else
    MSG="Update blog ($(date +%Y-%m-%d))"
  fi
  FORCE=0
fi

echo "Commit message: $MSG"
git commit -m "$MSG"

echo "Pushing to origin/$BRANCH ..."
if [[ "${FORCE:-0}" == "1" ]]; then
  git push -u --force origin "$BRANCH"
else
  git push -u origin "$BRANCH"
fi

echo ""
echo "Published. View at: https://jimmysue.github.io/"
