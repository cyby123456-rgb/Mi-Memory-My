#!/bin/zsh
set -u
cd /Users/cyby/Documents/Codex/2026-07-29/https-arxiv-org-pdf-2607-18975 || exit 1
exec >> tmp/longmemeval/run/full.log 2>&1
set -a
source .env
set +a
export MIMEMORY_LIVE_PROVIDER_APPROVED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src
exec /Users/cyby/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  scripts/run_longmemeval_s.py \
  --data tmp/longmemeval/data/longmemeval_s_cleaned.json \
  --root tmp/longmemeval/run/full
