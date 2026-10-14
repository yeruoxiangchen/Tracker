#!/usr/bin/env bash
# Recreate the exact upstream code used by the archived driving experiments.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
REVISION=baf617b0f5217442ba20721be9aed710cdf999e2
PATCH="$PROJECT_ROOT/patches/mpp_llava_baf617b_local.patch"
if [[ ! -d "$UPSTREAM/.git" ]]; then
  if [[ -e "$UPSTREAM" ]]; then
    echo "Existing upstream directory is not a checkout: $UPSTREAM" >&2
    exit 2
  fi
  git clone https://github.com/Coobiw/MPP-LLaVA.git "$UPSTREAM"
  git -C "$UPSTREAM" checkout --detach "$REVISION"
fi
if [[ "$(git -C "$UPSTREAM" rev-parse HEAD)" != "$REVISION" ]]; then
  echo "Expected upstream $REVISION; choose a separate MPP_UPSTREAM_ROOT." >&2
  exit 2
fi
if git -C "$UPSTREAM" apply --reverse --check "$PATCH" 2>/dev/null; then
  printf 'Archived MPP patch is already applied: %s\n' "$UPSTREAM"
else
  git -C "$UPSTREAM" apply --check "$PATCH"
  git -C "$UPSTREAM" apply "$PATCH"
  printf 'Applied archived MPP patch: %s\n' "$UPSTREAM"
fi
