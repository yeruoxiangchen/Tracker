#!/usr/bin/env bash
# Recreate the exact upstream code used by the archived driving experiments.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="${MPP_UPSTREAM_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/third_party/MPP-LLaVA}"
REVISION=baf617b0f5217442ba20721be9aed710cdf999e2
PATCH="$PROJECT_ROOT/patches/mpp_llava_baf617b_local.patch"
ASSET_ROOT="${MPP_PERSISTENT_CHECKPOINT_ROOT:-$PROJECT_ROOT/../mpp_llava_video_workspace/checkpoints}"
if [[ ! -d "$UPSTREAM/.git" ]]; then
  if [[ -e "$UPSTREAM" ]]; then
    echo "Existing upstream directory is not a checkout: $UPSTREAM" >&2
    exit 2
  fi
  git clone https://github.com/Coobiw/MPP-LLaVA.git "$UPSTREAM"
  git -C "$UPSTREAM" checkout --detach "$REVISION"
fi
mkdir -p "$ASSET_ROOT/ckpt"
ASSET_ROOT="$(cd "$ASSET_ROOT" && pwd)"
for name in cache ckpt; do
  target="$ASSET_ROOT"
  if [[ "$name" == ckpt ]]; then target="$ASSET_ROOT/ckpt"; fi
  if [[ -L "$UPSTREAM/$name" ]]; then
    if [[ "$(readlink -f "$UPSTREAM/$name")" != "$target" ]]; then
      echo "Existing $name symlink points elsewhere; set MPP_PERSISTENT_CHECKPOINT_ROOT accordingly." >&2
      exit 2
    fi
  elif [[ -e "$UPSTREAM/$name" ]]; then
    echo "Existing $name directory needs its assets configured manually: $UPSTREAM/$name" >&2
    exit 2
  else
    ln -s "$target" "$UPSTREAM/$name"
  fi
done
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
