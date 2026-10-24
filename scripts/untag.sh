#!/usr/bin/env bash

set -euo pipefail

delete_remote=false
if [[ "${1:-}" == "--remote" ]]; then
  delete_remote=true
  shift
fi

if [[ "$#" -ne 1 ]]; then
  echo "Usage: bash scripts/untag.sh [--remote] <version>" >&2
  exit 1
fi

version="$1"
tag_ref="refs/tags/$version"
if [[ "$version" == -* ]] || ! git check-ref-format "$tag_ref"; then
  echo "Invalid tag name: $version" >&2
  exit 1
fi

git rev-parse --git-dir >/dev/null

if git show-ref --verify --quiet "$tag_ref"; then
  git tag --delete -- "$version"
else
  echo "Local tag $version does not exist."
fi

if [[ "$delete_remote" == true ]]; then
  if git ls-remote --exit-code --refs --tags origin "$tag_ref" >/dev/null; then
    git push origin --delete "$tag_ref"
  else
    status=$?
    if [[ "$status" -eq 2 ]]; then
      echo "Remote tag $version does not exist."
    else
      echo "Could not check tag $version on origin." >&2
      exit "$status"
    fi
  fi
else
  echo "Remote tags are unchanged. Use --remote to delete the tag from origin."
fi
