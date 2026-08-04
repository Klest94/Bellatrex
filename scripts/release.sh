#!/usr/bin/env bash

set -e

if [[ "$#" -ne 1 ]]; then
  echo "Usage: git release <version>" >&2
  exit 1
fi

version="$1"

git switch main
git pull --ff-only
git tag "$version"
git push origin "$version"
