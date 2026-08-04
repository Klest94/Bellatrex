#!/usr/bin/env bash

set -e

if [[ "$#" -ne 1 ]]; then
  echo "Usage: git untag <version>" >&2
  exit 1
fi

version="$1"

if git rev-parse -q --verify "refs/tags/$version" >/dev/null; then
  git tag -d "$version"
else
  echo "Local tag $version does not exist."
fi

remote_tag="$(git ls-remote --tags origin "refs/tags/$version")"
if [[ -n "$remote_tag" ]]; then
  git push origin --delete "$version"
else
  echo "Remote tag $version does not exist."
fi
