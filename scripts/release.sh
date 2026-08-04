#!/usr/bin/env bash

set -e

if [[ "$#" -ne 1 ]]; then
  echo "Usage: git release <version>" >&2
  exit 1
fi

version="$1"
package_version="${version#v}"
version_file="app/bellatrex/__version__.py"

git switch main
git pull --ff-only

git add "$version_file"
git commit -m "chore: bump version to ${package_version}"
git push origin main

git tag "$version"
git push origin "$version"
