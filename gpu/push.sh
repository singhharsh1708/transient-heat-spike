#!/bin/bash
# Push the T4 sweep to Kaggle as a private script kernel, pinned to the current commit (which must be on origin/main).
#   gpu/push.sh            then: kaggle kernels status harshsingh1708/transient-heat-t4
#                                kaggle kernels output harshsingh1708/transient-heat-t4 -p <dir>
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
git fetch -q origin main
git merge-base --is-ancestor "$sha" origin/main || { echo "commit $sha is not on origin/main" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git show "$sha:gpu/kernel-metadata.json" > "$tmp/kernel-metadata.json"
git show "$sha:gpu/kaggle_t4.py" | sed "s/^COMMIT = \"main\"$/COMMIT = \"$sha\"/" > "$tmp/kaggle_t4.py"
grep -q "^COMMIT = \"$sha\"$" "$tmp/kaggle_t4.py"
kaggle kernels push -p "$tmp" --accelerator NvidiaTeslaT4
