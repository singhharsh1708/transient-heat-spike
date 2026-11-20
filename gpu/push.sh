#!/bin/bash
# Push the T4 sweep to Kaggle as a private script kernel, pinned to the current commit (which must be on GitHub).
#   gpu/push.sh            then: kaggle kernels status harshsingh1708/transient-heat-t4
#                                kaggle kernels output harshsingh1708/transient-heat-t4 -p <dir>
set -euo pipefail
cd "$(dirname "$0")/.."
sha=$(git rev-parse HEAD)
git branch -r --contains "$sha" | grep -q origin/main || { echo "commit $sha is not on origin/main" >&2; exit 1; }
tmp=$(mktemp -d)
cp gpu/kernel-metadata.json "$tmp/"
sed "s/^COMMIT = \"main\"$/COMMIT = \"$sha\"/" gpu/kaggle_t4.py > "$tmp/kaggle_t4.py"
grep -q "^COMMIT = \"$sha\"$" "$tmp/kaggle_t4.py"
kaggle kernels push -p "$tmp" --accelerator NvidiaTeslaT4
rm -rf "$tmp"
