#!/usr/bin/env bash
# create_release_branch.sh
#
# Creates a release branch from the current branch with a UTC timestamp in the
# branch name. The following directories are removed from the release branch
# history because they contain private or internal-only content:
#
#   - popsim/modules/private
#   - docs/notebooks/doc_ignore

set -euo pipefail

TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
RELEASE_NAME="release/${TIMESTAMP}"
ORIGIN_URL=$(git remote get-url origin)
REPO_ROOT=$(git rev-parse --show-toplevel)
TMPDIR=$(mktemp -d)

trap 'rm -rf "$TMPDIR"' EXIT

echo "Creating release branch: $RELEASE_NAME"

# Local clone is fast (hard-links objects), and gives us an independent .git dir
# so filter-repo can't corrupt the original repo
git clone --branch main --single-branch "$REPO_ROOT" "$TMPDIR/repo"

cd "$TMPDIR/repo"

git config advice.objectNameWarning false

# Rewrite history to remove private/internal directories entirely
uvx git-filter-repo \
  --invert-paths \
  --path popsim/modules/private \
  --path popsim/tests/test_modules/test_private \
  --path docs/notebooks/doc_ignore \
  --quiet \
  --force

# Point remote at the real origin (local clone sets it to the local path)
git remote add origin "$ORIGIN_URL"

git checkout -b "$RELEASE_NAME"
git push origin "$RELEASE_NAME"

echo "Done. Branch '$RELEASE_NAME' pushed to origin with private directories purged from history."
