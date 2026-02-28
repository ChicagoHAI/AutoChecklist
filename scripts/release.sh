#!/bin/bash
set -euo pipefail

VERSION="${1:?Usage: $0 VERSION (e.g., 0.2.0)}"
TAG="v${VERSION}"
PUBLIC_URL="https://github.com/ChicagoHAI/AutoChecklist.git"
PRIVATE_REMOTE="private"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

echo "=== Releasing ${TAG} ==="

# 1. Validate state
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "main" ]; then
    echo "ERROR: Must be on main branch (currently on ${BRANCH})"; exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: Working tree not clean. Commit or stash first."; exit 1
fi
git pull "$PRIVATE_REMOTE" main

# 2. Run tests
echo "--- Running tests ---"
uv run pytest -v -m "not integration and not vllm_offline and not openai_api"

# 3. Build package
echo "--- Building package ---"
uv build

# 4. Clone public repo
echo "--- Preparing public release ---"
git clone "$PUBLIC_URL" "$TMPDIR/public-repo"

# 5. Replace contents with allowlisted paths
cd "$TMPDIR/public-repo"
find . -maxdepth 1 -not -name '.git' -not -name '.' -print0 | xargs -0 rm -rf
cd -
grep -vE '^\s*(#|$)' .public-include > "$TMPDIR/public-files"
rsync -a -r --files-from="$TMPDIR/public-files" . "$TMPDIR/public-repo/"

# 6. Validate (defense-in-depth)
echo "--- Validating ---"
cd "$TMPDIR/public-repo"
git add -A
FAILED=0
for pattern in .env CLAUDE.md AGENTS.md PLAN.md agent_logs .claude .public-include openai_api_costs.tsv results.jsonl uv.lock .python-version; do
    if git ls-files "$pattern" | grep -q .; then
        echo "ERROR: sensitive path '$pattern' found"; FAILED=1
    fi
done
[ "$FAILED" -eq 0 ] || { echo "ABORTING"; exit 1; }
echo "Validation passed"

# 7. Commit, tag, push
git config user.name "$(git -C "$OLDPWD" config user.name)"
git config user.email "$(git -C "$OLDPWD" config user.email)"
git diff --staged --quiet && { echo "No changes — public repo already up to date"; exit 0; }
git commit -m "Release ${TAG}"
git branch -M main
if git ls-remote --tags origin "$TAG" | grep -q .; then
    echo "Tag $TAG already exists — skipping"
else
    git tag "$TAG"
fi
git push origin main
git push origin "$TAG" 2>/dev/null || echo "Tag already pushed"
cd -

# 8. Create GitHub Release (triggers PyPI publish)
if gh release view "$TAG" --repo ChicagoHAI/AutoChecklist &>/dev/null; then
    echo "Release $TAG already exists — skipping"
else
    gh release create "$TAG" --repo ChicagoHAI/AutoChecklist --title "$TAG" --generate-notes
fi

echo "=== Released ${TAG} successfully ==="
echo "Return to dev: git checkout dev"
