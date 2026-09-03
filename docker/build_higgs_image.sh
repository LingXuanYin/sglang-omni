#!/usr/bin/env bash
# Build (and optionally push) the Higgs TTS inference image for this fork.
#
# The image bakes in the current checkout, so this refuses to build from a
# dirty or unknown tree unless you say otherwise: an image whose recorded
# commit does not match its contents is worse than no provenance at all.
#
# Usage:
#   docker/build_higgs_image.sh                        # build, tag by commit
#   IMAGE=registry.example.com/team/higgs-tts \
#   PUSH=1 docker/build_higgs_image.sh                 # build and push
#   ALLOW_DIRTY=1 docker/build_higgs_image.sh          # build a dirty tree
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-higgs-tts}"
PUSH="${PUSH:-0}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"

commit="$(git rev-parse --short HEAD)"
branch="$(git rev-parse --abbrev-ref HEAD)"
build_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [ -n "$(git status --porcelain)" ]; then
    if [ "${ALLOW_DIRTY}" != "1" ]; then
        echo "Working tree is dirty; the image would not match commit ${commit}." >&2
        echo "Commit first, or re-run with ALLOW_DIRTY=1." >&2
        exit 1
    fi
    commit="${commit}-dirty"
fi

tag_commit="${IMAGE}:${commit}"
# Branch tags carry '/' (feat/voice-timbre-fusion); Docker tags cannot.
tag_branch="${IMAGE}:$(echo "${branch}" | tr '/' '-')"

echo "Building ${tag_commit}"
echo "  branch : ${branch}"
echo "  context: $(pwd)"

docker build \
    -f docker/Dockerfile.higgs \
    --build-arg "GIT_COMMIT=${commit}" \
    --build-arg "GIT_BRANCH=${branch}" \
    --build-arg "BUILD_TIME=${build_time}" \
    -t "${tag_commit}" \
    -t "${tag_branch}" \
    .

echo "Built: ${tag_commit}  ${tag_branch}"

# Fail loudly here rather than at deploy time: an image that cannot import the
# package, or that silently carries a different commit, is not shippable.
echo "Verifying image..."
docker run --rm --entrypoint /bin/bash "${tag_commit}" -lc '
    set -e
    python3 -c "import sglang_omni, sglang, pyworld; print(\"sglang_omni ok\")"
    python3 -c "
import sglang_omni.models.higgs_tts.stages as s
import sglang_omni.models.higgs_tts.fusion_reference as f
print(\"higgs stages + fusion import ok; long-ref mode:\", s._long_reference_mode())
"
    test "${SGLANG_OMNI_GIT_COMMIT}" = "'"${commit}"'" \
        || { echo "commit label mismatch"; exit 1; }
    echo "verification passed"
'

if [ "${PUSH}" = "1" ]; then
    echo "Pushing ${tag_commit} and ${tag_branch}"
    docker push "${tag_commit}"
    docker push "${tag_branch}"
fi
