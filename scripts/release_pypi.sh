#!/usr/bin/env bash
#
# Local release script for `codehelper`.
#
# First-party package (not a fork), so the version is NOT derived from a git
# tag: it lives in src/codehelper/__init__.py (`dynamic = ["version"]` +
# hatchling), and this script reads it straight from there. Same token-based
# release_pypi.sh pattern as the other axisrow projects (direct-cli,
# hermes-agent-axisrow, tg_content_factory, tg_messenger, ...).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
INIT_FILE="${ROOT_DIR}/src/codehelper/__init__.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/release_pypi.sh testpypi
  scripts/release_pypi.sh pypi
  scripts/release_pypi.sh all

Behavior:
  - loads .env from the repository root when present
  - derives the package version from src/codehelper/__init__.py
    (`__version__`), the single source of truth for the hatch `dynamic` field
  - rebuilds dist artifacts from scratch
  - runs twine checks before upload
  - uploads to TestPyPI, PyPI, or both

Required .env variables:
  TWINE_USERNAME=__token__
  TEST_PYPI_TOKEN=pypi-...   # for testpypi/all
  PYPI_TOKEN=pypi-...        # for pypi/all
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

TARGET="$1"

case "${TARGET}" in
  testpypi|pypi|all)
    ;;
  *)
    usage
    exit 1
    ;;
esac

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

TWINE_USERNAME="${TWINE_USERNAME:-__token__}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 1
  fi
}

require_command() {
  local name="$1"
  if ! command -v "${name}" >/dev/null 2>&1; then
    echo "Required command not found: ${name}" >&2
    exit 1
  fi
}

resolve_version() {
  local version
  version="$(grep -E '^__version__ = ' "${INIT_FILE}" | sed -E 's/^__version__ = "(.*)"/\1/')"
  if [[ -z "${version}" ]]; then
    echo "Could not read __version__ from ${INIT_FILE}" >&2
    exit 1
  fi
  if [[ ! "${version}" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
    echo "Invalid PEP 440 version read from ${INIT_FILE}: '${version}'" >&2
    exit 1
  fi
  echo "${version}"
}

build_artifacts() {
  require_command python3
  require_command uv

  local version
  version="$(resolve_version)"
  echo "package version -> ${version} (from src/codehelper/__init__.py)"

  echo "Cleaning old build artifacts"
  rm -rf "${ROOT_DIR}/dist" "${ROOT_DIR}/build" "${ROOT_DIR}"/*.egg-info

  echo "Building package"
  (
    cd "${ROOT_DIR}"
    # uv build, not `python -m build --no-isolation`: build-system.requires
    # pins hatchling>=1.21, and uv resolves an isolated build env matching
    # that pin, same as the CI workflow.
    uv build --sdist --wheel
  )

  echo "Checking artifacts with twine"
  (
    cd "${ROOT_DIR}"
    python3 -m twine check dist/*
  )
}

upload_target() {
  local repository="$1"
  local password_var="$2"

  require_var "${password_var}"

  echo "Uploading to ${repository}"
  (
    cd "${ROOT_DIR}"
    TWINE_USERNAME="${TWINE_USERNAME}" \
    TWINE_PASSWORD="${!password_var}" \
    python3 -m twine upload --non-interactive --skip-existing --repository "${repository}" dist/*
  )
}

build_artifacts

case "${TARGET}" in
  testpypi)
    upload_target "testpypi" "TEST_PYPI_TOKEN"
    ;;
  pypi)
    upload_target "pypi" "PYPI_TOKEN"
    ;;
  all)
    upload_target "testpypi" "TEST_PYPI_TOKEN"
    upload_target "pypi" "PYPI_TOKEN"
    ;;
esac
