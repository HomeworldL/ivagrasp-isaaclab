#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ISAACLAB_ROOT="/home/ccs/github/IsaacLab"

if [[ -n "${ISAACLAB_ROOT:-}" ]]; then
    upstream_root="${ISAACLAB_ROOT}"
else
    upstream_root="${DEFAULT_ISAACLAB_ROOT}"
fi

upstream_script="${upstream_root}/isaaclab.sh"

if [[ ! -x "${upstream_script}" ]]; then
    echo "Cannot find executable IsaacLab launcher at: ${upstream_script}" >&2
    echo "Set ISAACLAB_ROOT to your IsaacLab checkout root." >&2
    exit 1
fi

exec "${upstream_script}" "$@"
