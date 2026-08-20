#!/usr/bin/env bash
set -euo pipefail

download() {
  local url="$1"
  local destination="$2"
  local partial="${destination}.part"
  mkdir -p "$(dirname "${destination}")"
  if [[ -s "${destination}" ]]; then
    echo "exists, keeping: ${destination}"
    return
  fi
  curl --fail --location --retry 5 --retry-delay 5 --continue-at - \
    --output "${partial}" "${url}"
  mv "${partial}" "${destination}"
}

download \
  "https://download.europe.naverlabs.com/dune/dunemast3r_cvpr25_vitsmall.pth" \
  "checkpoints/mast3r/dunemast3r_cvpr25_vitsmall.pth"
download \
  "https://download.europe.naverlabs.com/dune/dune_vitsmall14_448.pth" \
  "checkpoints/dune/dune_vitsmall14_448.pth"

python scripts/verify_vggtomast3r_environment.py --imports-only
