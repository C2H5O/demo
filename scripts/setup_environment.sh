#!/usr/bin/env bash
set -euo pipefail

git submodule sync --recursive
git submodule update --init --recursive
conda env create --file environment.yml
conda run --no-capture-output -n vggtodistill3r python scripts/verify_environment.py
