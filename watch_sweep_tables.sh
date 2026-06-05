#!/bin/bash
set -euo pipefail

cd /storage/home/hcoda1/9/qdai41/scratch/cosmos/LingBot-VA-Modification
python scripts/lqr/write_live_sweep_tables.py --repo-root . --watch --interval-sec "${INTERVAL_SEC:-60}"
