#!/bin/sh
# Teacher corpus for the six new British voices, two processes at a time (six at once filled 70 GB of swap).
# Run from upstream/training. Resumable: finished shards are skipped.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2   # fail with an error instead of swapping if MPS memory grows anyway
N=${1:-1500}
run() { for v in "$@"; do uv run python ../../british/gen_teacher_voice.py "$v" "$N" 4 || exit 1; done; }
run bm_daniel bf_alice bf_emma > logs/gen_a.log 2>&1 &
run bf_lily bm_george bm_lewis > logs/gen_b.log 2>&1 &
wait
