#!/bin/sh
# After gen_all.sh: reference held-out sets for the released British voices, then the full training run.
# Run from upstream/training.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.4 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3
while pgrep -f gen_all.sh > /dev/null; do sleep 30; done
uv run python ../../british/gen_teacher_voice.py bf_isabella 200 4 > logs/gen_ref_f.log 2>&1 &
uv run python ../../british/gen_teacher_voice.py bm_fable 200 4 > logs/gen_ref_m.log 2>&1 &
wait
uv run python ../../british/train_new_voices.py --text-steps 600 --dec-steps 3000 --out checkpoints/british > logs/train.log 2>&1
echo "pipeline exit $?"
