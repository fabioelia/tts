# Paradee + all eight British voices

[Paradee-8M v1.3](https://github.com/sahilmahendrakar/paradee) speaks two of Kokoro's British voices
(`bf_isabella`, `bm_fable`). This adds the other six (`bf_alice`, `bf_emma`, `bf_lily`, `bm_daniel`,
`bm_george`, `bm_lewis`) to the same model: one 8.5 MB int8 ONNX file with 14 voices. The original
eight voices are unchanged (same weights).

## Try it

```
pip install "paradee @ git+https://github.com/sahilmahendrakar/paradee" soundfile
cd british && python demo.py          # page at http://localhost:8000 to try every voice
```

```python
from paradee_british import load
tts = load()                                  # british/models/paradee_int8.onnx + british/config.json
audio = tts("Lovely weather today.", voice="bm_george")   # 24 kHz float32
```

`british/compare/` has Paradee and Kokoro reading the same sentences, per British voice.

## How it was made

Only the six new rows of the voice tables in each half of the model were trained (everything else
frozen), on 1,500 sentences of Kokoro reading Wikipedia in each new voice.

On 8 held-out sentences per voice (UTMOS, Kokoro in brackets; DTW log-mel distance to Kokoro, lower is closer):

| Voice | UTMOS | DTW |
|---|---:|---:|
| `bf_alice` | 3.74 (4.40) | 1.68 |
| `bf_emma` | 4.16 (4.42) | 1.31 |
| `bf_lily` | 4.16 (4.37) | 1.66 |
| `bm_daniel` | 3.98 (4.23) | 1.49 |
| `bm_george` | 4.04 (4.30) | 1.56 |
| `bm_lewis` | 4.34 (4.14) | 1.48 |
| `bf_isabella` (released) | 3.99 (4.42) | 0.78 |
| `bm_fable` (released) | 4.16 (4.32) | 0.78 |

The new voices sound natural but are not yet close copies of their Kokoro originals; that needs a
full fine-tune of the shared weights, as upstream did for v1.2 and v1.3.

To rebuild: clone the upstream repo into `upstream/`, set up `upstream/training` (uv), then from
`upstream/training` run `../../british/gen_all.sh` (teacher recordings) and
`../../british/train_new_voices.py --text-steps 600 --dec-steps 3000 --out checkpoints/british`,
and export with upstream's `scripts/export_paradee_v1_2.py --voices ...`.

Licence: Kokoro-82M and Paradee are Apache 2.0.
