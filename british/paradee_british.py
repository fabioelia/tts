"""Paradee v1.3 + six more British voices (14 voices, one 8.5 MB ONNX file). Uses the upstream `paradee` runtime
unchanged; only its voice list comes from our config.json instead of the hard-coded v1.3 list.

    from paradee_british import load
    tts = load()                                    # models/paradee_int8.onnx + config.json next to this file
    audio = tts("Lovely weather today.", voice="bm_george")   # 24 kHz float32

usage: python paradee_british.py ["text"]   writes one WAV per British voice into samples/"""
import json, os, sys
from paradee import Paradee, SAMPLE_RATE

HERE = os.path.dirname(os.path.abspath(__file__))

def load(model_path=os.path.join(HERE, "models", "paradee_int8.onnx"), config_path=os.path.join(HERE, "config.json"), **kw):
    tts = Paradee(model_path=model_path, config_path=config_path, **kw)
    tts.voices = json.load(open(config_path, encoding="utf-8"))["voices"]   # row order of the model's style tables
    return tts

if __name__ == "__main__":
    import soundfile as sf
    text = sys.argv[1] if len(sys.argv) > 1 else "The quick brown fox jumps over the lazy dog, then has a cup of tea."
    tts = load(); os.makedirs(os.path.join(HERE, "samples"), exist_ok=True)
    for v in (v for v in tts.voices if v.startswith("b")):
        path = os.path.join(HERE, "samples", f"{v}.wav"); sf.write(path, tts(text, voice=v), SAMPLE_RATE); print(path)
