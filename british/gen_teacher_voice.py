"""Teacher corpus for one extra Kokoro voice, same row format as upstream gen_teacher.py + regen_audio.py,
but on MPS with the CPU-canonical source STFT (common.cpu_source_stft, matches CPU to ~1e-5) and a fixed seed per sentence (so no separate regen pass) and without "d" (no trainer reads it).
Run from upstream/training.  usage: gen_teacher_voice.py VOICE [n_sentences] [threads]
Writes data/teacher_VOICE/shard_*.pt (500 sentences each, resumable). Rows 0-199 of shard 0 are the held-out set."""
import sys, os, json, time, warnings, torch
sys.path.insert(0, "scripts")
from common import cpu_source_stft
warnings.filterwarnings("ignore")
from kokoro import KPipeline
VOICE = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 5)
SHARD = 500; OUT = f"data/teacher_{VOICE}"
assert VOICE[0] in "ab", "English voices only (a = American, b = British misaki dictionary)"
sents = json.load(open("data/wikitext_sents.json"))[:N]
DEV = "mps"
pipe = KPipeline(lang_code=VOICE[0], device=DEV, repo_id="hexgrad/Kokoro-82M"); m = pipe.model.eval(); cpu_source_stft(m.decoder)
ref = pipe.load_voice(VOICE).to(DEV)
os.makedirs(OUT, exist_ok=True)

@torch.no_grad()
def run(ps, seed):
    ids = torch.LongTensor([[0, *[m.vocab[c] for c in ps if c in m.vocab], 0]]).to(DEV)
    T = ids.shape[1]; r = ref[T - 3]  # style indexed by phoneme count, as upstream
    L = torch.full((1,), T, device=DEV); mask = torch.zeros(1, T, dtype=torch.bool, device=DEV)
    d_en = m.bert_encoder(m.bert(ids, attention_mask=(~mask).int())).transpose(-1, -2)
    s = r[:, 128:]; d = m.predictor.text_encoder(d_en, s, L, mask); x, _ = m.predictor.lstm(d)
    dur = torch.sigmoid(m.predictor.duration_proj(x)).sum(-1)
    pred_dur = torch.round(dur).clamp(min=1).long().squeeze(0)
    idx = torch.repeat_interleave(torch.arange(T, device=DEV), pred_dur)
    aln = torch.zeros(T, idx.shape[0], device=DEV); aln[idx, torch.arange(idx.shape[0], device=DEV)] = 1; aln = aln[None]
    F0, Nn = m.predictor.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    t_en = m.text_encoder(ids, L, mask)
    torch.manual_seed(seed); audio = m.decoder(t_en @ aln, F0, Nn, r[:, :128]).squeeze()
    c = lambda t: t.squeeze(0).cpu()
    return dict(ids=c(ids), dur=c(dur), pred_dur=pred_dur.cpu(), t_en=c(t_en).half(), F0=c(F0), N=c(Nn),
                audio=(audio.cpu().clamp(-1, 1) * 32767).short(), audio_cpu=True)

t0 = time.time(); secs = 0.0
for sh in range(0, len(sents), SHARD):
    path = f"{OUT}/shard_{sh // SHARD:04d}.pt"
    if os.path.exists(path): continue
    rows = []
    for i, text in enumerate(sents[sh:sh + SHARD]):
        ps, _ = pipe.g2p(text)
        if not ps or len(ps) > 400: continue
        r = run(ps, sh + i); r.update(text=text, ps=ps, voice=VOICE); rows.append(r); secs += len(r["audio"]) / 24000
        torch.mps.empty_cache()   # every sentence has a new shape; without this the MPS cache grows until the machine swaps
    torch.save(rows, path + ".tmp"); os.replace(path + ".tmp", path)
    print(f"{VOICE} {path} {len(rows)} sents  {secs/3600:.2f} h audio  {time.time()-t0:.0f}s  {secs/(time.time()-t0):.1f}x RT", flush=True)
print(VOICE, "done")
