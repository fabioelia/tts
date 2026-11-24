"""Add voices to Paradee v1.3 by learning only their rows of the two style tables. Every shared weight and the 8
released rows stay frozen, so the released voices come out bit-for-bit unchanged (asserted before saving).
Each half is fitted to the teacher the same way upstream trains it (teacher forcing, see train_text.py /
train_vhead.py): the text side to the teacher's durations, pitch and energy; the decoder to the teacher's audio
from the teacher's own features, with upstream's spectral + harmonicity + pitch-pulse losses (no GAN: with the
shared weights frozen there is little for critics to add).
Run from upstream/training.  usage: train_new_voices.py [--text-steps N] [--dec-steps N] [--out DIR]"""
import sys, glob, math, os, json, time, random, argparse, warnings
warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
import torch, torch.nn.functional as F, torchaudio
from student import TextStudent, alignment
from vhead import VDecoder

BASE = ["af_heart", "bf_isabella", "am_puck", "bm_fable", "hf_alpha", "hm_omega", "ff_siwis", "em_alex"]
NEW = ["bf_alice", "bf_emma", "bf_lily", "bm_daniel", "bm_george", "bm_lewis"]
INIT = {"f": BASE.index("bf_isabella"), "m": BASE.index("bm_fable")}   # each new row starts from the British voice of its sex
REF = ["bf_isabella", "bm_fable"]   # released British voices, scored the same way as a yardstick (never trained here)
VOICES = BASE + NEW; NB = len(BASE)
ap = argparse.ArgumentParser(); ap.add_argument("--text-steps", type=int, default=2000); ap.add_argument("--dec-steps", type=int, default=3000)
ap.add_argument("--lr", type=float, default=1e-2); ap.add_argument("--seg", type=int, default=64)
ap.add_argument("--src", default="hf13/pytorch"); ap.add_argument("--out", default="checkpoints/british"); ap.add_argument("--device", default="mps")
ap.add_argument("--text-device", default="mps", help="text batches are padded to a few bucket sizes (see text_batch) so the MPS graph cache stays bounded; on CPU this phase is >30 s/step")
A = ap.parse_args(); DEV = A.device; TDEV = A.text_device; os.makedirs(A.out, exist_ok=True)

def load(v):
    rows = [r for sh in sorted(glob.glob(f"data/teacher_{v}/shard_*.pt")) for r in torch.load(sh)]
    for r in rows: r["vid"] = VOICES.index(v)
    return rows[:200], rows[200:]
held, train = {}, []
for v in NEW: held[v], tr = load(v); train += tr
for v in REF:
    if glob.glob(f"data/teacher_{v}/shard_*.pt"): held[v] = load(v)[0]
print(f"{len(train)} train sentences, {sum(map(len, held.values()))} held out, voices {NEW}", flush=True)

def grow(style):
    """8-row table -> 14 rows, new rows copied from INIT; returns the table and a mask that lets only new rows learn."""
    new = torch.stack([style[INIT[v[1]]] for v in NEW]); w = torch.cat([style, new]).clone()
    mask = torch.zeros(len(VOICES), 1); mask[NB:] = 1
    return torch.nn.Parameter(w), mask

def freeze_all_but_style(model, src_state, dev):
    st = dict(src_state); w, mask = grow(st.pop("style"))
    res = model.load_state_dict({**st, "style": w.data}); assert not res.missing_keys and not res.unexpected_keys
    for p in model.parameters(): p.requires_grad_(False)
    model.style.requires_grad_(True); model.style.register_hook(lambda g, m=mask.to(dev): g * m)   # released rows get zero gradient
    return model.to(dev).eval()   # eval: no dropout, the shared modules behave exactly as at inference

def fit(name, model, steps, batch_fn, loss_fn, bs):
    opt = torch.optim.Adam([model.style], lr=A.lr)   # Adam with zero grad + no weight decay leaves masked rows exactly unchanged
    sched = lambda s: min(1, s / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)); t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = A.lr * sched(step)
        ls = loss_fn(model, batch_fn(random.sample(train, bs))); loss = sum(ls.values())
        opt.zero_grad(); loss.backward(); opt.step()
        if DEV == "mps" and step % 20 == 0: torch.mps.empty_cache()   # variable batch shapes grow the MPS cache otherwise
        if step % 100 == 0: print(f"{name} step {step} " + " ".join(f"{k} {v.item():.4f}" for k, v in ls.items()) + f" {(time.time() - t0) / step:.2f}s/step", flush=True)

# ---- text side: same targets and losses as upstream train_text.py, plus the voice id
def text_batch(batch):
    B = len(batch); T = max(len(r["ids"]) for r in batch); Fm = max(len(r["F0"]) for r in batch) // 2
    T, Fm = -(-T // 128) * 128, -(-Fm // 512) * 512   # bucket sizes: every new shape costs MPS graph-cache memory; padding is masked out
    ids = torch.zeros(B, T, dtype=torch.long); L = torch.tensor([len(r["ids"]) for r in batch]); dur = torch.zeros(B, T); aln = torch.zeros(B, T, Fm)
    F0 = torch.zeros(B, 2 * Fm); N = torch.zeros(B, 2 * Fm); fm = torch.zeros(B, 2 * Fm, dtype=torch.bool)
    for i, r in enumerate(batch):
        t = len(r["ids"]); f = len(r["F0"]) // 2
        ids[i, :t] = r["ids"]; dur[i, :t] = r["dur"]; aln[i, :t, :f] = alignment(r["pred_dur"], f); F0[i, :2 * f] = r["F0"]; N[i, :2 * f] = r["N"]; fm[i, :2 * f] = True
    tm = torch.arange(T)[None] < L[:, None]; vid = torch.tensor([r["vid"] for r in batch])
    return [x.to(TDEV) for x in (ids, L, dur, aln, F0, N, tm, fm, vid)]
def text_loss(model, b):
    ids, L, dur, aln, F0, N, tm, fm, vid = b; pd, pF0, pN, _ = model(ids, L, aln, vid)   # asr features don't depend on the voice
    return dict(dur=(F.l1_loss(torch.log(pd.clamp_min(1e-3)), torch.log(dur.clamp_min(1e-3)), reduction="none") * tm).sum() / tm.sum(),
                f0=(F.l1_loss(pF0, F0, reduction="none") * fm).sum() / fm.sum() / 100, n=(F.l1_loss(pN, N, reduction="none") * fm).sum() / fm.sum())

# ---- decoder: teacher-forced segments and upstream's train_vhead.py losses (--fine-stft --harm-weight 2 --mod-weight 5)
def segment(r, seg):
    aln = alignment(r["pred_dur"]); asr = r["t_en"].float() @ aln; Fr = aln.shape[1]
    F0, N, au = r["F0"][:2 * Fr], r["N"][:2 * Fr], r["audio"][:600 * Fr].float() / 32767
    if Fr <= seg: p = seg - Fr; asr, F0, N, au, a = F.pad(asr, (0, p)), F.pad(F0, (0, 2 * p)), F.pad(N, (0, 2 * p)), F.pad(au, (0, 600 * p)), 0
    else: a = random.randint(0, Fr - seg)
    c0 = (F0[:2 * a].double() * 300 / 24000).sum() % 1
    return asr[:, a:a + seg], F0[2 * a:2 * (a + seg)], N[2 * a:2 * (a + seg)], au[600 * a:600 * (a + seg)], c0.float()
def dec_batch(batch):
    x = [segment(r, A.seg) for r in batch]
    return [torch.stack([t[i] for t in x]).to(DEV) for i in range(5)] + [torch.tensor([r["vid"] for r in batch], device=DEV)]
mel = torchaudio.transforms.MelSpectrogram(24000, n_fft=1024, hop_length=256, n_mels=80).to(DEV)
def spec_losses(y, t):
    lm = F.l1_loss(torch.log(mel(y).clamp_min(1e-5)), torch.log(mel(t).clamp_min(1e-5))); ls = 0; SZ = (64, 128, 512, 1024, 2048)
    for n in SZ:
        w = torch.hann_window(n, device=DEV); Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs(); T = torch.stft(t, n, n // 4, window=w, return_complex=True).abs()
        ls = ls + (T - Y).norm() / T.norm().clamp_min(1e-5) + F.l1_loss(torch.log(Y.clamp_min(1e-5)), torch.log(T.clamp_min(1e-5)))
    return lm, ls / len(SZ)
def harm_loss(y, t, f0, n=2048, hop=512):   # verbatim from train_vhead.py
    w = torch.hann_window(n, device=DEV); nfr = y.shape[-1] // hop + 1
    idx = (torch.arange(nfr, device=DEV) * hop // 300).clamp(max=f0.shape[1] - 1); lo = (idx - 2).clamp(min=0); hi = (idx + 2).clamp(max=f0.shape[1] - 1)
    f = f0[:, idx]; v = (torch.minimum(torch.minimum(f0[:, lo], f), f0[:, hi]) > 100).float()
    fr = torch.arange(n // 2 + 1, device=DEV) * 24000 / n; ratio = fr[None, :, None] / f.clamp_min(60)[:, None]; frac = (ratio - ratio.round()).abs()
    mh, mi = (frac < 0.15).float(), ((frac - 0.5).abs() < 0.15).float(); bands = torch.stack([((fr >= 1000 * b) & (fr < 1000 * (b + 1))).float() for b in range(1, 9)])
    out = []
    for x in (y, t):
        P = torch.stft(x, n, hop, window=w, return_complex=True).abs().pow(2)
        out.append(torch.log((torch.einsum("bft,kf->bkt", P * mh, bands) + 1e-8) / (torch.einsum("bft,kf->bkt", P * mi, bands) + 1e-8)))
    return ((out[0] - out[1]).abs() * v[:, None]).sum() / (v.sum() * 8).clamp_min(1)
MODB = [(4, 8), (8, 12), (12, 16), (16, 21), (21, 29)]
def mod_loss(y, t, cyc, uv, n=64, hop=16, win=75):   # verbatim from train_vhead.py
    w = torch.hann_window(n, device=DEV); k = y.shape[-1] // hop + 1
    ph = 2 * torch.pi * F.pad(cyc, (0, hop))[:, ::hop][:, :k]; v = F.pad(uv, (0, hop))[:, ::hop][:, :k]
    carr = torch.stack([fn(m * ph) for m in (1, 2) for fn in (torch.cos, torch.sin)], 1); D = []
    for x in (y, t):
        X = torch.stft(x, n, hop, window=w, return_complex=True).abs(); e = torch.stack([X[:, a:b].sum(1) for a, b in MODB], 1)
        num = F.avg_pool1d((e[:, :, None] * carr[:, None]).flatten(1, 2), win, win // 2); den = F.avg_pool1d(e, win, win // 2).clamp_min(1e-5)
        D.append(2 * num.view(e.shape[0], 5, 4, -1) / den[:, :, None])
    vm = (F.avg_pool1d(v[:, None], win, win // 2) > 0.99).float()[:, :, None]
    return ((D[0] - D[1]).abs() * vm).sum() / (vm.sum() * 20).clamp_min(1)
def dec_loss(model, b):
    asr, F0, N, au, c0, vid = b; y = model(asr, F0, N, c0=c0, voice=vid); lm, ls = spec_losses(y, au)
    return dict(mel=lm, stft=ls, harm=2 * harm_loss(y, au, F0), mod=5 * mod_loss(y, au, model.head.last["cyc"], model.head.last["uv"]))

src_t = torch.load(f"{A.src}/text_side.pt", map_location="cpu"); src_d = torch.load(f"{A.src}/decoder.pt", map_location="cpu")
assert src_t["voices"] == BASE and src_d["voices"] == BASE, (src_t["voices"], src_d["voices"])
ts = freeze_all_but_style(TextStudent("s+mlp", n_voices=len(VOICES)), src_t["model"], TDEV)
vd = freeze_all_but_style(VDecoder(head="v2", dim=160, blocks=4, n_voices=len(VOICES)), src_d["model"], DEV)

@torch.no_grad()
def render(tag, n=8):
    """Free-running: student text side -> student decoder on held-out sentences, next to the teacher's audio."""
    import soundfile as sf; from common import compare
    d = f"{A.out}/eval_{tag}"; os.makedirs(d, exist_ok=True); out = {}
    for v in held:
        refs, outs = [], []
        for i, r in enumerate(held[v][:n]):
            ids = r["ids"][None].to(TDEV); L = torch.tensor([ids.shape[1]], device=TDEV); vo = torch.tensor([r["vid"]], device=TDEV)
            pd, _, _, pasr = ts(ids, L, voice=vo); aln = alignment(torch.round(pd[0]).clamp(min=1).long())[None]
            _, pF0, pN, _ = ts(ids, L, aln, vo); torch.manual_seed(i); y = vd((pasr @ aln).to(DEV), pF0.to(DEV), pN.to(DEV), voice=vo.to(DEV)).squeeze().cpu()
            ref = r["audio"].float() / 32767; refs.append(ref); outs.append(y)
            sf.write(f"{d}/held_{i + 1}_{v}.wav", y.numpy(), 24000)
            if DEV == "mps": torch.mps.empty_cache()
            if tag == "init": sf.write(f"{A.out}/teacher_{i + 1}_{v}.wav", ref.numpy(), 24000)
        out[v] = round(compare(refs, outs)[1], 3)
    print(f"EVAL {tag} DTW log-mel vs teacher (lower is better):", json.dumps(out), flush=True); return out

ev = {"init": render("init")}
fit("text", ts, A.text_steps, text_batch, text_loss, 32)
fit("dec", vd, A.dec_steps, dec_batch, dec_loss, 16)
ev["trained"] = render("trained")
for name, m, s in (("text_side", ts, src_t), ("decoder", vd, src_d)):
    assert torch.equal(m.style.data[:NB].cpu(), s["model"]["style"]), f"{name}: a released voice row changed"
    torch.save({"model": {k: v.cpu() for k, v in m.state_dict().items()}, "voices": VOICES}, f"{A.out}/{name}.pt")
json.dump(ev, open(f"{A.out}/eval.json", "w"), indent=1); print("saved", A.out)
