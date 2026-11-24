"""Local page to try every voice in the 14-voice model: type text, click a voice.
usage (from upstream/training):  uv run python ../../british/demo.py [port]   then open http://localhost:8000"""
import io, json, sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import soundfile as sf
from paradee_british import load, SAMPLE_RATE

tts = load()
NEW = {"bf_alice", "bf_emma", "bf_lily", "bm_daniel", "bm_george", "bm_lewis"}
LANG = {"a": "American", "b": "British", "h": "Hindi", "f": "French", "e": "Spanish"}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paradee Voices</title><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dc;--accent:#2f6f5e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#9a9993;--card:#20201e;--line:#33332f;--accent:#7cc4ad}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:900px;margin:0 auto;padding:24px 16px}
h1{font-size:1.4rem;margin:0 0 4px}p{color:var(--muted);margin:0 0 16px}
textarea{width:100%;box-sizing:border-box;min-height:90px;padding:10px;font:inherit;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:10px 0 20px}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button:hover{border-color:var(--accent)}button.primary{background:var(--accent);color:var(--bg);border-color:var(--accent)}
h2{font-size:1rem;margin:20px 0 8px;color:var(--muted);font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;display:flex;flex-direction:column;gap:8px}
.card.playing{border-color:var(--accent)}.name{font-weight:600}.tag{font-size:.75rem;color:var(--accent);margin-left:6px}
.meta{font-size:.8rem;color:var(--muted)}audio{width:100%;height:32px}
</style></head><body><main>
<h1>Paradee voices</h1><p>14 voices in one 8.5 MB model. <b>new</b> = the six British voices added here.</p>
<textarea id="text">The quick brown fox jumps over the lazy dog, then has a cup of tea.</textarea>
<div class="row"><label>Speed <input id="speed" type="range" min="0.6" max="1.6" step="0.05" value="1"> <span id="sv">1.00</span></label>
<button class="primary" id="all">Play all British</button></div>
<div id="groups"></div></main><script>
const VOICES = __VOICES__;
const $ = id => document.getElementById(id);
$("speed").oninput = e => $("sv").textContent = (+e.target.value).toFixed(2);
const groups = {};
for (const v of VOICES) {
  const g = groups[v.lang] ??= (() => { const s = document.createElement("section");
    s.innerHTML = `<h2>${v.lang}</h2><div class="grid"></div>`; $("groups").append(s); return s.querySelector(".grid"); })();
  const c = document.createElement("div"); c.className = "card"; c.id = "c_" + v.id;
  c.innerHTML = `<div><span class="name">${v.id}</span>${v.isNew ? '<span class="tag">new</span>' : ""}</div>
    <button>▶ Speak</button><audio controls hidden></audio><div class="meta"></div>`;
  c.querySelector("button").onclick = () => speak(v.id);
  g.append(c);
}
async function speak(id) {
  const c = $("c_" + id), a = c.querySelector("audio"), m = c.querySelector(".meta");
  m.textContent = "generating…";
  const t0 = performance.now();
  try {
    const r = await fetch(`/tts?voice=${id}&speed=${$("speed").value}&text=${encodeURIComponent($("text").value)}`);
    if (!r.ok) { m.textContent = "error: " + await r.text(); return; }
    a.src = URL.createObjectURL(await r.blob()); a.hidden = false;
    await new Promise((res, rej) => { a.onloadedmetadata = res; a.onerror = () => rej(new Error("browser could not decode the audio")); });
  } catch (e) { m.textContent = "error: " + e.message; return; }
  document.querySelectorAll(".card").forEach(x => x.classList.remove("playing")); c.classList.add("playing");
  m.textContent = `${a.duration.toFixed(1)} s audio in ${((performance.now() - t0) / 1000).toFixed(2)} s`;
  try { await a.play(); } catch (e) { m.textContent += " — browser blocked autoplay, press ▶ on the player"; return; }
  return new Promise(res => a.onended = res);
}
$("all").onclick = async () => { for (const v of VOICES.filter(v => v.lang === "British")) await speak(v.id); };
</script></body></html>"""

def page():
    voices = [dict(id=v, lang=LANG.get(v[0], v[0]), isNew=v in NEW) for v in tts.voices]
    return PAGE.replace("__VOICES__", json.dumps(voices)).encode()

class H(BaseHTTPRequestHandler):
    def send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", len(body)); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/": return self.send(200, page(), "text/html; charset=utf-8")
        if u.path != "/tts": return self.send(404, b"not found", "text/plain")
        try:
            voice, text = q.get("voice", ""), q.get("text", "").strip()[:2000]
            if voice not in tts.voices or not text: raise ValueError("need a known voice and some text")
            speed = min(max(float(q.get("speed", 1)), 0.5), 2.0)
            buf = io.BytesIO(); sf.write(buf, tts(text, speed=speed, voice=voice), SAMPLE_RATE, format="WAV")
            self.send(200, buf.getvalue(), "audio/wav")
        except Exception as e:
            self.send(400, str(e).encode(), "text/plain")
    def log_message(self, fmt, *a): print(fmt % a, flush=True)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"http://localhost:{port}"); HTTPServer(("127.0.0.1", port), H).serve_forever()
