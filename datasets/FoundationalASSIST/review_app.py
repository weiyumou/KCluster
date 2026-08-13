"""A local review UI for the problems the answerability screen could not settle.

Serves one problem at a time on ``localhost`` with everything needed to judge it
side by side — the problem text, the stored key, Gemini's answer, and what
students who were marked correct actually typed — and records the verdict to a
JSON file after every click, so the pass can be interrupted and resumed.

Nothing leaves this machine. The dataset is gated and carries a data-security
undertaking, so the problem text must not be sent to a hosted service; this is a
stdlib ``http.server`` bound to loopback, with no external requests and no CDN
assets. Do not repoint it at a public interface.

    python review_app.py --triage <dir>/triage.csv

Then open the URL it prints. Decisions land in ``<dir>/review-decisions.json``.
"""

import argparse
import csv
import http.server
import json
import os
import socketserver
import threading
import webbrowser

VERDICTS = {"keep": "Keep as is", "drop": "Drop", "fix": "Fix the key", "unsure": "Unsure"}


def jsonable(value):
    """Recursively replace NaN/Infinity with None.

    pandas hands back float('nan') for every missing cell, and ``json.dumps``
    serialises that as a bare ``NaN``. Python's own parser accepts it, but
    JSON.parse in the browser rejects it and the page renders nothing — so the
    payload has to be scrubbed before it is serialised.
    """
    if isinstance(value, float):
        return None if value != value or value in (float("inf"), float("-inf")) else value
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def dump_strict(value) -> str:
    """Serialise for the browser, refusing to emit anything JSON.parse would reject."""
    return json.dumps(jsonable(value), allow_nan=False)


def build_review_set(triage_path, questions_path, interactions_path, cache_path, verdict="review"):
    """Assemble everything the reviewer needs, caching the result beside the decisions."""
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    import pandas as pd

    triage = pd.read_csv(triage_path, keep_default_na=False, na_values=[""])
    todo = triage[triage["verdict"].eq(verdict)].copy()
    ids = set(todo["id"])

    questions = {}
    with open(questions_path) as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                if q["id"] in ids:
                    questions[q["id"]] = q

    # what students who were graded correct actually typed
    print("** Summarising student answers (one pass over the interaction log) **")
    ix = pd.read_csv(interactions_path, usecols=["problem_id", "answer_text", "discrete_score"])
    ix = ix[ix["discrete_score"].eq(1.0) & ix["problem_id"].isin(todo["problem_id"])]
    top = (ix.groupby("problem_id")["answer_text"].value_counts().rename("n")
             .reset_index().sort_values(["problem_id", "n"], ascending=[True, False]))
    students = {pid: g[["answer_text", "n"]].head(4).to_dict("records") for pid, g in top.groupby("problem_id")}

    items = []
    for _, r in todo.iterrows():
        q = questions.get(r["id"], {})
        stem = q.get("question", {}).get("stem", "")
        key, model = str(r["key"]), str(r["model_answer"])
        choices = [{"label": c["label"], "text": c["text"],
                    "is_key": c["label"] in key.split(", "),
                    "is_model": c["label"] in model.split(", ")}
                   for c in q.get("question", {}).get("choices", [])]
        items.append({
            "id": r["id"], "problem_id": int(r["problem_id"]), "type": r["type"],
            "part": int(r["part"]), "set": r["set"], "context": r["context"],
            "stem": stem, "choices": choices, "key": key,
            "model_answer": model, "model_answer_raw": r["model_answer_raw"],
            "student": r["student"], "students": students.get(int(r["problem_id"]), []),
            "p_key": r["p_key"], "p_nota": r["p_nota"], "p_self_contained": r["p_self_contained"],
            "skill": q.get("skill", []),
        })
    items.sort(key=lambda x: (x["type"], x["problem_id"]))
    items = jsonable(items)
    with open(cache_path, "w") as f:
        json.dump(items, f, allow_nan=False)
    print(f"** Built a review set of {len(items)} problems **")
    return items


def load_decisions(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_decisions(path, decisions):
    """Write via a temp file so an interrupted save cannot truncate the record."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(decisions, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def export_csv(path, items, decisions):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem_id", "id", "type", "verdict", "new_key", "note", "original_key"])
        for it in items:
            d = decisions.get(it["id"])
            if d:
                w.writerow([it["problem_id"], it["id"], it["type"], d.get("verdict", ""),
                            d.get("new_key", ""), d.get("note", ""), it["key"]])


def make_handler(items, state):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            payload = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if self.path == "/api/items":
                return self._send(200, dump_strict({"items": items, "decisions": state["decisions"]}))
            self._send(404, dump_strict({"error": "not found"}))

        def do_POST(self):
            if self.path != "/api/decision":
                return self._send(404, dump_strict({"error": "not found"}))
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            qid = body.get("id")
            if not qid:
                return self._send(400, dump_strict({"error": "missing id"}))
            if body.get("verdict") is None:
                state["decisions"].pop(qid, None)
            else:
                state["decisions"][qid] = {k: body.get(k, "") for k in ("verdict", "new_key", "note")}
            with state["lock"]:
                save_decisions(state["path"], state["decisions"])
                export_csv(state["csv"], items, state["decisions"])
            self._send(200, dump_strict({"ok": True, "done": len(state["decisions"])}))

        def log_message(self, *args):  # keep the console readable
            pass

    return Handler


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Answerability review</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#15181d;--muted:#6b7280;--line:#e2e5ea;--accent:#2563eb;
--keep:#15803d;--drop:#b91c1c;--fix:#b45309;--unsure:#6b7280;--chip:#eef1f5}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a20;--ink:#e8eaed;--muted:#9aa3af;
--line:#272b33;--chip:#222630;--keep:#4ade80;--drop:#f87171;--fix:#fbbf24;--unsure:#9aa3af}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 20px;
display:flex;align-items:center;gap:16px;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600}
.bar{flex:1;min-width:160px;height:6px;background:var(--chip);border-radius:3px;overflow:hidden}
.bar>i{display:block;height:100%;background:var(--accent);width:0;transition:width .2s}
.count{font-variant-numeric:tabular-nums;color:var(--muted);font-size:13px}
main{max-width:900px;margin:22px auto;padding:0 20px 120px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:14px}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}
.chip{background:var(--chip);color:var(--muted);border-radius:999px;padding:2px 10px;font-size:12px}
.stem{font-size:17px;white-space:pre-wrap;overflow-wrap:anywhere}
ul.choices{list-style:none;padding:0;margin:14px 0 0}
ul.choices li{padding:7px 10px;border:1px solid var(--line);border-radius:8px;margin-bottom:6px;
display:flex;gap:10px;align-items:baseline;overflow-wrap:anywhere}
li.key{border-color:var(--keep);background:color-mix(in srgb,var(--keep) 9%,transparent)}
li.model{border-color:var(--fix);background:color-mix(in srgb,var(--fix) 9%,transparent)}
li.key.model{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 10%,transparent)}
.tag{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-left:auto;white-space:nowrap}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.box{border:1px solid var(--line);border-radius:10px;padding:12px}
.box h3{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:600}
.val{font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;overflow-wrap:anywhere}
.sub{color:var(--muted);font-size:12px;margin-top:4px}
table{width:100%;border-collapse:collapse;font-size:13px}
td{padding:2px 0}td:last-child{text-align:right;color:var(--muted);font-variant-numeric:tabular-nums}
.actions{position:fixed;left:0;right:0;bottom:0;background:var(--card);border-top:1px solid var(--line);
padding:12px 20px;display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
button{font:inherit;padding:9px 16px;border-radius:9px;border:1px solid var(--line);background:var(--card);
color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button.on{color:#fff;border-color:transparent}
button[data-v=keep].on{background:var(--keep)}button[data-v=drop].on{background:var(--drop)}
button[data-v=fix].on{background:var(--fix)}button[data-v=unsure].on{background:var(--unsure)}
kbd{font:11px ui-monospace,monospace;background:var(--chip);border-radius:4px;padding:1px 5px;margin-left:6px;color:var(--muted)}
input,textarea{width:100%;font:14px ui-monospace,SFMono-Regular,Menlo,monospace;padding:8px;border-radius:8px;
border:1px solid var(--line);background:var(--bg);color:var(--ink);margin-top:8px}
.nav{display:flex;gap:10px;align-items:center}
.hint{color:var(--muted);font-size:12px;text-align:center;margin-top:10px}
.done{text-align:center;padding:50px 20px}
</style></head><body>
<header>
  <h1>Answerability review</h1>
  <div class="bar"><i id="bar"></i></div>
  <div class="count" id="count"></div>
  <div class="nav"><button id="prev">←</button><button id="next">→</button>
  <button id="jump" title="Next undecided">Next undecided</button></div>
</header>
<main id="main"></main>
<div class="actions" id="actions"></div>
<script>
let ITEMS=[],DEC={},i=0;
const $=s=>document.querySelector(s);
const esc=s=>String(s==null?"":s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const num=v=>(v==null||v===""||isNaN(+v))?"—":(+v).toFixed(3);

async function boot(){
  try{
    const r=await fetch("/api/items");
    if(!r.ok) throw new Error("server returned "+r.status);
    const d=JSON.parse(await r.text());
    ITEMS=d.items;DEC=d.decisions||{};
    if(!ITEMS.length){$("#main").innerHTML="<div class='done'>Nothing to review.</div>";return;}
    const first=ITEMS.findIndex(x=>!DEC[x.id]); i=first<0?0:first; render();
  }catch(err){
    $("#main").innerHTML="<div class='card'><h3>Could not load the review set</h3><pre>"+
      esc(err&&err.message||err)+"</pre></div>";
  }
}
function render(){
  const it=ITEMS[i];
  if(!it){$("#main").innerHTML="<div class='done'>Nothing to review.</div>";return;}
  const d=DEC[it.id]||{};
  const done=Object.keys(DEC).length;
  $("#bar").style.width=(100*done/ITEMS.length)+"%";
  $("#count").textContent=`${i+1} / ${ITEMS.length} · ${done} decided`;

  const choices=it.choices.length?`<ul class="choices">${it.choices.map(c=>{
    const cls=[c.is_key?"key":"",c.is_model?"model":""].join(" ").trim();
    const tag=c.is_key&&c.is_model?"key + Gemini":c.is_key?"answer key":c.is_model?"Gemini":"";
    return `<li class="${cls}"><b>${esc(c.label)})</b><span>${esc(c.text)}</span>${tag?`<span class="tag">${tag}</span>`:""}</li>`;
  }).join("")}</ul>`:"";

  const students=it.students.length?`<table>${it.students.map(s=>
    `<tr><td class="val">${esc(s.answer_text)}</td><td>${s.n}</td></tr>`).join("")}</table>`
    :`<div class="sub">no correct interactions</div>`;

  $("#main").innerHTML=`
    <div class="card">
      <div class="meta">
        <span class="chip">${esc(it.id)}</span>
        <span class="chip">${esc(it.type)}</span>
        <span class="chip">part ${it.part} · ${esc(it.context)}</span>
        ${it.skill.map(s=>`<span class="chip">${esc(s)}</span>`).join("")}
      </div>
      <div class="stem">${esc(it.stem)}</div>
      ${choices}
    </div>
    <div class="grid">
      <div class="box"><h3>Stored answer key</h3><div class="val">${esc(it.key)}</div></div>
      <div class="box"><h3>Gemini answered</h3><div class="val">${esc(it.model_answer_raw||it.model_answer)}</div>
        <div class="sub">P(key) ${num(it.p_key)} · P(none of the above) ${num(it.p_nota)}</div></div>
      <div class="box"><h3>Students marked correct</h3>${students}</div>
      <div class="box"><h3>Self-contained?</h3><div class="val">P = ${num(it.p_self_contained)}</div>
        <div class="sub">the probe found the text complete</div></div>
    </div>
    <div class="card">
      <label class="sub">Corrected key (used when the verdict is “Fix the key”)</label>
      <input id="newkey" value="${esc(d.new_key||"")}" placeholder="${esc(it.key)}">
      <label class="sub">Note (optional)</label>
      <textarea id="note" rows="2">${esc(d.note||"")}</textarea>
    </div>`;

  $("#actions").innerHTML=[["keep","Keep as is","K"],["drop","Drop","D"],["fix","Fix the key","F"],
    ["unsure","Unsure","U"]].map(([v,label,k])=>
    `<button data-v="${v}" class="${d.verdict===v?"on":""}">${label}<kbd>${k}</kbd></button>`).join("")
    +`<div class="hint" style="flex-basis:100%">←/→ move · a verdict advances automatically</div>`;
  document.querySelectorAll("#actions button").forEach(b=>b.onclick=()=>decide(b.dataset.v));
}
async function decide(v){
  const it=ITEMS[i];
  const body={id:it.id,verdict:v,new_key:$("#newkey").value.trim(),note:$("#note").value.trim()};
  DEC[it.id]={verdict:v,new_key:body.new_key,note:body.note};
  await fetch("/api/decision",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)});
  if(i<ITEMS.length-1){i++;} render();
}
function go(n){i=Math.min(ITEMS.length-1,Math.max(0,i+n));render();}
$("#prev").onclick=()=>go(-1);$("#next").onclick=()=>go(1);
$("#jump").onclick=()=>{const n=ITEMS.findIndex(x=>!DEC[x.id]);if(n>=0){i=n;render();}};
document.onkeydown=e=>{
  if(/^(INPUT|TEXTAREA)$/.test(e.target.tagName))return;
  const m={k:"keep",d:"drop",f:"fix",u:"unsure"}[e.key.toLowerCase()];
  if(m){e.preventDefault();decide(m);}
  else if(e.key==="ArrowLeft")go(-1); else if(e.key==="ArrowRight")go(1);
};
boot();
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--triage", required=True, type=str, help="triage.csv from the answerability run")
    parser.add_argument("--questions", default=None, type=str, help="question jsonl (default: alongside the data)")
    parser.add_argument("--interactions", default=None, type=str, help="interactions.csv (default: alongside)")
    parser.add_argument("--verdict", default="review", type=str, help="Which verdict to review (default: review)")
    parser.add_argument("--port", default=8765, type=int)
    parser.add_argument("--no_browser", action="store_true")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.triage))
    processed = os.path.abspath(os.path.join(out_dir, "..", ".."))
    questions = args.questions or os.path.join(processed, "foundational-assist.jsonl")
    interactions = args.interactions or os.path.join(processed, "interactions.csv")

    items = build_review_set(args.triage, questions, interactions,
                             os.path.join(out_dir, "review-set.json"), args.verdict)
    state = {"decisions": load_decisions(os.path.join(out_dir, "review-decisions.json")),
             "path": os.path.join(out_dir, "review-decisions.json"),
             "csv": os.path.join(out_dir, "review-decisions.csv"),
             "lock": threading.Lock()}

    url = f"http://127.0.0.1:{args.port}/"
    print(f"** {len(items)} problems to review, {len(state['decisions'])} already decided **")
    print(f"** Decisions:  {state['path']}\n**            {state['csv']} **")
    print(f"\n  Open {url}   (Ctrl-C to stop; every click is saved immediately)\n")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    socketserver.TCPServer.allow_reuse_address = True
    # Loopback only: the problem text must not leave this machine.
    with socketserver.TCPServer(("127.0.0.1", args.port), make_handler(items, state)) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print(f"\n** Stopped. {len(state['decisions'])}/{len(items)} decided. **")


if __name__ == "__main__":
    main()
