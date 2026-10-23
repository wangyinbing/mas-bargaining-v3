"""Build a click-through review page for the 42 hand-audited free-dialogue episodes.

Each card shows: the blind label, the evidence rounds, and two buttons.
Marks are kept in localStorage; the export button copies a compact result
string that can be pasted back into the chat.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUD = ROOT / "output_v3" / "main" / "audit"
TDIR = ROOT / "output_v3" / "main" / "transcripts"


def load_key(name):
    return json.loads((AUD / name).read_text(encoding="utf-8"))


def evidence(transcript, focus=None, n_last=3, limit=700):
    rounds = {e.get("round"): e for e in transcript if e.get("round")}
    keys = sorted(rounds)
    sel = keys[-n_last:]
    if focus and focus in rounds and focus not in sel:
        sel = [focus] + sel
    out = []
    for r in sel:
        e = rounds[r]
        out.append(dict(
            round=r,
            pie=max(1, int(round(10 * 0.9 ** (r - 1)))),
            a=(e.get("a") or "").replace("\n", " ")[:limit],
            b=(e.get("b") or "").replace("\n", " ")[:limit],
        ))
    return out


def build():
    key_main = load_key("audit_key.json")
    key_neg = load_key("audit_key_neg.json")
    labels = {r["id"]: r for r in csv.DictReader(open(AUD / "labels.csv", encoding="utf-8"))}
    labels_neg = {r["id"]: r for r in csv.DictReader(open(AUD / "labels_neg.csv", encoding="utf-8"))}

    cards = []
    for keyfile, labs, group in [(key_main, labels, "主包（28 成交 / 2 未成交）"),
                                 (key_neg, labels_neg, "负例补审（12 条，全部判未成交）")]:
        for eid, meta in sorted(keyfile.items()):
            if eid not in labs:
                continue
            lab = labs[eid]
            deal = lab["deal"] == "1"
            t = json.loads((TDIR / meta["file"]).read_text(encoding="utf-8"))
            ev = evidence(t["transcript"], focus=meta.get("det_round"))
            m = __import__("re").search(r"_(\d+)_e(\d+)\.json$", meta["file"])
            trial = meta.get("trial", int(m.group(1)) if m else 0)
            episode = meta.get("episode", int(m.group(2)) if m else 0)
            cards.append(dict(
                id=eid, group=group, model=meta["model"],
                trial=trial, episode=episode,
                file=meta["file"],
                label_deal=deal,
                label_split=f'{lab["split_a"]}-{lab["split_b"]}' if deal else "未成交",
                evidence=ev,
            ))

    data = json.dumps(cards, ensure_ascii=False)
    html = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>审计复核 · 42 条自由对话</title>
<style>
  :root { --fg:#1a1a1a; --muted:#666; --line:#ddd; --bg:#fff; --ok:#1a7f37; --bad:#b42318; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; margin:0; padding:24px 32px 80px; color:var(--fg); background:var(--bg); line-height:1.65; }
  h1 { font-size:20px; margin:0 0 6px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:18px; }
  .bar { position:sticky; top:0; background:var(--bg); padding:10px 0 14px; border-bottom:1px solid var(--line); z-index:5; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  .bar b { font-variant-numeric: tabular-nums; }
  button { font:inherit; padding:6px 12px; border:1px solid var(--line); background:#fff; border-radius:6px; cursor:pointer; }
  button:hover { border-color:#999; }
  button.ok.on { background:#e6f4ea; border-color:var(--ok); color:var(--ok); }
  button.bad.on { background:#fdecea; border-color:var(--bad); color:var(--bad); }
  .card { border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:16px 0; }
  .card.done { border-color:#bbb; background:#fafafa; }
  .head { display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }
  .id { font-weight:700; font-size:16px; }
  .meta { color:var(--muted); font-size:12px; }
  .label { margin:10px 0 4px; font-size:14px; }
  .label b { font-size:15px; }
  .ev { margin-top:10px; border-left:3px solid var(--line); padding-left:12px; }
  .r { margin:10px 0; }
  .r .rh { font-size:12px; color:var(--muted); }
  .m { margin:3px 0; font-size:14px; }
  .m span { font-weight:600; }
  .acts { margin-top:12px; display:flex; gap:10px; align-items:center; }
  .note { font-size:12px; color:var(--muted); }
  textarea { width:100%; height:90px; margin-top:14px; font-family: ui-monospace, Consolas, monospace; font-size:12px; }
</style></head><body>
<h1>审计复核：42 条自由对话</h1>
<div class="sub">逐条读“证据”，判断：<b>双方是否在同一轮明确接受了同一个分割？</b> 卡片上方是待复核的判定。正确点「✓ 正确」，有误点「✗ 有误」。全部看完后点「导出结果」，把文本发回。</div>
<div class="bar">
  <span>进度：<b id="prog">0</b> / <b id="total">0</b></span>
  <button onclick="markAll('ok')">全部标为正确</button>
  <button onclick="exportRes()">导出结果</button>
  <span class="note" id="hint"></span>
</div>
<div id="cards"></div>
<textarea id="out" placeholder="导出结果会出现在这里"></textarea>
<script>
const DATA = __DATA__;
const KEY = 'audit-review-v1';
let state = JSON.parse(localStorage.getItem(KEY) || '{}');
const root = document.getElementById('cards');

function save(){ localStorage.setItem(KEY, JSON.stringify(state)); updateProg(); }

function card(c){
  const el = document.createElement('div');
  el.className = 'card' + (state[c.id] ? ' done' : '');
  const lab = c.label_deal ? `成交 ${c.label_split}` : '未成交';
  let ev = '';
  for (const r of c.evidence){
    ev += `<div class="r"><div class="rh">第 ${r.round} 轮 · 饼 ${r.pie}</div>
      <div class="m"><span>A：</span>${escapeHtml(r.a)}</div>
      <div class="m"><span>B：</span>${escapeHtml(r.b)}</div></div>`;
  }
  el.innerHTML = `<div class="head"><span class="id">${c.id}</span>
      <span class="meta">${c.model} · trial ${c.trial} · episode ${c.episode}</span></div>
    <div class="label">待复核判定：<b>${lab}</b></div>
    <div class="ev">${ev}</div>
    <div class="acts">
      <button class="ok" onclick="mark('${c.id}','ok',this)">✓ 正确</button>
      <button class="bad" onclick="mark('${c.id}','bad',this)">✗ 有误</button>
      <span class="note" id="st-${c.id}">${state[c.id] ? (state[c.id]==='ok'?'已标：正确':'已标：有误') : '未复核'}</span>
    </div>`;
  if (state[c.id]==='ok') el.querySelector('.ok').classList.add('on');
  if (state[c.id]==='bad') el.querySelector('.bad').classList.add('on');
  return el;
}

function escapeHtml(s){ return (s||'').replace(/[&<>"]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m])); }

function mark(id, v, btn){
  state[id] = v; save();
  const card = btn.closest('.card'); card.classList.add('done');
  card.querySelectorAll('button').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.getElementById('st-'+id).textContent = v==='ok' ? '已标：正确' : '已标：有误';
}

function markAll(v){ DATA.forEach(c=>state[c.id]=v); save(); document.querySelectorAll('.card').forEach((el,i)=>{ el.classList.add('done'); el.querySelectorAll('button').forEach(b=>b.classList.remove('on')); el.querySelector('.'+(v==='ok'?'ok':'bad')).classList.add('on'); }); }

function updateProg(){
  document.getElementById('prog').textContent = Object.keys(state).filter(k=>state[k]).length;
}

function exportRes(){
  const lines = DATA.map(c => `${c.id},${state[c.id]||'unreviewed'}`).join('\\n');
  const out = document.getElementById('out'); out.value = lines; out.select();
  navigator.clipboard?.writeText(lines);
  document.getElementById('hint').textContent = '已复制到剪贴板，也可从下方文本框复制';
}

DATA.forEach(c => root.appendChild(card(c)));
document.getElementById('total').textContent = DATA.length;
updateProg();
</script></body></html>"""
    html = html.replace("__DATA__", data)
    (AUD / "review.html").write_text(html, encoding="utf-8")
    print(f"wrote {AUD/'review.html'} ({len(cards)} cards)")


if __name__ == "__main__":
    build()
