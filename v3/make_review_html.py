"""Build the Chinese-language review page for the 42 hand-audited episodes.

Chinese summary first (reading aid), English transcript verbatim underneath.
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUD = ROOT / "output_v3" / "main" / "audit"
TDIR = ROOT / "output_v3" / "main" / "transcripts"

MODEL_ZH = {
    "flash": "DeepSeek-V4.1-Flash（小模型）",
    "pro": "DeepSeek-V4-Pro（大模型）",
    "qwen32b": "Qwen3-32B",
    "glm45air": "GLM-4.5-Air",
}


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
    zh_path = AUD / "summaries_zh.json"
    zh = json.loads(zh_path.read_text(encoding="utf-8")) if zh_path.exists() else {}

    cards = []
    for keyfile, labs, group in [(key_main, labels, "主审计（30 条：28 成交 + 2 未成交）"),
                                 (key_neg, labels_neg, "负例补审（12 条，全部为未成交）")]:
        for eid, meta in sorted(keyfile.items()):
            if eid not in labs:
                continue
            lab = labs[eid]
            deal = lab["deal"] == "1"
            t = json.loads((TDIR / meta["file"]).read_text(encoding="utf-8"))
            m = re.search(r"_(\d+)_e(\d+)\.json$", meta["file"])
            cards.append(dict(
                id=eid, group=group,
                model=MODEL_ZH.get(meta["model"], meta["model"]),
                trial=meta.get("trial", int(m.group(1)) if m else 0),
                episode=meta.get("episode", int(m.group(2)) if m else 0),
                label_deal=deal,
                label_split=f"A 拿 {lab['split_a']}、B 拿 {lab['split_b']}" if deal else "未成交",
                summary=zh.get(eid, ""),
                evidence=evidence(t["transcript"], focus=meta.get("det_round")),
            ))

    data = json.dumps(cards, ensure_ascii=False)
    html = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>审计复核（中文版）· 42 条自由对话</title>
<style>
  :root { --fg:#1a1a1a; --muted:#666; --line:#ddd; --bg:#fff; --ok:#1a7f37; --bad:#b42318; --blue:#0b57d0; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; margin:0; padding:24px 32px 80px; color:var(--fg); background:var(--bg); line-height:1.7; }
  h1 { font-size:21px; margin:0 0 4px; }
  .sub { color:var(--muted); font-size:13px; margin-bottom:16px; }
  .rules { border:1px solid #c9d8f0; border-radius:10px; padding:14px 18px; margin:14px 0 14px; background:#f4f8ff; font-size:14px; }
  .rules h3 { margin:0 0 8px; font-size:15px; color:#0b57d0; }
  .rules table { border-collapse:collapse; margin:8px 0 4px; font-size:13px; }
  .rules th, .rules td { border:1px solid #c9d8f0; padding:4px 8px; text-align:left; }
  .rules th { background:#e8f0ff; font-weight:600; }
  .rules code { background:#e8f0ff; padding:0 4px; border-radius:3px; }
  .criteria { border:1px solid var(--line); border-radius:10px; padding:14px 18px; margin:14px 0 20px; background:#fafafa; font-size:14px; }
  .criteria h3 { margin:0 0 8px; font-size:15px; }
  .criteria .yes { color:var(--ok); font-weight:600; }
  .criteria .no { color:var(--bad); font-weight:600; }
  .criteria ul { margin:6px 0 0 18px; padding:0; }
  .bar { position:sticky; top:0; background:var(--bg); padding:10px 0 14px; border-bottom:1px solid var(--line); z-index:5; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  .bar b { font-variant-numeric: tabular-nums; }
  button { font:inherit; padding:6px 12px; border:1px solid var(--line); background:#fff; border-radius:6px; cursor:pointer; }
  button:hover { border-color:#999; }
  button.ok.on { background:#e6f4ea; border-color:var(--ok); color:var(--ok); }
  button.bad.on { background:#fdecea; border-color:var(--bad); color:var(--bad); }
  .card { border:1px solid var(--line); border-radius:10px; padding:16px 18px; margin:16px 0; }
  .card.done { border-color:#bbb; background:#fafafa; }
  .head { display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }
  .id { font-weight:700; font-size:17px; }
  .meta { color:var(--muted); font-size:12px; }
  .zh { margin:10px 0; padding:10px 12px; background:#eef4ff; border-left:3px solid var(--blue); border-radius:0 6px 6px 0; font-size:14px; }
  .zh b { color:var(--blue); }
  .label { margin:10px 0 4px; font-size:14px; }
  .label b { font-size:15px; }
  details { margin-top:8px; }
  summary { cursor:pointer; color:var(--muted); font-size:13px; }
  .ev { margin-top:8px; border-left:3px solid var(--line); padding-left:12px; }
  .r { margin:10px 0; }
  .r .rh { font-size:12px; color:var(--muted); }
  .m { margin:3px 0; font-size:13.5px; color:#333; }
  .m span { font-weight:600; }
  .acts { margin-top:12px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  .note { font-size:12px; color:var(--muted); }
  textarea { width:100%; height:110px; margin-top:14px; font-family: ui-monospace, Consolas, monospace; font-size:12px; }
</style></head><body>

<h1>审计复核（中文版）：42 条自由对话</h1>
<div class="sub">英文原文原样保留在每条下面，可展开核对。判定标准见下。</div>

<div class="rules">
  <h3>这个游戏是什么（先看这里）</h3>
  <div>桌上有一堆筹码，叫“<strong>饼</strong>”（英文 pie，不是吃的）。两个 AI 代理 A 和 B 谈怎么分：</div>
  <ul style="margin:6px 0 0 18px;padding:0;">
    <li>第 1 轮饼 = <strong>10 分</strong>；每过一轮缩 10%，即 10 → 9 → 8 → 7 → 7 → 6 → 5 → 5 → 4 → 4。</li>
    <li>第 r 轮谈成，就分第 r 轮剩下的那些分；<strong>第 10 轮还没谈成，两人都是 0</strong>。</li>
    <li>每个人都只想让自己拿得多，不在乎对方。</li>
  </ul>
  <div style="margin-top:8px;">E01 的实际过程：</div>
  <table>
    <tr><th>轮次</th><th>饼</th><th>A 说</th><th>B 说</th></tr>
    <tr><td>第 8 轮</td><td>5 分</td><td>我拿 3，你拿 2</td><td>拒绝，反手“我拿 3，你拿 2”</td></tr>
    <tr><td>第 9 轮</td><td>4 分</td><td>2 比 2，平分</td><td><strong>接受</strong></td></tr>
    <tr><td>第 10 轮</td><td>4 分</td><td>还是 2 比 2</td><td><strong>再次接受</strong></td></tr>
  </table>
  <div>→ 最终 A 拿 2、B 拿 2，4 分发完，判“成交”。</div>
  <div style="margin-top:6px;color:#555;">为什么第 10 轮又重复一遍？因为“自由对话”协议<strong>没有终止规则</strong>，第 9 轮谈成了也不会自动停，会一直说到第 10 轮。论文里这正是自由对话效率低的原因：成交被拖到最后一轮，饼已经缩掉大半。</div>
</div>

<div class="criteria">
  <h3>你要判断的一句话</h3>
  <div>在<strong>同一轮</strong>里，双方是否都明确接受了<strong>同一个分割</strong>？</div>
  <ul>
    <li><span class="yes">算成交</span>：A 说“我接受 2-2”，B 说“I accept 2-2 / deal”。双方接受的数字一致。</li>
    <li><span class="no">不算</span>：一方提议 4-0，另一方说“I reject 4-0”——没人接受对方的方案。</li>
    <li><span class="no">不算</span>：A 提议 2-2，B 说“可以，但我要 3”——接受的不是同一个数字。</li>
    <li><span class="no">不算</span>：一方说“我接受”但接受的是自己提的方案，对方从未同意。</li>
  </ul>
  <div style="margin-top:8px;color:#666;font-size:13px;">注：每轮的饼会缩水（第 r 轮 = 10 × 0.9<sup>r-1</sup>，四舍五入）。所以第 9 轮的 2-2 是 4 分饼的均分，第 1 轮的 5-5 是 10 分饼的均分。</div>
</div>

<div class="bar">
  <span>进度：<b id="prog">0</b> / <b id="total">0</b></span>
  <button onclick="markAll('ok')">全部标为正确</button>
  <button onclick="exportRes()">导出结果</button>
  <span class="note" id="hint"></span>
</div>

<div id="cards"></div>
<textarea id="out" placeholder="点“导出结果”后，结果会出现在这里，同时复制到剪贴板"></textarea>

<script>
const DATA = __DATA__;
const KEY = 'audit-review-zh-v1';
let state = JSON.parse(localStorage.getItem(KEY) || '{}');
const root = document.getElementById('cards');

function save(){ localStorage.setItem(KEY, JSON.stringify(state)); updateProg(); }
function escapeHtml(s){ return (s||'').replace(/[&<>"]/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m])); }

function card(c){
  const el = document.createElement('div');
  el.className = 'card' + (state[c.id] ? ' done' : '');
  const lab = c.label_deal ? `成交（${c.label_split}）` : '未成交';
  let ev = '';
  for (const r of c.evidence){
    ev += `<div class="r"><div class="rh">第 ${r.round} 轮 · 饼 ${r.pie} 分</div>
      <div class="m"><span>A：</span>${escapeHtml(r.a)}</div>
      <div class="m"><span>B：</span>${escapeHtml(r.b)}</div></div>`;
  }
  el.innerHTML = `<div class="head"><span class="id">${c.id}</span>
      <span class="meta">${c.model} · 第 ${c.trial} 组 · 第 ${c.episode} 局</span></div>
    <div class="zh"><b>中文复述：</b>${escapeHtml(c.summary || '（无摘要，请直接看英文原文）')}</div>
    <div class="label">待你复核的判定：<b>${lab}</b></div>
    <details><summary>展开英文原文（最后几轮）</summary><div class="ev">${ev}</div></details>
    <div class="acts">
      <button class="ok" onclick="mark('${c.id}','ok',this)">✓ 判定正确</button>
      <button class="bad" onclick="mark('${c.id}','bad',this)">✗ 判定有误</button>
      <span class="note" id="st-${c.id}">${state[c.id] ? (state[c.id]==='ok'?'已标：正确':'已标：有误') : '未复核'}</span>
    </div>`;
  if (state[c.id]==='ok') el.querySelector('.ok').classList.add('on');
  if (state[c.id]==='bad') el.querySelector('.bad').classList.add('on');
  return el;
}

function mark(id, v, btn){
  state[id] = v; save();
  const card = btn.closest('.card'); card.classList.add('done');
  card.querySelectorAll('button').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.getElementById('st-'+id).textContent = v==='ok' ? '已标：正确' : '已标：有误';
}

function markAll(v){
  DATA.forEach(c=>state[c.id]=v); save();
  document.querySelectorAll('.card').forEach(el=>{
    el.classList.add('done');
    el.querySelectorAll('button').forEach(b=>b.classList.remove('on'));
    el.querySelector('.'+(v==='ok'?'ok':'bad')).classList.add('on');
  });
}

function updateProg(){ document.getElementById('prog').textContent = Object.keys(state).filter(k=>state[k]).length; }

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
    (AUD / "review_zh.html").write_text(html, encoding="utf-8")
    print(f"wrote {AUD/'review_zh.html'} ({len(cards)} cards, {len(zh)} summaries)")


if __name__ == "__main__":
    build()
