#!/usr/bin/env python3
"""Offline HTML: automatic candidate overlays/crops and original QA, no server."""
import argparse
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mpp_video_qa.core import load_jsonl
from mpp_video_qa.region_views import validate_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--views", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=20)
    a = p.parse_args()
    rows = load_jsonl(a.data)
    plans = {r["question_id"]: r for r in load_jsonl(a.views)}
    # Mix targeted and full-only plans, including different question groups.
    ordered = sorted(rows, key=lambda r: (tuple(plans[r["question_id"]]["focus_groups"]), r["question_id"]))
    indices = sorted({round(i * (len(ordered) - 1) / max(1, a.count - 1)) for i in range(min(a.count, len(ordered)))})
    assets, cases = {}, []
    for i in indices:
        row = ordered[i]
        plan = plans[row["question_id"]]
        validate_plan(plan, row)
        for path in row["frames"]:
            if path not in assets:
                mime = "image/png" if Path(path).suffix.lower() == ".png" else "image/jpeg"
                assets[path] = "data:" + mime + ";base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")
        cases.append({"row": row, "plan": plan})
    data = json.dumps({"assets": assets, "cases": cases}, ensure_ascii=False).replace("<", "\\u003c")
    html = """<!doctype html><meta charset="utf-8"><title>Automatic region review</title>
<style>body{font:16px system-ui;margin:24px;background:#f4f6f8}section{background:white;padding:16px;margin:20px 0;border-radius:8px}.frames{display:flex;gap:10px;overflow:auto}.frame{min-width:300px;width:300px}canvas{width:100%;cursor:zoom-in}.crops{display:flex;gap:5px}.crops canvas{max-width:145px}small{color:#555}button{margin:8px}dialog{width:90vw}dialog img{max-width:100%}</style>
<h1>自动候选局部观察检查</h1><p>绿色框是程序候选，不代表已确认问题目标。按 T0–T4 保留原时间顺序；点击图查看原分辨率。参考答案仅用于此检查页面，不参与选区。</p>
<main id="main"></main><dialog id="zoom"><button onclick="zoom.close()">关闭</button><img id="large"></dialog>
<script id="data" type="application/json">DATA</script><script>
const data=JSON.parse(document.getElementById('data').textContent), root=document.getElementById('main');
for(const {row,plan} of data.cases){const s=document.createElement('section');root.append(s); const h=document.createElement('h3');h.textContent=row.question;s.append(h);const r=document.createElement('p');r.textContent='Reference: '+row.answers.join(' | ');s.append(r);const t=document.createElement('small');t.textContent=row.question_id+' · '+row.source_subset+' · '+plan.focus_groups.join(', ')+(plan.full_replay?' · full replay':'');s.append(t);const grid=document.createElement('div');grid.className='frames';s.append(grid);
row.frames.forEach((path,index)=>{const f=document.createElement('div');f.className='frame';grid.append(f);const text=document.createElement('p');text.textContent='T'+index;f.append(text);const canvas=document.createElement('canvas');f.append(canvas);const crops=document.createElement('div');crops.className='crops';f.append(crops);const image=new Image();image.onload=()=>{canvas.width=image.width;canvas.height=image.height;const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);ctx.lineWidth=Math.max(3,image.width/200);ctx.strokeStyle='#00ff70';for(const v of plan.views.filter(v=>v.kind==='region'&&v.frame_index===index)){const b=v.pixel_box;ctx.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);const c=document.createElement('canvas');c.width=b[2]-b[0];c.height=b[3]-b[1];c.getContext('2d').drawImage(image,b[0],b[1],c.width,c.height,0,0,c.width,c.height);crops.append(c);c.onclick=()=>{document.getElementById('large').src=c.toDataURL();document.getElementById('zoom').showModal();}}};canvas.onclick=()=>{document.getElementById('large').src=canvas.toDataURL();document.getElementById('zoom').showModal();};image.src=data.assets[path];});}
</script>""".replace("DATA", data)
    destination = Path(a.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html, encoding="utf-8")
    print(json.dumps({"html": str(destination.resolve()), "cases": len(cases)}))


if __name__ == "__main__":
    main()
