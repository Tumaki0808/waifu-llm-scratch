from __future__ import annotations

import re
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from tokenizers import Tokenizer

from model import Llama3Mini

app = FastAPI(title="Anime Companion")
tokenizer = Tokenizer.from_file("tokenizer.json")
model = Llama3Mini()
checkpoint = torch.load(Path("checkpoints/sft_best.pt"), map_location="cpu")
model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
model.eval()


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


def _prompt(request: ChatRequest) -> str:
    turns = request.history[-3:] + [{"role": "user", "content": request.message}]
    text = "<|bos|>"
    for turn in turns:
        role = "<|user|>" if turn.get("role") == "user" else "<|bot|>"
        text += role + turn.get("content", "")
    text += "<|bot|>"
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return tokenizer.decode(ids[-380:], skip_special_tokens=False)


@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, str]:
    ids = tokenizer.encode(_prompt(request), add_special_tokens=False).ids
    result = model.generate(torch.tensor([ids]), max_new_tokens=120)
    decoded = tokenizer.decode(result[0].tolist(), skip_special_tokens=False)
    reply = decoded.rsplit("<|bot|>", 1)[-1]
    reply = re.split(r"<\|(?:eos|user)\|>", reply, maxsplit=1)[0].strip()
    action = re.findall(r"\*([^*]+)\*", reply)
    return {"reply": reply, "emotion": action[-1] if action else "listening"}


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Anime Companion</title>
<style>body{margin:0;background:#151321;color:#fff;font:16px system-ui}main{max-width:780px;margin:4rem auto;padding:2rem;background:#211d35;border:1px solid #493866;border-radius:20px;box-shadow:0 20px 60px #0008}h1{color:#f39ac5}.status{color:#b99cff;margin-bottom:1rem}.chat{min-height:360px}.msg{padding:.7rem 1rem;margin:.6rem 0;border-radius:12px;background:#302947}.user{background:#3d3152}.action{color:#f092c8;font-style:italic}.composer{display:flex;gap:.6rem}input{flex:1;padding:1rem;border-radius:10px;border:0;background:#12101d;color:#fff}button{padding:0 1.3rem;border:0;border-radius:10px;background:#db6da6;color:white;font-weight:bold}</style></head>
<body><main><h1>✦ Anime Companion</h1><div id="status" class="status">● ready</div><div id="chat" class="chat"></div><form class="composer"><input autofocus placeholder="Say something..."><button>Send</button></form></main>
<script>const chat=document.querySelector('#chat'),status=document.querySelector('#status'),input=document.querySelector('input'),history=[];
function render(t){return t.replace(/\\*([^*]+)\\*/g,'<span class="action">*$1*</span>')}function add(role,text){const d=document.createElement('div');d.className='msg '+role;d.innerHTML=render(text);chat.append(d);chat.scrollTop=chat.scrollHeight}
document.querySelector('form').onsubmit=async e=>{e.preventDefault();let m=input.value.trim();if(!m)return;input.value='';add('user',m);const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:m,history})}).then(x=>x.json());add('bot',r.reply);status.textContent='● emotion: '+r.emotion;history.push({role:'user',content:m},{role:'assistant',content:r.reply});history.splice(0,Math.max(0,history.length-6))};</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML
