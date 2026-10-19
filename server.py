from __future__ import annotations

import re
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from tokenizers import Tokenizer
from model import Llama3Mini, ModelConfig

app = FastAPI(title="Aoi Companion LLM")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file("tokenizer.json")

ckpt = torch.load("checkpoints/sft_best.pt", map_location=device)
cfg = ModelConfig(vocab_size=tokenizer.get_vocab_size())
model = Llama3Mini(cfg).to(device)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()


class ChatRequest(BaseModel):
    history: list[dict[str, str]]
    message: str


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    prompt = "<|bos|>"
    for turn in req.history[-3:]:
        prompt += f"<|user|>{turn['user']}<|bot|>{turn['bot']}<|eos|>"
    prompt += f"<|user|>{req.message.strip()}<|bot|>"

    ids = tokenizer.encode(prompt, add_special_tokens=False).ids[-380:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=110, temperature=0.65, top_p=0.90, repetition_penalty=1.18)
    reply_ids = out[0, len(ids) :].tolist()
    reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

    actions = re.findall(r"\*([^*]+)\*", reply)
    mood = actions[0] if actions else "chatting happily"
    return {"reply": reply, "mood": mood}


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Aoi — Custom 42M Companion LLM</title>
  <style>
    body { background: #0f101a; color: #f1f5f9; font-family: system-ui, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
    .card { width: 92%; max-width: 640px; height: 86vh; background: #181a29; border: 1px solid #2e324b; border-radius: 16px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 20px 50px rgba(0,0,0,0.5); }
    .header { padding: 16px 20px; background: #1f2235; border-bottom: 1px solid #2e324b; display: flex; justify-content: space-between; align-items: center; }
    .badge { background: #ec489933; color: #f472b6; border: 1px solid #ec489966; padding: 4px 12px; border-radius: 999px; font-size: 13px; }
    #chat { flex: 1; padding: 20px; overflow-y: auto; display: flex; flex-direction: column; gap: 14px; }
    .msg { max-width: 80%; padding: 12px 16px; border-radius: 14px; line-height: 1.5; font-size: 15px; }
    .user { align-self: flex-end; background: #4f46e5; color: white; border-bottom-right-radius: 4px; }
    .bot { align-self: flex-start; background: #25283d; border: 1px solid #363a57; border-bottom-left-radius: 4px; }
    .emote { color: #f472b6; font-style: italic; display: block; margin-bottom: 4px; }
    .input-bar { padding: 14px; background: #1f2235; display: flex; gap: 10px; }
    input { flex: 1; background: #121420; border: 1px solid #363a57; color: white; padding: 12px 14px; border-radius: 10px; font-size: 15px; outline: none; }
    button { background: #ec4899; color: white; border: none; padding: 0 20px; border-radius: 10px; font-weight: 600; cursor: pointer; }
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <div><strong>Aoi</strong> <span style="color:#94a3b8;font-size:13px;">(42M Custom LLM)</span></div>
      <div class="badge" id="mood">Status: Ready~</div>
    </div>
    <div id="chat"></div>
    <div class="input-bar">
      <input id="inp" placeholder="Say something to Aoi..." onkeydown="if(event.key==='Enter')send()"/>
      <button onclick="send()">Send</button>
    </div>
  </div>
  <script>
    const history = [];
    const chat = document.getElementById('chat');
    function formatReply(text) {
      return text.replace(/\\*([^*]+)\\*/g, '<span class="emote">*$1*</span>');
    }
    async function send() {
      const inp = document.getElementById('inp');
      const msg = inp.value.trim();
      if (!msg) return;
      inp.value = '';
      chat.innerHTML += `<div class="msg user">${msg}</div>`;
      chat.scrollTop = chat.scrollHeight;
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({history, message: msg})
      });
      const data = await res.json();
      history.push({user: msg, bot: data.reply});
      document.getElementById('mood').innerText = 'Action: *' + data.mood + '*';
      chat.innerHTML += `<div class="msg bot">${formatReply(data.reply)}</div>`;
      chat.scrollTop = chat.scrollHeight;
    }
  </script>
</body>
</html>"""
