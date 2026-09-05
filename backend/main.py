from __future__ import annotations
import asyncio, json, os, sqlite3, time, uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

DATA = Path(os.getenv('STT_DATA_DIR', './data')); DATA.mkdir(parents=True, exist_ok=True)
DB = DATA / 'stt.sqlite3'; AUDIO = DATA / 'audio'; AUDIO.mkdir(exist_ok=True)

class Adapter(Protocol):
    name: str
    async def start(self, language: str): ...
    async def chunk(self, state, pcm: np.ndarray) -> str: ...
    async def finish(self, state) -> str: ...

class Placeholder:
    def __init__(self, name, model_id): self.name, self.model_id = name, model_id
    async def start(self, language): return {'text': '', 'started': time.perf_counter(), 'language': language}
    async def chunk(self, state, pcm):
        # Replace this adapter with the installed model runtime. The transport and UI remain unchanged.
        state['samples'] += len(pcm) if 'samples' in state else len(pcm)
        return state['text']
    async def finish(self, state): return state['text']

class QwenAdapter(Placeholder):
    def __init__(self): super().__init__('Qwen3-ASR 0.6B', 'Qwen/Qwen3-ASR-0.6B')

class NemotronAdapter(Placeholder):
    def __init__(self): super().__init__('Nemotron 3.5 ASR 0.6B', 'nvidia/nemotron-3.5-asr-streaming-0.6b')

MODELS = {'qwen3-asr-0.6b': QwenAdapter(), 'nemotron-3.5-asr-0.6b': NemotronAdapter()}

def init_db():
    with sqlite3.connect(DB) as c:
        c.execute('CREATE TABLE IF NOT EXISTS recordings (id TEXT PRIMARY KEY, created REAL, path TEXT, duration REAL)')
        c.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, recording_id TEXT, model_id TEXT, language TEXT, text TEXT, first_text_ms REAL, final_ms REAL, rtf REAL, reference TEXT, wer REAL, cer REAL)')
init_db()

@asynccontextmanager
async def lifespan(app): yield
app = FastAPI(title='Local STT Lab', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

@app.get('/health')
async def health(): return {'status':'ok', 'models': list(MODELS)}

@app.get('/models')
async def models(): return [{'id': k, 'name': v.name, 'model_id': v.model_id, 'streaming': True, 'available': True} for k,v in MODELS.items()]

@app.get('/recordings')
async def recordings():
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute('SELECT * FROM recordings ORDER BY created DESC')]

@app.post('/recordings')
async def upload_recording(audio: UploadFile = File(...)):
    rid = uuid.uuid4().hex; path = AUDIO / f'{rid}.wav'; data = await audio.read(); path.write_bytes(data)
    with sqlite3.connect(DB) as c: c.execute('INSERT INTO recordings VALUES (?,?,?,?)',(rid,time.time(),str(path),0))
    return {'id': rid}

@app.websocket('/ws')
async def ws(websocket: WebSocket):
    await websocket.accept(); state = None; adapter = None; run_id = None; first = None; started = None; samples = 0
    try:
        while True:
            msg = await websocket.receive()
            if msg.get('text'):
                cmd = json.loads(msg['text'])
                if cmd['type']=='start':
                    adapter = MODELS.get(cmd.get('model_id','qwen3-asr-0.6b'))
                    if not adapter: await websocket.send_json({'type':'error','message':'Unknown model'}); continue
                    state = await adapter.start(cmd.get('language','auto')); state['samples']=0; run_id=uuid.uuid4().hex; started=time.perf_counter(); first=None
                    await websocket.send_json({'type':'started','run_id':run_id,'model':adapter.name})
                elif cmd['type']=='stop' and adapter and state:
                    text = await adapter.finish(state); final=(time.perf_counter()-started)*1000; rtf=(final/1000)/(samples/16000) if samples else 0
                    await websocket.send_json({'type':'final','text':text,'final_ms':final,'rtf':rtf,'first_text_ms':first})
                    state=None
            elif msg.get('bytes') is not None and adapter and state:
                pcm=np.frombuffer(msg['bytes'],dtype=np.int16).astype(np.float32)/32768; samples += len(pcm)
                text=await adapter.chunk(state,pcm)
                if text and first is None: first=(time.perf_counter()-started)*1000
                await websocket.send_json({'type':'partial','text':text,'first_text_ms':first})
    except WebSocketDisconnect: return

