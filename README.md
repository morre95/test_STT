# Local STT Lab

En liten testmiljö för att jämföra lokala tal-till-text-modeller från en Android-telefon mot en FastAPI-server. Backendens adaptergränssnitt är avsiktligt neutralt så Qwen3-ASR, Nemotron och framtida molnleverantörer kan använda samma WebSocket-protokoll.

## Starta backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Modelladaptrarna i denna första scaffold är transportklara platshållare. Installera Qwen- och NeMo-runtime och fyll `QwenAdapter`/`NemotronAdapter` med faktisk inferens enligt modellernas officiella exempel innan mätningar används.

