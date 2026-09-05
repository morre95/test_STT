# Local STT Lab

En liten testmiljö för att jämföra lokala tal-till-text-modeller från en Android-telefon mot en FastAPI-server. Backendens adaptergränssnitt är avsiktligt neutralt så Qwen3-ASR, Nemotron och framtida molnleverantörer kan använda samma WebSocket-protokoll.

## Installera backend och modeller

```bash
cd backend
uv sync
uv venv --python 3.12 .venv-qwen
uv pip install --python .venv-qwen/bin/python -r requirements-qwen.txt
uv venv --python 3.12 .venv-nemotron
uv pip install --python .venv-nemotron/bin/python -r requirements-nemotron.txt
```

Kontrollera att båda miljöerna ser CUDA:

```bash
.venv-qwen/bin/python -c "import torch; print(torch.cuda.is_available())"
.venv-nemotron/bin/python -c "import torch; print(torch.cuda.is_available())"
```

Starta sedan servern från `backend/`:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Modellvikterna hämtas från Hugging Face första gången modellen väljs. Qwen använder officiell vLLM-streaming och Nemotron använder NeMos cache-aware RNNT-streaming. Endast en modell ligger i GPU-minnet åt gången.

Verifiera modellkedjan utan telefon med två sekunders tyst PCM:

```bash
uv run python smoke_model.py qwen-0.6b
uv run python smoke_model.py nemotron-0.6b
```

## Android

Kör `flutter pub get` och `flutter run` i `app/`. Standardadressen `10.0.2.2:8000` gäller Android-emulatorn. På en fysisk telefon anger du datorns LAN-adress, exempelvis `192.168.1.20:8000`.
