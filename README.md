# Local STT Lab

En liten testmiljö för att jämföra lokala tal-till-text-modeller från en Android-telefon mot en FastAPI-server. Backendens adaptergränssnitt är avsiktligt neutralt så Qwen3-ASR, Nemotron, Vosk och framtida molnleverantörer kan använda samma WebSocket-protokoll.

## Installera backend och modeller

```bash
cd backend
uv sync
uv venv --python 3.12 .venv-qwen
uv pip install --python .venv-qwen/bin/python -r requirements-qwen.txt
uv venv --python 3.12 .venv-nemotron
uv pip install --python .venv-nemotron/bin/python -r requirements-nemotron.txt
uv venv --python 3.12 .venv-vosk
uv pip install --python .venv-vosk/bin/python -r requirements-vosk.txt
```

Kontrollera att GPU-miljöerna ser CUDA (Vosk kör på CPU och behöver ingen):

```bash
.venv-qwen/bin/python -c "import torch; print(torch.cuda.is_available())"
.venv-nemotron/bin/python -c "import torch; print(torch.cuda.is_available())"
```

Starta sedan servern från `backend/`:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Modellvikterna hämtas första gången modellen väljs: Qwen och Nemotron från Hugging Face, Vosk från alphacephei.com till `~/.cache/vosk`. Qwen använder officiell vLLM-streaming, Nemotron använder NeMos cache-aware RNNT-streaming och Vosk streamar via Kaldi på CPU. Endast en modell är laddad åt gången.

Varje Vosk-checkpoint innehåller en språkspecifik avkodningsgraf, så `vosk-sv` accepterar bara `sv` och `vosk-en` bara `en`. Modellerna i katalogen anger sina språk i `/models`, och servern avvisar en körning med fel språk. Vosk har inga strömningsinställningar; `settings` måste vara tomt.

Verifiera modellkedjan utan telefon med två sekunders tyst PCM:

```bash
uv run python smoke_model.py qwen-0.6b
uv run python smoke_model.py nemotron-0.6b
uv run python smoke_model.py vosk-sv
```

## Android

Kör `flutter pub get` och `flutter run` i `app/`. Standardadressen `10.0.2.2:8000` gäller Android-emulatorn. På en fysisk telefon anger du datorns LAN-adress, exempelvis `192.168.1.20:8000`.

Modell- och språklistorna hämtas från serverns `/models`, så appen behöver inte ändras när katalogen växer. Adressfältet läser om listan när du trycker enter eller på uppdateringsikonen. Modeller vars runtime saknas visas som *ej installerad* och går inte att välja, och språkvalet begränsas till det modellen stöder.
