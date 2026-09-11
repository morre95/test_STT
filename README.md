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
uv venv --python 3.12 .venv-speaker
uv pip install --python .venv-speaker/bin/python -r requirements-speaker.txt
mkdir -p runtime-models
git clone https://github.com/modelscope/3D-Speaker.git runtime-models/3D-Speaker
git -C runtime-models/3D-Speaker checkout 065629c313eaf1a01c65c640c46d77e61e9607b4
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

Qwen anpassar automatiskt vLLM:s minnesbudget efter ledigt VRAM och lämnar
1 GiB marginal för skrivbordet. På en dedikerad GPU kan gränserna styras med
`STT_QWEN_GPU_MEMORY_UTILIZATION` (andel av totalt VRAM, högst `1`) och
`STT_QWEN_GPU_HEADROOM_GIB`.

Talarbenchmarken använder en separat miljö eftersom PyTorch, ModelScope,
3D-Speaker, Silero VAD och ONNX Runtime är betydligt tyngre än API-servern.
Installera en CUDA-matchad `torch`/`torchaudio`-kombination i
`.venv-speaker` före requirements-filen om GPU ska användas. Annars körs
modellerna på CPU.

Varje Vosk-checkpoint innehåller en språkspecifik avkodningsgraf, så `vosk-sv` accepterar bara `sv` och `vosk-en` bara `en`. Modellerna i katalogen anger sina språk i `/models`, och servern avvisar en körning med fel språk. Vosk har inga strömningsinställningar; `settings` måste vara tomt.

Verifiera modellkedjan utan telefon med två sekunders tyst PCM:

```bash
uv run python smoke_model.py qwen-0.6b
uv run python smoke_model.py nemotron-0.6b
uv run python smoke_model.py vosk-sv
```

Verifiera en riktig talarmodell mot en 16 kHz-fil:

```bash
uv run python smoke_speaker.py campplus /sökväg/möte.wav
uv run python smoke_speaker.py eres2net /sökväg/möte.wav
uv run python smoke_speaker.py eres2netv2 /sökväg/möte.wav
uv run python smoke_speaker.py wespeaker-resnet34-lm /sökväg/möte.wav
```

Första körningen hämtar checkpointen och kan därför ta tid. 3D-Speaker-
revisionerna är låsta i modellkatalogen och checkout-kommandot ovan låser
källkoden till en verifierad commit. Sätt `STT_3D_SPEAKER_ROOT` om
checkouten ligger någon annanstans. WeSpeakers officiella
`voxceleb_resnet34_LM.onnx` hämtas från projektets modellarkiv.

## Talarbenchmark

Appens flik **Profiler** används för att registrera kända personer. Automatisk
gräns mot *Okänd* kräver minst två personer och två separata röstprov per
person. Kalibreringen använder bara profilklippen; mötesfacit läcker aldrig in
i tröskeln.

I fliken **Talare**:

1. välj en befintlig mobilinspelning eller ladda upp WAV/FLAC,
2. lägg till facittalare och tidssegment med text,
3. välj embeddingmodeller och en fast STT-modell,
4. kör benchmarken och följ progressen.

Alla embeddingmodeller får samma Silero VAD, 1,5 s-fönster, 0,75 s-steg och
deterministisk spektral klustring. Resultaten visar DER med 250 ms collar både
med och utan överlapp, JER, missat/falskt/förväxlat tal, talarantalets fel,
cpWER, namnmedveten SA-WER, identitetsmått och RTF. V1 väljer en primär talare
åt gången; överlappande facit tillåts och syns därför i DER med överlapp.

Uppladdningar normaliseras till mono PCM16 16 kHz. Mötesfiler får vara högst
60 minuter/200 MiB och profilklipp 2–60 sekunder. Benchmarkjobb fortsätter om
appen kopplas ned, kan avbrytas och sparas i den lokala SQLite-databasen.

3D-Speakers kod är Apache-2.0. WeSpeakers VoxCeleb-vikt följer
modell-/datasetvillkoren som WeSpeaker dokumenterar som CC BY 4.0. Kontrollera
respektive modellkort innan resultat eller vikter distribueras.

## Android

Kör `flutter pub get` och `flutter run` i `app/`. Standardadressen `10.0.2.2:8000` gäller Android-emulatorn. På en fysisk telefon anger du datorns LAN-adress, exempelvis `192.168.1.20:8000`.

Modell- och språklistorna hämtas från serverns `/models`, så appen behöver inte ändras när katalogen växer. Adressfältet läser om listan när du trycker enter eller på uppdateringsikonen. Modeller vars runtime saknas visas som *ej installerad* och går inte att välja, och språkvalet begränsas till det modellen stöder.
