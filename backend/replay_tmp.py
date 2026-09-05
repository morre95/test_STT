"""Spela upp en sparad inspelning genom den riktiga worker-processen."""
import asyncio, sys, wave
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from stt_lab.catalog import models
from stt_lab.runtime import Runtime

async def main(model_id, wav):
    spec = models()[model_id]
    rt = Runtime(Path("data"))
    try:
        await rt.load(spec)
        await rt.start("sv", spec["defaults"])
        with wave.open(wav) as w:
            data = w.readframes(w.getnframes())
        partials = []
        for i in range(0, len(data), 3200):
            r = await rt.push(data[i:i + 3200])
            if r["text"] and (not partials or r["text"] != partials[-1]):
                partials.append(r["text"])
        final = await rt.finish()
        print("PARTIALS:", len(partials))
        for t in partials[:4]:
            print("   ", repr(t))
        print("SLUT:", repr(final["text"]), "| language:", final.get("language"))
    finally:
        await rt.close()

asyncio.run(main(sys.argv[1], sys.argv[2]))
