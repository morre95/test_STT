"""Load one model and run two seconds of silent PCM through its streaming API."""
import argparse
import asyncio
from pathlib import Path

from stt_lab.catalog import models
from stt_lab.runtime import Runtime


async def smoke(model_id: str) -> None:
    catalog = models()
    if model_id not in catalog:
        raise SystemExit(f"Unknown model {model_id!r}; choose one of: {', '.join(catalog)}")
    spec = catalog[model_id]
    runtime = Runtime(Path(__file__).with_name("data"))
    try:
        metadata = await runtime.load(spec)
        await runtime.start(spec["languages"][0], spec["defaults"])
        for _ in range(20):
            await runtime.push(bytes(3200))
        result = await runtime.finish()
        print({"model": model_id, "runtime": metadata, "result": result})
    finally:
        await runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models())
    asyncio.run(smoke(parser.parse_args().model))
