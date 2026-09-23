"""HTTP service: one Qwen3.5-9B replica per GPU, requests go to whichever replica is free.

    JEV_PHOTO_DEVICES=cuda:0,cuda:1 jev-photo-server --port 8790
"""
import argparse
import asyncio
import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .jev.schema import Request

STATIC = Path(__file__).resolve().parent / "static"


class EditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str = Field(min_length=1)
    instruction: str = Field(default="", max_length=2000)
    mode: Literal["shared", "independent"] = "shared"


class Worker:
    """Owns one Engine; all CUDA work for it runs on its single thread."""

    def __init__(self, device, model_path):
        from .jev import Engine
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"jev-{device}")
        self.engine = self.executor.submit(Engine, model_path, device).result()
        self.executor.submit(self.warmup).result()

    def warmup(self):
        """Compile the Triton kernels before the first real request."""
        from PIL import Image
        from .edit.pipeline import edit
        edit(self.engine, Image.new("RGB", (256, 192), (120, 110, 100)), "make it brighter")

    async def run(self, fn, *args, **kwargs):
        return await asyncio.get_running_loop().run_in_executor(self.executor, partial(fn, *args, **kwargs))


class Pool:
    def __init__(self, workers):
        self.workers = workers
        self.free = asyncio.Queue()
        for worker in workers:
            self.free.put_nowait(worker)

    async def run(self, fn, *args, **kwargs):
        worker = await self.free.get()
        try:
            return await worker.run(fn, worker.engine, *args, **kwargs)
        finally:
            self.free.put_nowait(worker)


@asynccontextmanager
async def lifespan(app):
    devices = [d.strip() for d in os.environ.get("JEV_PHOTO_DEVICES", "cuda:0,cuda:1").split(",") if d.strip()]
    model_path = os.environ.get("JEV_PHOTO_MODEL_PATH")
    loop = asyncio.get_running_loop()
    # Sequential loading: transformers' lazy imports and from_pretrained are not thread-safe.
    workers = [await loop.run_in_executor(None, Worker, d, model_path) for d in devices]
    app.state.pool = Pool(workers)
    yield
    for worker in workers:
        worker.executor.shutdown()


app = FastAPI(title="Jev Photo Control (local Qwen3.5-9B)", lifespan=lifespan)


@app.get("/health")
def health():
    pool = getattr(app.state, "pool", None)
    return {"ready": pool is not None, "devices": [w.device for w in pool.workers] if pool else [],
            "model": pool.workers[0].engine.model_source if pool else None, "calibrated": False}


def _judge(engine, request):
    return engine.judge(request, allow_path=False)


@app.post("/v1/judge")
async def judge(request: Request):
    try:
        return await app.state.pool.run(_judge, request)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=93)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _edit(engine, body: EditRequest):
    from .edit.pipeline import MAX_OUTPUT_SIDE, edit
    from .jev.preprocessing import read_image
    image = read_image(body.image, allow_path=False, max_side=MAX_OUTPUT_SIDE)
    result = edit(engine, image, body.instruction, mode=body.mode)
    out = result["image"]
    steps = result["steps"]
    return {
        "image": data_url(out), "width": out.width, "height": out.height,
        "request": result["request"], "auto": result["auto"], "params": result["params"].dict(),
        "gates": steps["gates"],
        "amounts": {k: {key: v[key] for key in ("type", "value", "written", "choice", "probabilities", "trace")
                        if key in v} for k, v in steps["amounts"].items()},
        "boxes": {k: {"bbox_2d": v["bbox_2d"], "box": v["box"]} for k, v in steps.get("boxes", {}).items()},
        "metrics": result["metrics"],
    }


@app.post("/v1/edit")
async def edit_endpoint(body: EditRequest):
    try:
        return await app.state.pool.run(_edit, body)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
