from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import base64
from image_processor import restore
from evaluation import evaluate

app = FastAPI(title="ReviveAI DIP Service", version="1.0")


def decode(data: bytes):
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded content is not a decodable image")
    if image.shape[0] * image.shape[1] > 25_000_000:
        raise ValueError("Image dimensions exceed the 25 megapixel processing limit")
    return image


def encode(image):
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("Could not encode processing result")
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


@app.get("/health")
def health():
    return {"status": "ok", "service": "dip-service"}


@app.post("/process")
async def process(image: UploadFile = File(...), options: str = Form("{}")):
    try:
        decoded = decode(await image.read())
        stages, duration, operations = restore(decoded, json.loads(options))
        final = stages["final"]
        return {"stages": {name: encode(img) for name, img in stages.items()},
                "durationMs": duration, "operations": operations,
                "dimensions": {"width": decoded.shape[1], "height": decoded.shape[0]},
                "outputDimensions": {"width": final.shape[1], "height": final.shape[0]}}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/evaluate")
async def evaluate_endpoint(reference: UploadFile = File(...), restored: UploadFile = File(...)):
    try:
        result = evaluate(decode(await reference.read()), decode(await restored.read()))
        return JSONResponse(result, allow_nan=False)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
