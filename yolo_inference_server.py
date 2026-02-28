"""
Local YOLO inference HTTP server.

Runs on 127.0.0.1:5000 by default and accepts:
- POST /infer (multipart/form-data with field name: image)
"""

import os
from io import BytesIO

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_IMPORT_ERROR = None
except Exception as import_exc:
    YOLO = None
    YOLO_IMPORT_ERROR = import_exc


def resolve_model_path() -> str:
    """
    Resolve model path for local runs.
    Priority:
    1) YOLO_MODEL_PATH env value
    2) best.pt
    3) best.2.pt
    4) best (2).pt (common Windows auto-rename)
    """
    env_path = os.getenv("YOLO_MODEL_PATH")
    if env_path:
        return env_path

    preferred = "best.pt"
    if os.path.exists(preferred):
        return preferred

    legacy_best_dot = "best.2.pt"
    if os.path.exists(legacy_best_dot):
        return legacy_best_dot

    windows_alt = "best (2).pt"
    if os.path.exists(windows_alt):
        return windows_alt

    return preferred


MODEL_PATH = resolve_model_path()
HOST = os.getenv("YOLO_HOST", "127.0.0.1")
PORT = int(os.getenv("YOLO_PORT", "5000"))
IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
CONF = float(os.getenv("YOLO_CONF", "0.25"))

app = Flask(__name__)
model = None
MODEL_ERROR = None

if YOLO is None:
    MODEL_ERROR = f"ultralytics import failed: {YOLO_IMPORT_ERROR}"
else:
    try:
        model = YOLO(MODEL_PATH)
    except Exception as model_exc:
        MODEL_ERROR = f"model load failed for '{MODEL_PATH}': {model_exc}"


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok" if model is not None else "degraded",
            "model": MODEL_PATH,
            "model_ready": model is not None,
            "detail": MODEL_ERROR,
        }
    )


@app.post("/infer")
def infer():
    if model is None:
        return jsonify(
            {
                "error": "YOLO model is not ready",
                "detail": MODEL_ERROR,
                "detections": [],
            }
        ), 503

    file_obj = request.files.get("image")
    if file_obj is None:
        return jsonify({"error": "form-data field 'image' is required"}), 400

    image_bytes = file_obj.read()
    if not image_bytes:
        return jsonify({"error": "image payload is empty"}), 400

    try:
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        return jsonify({"error": "invalid image file", "detail": str(exc)}), 400

    results = model.predict(pil_image, save=False, imgsz=IMGSZ, conf=CONF, verbose=False)
    detections = []

    if results and len(results) > 0:
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        names = getattr(r0, "names", {}) or {}
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = (
                boxes.conf.cpu().numpy()
                if getattr(boxes, "conf", None) is not None
                else np.zeros((len(xyxy),), dtype=float)
            )
            class_ids = (
                boxes.cls.cpu().numpy()
                if getattr(boxes, "cls", None) is not None
                else np.zeros((len(xyxy),), dtype=float)
            )

            for idx, bbox in enumerate(xyxy):
                x1, y1, x2, y2 = [float(v) for v in bbox]
                class_id = int(class_ids[idx]) if idx < len(class_ids) else None
                class_name = names.get(class_id) if class_id is not None else None
                confidence = float(confs[idx]) if idx < len(confs) else None
                detections.append(
                    {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                    }
                )

    return jsonify({"detections": detections})


if __name__ == "__main__":
    print(
        f"Starting YOLO inference server on http://{HOST}:{PORT} "
        f"with model '{MODEL_PATH}'"
    )
    app.run(host=HOST, port=PORT, debug=False)
