"""
Lightweight HTTP client for sending an image to a remote detection server.

Expected response:
- JSON list of detections, or
- JSON object containing one of these keys: boxes, bboxes, detections, predictions, results.
"""

import json
import mimetypes
import os
import uuid
from typing import Any, Dict, Optional
from urllib import error, request


class RemoteInferenceError(RuntimeError):
    """Raised when remote inference request/response fails."""


def _join_url(server_url: str, endpoint: str) -> str:
    if not server_url:
        raise RemoteInferenceError("Server URL cannot be empty.")
    endpoint = endpoint or ""
    if not endpoint:
        return server_url
    if server_url.endswith("/") and endpoint.startswith("/"):
        return server_url[:-1] + endpoint
    if (not server_url.endswith("/")) and (not endpoint.startswith("/")):
        return server_url + "/" + endpoint
    return server_url + endpoint


def _encode_multipart(image_path: str, fields: Optional[Dict[str, Any]] = None):
    boundary = f"----MIVBoundary{uuid.uuid4().hex}"
    chunks = []

    for key, value in (fields or {}).items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        chunks.append(str(value).encode("utf-8"))
        chunks.append(b"\r\n")

    filename = os.path.basename(image_path)
    guessed_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode("utf-8")
    )
    chunks.append(f"Content-Type: {guessed_type}\r\n\r\n".encode("utf-8"))
    chunks.append(image_bytes)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))

    body = b"".join(chunks)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def infer_image(
    image_path: str,
    server_url: str,
    endpoint: str = "/infer",
    timeout_seconds: float = 60.0,
    extra_fields: Optional[Dict[str, Any]] = None,
):
    """
    Send image to remote server and return parsed JSON payload.
    """
    if not image_path or not os.path.exists(image_path):
        raise RemoteInferenceError(f"Image file not found: {image_path}")

    timeout_seconds = float(timeout_seconds) if timeout_seconds else 60.0
    if timeout_seconds <= 0:
        timeout_seconds = 60.0

    url = _join_url(server_url, endpoint)
    body, content_type = _encode_multipart(image_path=image_path, fields=extra_fields)

    req = request.Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(body)))
    req.add_header("Accept", "application/json")

    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read()
            status = int(getattr(resp, "status", 200))
            if status >= 400:
                text = raw.decode("utf-8", errors="ignore")
                raise RemoteInferenceError(f"HTTP {status}: {text[:400]}")
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception as exc:
                snippet = raw[:200].decode("utf-8", errors="ignore")
                raise RemoteInferenceError(f"Server did not return valid JSON: {snippet}") from exc
    except error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = str(exc)
        raise RemoteInferenceError(f"HTTPError {exc.code}: {detail[:400]}") from exc
    except error.URLError as exc:
        raise RemoteInferenceError(f"Server connection error: {exc}") from exc
    except TimeoutError as exc:
        raise RemoteInferenceError("Server response timed out.") from exc
