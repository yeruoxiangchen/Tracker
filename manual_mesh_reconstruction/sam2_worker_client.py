"""Lifecycle-safe client for the persistent SAM2 Tiny video worker."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


WORKER_FORMAT = "manual_mesh_reconstruction.sam2_tiny_video_worker.v1"
DEFAULT_SAM2_PYTHON = Path(
    "/home/zjr/anaconda3/envs/any6d_sam3d/bin/python"
)


class Sam2TinyVideoWorkerClient:
    def __init__(
        self,
        *,
        python: Path = DEFAULT_SAM2_PYTHON,
        host: str = "127.0.0.1",
        port: int = 5091,
        log_path: Path,
        startup_timeout_seconds: float = 90.0,
    ) -> None:
        self.python = Path(python).expanduser().resolve()
        self.host = str(host)
        self.port = int(port)
        self.log_path = Path(log_path).expanduser().resolve()
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._log_handle = None
        self._prewarm_thread: threading.Thread | None = None
        # Local worker traffic must never inherit HTTP(S)_PROXY.  On this
        # machine urllib's environment proxy bypass does not reliably exempt
        # 127.0.0.1, which made a healthy worker look like a 90-second startup
        # timeout.  A dedicated no-proxy opener is deterministic.
        self._opener = build_opener(ProxyHandler({}))
        atexit.register(self.close)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _get_json(self, path: str, timeout: float) -> dict[str, Any]:
        with self._opener.open(self.base_url + path, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def health(self, *, timeout: float = 1.0) -> dict[str, Any] | None:
        try:
            payload = self._get_json("/health", timeout)
        except (OSError, HTTPError, URLError, TimeoutError, ValueError):
            return None
        if payload.get("format") != WORKER_FORMAT or payload.get("status") != "ok":
            return None
        return payload

    def ensure_running(self) -> dict[str, Any]:
        with self._lock:
            health = self.health()
            if health is not None and health.get("model_loaded") is True:
                return health
            # A worker that is still binding its socket or answering a previous
            # segmentation can briefly miss the one-second probe.  Give an
            # already-owned/externally-owned port a short adoption window before
            # spawning, otherwise two callers can race and the second process
            # exits with EADDRINUSE even though the first worker is healthy.
            for _ in range(8):
                time.sleep(0.25)
                health = self.health(timeout=1.0)
                if health is not None and health.get("model_loaded") is True:
                    return health
            if not self.python.is_file():
                raise FileNotFoundError(
                    f"SAM2 Tiny environment Python is missing: {self.python}"
                )
            if self._process is None or self._process.poll() is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log_handle = self.log_path.open("ab", buffering=0)
                environment = os.environ.copy()
                environment.setdefault("PYTHONUNBUFFERED", "1")
                local_hosts = "127.0.0.1,localhost,::1"
                environment["NO_PROXY"] = local_hosts
                environment["no_proxy"] = local_hosts
                self._process = subprocess.Popen(
                    [
                        str(self.python),
                        "-u",
                        "-m",
                        "manual_mesh_reconstruction.sam2_tiny_video_worker",
                        "--host",
                        self.host,
                        "--port",
                        str(self.port),
                        "--preload",
                    ],
                    cwd=str(Path(__file__).resolve().parents[1]),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )

            deadline = time.monotonic() + self.startup_timeout_seconds
            last_health = None
            while time.monotonic() < deadline:
                if self._process is not None and self._process.poll() is not None:
                    # A concurrently launched valid worker may have won the port
                    # race.  Adopt it when its immutable health contract matches.
                    adopted = self.health(timeout=2.0)
                    if adopted is not None and adopted.get("model_loaded") is True:
                        self._process = None
                        return adopted
                    raise RuntimeError(
                        "persistent SAM2 Tiny worker exited during startup; "
                        f"inspect {self.log_path}"
                    )
                last_health = self.health(timeout=1.0)
                if last_health is not None and last_health.get("model_loaded") is True:
                    return last_health
                time.sleep(0.25)
            raise TimeoutError(
                "persistent SAM2 Tiny worker did not become ready within "
                f"{self.startup_timeout_seconds:.1f}s; last_health={last_health}; "
                f"inspect {self.log_path}"
            )

    def prewarm_async(self) -> None:
        with self._lock:
            if self._prewarm_thread is not None and self._prewarm_thread.is_alive():
                return

            def target() -> None:
                try:
                    self.ensure_running()
                except Exception as error:
                    print(f">>> [SAM2 Tiny] 常驻 worker 预热失败: {error}", flush=True)

            self._prewarm_thread = threading.Thread(
                target=target,
                name="sam2-tiny-prewarm",
                daemon=True,
            )
            self._prewarm_thread.start()

    def segment(
        self,
        *,
        image_paths: list[Path],
        mask_paths: list[Path],
        prompts: list[dict[str, Any]],
        timeout_seconds: float = 180.0,
    ) -> dict[str, Any]:
        self.ensure_running()
        body = json.dumps(
            {
                "image_paths": [str(Path(value).resolve()) for value in image_paths],
                "mask_paths": [str(Path(value).resolve()) for value in mask_paths],
                "prompts": prompts,
            }
        ).encode("utf-8")
        request = Request(
            self.base_url + "/segment",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener.open(
                request, timeout=float(timeout_seconds)
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SAM2 Tiny worker rejected segmentation: {detail}") from error
        if payload.get("format") != WORKER_FORMAT or payload.get("passed") is not True:
            raise RuntimeError(f"invalid SAM2 Tiny worker result: {payload}")
        return payload

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
