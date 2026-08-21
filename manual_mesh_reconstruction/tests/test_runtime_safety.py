from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.request import ProxyHandler

from manual_mesh_reconstruction.sam2_worker_client import (
    Sam2TinyVideoWorkerClient,
    WORKER_FORMAT,
)
from trellis_point_prior_mv import server as transport_server


class RuntimeSafetyTests(unittest.TestCase):
    def test_sam2_local_health_never_uses_environment_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://127.0.0.1:1",
                "HTTPS_PROXY": "http://127.0.0.1:1",
                "NO_PROXY": "",
                "no_proxy": "",
            },
            clear=False,
        ):
            client = Sam2TinyVideoWorkerClient(
                host="127.0.0.1",
                port=5091,
                log_path=Path(temporary) / "sam2.log",
            )
            proxy_handlers = [
                handler
                for handler in client._opener.handlers
                if isinstance(handler, ProxyHandler)
            ]
            # urllib omits an empty ProxyHandler from the final handler list;
            # the important invariant is that no environment-derived proxy
            # handler is installed at all.
            self.assertEqual(proxy_handlers, [])

            response = MagicMock()
            response.__enter__.return_value = response
            response.read.return_value = json.dumps(
                {
                    "format": WORKER_FORMAT,
                    "status": "ok",
                    "model_loaded": True,
                }
            ).encode("utf-8")
            client._opener.open = MagicMock(return_value=response)
            try:
                health = client.health(timeout=2.0)
            finally:
                client.close()
            self.assertIsNotNone(health)
            self.assertTrue(health["model_loaded"])
            client._opener.open.assert_called_once_with(
                "http://127.0.0.1:5091/health", timeout=2.0
            )

    def test_current_session_json_is_atomically_published(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            flag = Path(temporary) / "flags/current_session.json"
            transport_server._atomic_write_json(
                str(flag), {"session_id": "session_0000", "ordinal": 0}
            )
            writer_done = threading.Event()
            failures: list[Exception] = []

            def writer() -> None:
                try:
                    for ordinal in range(1, 250):
                        transport_server._atomic_write_json(
                            str(flag),
                            {
                                "session_id": f"session_{ordinal:04d}",
                                "ordinal": ordinal,
                            },
                        )
                except Exception as error:  # pragma: no cover - assertion path
                    failures.append(error)
                finally:
                    writer_done.set()

            worker = threading.Thread(target=writer)
            worker.start()
            reads = 0
            while not writer_done.is_set() or reads < 500:
                try:
                    payload = json.loads(flag.read_text(encoding="utf-8"))
                    self.assertIn("session_id", payload)
                    self.assertIn("ordinal", payload)
                except Exception as error:  # pragma: no cover - assertion path
                    failures.append(error)
                    break
                reads += 1
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
