"""Out-of-process policy server + remote policy client.

Why split the policy out of the control process at all?

* The control loop must never be blocked by a 450M-parameter forward pass. With a
  server, the arm keeps a fixed rate and the runtime decides what to do when
  inference is late (the safety layer holds position).
* One GPU host can serve several arms / several evaluation workers. Requests are
  serialised on a lock because GPU inference is not re-entrant, which is the real
  constraint and is stated explicitly instead of being discovered at runtime.
* It creates a clean seam for measuring **round-trip** latency: network + queue +
  inference, which is what actually limits control frequency on a real robot.

Transport is ``multiprocessing.connection`` (stdlib): length-prefixed pickles over
TCP with an HMAC handshake. That is appropriate for a trusted lab network. For
deployment outside a lab, swap the transport for gRPC/ZMQ -- the seam is one file.

Security note: pickle deserialises code. Only connect to servers you trust and
never expose this port to an untrusted network.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from evla.types import Observation, Policy

DEFAULT_AUTHKEY = b"evla-lab"


@dataclass
class ServerStats:
    requests: int = 0
    infer_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        if not self.infer_ms:
            return {"requests": 0}
        arr = np.asarray(self.infer_ms)
        return {
            "requests": self.requests,
            "infer_p50_ms": round(float(np.percentile(arr, 50)), 2),
            "infer_p95_ms": round(float(np.percentile(arr, 95)), 2),
        }


class PolicyServer:
    """Serve ``predict_chunk`` over TCP for one policy."""

    def __init__(self, policy: Policy, host: str = "127.0.0.1", port: int = 5599,
                 authkey: bytes = DEFAULT_AUTHKEY) -> None:
        self.policy = policy
        self.host = host
        self.port = int(port)
        self.authkey = authkey
        self.stats = ServerStats()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._listener = None

    # ------------------------------------------------------------------ server
    def serve_forever(self, announce: bool = True) -> None:
        from multiprocessing.connection import Listener

        self._listener = Listener((self.host, self.port), authkey=self.authkey)
        if announce:
            print(f"[evla] policy server listening on {self.host}:{self.port} "
                  f"({self.policy.describe().get('name', 'policy')})", flush=True)
        while not self._stop.is_set():
            try:
                conn = self._listener.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn) -> None:
        with conn:
            while not self._stop.is_set():
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    return
                cmd = msg[0]
                if cmd == "ping":
                    conn.send(("ok", {"name": self.policy.name}))
                elif cmd == "describe":
                    conn.send(("ok", self.policy.describe()))
                elif cmd == "reset":
                    with self._lock:
                        self.policy.reset(msg[1] if len(msg) > 1 else "")
                    conn.send(("ok", None))
                elif cmd == "predict":
                    obs: Observation = msg[1]
                    t0 = time.perf_counter()
                    with self._lock:  # GPU inference is serialised on purpose
                        chunk = self.policy.predict_chunk(obs)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    self.stats.requests += 1
                    self.stats.infer_ms.append(dt_ms)
                    conn.send(("ok", np.asarray(chunk, np.float32), dt_ms))
                elif cmd == "shutdown":
                    self._stop.set()
                    return
                else:
                    conn.send(("error", f"unknown command {cmd!r}"))

    def shutdown(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass


class RemotePolicy(Policy):
    """Client-side proxy that satisfies the :class:`~evla.types.Policy` contract."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5599,
                 authkey: bytes = DEFAULT_AUTHKEY) -> None:
        from multiprocessing.connection import Client

        self.conn = Client((host, port), authkey=authkey)
        self.conn.send(("describe",))
        _, info = self.conn.recv()
        self.info = info
        self.name = f"remote:{info.get('name', 'policy')}"
        self.chunk_size = int(info.get("chunk_size", 1))
        self.action_dim = int(info.get("action_dim", 2))
        self.rtt_ms: list[float] = []
        self.server_ms: list[float] = []

    def reset(self, task: str = "") -> None:
        self.conn.send(("reset", task))
        self.conn.recv()

    def predict_chunk(self, obs: Observation) -> np.ndarray:
        t0 = time.perf_counter()
        self.conn.send(("predict", obs))
        status, chunk, dt_ms = self.conn.recv()
        self.rtt_ms.append((time.perf_counter() - t0) * 1000.0)
        self.server_ms.append(dt_ms)
        if status != "ok":
            raise RuntimeError(f"policy server error: {chunk}")
        return np.asarray(chunk, np.float32)

    def describe(self) -> dict:
        info = dict(self.info)
        info["transport"] = "multiprocessing.connection"
        if self.rtt_ms:
            arr = np.asarray(self.rtt_ms)
            info["rtt_ms"] = {
                "p50": round(float(np.percentile(arr, 50)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
                "max": round(float(arr.max()), 2),
            }
        return info

    def close(self) -> None:
        try:
            self.conn.send(("shutdown",))
            self.conn.close()
        except (OSError, EOFError):
            pass
