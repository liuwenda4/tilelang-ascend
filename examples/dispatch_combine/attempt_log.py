"""Append-only attempt telemetry for Dispatch/Combine experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Sequence

import torch


def tensor_digest(tensor: torch.Tensor) -> str:
    """Return a stable digest after the caller has synchronized the device."""
    value = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def environment_snapshot() -> str:
    keys = (
        "HOSTNAME",
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "NPU_MEMORY_FRACTION",
        "HCCL_EXEC_TIMEOUT",
    )
    lines = [
        f"python={sys.version}",
        f"platform={platform.platform()}",
        f"machine={platform.machine()}",
        f"cwd={Path.cwd()}",
    ]
    lines.extend(f"{key}={os.environ.get(key, '')}" for key in keys)
    return "\n".join(lines) + "\n"


@dataclass
class AttemptLogger:
    root: Path
    attempt_id: str

    @classmethod
    def create(cls, base: str | Path = "artifacts/moe", attempt_id: str | None = None) -> "AttemptLogger":
        now = datetime.now(timezone.utc)
        name = attempt_id or f"{now.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
        root = Path(base) / name
        root.mkdir(parents=True, exist_ok=True)
        logger = cls(root=root, attempt_id=name)
        logger.write_text("environment.txt", environment_snapshot())
        logger.event("attempt_started", pid=os.getpid())
        return logger

    @property
    def events_path(self) -> Path:
        return self.root / "events.jsonl"

    def event(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "attempt_id": self.attempt_id,
            "pid": os.getpid(),
            "event": event,
            **fields,
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def command(self, argv: Sequence[str], *, cwd: str | Path | None = None, timeout: int = 120) -> int:
        """Run and capture a command without hiding stdout/stderr from the log."""
        command = [str(value) for value in argv]
        self.event("command_started", argv=command, cwd=str(cwd or Path.cwd()))
        try:
            result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            self.event("command_timeout", argv=command, timeout_seconds=timeout, stdout=exc.stdout, stderr=exc.stderr)
            return 124
        self.event(
            "command_finished",
            argv=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
        return result.returncode

    def rank_event(self, rank: int, event: str, **fields: Any) -> None:
        self.event(event, rank=rank, **fields)

    def write_json(self, name: str, value: Any) -> None:
        (self.root / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    def write_text(self, name: str, value: str) -> None:
        (self.root / name).write_text(value, encoding="utf-8")
