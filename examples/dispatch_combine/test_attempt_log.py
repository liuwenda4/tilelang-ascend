import json

import torch

from attempt_log import AttemptLogger, tensor_digest


def test_attempt_logger_records_events_and_digests(tmp_path):
    logger = AttemptLogger.create(tmp_path, attempt_id="unit-test")
    logger.event("kernel_finished", rank=1, generation=3, actual_count=4, timeout=False)
    logger.write_json("summary.json", {"status": "PASS"})
    records = [json.loads(line) for line in logger.events_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["attempt_started", "kernel_finished"]
    assert records[1]["generation"] == 3
    assert (logger.root / "summary.json").exists()
    assert len(tensor_digest(torch.arange(4))) == 64
