from __future__ import annotations

import argparse
import re
from pathlib import Path


TRAIN_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iters>\d+) train_loss:(?P<train_loss>[0-9.]+) "
    r"train_time:(?P<train_time_ms>[0-9.]+)ms step_avg:(?P<step_avg_ms>[0-9.]+)ms"
)
VAL_RE = re.compile(
    r"step:(?P<step>\d+)/(?P<iters>\d+) val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+) "
    r"train_time:(?P<train_time_ms>[0-9.]+)ms step_avg:(?P<step_avg_ms>[0-9.]+)ms"
)
FINAL_RE = re.compile(
    r"final_int8_zlib_roundtrip_exact val_loss:(?P<val_loss>[0-9.]+) val_bpb:(?P<val_bpb>[0-9.]+)"
)
SIZE_RE = re.compile(r"Total submission size int8\+zlib: (?P<bytes_total>\d+) bytes")


def latest_match(pattern: re.Pattern[str], lines: list[str]) -> dict[str, str] | None:
    for line in reversed(lines):
        match = pattern.search(line)
        if match is not None:
            return match.groupdict()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path")
    args = parser.parse_args()

    path = Path(args.log_path).expanduser().resolve()
    lines = path.read_text(encoding="utf-8").splitlines()

    train = latest_match(TRAIN_RE, lines)
    val = latest_match(VAL_RE, lines)
    final = latest_match(FINAL_RE, lines)
    size = latest_match(SIZE_RE, lines)

    print(f"log={path}")
    if train is not None:
        print(
            "latest_train "
            f"step={train['step']}/{train['iters']} "
            f"loss={train['train_loss']} step_avg_ms={train['step_avg_ms']} "
            f"train_time_ms={train['train_time_ms']}"
        )
    if val is not None:
        print(
            "latest_val "
            f"step={val['step']}/{val['iters']} "
            f"loss={val['val_loss']} bpb={val['val_bpb']} "
            f"step_avg_ms={val['step_avg_ms']} train_time_ms={val['train_time_ms']}"
        )
    if final is not None:
        print(f"final_int8 val_loss={final['val_loss']} val_bpb={final['val_bpb']}")
    if size is not None:
        print(f"artifact_bytes_total={size['bytes_total']}")
    if train is None and val is None and final is None:
        print("no_step_metrics_found")


if __name__ == "__main__":
    main()
